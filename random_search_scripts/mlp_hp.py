#pip install -r requirements.txt # Keep this reminder if needed
import numpy as np
import pandas as pd
import h5py
import tqdm
import math #
import gc # Garbage collector
import os # For creating directories
import sys 
import json # For saving results
import time # For timestamping runs
import datetime # For timer formatting
import random # For seeding
from scipy.stats import loguniform, uniform # For hyperparam sampling

import torch
import torch.nn as nn
import torch.nn.functional as F # for activation functions like SiLU/ReLU
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR # For warm up LR

from sklearn.model_selection import KFold, ParameterSampler # For CV and Random Search
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix

# --- Start Timer ---
script_start_time = time.time()

def aggressive_cleanup():
    """More aggressive GPU memory cleanup"""
    # Clear all gradients explicitly
    torch.cuda.empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Make sure CUDA operations are complete

# --- Dataset Class (Unchanged) ---
class Embedding_retriever(Dataset):
    def __init__(self, h5_path, protein_keys = None):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            all_keys = list(f['embeddings_folder'].keys())
            all_keys.sort()
            self.protein_keys = protein_keys if protein_keys is not None else all_keys

    def __len__(self):
        return len(self.protein_keys)

    def __getitem__(self, index):
        protein_key = self.protein_keys[index]
        with h5py.File(self.h5_path, 'r') as f:
            protein_group = f['embeddings_folder'][protein_key]
            embeddings = protein_group['embeddings'][:]
            labels = protein_group['labels'][:]
            labels = labels.astype(np.float32)

        return {
            'name': protein_key,
            'embeddings': torch.tensor(embeddings, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'length': len(labels)
        }

# --- Padding Collate Function (Unchanged) ---
def collate_fn(batch):
    embeddings = [item['embeddings'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]
    names = [item['name'] for item in batch]

    padded_embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value= 0.0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1) # -1 indicates padding

    max_len = padded_embeddings.size(1)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    # Mask is True for padded positions, False for actual data
    padding_mask = torch.arange(max_len)[None, :] >= lengths_tensor[:, None]

    return {
        'names': names,
        'embeddings': padded_embeddings, # Shape: [B, SeqLen, EmbDim]
        'labels': padded_labels,         # Shape: [B, SeqLen]
        'padding_mask': padding_mask,    # Shape: [B, SeqLen] - True where padded
        'lengths': lengths_tensor        # Shape: [B]
    }


# --- MLP Model ---
class EpitopeMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dims, dropout=0.1, activation_fn=nn.SiLU, use_seq_length=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout
        self.activation_fn = activation_fn
        self.use_seq_length = use_seq_length

        layers = []
        input_dim = self.embed_dim +1 if self.use_seq_length else self.embed_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim)) # LayerNorm 
            layers.append(self.activation_fn())
            layers.append(nn.Dropout(self.dropout_p))
            input_dim = h_dim # Set input dim for the next layer

        # Final output layer
        layers.append(nn.Linear(input_dim, 1))

        self.network = nn.Sequential(*layers)
        self.init_weights()

    def init_weights(self):
        # Initialize weights for linear layers
        for module in self.network:
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src_embeddings, seq_lengths=None): # Added seq_lengths
        """
        Forward pass for the MLP.
        Args:
            src_embeddings (Tensor): Input embeddings [Batch, SeqLen, EmbedDim].
            seq_lengths (Tensor, optional): Sequence lengths [Batch]. Required if use_seq_length=True.
        Returns:
            Tensor: Output logits [Batch, SeqLen, 1].
        """
        if self.use_seq_length:
            if seq_lengths is None:
                raise ValueError("seq_lengths must be provided when use_seq_length is True")

            batch_size, seq_len, _ = src_embeddings.shape
            # Normalize lengths (using a predefined MAX_LEN or calculated max)
            # Using a large constant avoids recalculating max, but might compress values
            MAX_LEN_NORM = 5000.0 # Or calculate from data
            norm_seq_lengths = seq_lengths.float().to(src_embeddings.device) / MAX_LEN_NORM
            # Clamp values between 0 and 1 just in case lengths exceed MAX_LEN_NORM
            norm_seq_lengths = torch.clamp(norm_seq_lengths, 0.0, 1.0)


            # Expand sequence length to match embedding dimensions
            # Shape: [B] -> [B, 1, 1] -> [B, SeqLen, 1]
            expanded_seq_lengths = norm_seq_lengths.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)

            # Concatenate expanded sequence lengths with embeddings
            # Shape: [B, SeqLen, EmbDim + 1]
            mlp_input = torch.cat([src_embeddings, expanded_seq_lengths], dim=2)
        else:
            mlp_input = src_embeddings

        logits = self.network(mlp_input)
        return logits
    
# --- Training Function ---
def train_epoch(model, dataloader, optimizer, criterion, device, epoch, trial_num, n_iterations):
    model.train()
    total_loss = 0
    num_batches_processed = 0 
    progress_bar = tqdm.tqdm(dataloader, desc=f'Trial {trial_num}/{n_iterations} Epoch {epoch+1} Training', leave=False, ncols=100)
    for batch in progress_bar:
        if batch is None:
            print(f"Skipping empty batch in epoch {epoch+1} training.")
            continue

        embeddings = batch['embeddings'].to(device) # Shape: [B, SeqLen, EmbDim]
        labels = batch['labels'].to(device)         # Shape: [B, SeqLen], contains -1 for padding
        lengths = batch['lengths'].to(device) # Extract lengths


        optimizer.zero_grad()

        # --- Model Forward Pass ---
        outputs = model(embeddings, seq_lengths=lengths)
        outputs = outputs.squeeze(-1)

        # --- Loss Calculation  ---
        active_mask = (labels != -1) # Mask is True for non-padded positions
        if active_mask.sum() == 0:
            continue # Skip batch if no active labels (e.g., all padding)

        active_logits = outputs[active_mask] # Select logits corresponding to actual residues
        active_labels = labels[active_mask] # Select labels corresponding to actual residues

        if active_labels.numel() == 0: # Double check after masking
            continue

        loss = criterion(active_logits, active_labels)

        # --- Gradient Handling ---
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss encountered during training epoch {epoch+1}. Skipping batch gradient update.")
            optimizer.zero_grad() # Prevent propagation of invalid gradients
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Gradient clipping
        optimizer.step()

        total_loss += loss.item()
        num_batches_processed += 1
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
    return avg_loss


# --- Evaluation Function ---
def evaluate(model, dataloader, criterion, device, epoch, threshold, trial_num, n_iterations):
    model.eval()
    total_loss = 0
    all_preds_prob = []
    all_labels_list = []
    num_batches_processed = 0 # Renamed for clarity

    with torch.no_grad():
        progress_bar = tqdm.tqdm(dataloader, desc=f"Trial {trial_num}/{n_iterations} Epoch {epoch+1} Evaluating", leave=False, ncols=100)
        for batch in progress_bar:
            embeddings = batch['embeddings'].to(device) # Shape: [B, SeqLen, EmbDim]
            labels = batch['labels'].to(device)         # Shape: [B, SeqLen]
            lengths = batch['lengths'].to(device)

            # --- Model Forward Pass ---
            outputs = model(embeddings, seq_lengths=lengths).squeeze(-1) # Shape: [B, SeqLen]

            # --- Loss and Metrics Calculation (only on non-padded positions) ---
            active_mask = (labels != -1)
            if active_mask.sum() == 0:
                continue # Skip if no active labels

            active_logits = outputs[active_mask]
            active_labels = labels[active_mask]

            if active_labels.numel() == 0:
                continue # Skip if masking resulted in zero elements

            loss = criterion(active_logits, active_labels)

            # Check for NaN/inf loss during validation
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches_processed += 1

                # Store predictions and labels for metrics
                probs = torch.sigmoid(active_logits).cpu().numpy()
                all_preds_prob.extend(probs)
                all_labels_list.extend(active_labels.cpu().numpy())
            else:
                 print(f"Warning: NaN or Inf loss encountered during evaluation epoch {epoch+1}. Skipping batch metrics.")

    # --- Metric Calculation ---
    avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
    precision, recall, f1, auc_roc, auc_pr, auc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cm = np.zeros((2, 2), dtype=int) # Default confusion matrix

    if len(all_labels_list) > 0 and len(all_preds_prob) > 0:
        all_labels_np = np.array(all_labels_list).astype(int)
        all_preds_prob_np = np.array(all_preds_prob)

        # Ensure probabilities are valid before metric calculation
        if np.any(np.isnan(all_preds_prob_np)) or np.any(np.isinf(all_preds_prob_np)):
            print(f"Warning: NaNs or Infs found in predicted probabilities for epoch {epoch+1}. Metrics calculation skipped.")
        # Ensure there's variation in labels for AUC calculations
        elif len(np.unique(all_labels_np)) < 2:
             print(f"Warning: Only one class ({np.unique(all_labels_np)}) present in TRUE labels for epoch {epoch+1}. AUC ROC/AUC10 cannot be computed reliably.")
             # Calculate other metrics that might still work
             all_preds_binary = (all_preds_prob_np >= threshold).astype(int)
             precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds_binary, average='binary', zero_division=0, labels=[0, 1])
             try: # AUC PR might still work if probabilities vary
                auc_pr = average_precision_score(all_labels_np, all_preds_prob_np)
             except ValueError:
                auc_pr = 0.0 # Set to 0 if it fails
             auc_roc = 0.0 # Cannot calculate ROC with single class
             auc10 = 0.0 # Cannot calculate AUC10 with single class
             try:
                 cm = confusion_matrix(all_labels_np, all_preds_binary, labels=[0, 1])
             except ValueError:
                 print(f"Warning: Could not compute confusion matrix for epoch {epoch+1} (likely due to single class). Using default zero matrix.")
        else:
            # Calculate binary predictions using threshold
            all_preds_binary = (all_preds_prob_np >= threshold).astype(int)

            # Calculate Confusion Matrix
            try:
                cm = confusion_matrix(all_labels_np, all_preds_binary, labels=[0, 1])
            except ValueError as e:
                print(f"Warning: Could not compute confusion matrix for epoch {epoch+1}: {e}. Using default zero matrix.")

            # Calculate other metrics
            precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds_binary, average='binary', zero_division=0, labels=[0, 1])
            try:
                auc_pr = average_precision_score(all_labels_np, all_preds_prob_np)
            except ValueError as e:
                 print(f"Warning: ValueError during AUC-PR calculation for epoch {epoch+1}: {e}")
                 auc_pr = 0.0
            try:
                auc_roc = roc_auc_score(all_labels_np, all_preds_prob_np)
                # Calculate AUC10 
                try:
                    auc10 = roc_auc_score(all_labels_np, all_preds_prob_np, max_fpr=0.1)
                except ValueError:
                    print(f"Warning: ValueError during AUC10 calculation (max_fpr=0.1) for epoch {epoch+1}. Setting AUC10 to 0.0.")
                    auc10 = 0.0 # Set to 0 if AUC10 calculation fails
            except ValueError as e:
                print(f"Warning: ValueError during AUC-ROC calculation for epoch {epoch+1}: {e}")
                auc_roc = 0.0
                auc10 = 0.0 # Also set AUC10 to 0 if AUC-ROC fails
    else:
        print(f"Warning: No valid predictions/labels collected for evaluation epoch {epoch+1}. Metrics calculation skipped.")


    # --- Print Evaluation Results ---
    print(f"Epoch {epoch+1} Eval Results (Trial {trial_num}/{n_iterations}):")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, AUC10: {auc10:.4f}")
    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Threshold used for P/R/F1/CM: {threshold}")
    print("\n  Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"        Predicted 0    Predicted 1")
    print(f"Actual 0 | {cm[0, 0]:<12} | {cm[0, 1]:<12} | (TN={cm[0, 0]}, FP={cm[0, 1]})")
    print(f"Actual 1 | {cm[1, 0]:<12} | {cm[1, 1]:<12} | (FN={cm[1, 0]}, TP={cm[1, 1]})")
    print("-" * 30) # Separator

    return avg_loss, precision, recall, f1, auc_roc, auc_pr, auc10


# --- Configuration ---
# Ask for GPU and Threshold upfront
target_gpu = -1
while target_gpu not in [0, 1, 2, 3]: # Assuming GPUs 0-3 exist
    try:
        target_gpu = int(input("Which GPU to use? [0, 1, 2, 3]: "))
        if target_gpu not in [0, 1, 2, 3]:
            print("Invalid GPU index.")
    except ValueError:
        print("Please enter a number.")

threshold = -1.0
while not (0.0 < threshold < 1.0):
    try:
        threshold = float(input("Enter prediction threshold value (e.g., 0.5): "))
        if not (0.0 < threshold < 1.0):
            print("Threshold must be between 0 and 1 (exclusive).")
    except ValueError:
        print("Please enter a valid number.")

# MAPPING (Simplified for this example, assuming paths are known)
EMBEDDING_CONFIG = {
    'esm2': {
        'h5_path': 'esm2_protein_embeddings.h5', # Adjust path as needed
        'embed_dim': 1280,
    },
    'esmc': {
        'h5_path': 'esmc_protein_embeddings.h5', # Adjust path as needed
        'embed_dim': 960,
    }
}

# Ask the user for embedding type
EMBEDDING_TYPE = ""
print("-----------------------------------------------------")
while EMBEDDING_TYPE not in EMBEDDING_CONFIG:
    EMBEDDING_TYPE = input(f"Select embedding type ({'/'.join(EMBEDDING_CONFIG.keys())}): ").lower().strip()
    if EMBEDDING_TYPE not in EMBEDDING_CONFIG:
        print("Invalid choice. Please try again.")
print("-----------------------------------------------------")

# Get config based on user choice
config = EMBEDDING_CONFIG[EMBEDDING_TYPE]
H5_FILE_PATH = config['h5_path']
EMBED_DIM = config['embed_dim']
# target_gpu is already defined from user input

# Paths and Directories Setup
run_timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M")
BASE_RUNS_DIR = f'{EMBEDDING_TYPE}_MLP_runs_{run_timestamp}' # Clearly label as MLP runs
RUN_TYPE_DIR = os.path.join(BASE_RUNS_DIR, EMBEDDING_TYPE)
MODEL_SAVE_DIR = os.path.join(RUN_TYPE_DIR, 'saved_models')
TENSORBOARD_LOG_DIR = os.path.join(RUN_TYPE_DIR, 'tensorboard_logs')
RESULTS_FILE = os.path.join(RUN_TYPE_DIR, f'{EMBEDDING_TYPE}_mlp_hyperparam_search_results.json')

os.makedirs(RUN_TYPE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
print(f"Created directories. Models will be saved in: {MODEL_SAVE_DIR}")
print(f"TensorBoard logs will be saved in: {TENSORBOARD_LOG_DIR}")
print(f"Results JSON will be saved at: {RESULTS_FILE}")

# Fixed Training Settings
N_SPLITS = 4
N_EPOCHS = 20 # Maybe MLP trains faster/slower? Adjust if needed
BATCH_SIZE = 16 # MLP might tolerate larger batches
RANDOM_SEED = 21
EARLY_STOPPING_PATIENCE = 4 # Adjust patience if needed
N_WARMUP_EPOCHS = 3

# --- MLP Random Search Space ---
param_dist = {
    'learning_rate': loguniform(1e-5, 1e-3), # MLP might prefer higher LR
    'dropout': uniform(0.1, 0.4),    # Dropout range 
    'hidden_dims_config': ['small', 'medium', 'large'], # Sample configurations
    'weight_decay': loguniform(1e-4, 1e-1),
    'activation_fn': [nn.ReLU, nn.SiLU] # Sample activation function
}
# Define hidden layer configurations corresponding to the sampled strings
HIDDEN_CONFIGS = {
    'small': [EMBED_DIM // 2, EMBED_DIM // 4],          # Example: [640, 320] for ESM2
    'medium': [EMBED_DIM, EMBED_DIM // 2],            # Example: [1280, 640] for ESM2
    'large': [EMBED_DIM * 2, EMBED_DIM, EMBED_DIM//2] # Example: [2560, 1280, 640]
}

N_SEARCH_ITERATIONS = 30 # Number of random combinations to try

# --- Device Setup ---
device_str = "cpu" # Default to CPU
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Found {num_gpus} GPU(s).")
    if target_gpu < num_gpus:
        device_str = f"cuda:{target_gpu}"
    elif num_gpus > 0:
        print(f"Warning: Target GPU {target_gpu} not available ({num_gpus} GPUs found). Using cuda:0.")
        device_str = "cuda:0"
    else:
        print("Warning: CUDA reports available but no devices found? Using CPU.")
else:
    print("CUDA not available. Using CPU.")

DEVICE = torch.device(device_str)

# --- Print Configuration Summary ---
print("-----------------------------------------------------")
print(f"--- Running MLP Experiment for: {EMBEDDING_TYPE} ---")
print(f"Using device: {DEVICE}")
print(f"Embeddings Path: {H5_FILE_PATH}")
print(f"Embeddings Dimension: {EMBED_DIM}")
print(f"Output Directory: {RUN_TYPE_DIR}")
print(f"Prediction Threshold: {threshold}")
print(f"Using Model: EpitopeMLP")
print("-----------------------------------------------------")

# --- SEEDING ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# --- Load Protein Keys ---
try:
    with h5py.File(H5_FILE_PATH, 'r') as f:
        if 'embeddings_folder' not in f:
             raise KeyError("'embeddings_folder' not found in HDF5 file.")
        all_protein_keys = list(f['embeddings_folder'].keys())
        all_protein_keys.sort()
        all_protein_keys = np.array(all_protein_keys)
    print(f"Loaded {len(all_protein_keys)} protein keys from {H5_FILE_PATH}")
except FileNotFoundError:
    print(f"Error: HDF5 file not found at {H5_FILE_PATH}")
    exit()
except KeyError as e:
    print(f"Error loading keys from HDF5 file: {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading keys: {e}")
    exit()

# --- Hyperparameter Search Initialization ---
sampler = ParameterSampler(param_dist, n_iter=N_SEARCH_ITERATIONS, random_state=RANDOM_SEED)
all_trial_results = []
best_overall_avg_auc_pr = -1 # Track best average AUC-PR across all trials
best_trial_config = None

print(f"\n--- Starting MLP Hyperparameter Search ({N_SEARCH_ITERATIONS} trials) for {EMBEDDING_TYPE} ---")

# --- Setup Main TensorBoard Writer ---
main_tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
print(f"TensorBoard HParam logs will be stored in: {TENSORBOARD_LOG_DIR}")

# --- Trial Loop ---
for trial_num, params in enumerate(sampler, 1):
    aggressive_cleanup()
    print(f"\n----- Trial {trial_num}/{N_SEARCH_ITERATIONS} ({EMBEDDING_TYPE} - MLP) -----")

    # Extract params for this trial
    LEARNING_RATE = params['learning_rate']
    DROPOUT = params['dropout']
    HIDDEN_DIMS_CONFIG_KEY = params['hidden_dims_config']
    HIDDEN_DIMS = HIDDEN_CONFIGS[HIDDEN_DIMS_CONFIG_KEY] # Get the actual list of hidden dims
    WEIGHT_DECAY = params['weight_decay']
    ACTIVATION_FN = params['activation_fn']

    # Log the chosen parameters clearly
    print(f"Parameters:")
    print(f"  Learning Rate: {LEARNING_RATE:.2e}")
    print(f"  Dropout: {DROPOUT:.3f}")
    print(f"  Hidden Dims Key: '{HIDDEN_DIMS_CONFIG_KEY}' -> {HIDDEN_DIMS}")
    print(f"  Weight Decay: {WEIGHT_DECAY:.2e}")
    print(f"  Activation: {ACTIVATION_FN.__name__}") # Print activation function name

    # K-Fold Cross-Validation Loop
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = [] # Store metrics for each fold in this trial

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_protein_keys)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        set_seed(RANDOM_SEED + fold) # Ensure reproducibility within the fold

        train_keys = all_protein_keys[train_idx]
        val_keys = all_protein_keys[val_idx]

        # Create datasets and dataloaders
        train_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=list(train_keys))
        val_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=list(val_keys))
        # Consider using more workers if I/O is a bottleneck
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)

        # Calculate pos_weight for class imbalance
        print("Calculating pos_weight for imbalance...")
        num_pos, num_neg = 0, 0
        # Use a temporary loader for efficiency if dataset is large
        temp_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE*2, collate_fn=collate_fn, num_workers=2)
        for batch in tqdm.tqdm(temp_loader, desc="Calculating pos_weight", leave=False, ncols=80):
            labels = batch['labels'] # Labels are already tensors
            active_mask = (labels != -1)
            active_labels = labels[active_mask].numpy() # Calculate on active labels only
            num_pos += np.sum(active_labels == 1)
            num_neg += np.sum(active_labels == 0)
        del temp_loader # Free memory

        if num_pos > 0 and num_neg > 0:
            pos_weight = num_neg / num_pos
        else:
            print(f"Warning: Fold {fold+1} - num_pos={num_pos}, num_neg={num_neg}. Using pos_weight=1.0")
            pos_weight = 1.0 # Default if one class is missing in training fold
        print(f"Fold {fold+1} - pos_weight: {pos_weight:.2f}")
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # --- Instantiate NEW MLP model and optimizer for the fold ---
        model = EpitopeMLP(
            embed_dim=EMBED_DIM,
            hidden_dims=HIDDEN_DIMS,
            dropout=DROPOUT,
            activation_fn=ACTIVATION_FN,
            use_seq_length=True
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Learning Rate Scheduler (Warmup + Cosine Decay)
        scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=N_WARMUP_EPOCHS)
        scheduler_decay = CosineAnnealingLR(optimizer, T_max=(N_EPOCHS - N_WARMUP_EPOCHS), eta_min=LEARNING_RATE * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[N_WARMUP_EPOCHS])

        # Tracking best performance within the fold
        best_val_f1 = -1
        best_fold_metrics = {}
        epochs_no_improve = 0

        # Epoch Loop for the Fold
        print(f"Starting training for max {N_EPOCHS} epochs (Warmup: {N_WARMUP_EPOCHS}, Patience: {EARLY_STOPPING_PATIENCE})...")
        for epoch in range(N_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch, trial_num, N_SEARCH_ITERATIONS)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{N_EPOCHS} - Train Loss: {train_loss:.4f} | Current LR: {current_lr:.2e}")

            # Evaluate on validation set
            val_loss, precision, recall, f1, auc_roc, auc_pr, auc10 = evaluate(model, val_loader, criterion, DEVICE, epoch, threshold, trial_num, N_SEARCH_ITERATIONS)

            # --- TensorBoard Logging for this epoch (within fold) ---
            # Create a unique identifier for this fold's run in TensorBoard
            fold_run_name = f'trial_{trial_num}/fold_{fold+1}'
            main_tb_writer.add_scalar(f'{fold_run_name}/Loss/Train', train_loss, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/Loss/Val', val_loss, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/Metrics/Val_AUC_PR', auc_pr, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/Metrics/Val_AUC_ROC', auc_roc, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/Metrics/Val_AUC10', auc10, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/Metrics/Val_F1', f1, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/Metrics/Val_Precision', precision, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/Metrics/Val_Recall', recall, epoch)
            main_tb_writer.add_scalar(f'{fold_run_name}/LearningRate', current_lr, epoch)


            # --- Early Stopping Logic & Model Saving ---
            if f1 > best_val_f1:
                print(f"Epoch {epoch+1}: Val f1 improved ({best_val_f1:.4f} -> {f1:.4f}). Saving model & resetting patience.")
                best_val_f1 = f1
                best_fold_metrics = {
                    'loss': val_loss, 'precision': precision, 'recall': recall,
                    'f1': f1, 'auc_roc': auc_roc, 'auc_pr': auc_pr, 'auc10': auc10,
                    'epoch': epoch + 1 # Record the best epoch
                }
                # Save the best model state for this fold
                model_save_path = os.path.join(MODEL_SAVE_DIR, f'{EMBEDDING_TYPE}_mlp_trial_{trial_num}_fold_{fold+1}_best.pth')
                try:
                    torch.save(model.state_dict(), model_save_path)
                    # print(f"Model saved to {model_save_path}") # Optional: uncomment for verbose saving confirmation
                except Exception as e:
                    print(f"Error saving model for trial {trial_num}, fold {fold+1}: {e}")
                epochs_no_improve = 0 # Reset patience
            else:
                epochs_no_improve += 1
                print(f"Epoch {epoch+1}: Val F1 ({f1:.4f}) did not improve from best ({best_val_f1:.4f}). Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"\n--- Early stopping triggered for Fold {fold+1} at epoch {epoch+1} ---")
                    break # Exit the epoch loop for this fold

            scheduler.step() # Update learning rate

        # --- End of Epoch Loop ---

        if best_fold_metrics: # Check if any improvement was found
            print(f"\nBest Validation Metrics for Trial {trial_num}, Fold {fold + 1} (Found at Epoch {best_fold_metrics.get('epoch', 'N/A')}):")
            # Print metrics clearly
            print(f"  Loss: {best_fold_metrics['loss']:.4f} | AUC-PR: {best_fold_metrics['auc_pr']:.4f} | AUC-ROC: {best_fold_metrics['auc_roc']:.4f} | AUC10: {best_fold_metrics['auc10']:.4f}")
            print(f"  F1: {best_fold_metrics['f1']:.4f} | Precision: {best_fold_metrics['precision']:.4f} | Recall: {best_fold_metrics['recall']:.4f}")
            fold_results.append(best_fold_metrics)
        else:
            print(f"\nFold {fold+1} completed, but no improvement detected based on AUC-PR. No best model saved for this fold.")

        # Cleanup for the fold
# Replace with:
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None  # Clear gradients explicitly

        del train_loader, val_loader  # DataLoaders first
        del train_dataset, val_dataset  # Then datasets
        del criterion, pos_weight_tensor  # Small objects
        del optimizer, scheduler  # Optimizer and scheduler
        del model  # Model last

        aggressive_cleanup()  # Call our new function

    # --- End of Fold Loop ---

    # --- Summarize Fold Results & Log HParams for the Trial ---
    trial_avg_metrics = {}
    trial_std_metrics = {}
    if fold_results: # Check if any folds completed successfully and yielded metrics
        metric_keys = [k for k in fold_results[0].keys() if k != 'epoch'] # Exclude 'epoch' from averaging

        trial_avg_metrics = {key: np.mean([fold[key] for fold in fold_results]) for key in metric_keys}
        trial_std_metrics = {key: np.std([fold[key] for fold in fold_results]) for key in metric_keys}

        print(f"\n----- Trial {trial_num} ({EMBEDDING_TYPE} - MLP) Cross-Validation Summary -----")
        print("Average Metrics across folds:")
        for key, value in trial_avg_metrics.items():
            print(f"  Avg {key.upper()}: {value:.4f} (+/- {trial_std_metrics[key]:.4f})")

        # --- Log Hyperparameters and Aggregated Metrics to TensorBoard ---
        # Prepare hparams dict (need to handle non-scalar types like list/class)
        hparams_for_log = {
            'learning_rate': LEARNING_RATE,
            'dropout': DROPOUT,
            'hidden_dims_config': HIDDEN_DIMS_CONFIG_KEY, # Log the key
             'weight_decay': WEIGHT_DECAY,
             'activation_fn': ACTIVATION_FN.__name__ # Log the name
        }
        # Add hidden_dims as string for logging if needed: hparams_for_log['hidden_dims'] = str(HIDDEN_DIMS)

        # Prepare metrics dict
        metrics_for_log = {f'hparam/avg_{key}': value for key, value in trial_avg_metrics.items()}
        metrics_for_log.update({f'hparam/std_{key}': value for key, value in trial_std_metrics.items()})

        try:
            # Use add_hparams to link parameters and final metrics
            main_tb_writer.add_hparams(hparams_for_log, metrics_for_log, run_name=f'trial_{trial_num}_summary')
            print(f"Logged Trial {trial_num} HParam summary to TensorBoard.")
        except Exception as e:
            print(f"Error logging Trial {trial_num} HParams to TensorBoard: {e}")
            print("HParams dict:", hparams_for_log)
            print("Metrics dict:", metrics_for_log)


        # Update overall best trial if current one is better
        current_avg_auc_pr = trial_avg_metrics.get('auc_pr', -1)
        if current_avg_auc_pr > best_overall_avg_auc_pr:
            print(f"--- *** New Best Trial Found (Trial {trial_num}) *** ---")
            print(f"--- *** Average AUC-PR improved from {best_overall_avg_auc_pr:.4f} to {current_avg_auc_pr:.4f} *** ---")
            best_overall_avg_auc_pr = current_avg_auc_pr
            # Store the config and avg metrics of the best trial
            best_trial_config = {
                'trial_num': trial_num,
                'params': { # Log the actual params used
                    'learning_rate': LEARNING_RATE,
                    'dropout': DROPOUT,
                    'hidden_dims_config': HIDDEN_DIMS_CONFIG_KEY,
                    'hidden_dims': HIDDEN_DIMS, # Store the list too
                    'weight_decay': WEIGHT_DECAY,
                    'activation_fn': ACTIVATION_FN.__name__
                 },
                'avg_metrics': trial_avg_metrics
            }

        # Prepare summary for JSON logging
        trial_summary = {
            'trial_num': trial_num,
            'params': best_trial_config['params'] if trial_num == best_trial_config.get('trial_num') else {
                    'learning_rate': LEARNING_RATE, 'dropout': DROPOUT, 'hidden_dims_config': HIDDEN_DIMS_CONFIG_KEY,
                    'hidden_dims': HIDDEN_DIMS, 'weight_decay': WEIGHT_DECAY, 'activation_fn': ACTIVATION_FN.__name__},
            'status': 'completed',
            'avg_metrics': trial_avg_metrics,
            'std_metrics': trial_std_metrics,
            'individual_fold_metrics': fold_results # Keep individual fold results
        }

    else:
        # Handle case where no folds completed successfully for this trial
        print(f"----- Trial {trial_num} ({EMBEDDING_TYPE} - MLP) had no successful folds -----")
        trial_summary = {
            'trial_num': trial_num,
            'params': { # Log params even if failed
                 'learning_rate': LEARNING_RATE, 'dropout': DROPOUT, 'hidden_dims_config': HIDDEN_DIMS_CONFIG_KEY,
                 'hidden_dims': HIDDEN_DIMS, 'weight_decay': WEIGHT_DECAY, 'activation_fn': ACTIVATION_FN.__name__},
            'status': 'failed',
            'reason': 'No folds completed successfully or yielded improving metrics based on AUC-PR.',
            'avg_metrics': {}, 'std_metrics': {}, 'individual_fold_metrics': []
        }

    all_trial_results.append(trial_summary)

    # --- Save results incrementally after each trial ---
    try:
        # Helper function to convert numpy types for JSON
        def default_serializer(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                 return obj.tolist()
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.void)):
                return None
            # Add handling for PyTorch classes if they accidentally get in
            elif hasattr(obj, '__name__'): # Crude check for classes/functions
                return obj.__name__
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_trial_results, f, indent=4, default=default_serializer)
        print(f"Trial {trial_num} results updated in {RESULTS_FILE}")
    except IOError as e:
        print(f"Error saving results to {RESULTS_FILE}: {e}")
    except TypeError as e:
        print(f"Error serializing results to JSON for trial {trial_num}: {e}")
        # print("Last trial data:", all_trial_results[-1]) # Uncomment to debug the problematic data


# --- End of Trial Loop ---

# Close the main TensorBoard writer
main_tb_writer.close()

# --- Final Hyperparameter Search Summary ---
print(f"\n--- MLP Hyperparameter Search Complete for {EMBEDDING_TYPE} ---")
if best_trial_config:
    print("\nBest Overall Trial Found:")
    print(f"  Trial Number: {best_trial_config['trial_num']}")
    print(f"  Parameters: {best_trial_config['params']}")
    print("  Best Average Metrics:")
    for key, value in best_trial_config['avg_metrics'].items():
         print(f"    Avg {key.upper()}: {value:.4f}")
else:
     print("No trial resulted in an improvement based on average AUC-PR across folds.")


print(f"\nFull results saved in: {RESULTS_FILE}")
print(f"TensorBoard logs in: {TENSORBOARD_LOG_DIR}")
print(f"Best model checkpoints saved in: {MODEL_SAVE_DIR}")

# --- Calculate and Print Elapsed Time ---
script_end_time = time.time()
total_elapsed_seconds = script_end_time - script_start_time
elapsed_time_str = str(datetime.timedelta(seconds=int(total_elapsed_seconds))) # Format as H:M:S
print("\n-----------------------------------------------------")
print(f"MLP Script Finished.")
print(f"Total Execution Time: {elapsed_time_str}")
print("-----------------------------------------------------")