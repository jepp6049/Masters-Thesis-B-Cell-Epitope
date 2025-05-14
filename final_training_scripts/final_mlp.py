#pip install -r requirements.txt # Keep this reminder if needed
import numpy as np
import pandas as pd
import h5py
import tqdm
import math
import gc # Garbage collector
import os # For creating directories
import sys
import json # For saving/loading results
import time # For timestamping runs
import datetime # For timer formatting
import random # For seeding
# Removed scipy.stats imports as we load fixed params now

import torch
import torch.nn as nn
import torch.nn.functional as F # for activation functions like SiLU/ReLU
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset # Subset removed as we use train_test_split now
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR # For warm up LR

# Use train_test_split for a single train/val split
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix

# --- Start Timer ---
script_start_time = time.time()

# --- Dataset Class (Unchanged) ---
class Embedding_retriever(Dataset):
    def __init__(self, h5_path, protein_keys = None):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            all_keys = list(f['embeddings_folder'].keys())
            all_keys.sort()
            # Ensure protein_keys is None or a list/tuple
            if protein_keys is not None and not isinstance(protein_keys, (list, tuple, np.ndarray)):
                 raise TypeError("protein_keys must be a list, tuple, numpy array, or None")
            self.protein_keys = list(protein_keys) if protein_keys is not None else all_keys


    def __len__(self):
        return len(self.protein_keys)

    def __getitem__(self, index):
        protein_key = self.protein_keys[index]
        with h5py.File(self.h5_path, 'r') as f:
            try:
                protein_group = f['embeddings_folder'][protein_key]
                embeddings = protein_group['embeddings'][:]
                labels = protein_group['labels'][:]
                labels = labels.astype(np.float32)
            except KeyError:
                print(f"Error: Key '{protein_key}' not found in embeddings_folder.")
                # Return dummy data or raise error, depending on desired handling
                # Returning None might cause issues in collate_fn, better to raise
                raise KeyError(f"Protein key '{protein_key}' not found in HDF5 file {self.h5_path}")
            except Exception as e:
                print(f"Error reading data for key '{protein_key}': {e}")
                raise e

        return {
            'name': protein_key,
            'embeddings': torch.tensor(embeddings, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.float32),
            'length': len(labels)
        }


# --- Padding Collate Function (Unchanged) ---
def collate_fn(batch):
    # Filter out None items if __getitem__ could potentially return None
    # batch = [item for item in batch if item is not None]
    # if not batch: return None # Return None if batch becomes empty

    embeddings = [item['embeddings'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]
    names = [item['name'] for item in batch]

    try:
        padded_embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value=0.0)
        padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1) # -1 indicates padding
    except RuntimeError as e:
        print(f"Error during padding: {e}")
        print("Lengths in batch:", lengths)
        # Potentially print shapes of embeddings/labels for debugging
        # for i, emb in enumerate(embeddings): print(f"Emb {i} shape: {emb.shape}")
        # for i, lab in enumerate(labels): print(f"Lab {i} shape: {lab.shape}")
        raise e # Re-raise the error after printing info

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

# --- MLP Model (Unchanged) ---
class EpitopeMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dims, dropout=0.1, activation_fn=nn.SiLU, use_seq_length=True):
        super().__init__()
        self.embed_dim = embed_dim
        # Ensure hidden_dims is a list/tuple
        if not isinstance(hidden_dims, (list, tuple)):
             raise TypeError(f"hidden_dims must be a list or tuple, got {type(hidden_dims)}")
        self.hidden_dims = hidden_dims
        self.dropout_p = dropout
        # Ensure activation_fn is a class like nn.ReLU or nn.SiLU
        if not isinstance(activation_fn, type) or not issubclass(activation_fn, nn.Module):
             raise TypeError(f"activation_fn must be a torch.nn activation class (e.g., nn.ReLU), got {activation_fn}")
        self.activation_fn = activation_fn
        self.use_seq_length = use_seq_length

        layers = []
        # Adjust input_dim based on flag
        input_dim = self.embed_dim + 1 if self.use_seq_length else self.embed_dim

        # Create layers
        current_dim = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(current_dim, h_dim))
            layers.append(nn.LayerNorm(h_dim))
            layers.append(self.activation_fn())
            layers.append(nn.Dropout(self.dropout_p))
            current_dim = h_dim

        # Final output layer
        layers.append(nn.Linear(current_dim, 1))

        self.network = nn.Sequential(*layers)
        self.init_weights() # Optional: Initialize weights

    def init_weights(self):
        # Initialize weights for linear layers
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, src_embeddings, seq_lengths=None):
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

            MAX_LEN_NORM = 5000.0 # Consistent global normalization
            lengths_float = seq_lengths.float().to(src_embeddings.device)
            norm_seq_lengths = torch.clamp(lengths_float / MAX_LEN_NORM, 0.0, 1.0)

            expanded_seq_lengths = norm_seq_lengths.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
            mlp_input = torch.cat([src_embeddings, expanded_seq_lengths], dim=2)
        else:
            mlp_input = src_embeddings

        logits = self.network(mlp_input)
        return logits

# --- Training Function (Simplified Description) ---
def train_epoch(model, dataloader, optimizer, criterion, device, epoch): # Removed trial_num, n_iterations
    model.train()
    total_loss = 0
    num_batches_processed = 0
    # Updated progress bar description
    progress_bar = tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1} Training', leave=False, ncols=100)
    for batch in progress_bar:
        if batch is None: # Should not happen if collate_fn handles errors
            print(f"Skipping empty batch in epoch {epoch+1} training.")
            continue

        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()
        outputs = model(embeddings, seq_lengths=lengths).squeeze(-1)

        active_mask = (labels != -1)
        if active_mask.sum() == 0: continue
        active_logits = outputs[active_mask]
        active_labels = labels[active_mask]
        if active_labels.numel() == 0: continue

        loss = criterion(active_logits, active_labels)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss encountered during training epoch {epoch+1}. Skipping batch gradient update.")
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches_processed += 1
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / num_batches_processed if num_batches_processed > 0 else 0
    return avg_loss

# --- Evaluation Function (Simplified Description) ---
import math # Make sure math is imported at the top

# --- Evaluation Function (Revised) ---
def evaluate(model, dataloader, criterion, device, epoch, threshold, eval_type="Validation"): # Added eval_type
    model.eval()
    total_loss = 0.0
    all_preds_prob = []
    all_labels_list = []
    num_batches_processed = 0 # Batches where predictions were successfully collected
    num_loss_batches = 0    # Batches where loss was successfully calculated

    # Determine progress bar description based on eval_type
    progress_desc = f"{eval_type} Evaluating"
    # Only include epoch in description for Validation runs linked to training epochs
    if eval_type == "Validation":
        progress_desc = f"Epoch {epoch+1} {eval_type} Evaluating"

    with torch.no_grad():
        progress_bar = tqdm.tqdm(dataloader, desc=progress_desc, leave=False, ncols=100)
        for batch_idx, batch in enumerate(progress_bar):
            if batch is None:
                print(f"Warning: Skipping empty batch {batch_idx} during {eval_type} evaluation.")
                continue
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            lengths = batch['lengths'].to(device)

            try:
                outputs = model(embeddings, seq_lengths=lengths).squeeze(-1)

                active_mask = (labels != -1)
                if active_mask.sum() == 0:
                    # print(f"Info: Skipping batch {batch_idx} in {eval_type} - no active labels.")
                    continue
                active_logits = outputs[active_mask]
                active_labels = labels[active_mask]
                if active_labels.numel() == 0:
                    # print(f"Info: Skipping batch {batch_idx} in {eval_type} - zero active elements after masking.")
                    continue

                # Check for NaN/Inf logits *before* calculating loss or sigmoid
                if torch.any(torch.isnan(active_logits)) or torch.any(torch.isinf(active_logits)):
                     print(f"Warning: NaN or Inf logits encountered in batch {batch_idx} during {eval_type} evaluation. Skipping batch metrics & loss.")
                     continue # Skip entire batch if logits are bad

                # --- Loss Calculation (Conditional) ---
                batch_loss = None
                if criterion is not None:
                    try:
                        current_loss_tensor = criterion(active_logits, active_labels)
                        if not torch.isnan(current_loss_tensor) and not torch.isinf(current_loss_tensor):
                            batch_loss = current_loss_tensor.item() # Get scalar value
                            total_loss += batch_loss
                            num_loss_batches += 1
                        else:
                            print(f"Warning: NaN or Inf loss value computed in batch {batch_idx} during {eval_type} evaluation. Loss for batch ignored.")
                    except Exception as e:
                        print(f"Error calculating loss in batch {batch_idx} during {eval_type} evaluation: {e}. Loss for batch ignored.")

                # --- Collect Predictions ---
                # We proceed to collect predictions even if loss calculation failed, as long as logits were okay
                probs = torch.sigmoid(active_logits).cpu().numpy()
                all_preds_prob.extend(probs)
                all_labels_list.extend(active_labels.cpu().numpy())
                num_batches_processed += 1

                # Update progress bar postfix (show loss only if calculated)
                postfix_dict = {}
                if batch_loss is not None:
                   postfix_dict['loss'] = f"{batch_loss:.4f}"
                if postfix_dict:
                   progress_bar.set_postfix(postfix_dict)

            except Exception as e:
                print(f"Error processing batch {batch_idx} during {eval_type} evaluation: {e}")
                # Depending on the error, you might want to `continue` or `raise e`

    # --- Metric Calculation ---
    # Calculate avg_loss based only on batches where it was validly computed
    # Use float('nan') if no loss batches occurred (e.g., test set or all batches failed loss calculation)
    avg_loss = total_loss / num_loss_batches if num_loss_batches > 0 else float('nan')

    # Initialize metrics
    precision, recall, f1, auc_roc, auc_pr, auc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    cm = np.zeros((2, 2), dtype=int)
    metrics_calculated = False # Flag to check if we entered the calculation block

    # Ensure we have predictions to calculate metrics
    if len(all_labels_list) > 0 and len(all_preds_prob) > 0:
        all_labels_np = np.array(all_labels_list).astype(int)
        all_preds_prob_np = np.array(all_preds_prob)

        # Final check for NaNs/Infs in collected predictions (should be less likely now)
        if np.any(np.isnan(all_preds_prob_np)) or np.any(np.isinf(all_preds_prob_np)):
            print(f"Warning: NaNs or Infs found in FINAL collected predicted probabilities for {eval_type}. Metrics calculation skipped.")
        # Check for single class in true labels
        elif len(np.unique(all_labels_np)) < 2:
             print(f"Warning: Only one class ({np.unique(all_labels_np)}) present in TRUE labels for {eval_type}. AUC ROC/AUC10 cannot be computed reliably.")
             all_preds_binary = (all_preds_prob_np >= threshold).astype(int)
             try:
                 # Use labels present, handle potential warnings/errors if only one class predicted
                 present_labels = np.unique(all_labels_np)
                 precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds_binary, average='binary', zero_division=0, labels=present_labels, pos_label=1 if 1 in present_labels else present_labels[0])
                 cm = confusion_matrix(all_labels_np, all_preds_binary, labels=[0, 1])
                 metrics_calculated = True
             except ValueError as e:
                 print(f"Warning: Error calculating P/R/F1/CM for single class ({eval_type}): {e}")
             try:
                 auc_pr = average_precision_score(all_labels_np, all_preds_prob_np)
                 metrics_calculated = True # AUC PR can sometimes be calculated for single class
             except ValueError as e:
                 print(f"Warning: ValueError calculating AUC-PR for single class ({eval_type}): {e}"); auc_pr = 0.0
             auc_roc, auc10 = 0.0, 0.0 # Cannot calculate these
        # Standard case: multiple classes, valid predictions
        else:
            all_preds_binary = (all_preds_prob_np >= threshold).astype(int)
            try: cm = confusion_matrix(all_labels_np, all_preds_binary, labels=[0, 1])
            except ValueError as e: print(f"Warning: Could not compute CM for {eval_type}: {e}. Using default.")
            try:
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds_binary, average='binary', zero_division=0, labels=[0, 1])
            except ValueError as e: print(f"Warning: Error calculating P/R/F1 for {eval_type}: {e}")
            try: auc_pr = average_precision_score(all_labels_np, all_preds_prob_np)
            except ValueError as e: print(f"Warning: ValueError AUC-PR for {eval_type}: {e}"); auc_pr = 0.0
            try:
                auc_roc = roc_auc_score(all_labels_np, all_preds_prob_np)
                try: auc10 = roc_auc_score(all_labels_np, all_preds_prob_np, max_fpr=0.1)
                except ValueError as e: print(f"Warning: ValueError AUC10 for {eval_type}: {e}. Setting AUC10 to 0.0."); auc10 = 0.0
            except ValueError as e: print(f"Warning: ValueError AUC-ROC for {eval_type}: {e}"); auc_roc, auc10 = 0.0, 0.0
            metrics_calculated = True # Metrics were attempted
    else:
        # This case handles if no valid predictions/labels were collected across all batches
        print(f"Warning: No valid predictions/labels collected during {eval_type} evaluation. Metrics calculation skipped.")

    # --- Print Evaluation Results ---
    # Adjust title based on type and epoch
    epoch_str = f" (Epoch {epoch+1})" if eval_type == 'Validation' else ""
    print(f"\n--- {eval_type} Results{epoch_str} ---")

    # Only print loss if it was calculated (num_loss_batches > 0) and the avg is not NaN
    if num_loss_batches > 0 and not math.isnan(avg_loss):
        print(f"  Average Loss: {avg_loss:.4f} (calculated over {num_loss_batches} batches)")
    else:
        print(f"  Average Loss: N/A (criterion was None or loss calculation failed)")

    # Print other metrics (show 0.0 if calculation failed or wasn't possible)
    print(f"  AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, AUC10: {auc10:.4f}")
    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  Threshold used for P/R/F1/CM: {threshold}")

    # Print CM only if it was computed successfully (i.e., not the default zeros) AND metrics were calculated
    if metrics_calculated and np.any(cm): # Check if CM is not just zeros
        print("\n  Confusion Matrix (Rows: Actual, Cols: Predicted):")
        print(f"        Predicted 0    Predicted 1")
        print(f"Actual 0 | {cm[0, 0]:<12} | {cm[0, 1]:<12} | (TN={cm[0, 0]}, FP={cm[0, 1]})")
        print(f"Actual 1 | {cm[1, 0]:<12} | {cm[1, 1]:<12} | (FN={cm[1, 0]}, TP={cm[1, 1]})")
    elif metrics_calculated:
        print("\n  Confusion Matrix: Not computed correctly (potentially due to single class or other issues).")
    else:
         print("\n  Confusion Matrix: Not computed (no valid predictions/labels collected).")
    print("-" * 40) # Separator

    # Return metrics dictionary (loss will be NaN for test set or if calculation failed)
    return {'loss': avg_loss, 'precision': precision, 'recall': recall, 'f1': f1,
            'auc_roc': auc_roc, 'auc_pr': auc_pr, 'auc10': auc10}

# --- Configuration ---
# Ask for GPU and Threshold upfront (kept)
target_gpu = -1
while target_gpu not in [0, 1, 2, 3]:
    try:
        target_gpu = int(input("Which GPU to use? [0, 1, 2, 3]: "))
        if target_gpu not in [0, 1, 2, 3]: print("Invalid GPU index.")
    except ValueError: print("Please enter a number.")

threshold = -1.0
while not (0.0 < threshold < 1.0):
    try:
        threshold = float(input("Enter prediction threshold value (e.g., 0.5): "))
        if not (0.0 < threshold < 1.0): print("Threshold must be between 0 and 1 (exclusive).")
    except ValueError: print("Please enter a valid number.")

# Embedding Config (kept)
EMBEDDING_CONFIG = {
    'esm2': {'h5_path': 'esm2_protein_embeddings.h5', 'embed_dim': 1280, 'test': 'esm2_test_protein_embeddings.h5'},
    'esmc': {'h5_path': 'esmc_protein_embeddings.h5', 'embed_dim': 960, 'test': 'esmc_test_protein_embeddings.h5'}
}
EMBEDDING_TYPE = ""
print("-----------------------------------------------------")
while EMBEDDING_TYPE not in EMBEDDING_CONFIG:
    EMBEDDING_TYPE = input(f"Select embedding type ({'/'.join(EMBEDDING_CONFIG.keys())}): ").lower().strip()
    if EMBEDDING_TYPE not in EMBEDDING_CONFIG: print("Invalid choice. Please try again.")
print("-----------------------------------------------------")
config = EMBEDDING_CONFIG[EMBEDDING_TYPE]
H5_FILE_PATH = config['h5_path']
EMBED_DIM = config['embed_dim']
TEST_H5_FILE_PATH = config['test']
# --- Load Best Hyperparameters ---
PATH_TO_JSON = ""
while not os.path.isfile(PATH_TO_JSON) or not PATH_TO_JSON.lower().endswith('.json'):
    PATH_TO_JSON = input("Enter path to JSON file with hyperparameter search results: ")
    if not os.path.isfile(PATH_TO_JSON):
        print(f"Error: File not found at '{PATH_TO_JSON}'")
    elif not PATH_TO_JSON.lower().endswith('.json'):
        print("Error: File must be a .json file")

try:
    with open(PATH_TO_JSON, 'r') as f:
        hp_search_data = json.load(f)
except Exception as e:
    print(f"Error reading or parsing JSON file {PATH_TO_JSON}: {e}")
    sys.exit(1)

METRIC_TYPE = ""
valid_metrics = ["f1", "auc_pr", "recall", "precision", "auc_roc", "auc10"] # lowercase
print(f"Available metrics for selecting best run: {', '.join(valid_metrics)}")
while METRIC_TYPE not in valid_metrics:
    METRIC_TYPE = input("Which metric should determine the best run? ").lower().strip()
    if METRIC_TYPE == "roc-auc": METRIC_TYPE = "auc_roc" # Allow alternative name
    if METRIC_TYPE not in valid_metrics:
        print(f"Invalid metric. Please choose from: {', '.join(valid_metrics)}")

# Find the best run based on the chosen metric
best_run_data = None
max_val_metric = -float('inf') # Initialize with negative infinity

for run_data in hp_search_data:
    # Check if run completed and has necessary data
    if run_data.get('status') != 'completed': continue
    avg_metrics = run_data.get('avg_metrics')
    if not isinstance(avg_metrics, dict): continue

    current_metric = avg_metrics.get(METRIC_TYPE)
    if current_metric is None: continue # Metric not found in this run's avg_metrics

    if current_metric > max_val_metric:
        max_val_metric = current_metric
        best_run_data = run_data

if best_run_data is None:
    print(f"Error: Could not find any completed run with metric '{METRIC_TYPE}' in {PATH_TO_JSON}")
    sys.exit(1)

best_params = best_run_data['params']
best_trial_num_source = best_run_data.get('trial_num', 'N/A') # Get original trial number if available

print("--- Loaded Best Parameters ---")
print(f"From source Trial: {best_trial_num_source} in {os.path.basename(PATH_TO_JSON)}")
print(f"Based on highest avg '{METRIC_TYPE}': {max_val_metric:.4f}")
print(f"Parameters: {json.dumps(best_params, indent=2)}")
print("------------------------------")

# --- Extract and Map Parameters ---
LEARNING_RATE = best_params['learning_rate']
DROPOUT = best_params['dropout']
# HIDDEN_DIMS_CONFIG_KEY = best_params['hidden_dims_config'] # Not needed if 'hidden_dims' list is stored
HIDDEN_DIMS = best_params.get('hidden_dims') # Directly get the list
if HIDDEN_DIMS is None: # Fallback if only the key was stored previously
    HIDDEN_DIMS_CONFIG_KEY = best_params['hidden_dims_config']
    HIDDEN_CONFIGS = { # Re-define here if needed, based on EMBED_DIM
        'small': [EMBED_DIM // 2, EMBED_DIM // 4],
        'medium': [EMBED_DIM, EMBED_DIM // 2],
        'large': [EMBED_DIM * 2, EMBED_DIM, EMBED_DIM // 2]
    }
    HIDDEN_DIMS = HIDDEN_CONFIGS[HIDDEN_DIMS_CONFIG_KEY]

WEIGHT_DECAY = best_params['weight_decay']
ACTIVATION_FN_NAME = best_params['activation_fn'] # Name stored as string
USE_SEQ_LENGTH = best_params.get('use_seq_length', True)
EARLY_STOPPING_EPSILON = 1e-5 # Small tolerance for improvement (NEW)



# Map activation function name back to class
activation_map = {
    'ReLU': nn.ReLU,
    'SiLU': nn.SiLU,
    # Add others if used in HP search, e.g., 'GELU': nn.GELU
}
if ACTIVATION_FN_NAME not in activation_map:
    raise ValueError(f"Unknown activation function name '{ACTIVATION_FN_NAME}' found in parameters. Add it to activation_map.")
ACTIVATION_FN = activation_map[ACTIVATION_FN_NAME]

# --- Paths and Directories Setup for Final Training ---
run_timestamp = datetime.datetime.now().strftime("%m-%d_%H%M")
# Modify base directory name to indicate final training
BASE_RUNS_DIR = f'{EMBEDDING_TYPE}_MLP_final_training_{run_timestamp}'
# Include metric used for selection in the run type directory for clarity
RUN_TYPE_DIR = os.path.join(BASE_RUNS_DIR, f'{EMBEDDING_TYPE}_best_{METRIC_TYPE}')
# Save only one final model
MODEL_SAVE_DIR = RUN_TYPE_DIR # Save model directly in the run directory
TENSORBOARD_LOG_DIR = os.path.join(RUN_TYPE_DIR, 'tensorboard_logs')
# Save final results/summary of this single run
FINAL_RESULTS_FILE = os.path.join(RUN_TYPE_DIR, f'{EMBEDDING_TYPE}_final_training_summary.json')

os.makedirs(RUN_TYPE_DIR, exist_ok=True)
# No need to create MODEL_SAVE_DIR separately if it's the same as RUN_TYPE_DIR
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
print(f"Created directories for final training run.")
print(f"Final model will be saved in: {MODEL_SAVE_DIR}")
print(f"TensorBoard logs will be saved in: {TENSORBOARD_LOG_DIR}")
print(f"Final run summary will be saved at: {FINAL_RESULTS_FILE}")

# --- Fixed Training Settings ---
N_EPOCHS = 200 # Increased epochs for final training
BATCH_SIZE = 16 # Keep batch size consistent or adjust as needed
RANDOM_SEED = 89 # Use a potentially different seed for the final run
EARLY_STOPPING_PATIENCE = 20 # Increased patience
N_WARMUP_EPOCHS = 3 # Keep warmup consistent
VALIDATION_SPLIT_RATIO = 0.15 # e.g., 15% for validation


# --- Device Setup ---
device_str = "cpu"
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Found {num_gpus} GPU(s).")
    if target_gpu < num_gpus: device_str = f"cuda:{target_gpu}"
    elif num_gpus > 0: print(f"Warning: Target GPU {target_gpu} unavailable. Using cuda:0."); device_str = "cuda:0"
    else: print("Warning: CUDA available but no devices found? Using CPU.")
else: print("CUDA not available. Using CPU.")
DEVICE = torch.device(device_str)

# --- Print Configuration Summary ---
print("-----------------------------------------------------")
print(f"--- Running FINAL MLP Training for: {EMBEDDING_TYPE} ---")
print(f"Using device: {DEVICE}")
print(f"Using parameters from: {os.path.basename(PATH_TO_JSON)} (Best trial by '{METRIC_TYPE}')")
print(f"Embeddings Path: {H5_FILE_PATH}")
print(f"Embeddings Dimension: {EMBED_DIM}")
print(f"Output Directory: {RUN_TYPE_DIR}")
print(f"Prediction Threshold: {threshold}")
print(f"Total Epochs: {N_EPOCHS}, Early Stopping Patience: {EARLY_STOPPING_PATIENCE}")
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
        if 'embeddings_folder' not in f: raise KeyError("'embeddings_folder' not found")
        all_protein_keys = np.array(sorted(list(f['embeddings_folder'].keys())))
    print(f"Loaded {len(all_protein_keys)} protein keys from {H5_FILE_PATH}")
except FileNotFoundError: print(f"Error: HDF5 file not found at {H5_FILE_PATH}"); sys.exit(1)
except KeyError as e: print(f"Error loading keys from HDF5: {e}"); sys.exit(1)
except Exception as e: print(f"Unexpected error loading keys: {e}"); sys.exit(1)

# --- Create Train/Validation Split ---
if len(all_protein_keys) < 2:
     print("Error: Need at least two proteins to create a train/validation split.")
     sys.exit(1)

train_keys, val_keys = train_test_split(
    all_protein_keys,
    test_size=VALIDATION_SPLIT_RATIO,
    random_state=RANDOM_SEED # Use the same seed for consistent split
)
print(f"Split data: {len(train_keys)} training keys, {len(val_keys)} validation keys.")

# --- Setup TensorBoard Writer ---
main_tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
print(f"TensorBoard logs active: {TENSORBOARD_LOG_DIR}")

# --- Prepare Dataloaders ---
train_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=train_keys)
val_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=val_keys)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False, persistent_workers=True if DEVICE.type == 'cuda' and int(torch.__version__.split('.')[0]) >= 1 and int(torch.__version__.split('.')[1]) >= 8 else False) # Added persistent_workers suggestion
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False, persistent_workers=True if DEVICE.type == 'cuda' and int(torch.__version__.split('.')[0]) >= 1 and int(torch.__version__.split('.')[1]) >= 8 else False)


# --- Calculate pos_weight on TRAINING data only ---
print("Calculating pos_weight for imbalance on training set...")
num_pos, num_neg = 0, 0
# Use a temporary loader for efficiency
temp_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE*2, collate_fn=collate_fn, num_workers=2)
for batch in tqdm.tqdm(temp_loader, desc="Calculating pos_weight", leave=False, ncols=80):
    if batch is None: continue
    labels = batch['labels']
    active_mask = (labels != -1)
    active_labels = labels[active_mask].numpy()
    num_pos += np.sum(active_labels == 1)
    num_neg += np.sum(active_labels == 0)
del temp_loader

if num_pos > 0 and num_neg > 0:
    pos_weight = num_neg / num_pos
else:
    print(f"Warning: Training set - num_pos={num_pos}, num_neg={num_neg}. Using pos_weight=1.0")
    pos_weight = 1.0
print(f"Training set pos_weight: {pos_weight:.2f}")
pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

# --- Instantiate FINAL Model and Optimizer ---
print("Instantiating final model with best parameters...")
model = EpitopeMLP(
    embed_dim=EMBED_DIM,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT,
    activation_fn=ACTIVATION_FN,
    use_seq_length=best_params.get('use_seq_length', True) # Assume True if not in JSON
).to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# --- Learning Rate Scheduler ---
scheduler_warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=N_WARMUP_EPOCHS)
scheduler_decay = CosineAnnealingLR(optimizer, T_max=(N_EPOCHS - N_WARMUP_EPOCHS), eta_min=LEARNING_RATE * 0.01)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_decay], milestones=[N_WARMUP_EPOCHS])

# --- Training Loop ---
best_val_metric_value = -float('inf') # Track the chosen metric (e.g., AUC PR)
best_epoch = -1
epochs_no_improve = 0
final_model_save_path = os.path.join(MODEL_SAVE_DIR, f'{EMBEDDING_TYPE}_mlp_final_best.pth')

print(f"\n--- Starting Final Training for {N_EPOCHS} Epochs ---")
print(f"Monitoring validation '{METRIC_TYPE}' for improvement.")
print(f"Best model will be saved to: {final_model_save_path}")

for epoch in range(N_EPOCHS):
    # Train
    train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}/{N_EPOCHS} - Train Loss: {train_loss:.4f} | Current LR: {current_lr:.2e}")
    main_tb_writer.add_scalar('Loss/Train', train_loss, epoch)
    main_tb_writer.add_scalar('LearningRate', current_lr, epoch)

    # Evaluate
    val_metrics = evaluate(model, val_loader, criterion, DEVICE, epoch, threshold)
    main_tb_writer.add_scalar('Loss/Val', val_metrics['loss'], epoch)
    main_tb_writer.add_scalar(f'Metrics/Val_{METRIC_TYPE}', val_metrics[METRIC_TYPE], epoch)
    # Log other val metrics too
    for m_key, m_val in val_metrics.items():
        if m_key != METRIC_TYPE and m_key != 'loss': # Avoid double logging loss/primary metric
             main_tb_writer.add_scalar(f'Metrics/Val_{m_key}', m_val, epoch)


    # --- Early Stopping & Model Saving ---
    current_val_metric = val_metrics[METRIC_TYPE]
    if current_val_metric > best_val_metric_value - EARLY_STOPPING_EPSILON:
        print(f"Epoch {epoch+1}: Validation {METRIC_TYPE.upper()} improved ({best_val_metric_value:.4f} -> {current_val_metric:.4f}). Saving model...")
        best_val_metric_value = current_val_metric
        best_epoch = epoch + 1
        epochs_no_improve = 0
        try:
            torch.save(model.state_dict(), final_model_save_path)
        except Exception as e:
            print(f"Error saving model at epoch {epoch+1}: {e}")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch+1}: Validation {METRIC_TYPE.upper()} ({current_val_metric:.4f}) did not improve from best ({best_val_metric_value:.4f}). Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n--- Early stopping triggered at epoch {epoch+1} ---")
            break # Exit training loop

    scheduler.step() # Update learning rate

# --- End of Training Loop ---
main_tb_writer.close()

print("\n--- Final Training Complete ---")
if best_epoch != -1:
    print(f"Best validation {METRIC_TYPE.upper()}: {best_val_metric_value:.4f} achieved at epoch {best_epoch}")
    print(f"Best model saved to: {final_model_save_path}")
else:
    print("No improvement observed during training. No model saved.")

# --- Save Final Run Summary ---
final_summary = {
    'embedding_type': EMBEDDING_TYPE,
    'best_params_source_file': PATH_TO_JSON,
    'best_params_source_trial': best_trial_num_source,
    'metric_used_for_selection': METRIC_TYPE,
    'loaded_parameters': best_params,
    'training_seed': RANDOM_SEED,
    'final_training_epochs_run': epoch + 1, # Actual epochs completed
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'best_validation_metric': {METRIC_TYPE: best_val_metric_value},
    'best_epoch': best_epoch,
    'final_model_path': final_model_save_path if best_epoch != -1 else None,
    'training_timestamp': run_timestamp
}
try:
    # Use the same serializer as before for consistency
    def default_serializer(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
        elif isinstance(obj, (np.ndarray,)): return obj.tolist()
        elif isinstance(obj, (np.bool_)): return bool(obj)
        elif isinstance(obj, (np.void)): return None
        elif hasattr(obj, '__name__'): return obj.__name__
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(FINAL_RESULTS_FILE, 'w') as f:
        json.dump(final_summary, f, indent=4, default=default_serializer)
    print(f"Final training summary saved to: {FINAL_RESULTS_FILE}")
except Exception as e:
    print(f"Error saving final training summary: {e}")

# --- Calculate and Print Elapsed Time ---
script_end_time = time.time()
total_elapsed_seconds = script_end_time - script_start_time
elapsed_time_str = str(datetime.timedelta(seconds=int(total_elapsed_seconds)))
print("\n-----------------------------------------------------")
print(f"MLP Final Training Script Finished.")
print(f"Total Execution Time: {elapsed_time_str}")
print("-----------------------------------------------------")
print("\n--- Final Training Complete ---")
if best_epoch != -1: print(f"Best validation {METRIC_TYPE.upper()}: {best_val_metric_value:.4f} (Epoch {best_epoch}). Model saved: {final_model_save_path}")
else: print("No improvement observed during training based on validation set. No model saved.")

# --- FINAL EVALUATION ON TEST SET ---
test_metrics = {} # Initialize empty dict for test metrics
test_protein_keys = []
test_metrics = {} # Initialize empty dict for test metrics
test_protein_keys = [] # Initialize before try block

if best_epoch != -1 and os.path.exists(final_model_save_path):
    print("\n--- Starting Evaluation on Test Set ---")

    # --- Debugging Test File Path ---
    print(f"Attempting to load test keys from HDF5 file: '{TEST_H5_FILE_PATH}'") # Print the path being used
    print(f"Checking existence of test file: {os.path.exists(TEST_H5_FILE_PATH)}") # Verify existence

    # Load test keys
    try:
        # Check if the test file path is valid and exists *before* trying to open
        if not TEST_H5_FILE_PATH or not os.path.exists(TEST_H5_FILE_PATH):
             raise FileNotFoundError(f"Test HDF5 file not found or path is invalid: '{TEST_H5_FILE_PATH}'")

        print(f"Opening test HDF5 file: {TEST_H5_FILE_PATH}...") # Added print
        with h5py.File(TEST_H5_FILE_PATH, 'r') as f:
            print(f"File opened successfully. Checking for 'embeddings_folder'...") # Added print
            if 'embeddings_folder' not in f:
                raise KeyError(f"'embeddings_folder' group not found in {TEST_H5_FILE_PATH}")

            print("'embeddings_folder' found. Reading keys...") # Added print
            keys_list = list(f['embeddings_folder'].keys())
            print(f"Found {len(keys_list)} keys in 'embeddings_folder'. Sorting...") # Added print

            if not keys_list:
                 print("Warning: 'embeddings_folder' exists but contains no protein keys.")
                 # test_protein_keys will remain empty or become an empty array

            # Assign to the pre-initialized variable only if keys were found
            test_protein_keys = np.array(sorted(keys_list))

        # Print after the with block, only if successful so far
        print(f"Successfully loaded and sorted {len(test_protein_keys)} test protein keys.")

    except FileNotFoundError as e:
        print(f"Error loading test keys (FileNotFoundError): {e}. Skipping test evaluation.")
        # test_protein_keys remains []
    except KeyError as e:
        print(f"Error loading test keys (KeyError): {e}. Skipping test evaluation.")
        # test_protein_keys remains []
    except Exception as e: # Catch other potential errors during loading
        print(f"An unexpected error occurred loading test keys: {e.__class__.__name__}: {e}. Skipping test evaluation.")
        # test_protein_keys remains []

    # --- Debugging After Loading Attempt ---
    print(f"Status after loading attempt: len(test_protein_keys) = {len(test_protein_keys)}")
    if len(test_protein_keys) > 0:
        # Create test dataset and dataloader
        test_dataset = Embedding_retriever(TEST_H5_FILE_PATH, protein_keys=test_protein_keys)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4, pin_memory=True if DEVICE.type == 'cuda' else False)

        # Instantiate model with best params and load state dict
        print("Loading best model state for test evaluation...")
        # ---------> THIS LINE DEFINES eval_model <---------
        eval_model = EpitopeMLP(
            embed_dim=EMBED_DIM, hidden_dims=HIDDEN_DIMS, dropout=DROPOUT,
            activation_fn=ACTIVATION_FN, use_seq_length=USE_SEQ_LENGTH
        ).to(DEVICE)
        # ---------> ENSURE IT IS PRESENT <---------
        try:
            # Load the weights into the newly created eval_model
            eval_model.load_state_dict(torch.load(final_model_save_path, map_location=DEVICE))
            eval_model.eval() # Set to evaluation mode

            # Evaluate on Test Set using eval_model
            test_metrics = evaluate(
                model=eval_model,  # Pass the correct model here
                dataloader=test_loader,
                criterion=None,
                device=DEVICE,
                epoch=0,
                threshold=threshold,
                eval_type="Test Set"
            )

        except FileNotFoundError:
            print(f"Error: Best model file not found at {final_model_save_path}. Cannot run test evaluation.")
            test_metrics = {}
        except Exception as e:
            print(f"Error during test evaluation execution: {e}")
            test_metrics = {}

        # Cleanup test loader etc., including eval_model
        del test_dataset, test_loader, eval_model
        gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
    else:
        print("Skipping test evaluation as no test keys were loaded or loading failed.")
# --- END OF FINAL EVALUATION ---



# --- Save Final Run Summary (including test metrics) ---
final_summary = {
    'embedding_type': EMBEDDING_TYPE,
    'best_params_source_file': os.path.basename(PATH_TO_JSON), # Just filename
    'best_params_source_trial': best_trial_num_source,
    'metric_used_for_selection': METRIC_TYPE,
    'loaded_parameters': best_params,
    'training_seed': RANDOM_SEED,
    'train_val_h5_path': H5_FILE_PATH,
    'test_h5_path': TEST_H5_FILE_PATH,
    'final_training_epochs_run': epoch + 1 if 'epoch' in locals() else 0, # Actual epochs completed
    'early_stopping_patience': EARLY_STOPPING_PATIENCE,
    'best_validation_metric': {METRIC_TYPE: best_val_metric_value},
    'best_epoch': best_epoch,
    'final_model_path': final_model_save_path if best_epoch != -1 else None,
    'test_set_metrics': test_metrics, # Add the test metrics dictionary
    'training_timestamp': run_timestamp,
    'total_runtime_seconds': time.time() - script_start_time,
    'threshold_used': threshold
}
try:
    def default_serializer(obj): # Copied serializer
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)): return float(obj)
        elif isinstance(obj, (np.ndarray,)): return obj.tolist()
        elif isinstance(obj, (np.bool_)): return bool(obj)
        elif isinstance(obj, (np.void)): return None
        elif hasattr(obj, '__name__'): return obj.__name__
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    with open(FINAL_RESULTS_FILE, 'w') as f:
        json.dump(final_summary, f, indent=4, default=default_serializer)
    print(f"\nFinal training summary (including test metrics) saved to: {FINAL_RESULTS_FILE}")
except Exception as e: print(f"Error saving final training summary: {e}")

# --- Calculate and Print Elapsed Time ---
script_end_time = time.time()
total_elapsed_seconds = script_end_time - script_start_time
elapsed_time_str = str(datetime.timedelta(seconds=int(total_elapsed_seconds)))
print("\n-----------------------------------------------------")
print(f"MLP Final Training & Evaluation Script Finished.")
print(f"Total Execution Time: {elapsed_time_str}")
print("-----------------------------------------------------")
