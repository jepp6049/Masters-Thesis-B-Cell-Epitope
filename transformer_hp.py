#pip install -r requirements.txt

import numpy as np
import pandas as pd
from Bio import SeqIO
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import h5py
import tqdm
import math
import gc # Garbage collector
import os # For creating directories
import sys # for choosing GPU
import json # For saving results
import time # For timestamping runs
import datetime # For timer formatting
import random # For seeding
from scipy.stats import loguniform, uniform

import torch
import torch.nn as nn
import torch.nn.functional as F # For Swish if needed
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.tensorboard import SummaryWriter # For TensorBoard
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR # For warm up LR


from sklearn.model_selection import KFold, ParameterSampler # For CV and Random Search
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix


# --- Email Sending Function ---




# --- Start Timer ---
script_start_time = time.time()

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
    
# ## testing that we can retrieve specified protein from HDF5
# protein_data = Embedding_retriever('/work/jgg/exp1137-improving-b-cell-antigen-predictions-using-plms/esm2_protein_embeddings.h5', ['3b9k_B'])
# protein_data_esmc = Embedding_retriever('/work/jgg/exp1137-improving-b-cell-antigen-predictions-using-plms/esmc_protein_embeddings.h5', ['5ggv_Y'])
# test_data = Embedding_retriever('/work/jgg/exp1137-improving-b-cell-antigen-predictions-using-plms/esm2_test_protein_embeddings.h5', ['7lj4_B'])
# test_data_escm = Embedding_retriever('/work/jnw/exp1137-improving-b-cell-antigen-predictions-using-plms/esmc_test_protein_embeddings.h5')

### Padding Collate Function
def collate_fn(batch):
    embeddings = [item['embeddings'] for item in batch]
    labels = [item['labels'] for item in batch]
    lengths = [item['length'] for item in batch]
    names = [item['name'] for item in batch]

    padded_embeddings = nn.utils.rnn.pad_sequence(embeddings, batch_first=True, padding_value= 0.0)
    padded_labels = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-1)

    max_len = padded_embeddings.size(1)
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    padding_mask  =torch.arange(max_len)[None, :] >= lengths_tensor[:, None]

    return {
        'names': names,
        'embeddings': padded_embeddings,
        'labels': padded_labels,
        'padding_mask': padding_mask,
        'lengths': lengths_tensor
    }

# Building Our Models

## Transformer Model
MAX_LEN = 5000

# --- Positional Encoding --- 
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len = MAX_LEN):
        super().__init__()
            #super(PositionalEncoder, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model) #generating empty tensor to later populate with positional encoding values.
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # Create a column vector of token positions, which will later be used to calculate sinusoidal encodings 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # the denominator of PE formula.

        #Slicing of both even and uneven terms (removed if/else function)
        pe[:, 0::2] = torch.sin(position * div_term) # generators tensor using sine function
        pe[:, 1::2] = torch.cos(position * div_term) # generators tensor using sine function

        self.register_buffer('pe', pe) # fixed paramter, not trainable.

    def forward(self, x):
        # apply positional encoding to input.
        pos_encoding_slice = self.pe[:x.size(1), :].unsqueeze(0)
        x = x + pos_encoding_slice
        return self.dropout(x) # regularization to input
    
        # OBS!!!: dimension output of PE is now [batch_size, seq_len, embed_dim]´

# --- SiLu (Swish) Feed Forward Network ---
class SiLuFFN(nn.Module):
    def __init__(self, embed_dim, ffn_hidden_dim, dropout=0.1):
        super().__init__()

        self.w1 = nn.Linear(embed_dim, ffn_hidden_dim, bias=True) # defines 1st layer
        self.w2 = nn.Linear(ffn_hidden_dim, embed_dim, bias=True) # defines 2nd layer
        self.dropout = nn.Dropout(dropout) # a good ol' dropout layer
        self.activation = F.silu # stores SWISH activation function

    def forward(self ,x):
        hidden = self.w1(x) #passing input through first (w1) layer    
        activated = self.activation(hidden) # applying SWISH    
        dropped = self.dropout(activated) # applying dropout
        output = self.w2(dropped) # pass through second (w2) layer

        return output # returns final processed tensors

# --- Transformer Encoder Layer with SiLU FFN ---
class TransformerEncoderLayer(nn.Module):                       # (vi kan måske fjerne lidt kommentarer)
    def __init__(self, embed_dim, nhead, ffn_hidden_dim, dropout=0.1, pre_norm=True):
        super().__init__()
        self.pre_norm = pre_norm

        #Sub layers
        self.self_attn = nn.MultiheadAttention(embed_dim, nhead, dropout=dropout, batch_first=True) # 1. MHA
        self.ffn = SiLuFFN(embed_dim, ffn_hidden_dim, dropout) # FFN part of transformer block.T_destination

        # Layer norm & dropout
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim) # will have different weights
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout) # will have different weights

    def forward(self, src, src_mask=None, src_key_padding_mask=None): # (gammel)
        if self.pre_norm:
            # Pre-LN: Norm -> Sublayer -> Dropout -> Residual
            # 1. Multi-Head Attention block
            src_norm1 = self.norm1(src) # normalize first (PRE)
            attention_output, _ = self.self_attn(src_norm1, src_norm1, src_norm1, # these are for the Q, K and V tensors
                                                 attn_mask=src_mask,
                                                 key_padding_mask=src_key_padding_mask,
                                                 need_weights=False)
            src = src + self.dropout1(attention_output) # We dropout some of our output from the attention layer

            # 2. Feed-Forward block
            src_norm2 = self.norm2(src) # again, normalize first (from prev block output)
            ffn_output = self.ffn(src_norm2)
            src = src + self.dropout2(ffn_output) # We dropout some of our output from the FFN layer (residual connection)
        
            return src # this output goes to decoder block
        
        else: 
            # Post-LN: Sublayer -> Dropout -> Residual -> Norm
            attention_output, _ = self.self_attn(src, src, src, # these are for the Q, K and V tensors
                                             attn_mask=src_mask,
                                             key_padding_mask=src_key_padding_mask,
                                             need_weights=False)
        
            # Add & Norm (Residual Connection 1) 
            src = src + self.dropout1(attention_output) # We dropout some of our output from the attention layer
            src = self.norm1(src) # and we apply layer normalization

            # FF block
            ffn_output = self.ffn(src)

            # Add & Norm (Residual Connection 2) 
            src = src + self.dropout2(ffn_output) # We dropout some of our output from the FFN layer
            src = self.norm2(src) # and we apply layer normalization

            return src # this output goes to decoder block
            

# --- Main Epitope Transformer Model using SiLU ---
class EpitopeTransformer(nn.Module):
    def __init__(self, embed_dim, nhead, num_encoder_layers, ffn_hidden_dim,
                 dropout=0.1, max_len=MAX_LEN, pre_norm=True, use_seq_length = True):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.pre_norm = pre_norm
        self.use_seq_length = use_seq_length


        self.pos_encoder = PositionalEncoder(embed_dim, dropout, max_len) # first PE layer

        # Encoder Stack
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, nhead, ffn_hidden_dim, dropout, pre_norm=self.pre_norm)
            for _ in range(num_encoder_layers)
        ])

        self.norm_final = nn.LayerNorm(embed_dim) if self.pre_norm else nn.Identity()  # if pre-ln: need one final norm after stack.
        
        if self.use_seq_length:
            self.output_layer = nn.Linear(embed_dim + 1, 1)  # embed_dim + 1 for sequence length
        else:
            self.output_layer = nn.Linear(embed_dim, 1)
        self.init_weights() # optional, but good practice

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.output_layer.bias)
        nn.init.uniform_(self.output_layer.weight, -initrange, initrange) # initialize bias with zeros, weights with uniform

    def forward(self, src, src_key_padding_mask=None, seq_lengths=None): #padding mask=None as we're only in encoder layer
        src = self.pos_encoder(src) # we encode input. Shape: [B, SeqLen, EmbDim] 

        output = src # we initialize output value and loop through each encoder layer passing the output from layer to layer.
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        output = self.norm_final(output) # apply final norm if only Pre-LN is used
        
        if self.use_seq_length and seq_lengths is not None:
            # Normalize sequence lengths to [0,1] range
            batch_size, seq_len, _ = output.shape
            norm_seq_lengths = seq_lengths.float() / MAX_LEN
            
            # Expand sequence length to match output dimensions
            # Shape: [batch_size, 1, 1] -> [batch_size, seq_len, 1]
            expanded_seq_lengths = norm_seq_lengths.view(batch_size, 1, 1).expand(batch_size, seq_len, 1)
            
            # Concatenate expanded sequence lengths with transformer output
            # Shape: [batch_size, seq_len, embed_dim + 1]
            output_with_len = torch.cat([output, expanded_seq_lengths], dim=2)
            
            # Apply the final output layer
            output_logits = self.output_layer(output_with_len)
        else:
            output_logits = self.output_layer(output) # Apply the final output layer (Prediction Head)
            
        # Output shape: [batch_size, seq_len, 1]
        return output_logits

# --- Training Function ---
def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0 # Count actual batches processed
    progress_bar = tqdm.tqdm(dataloader, desc=f'Epoch {epoch+1} Training', leave=False, ncols=100)
    for batch in progress_bar:
        if batch is None:
            print(f"Skipping empty batch in epoch {epoch+1} training.")
            continue
        embeddings = batch['embeddings'].to(device)
        labels = batch['labels'].to(device)
        padding_mask = batch['padding_mask'].to(device) # Mask is True for padded positions
        lengths = batch['lengths'].to(device)  # Get the sequence lengths

        optimizer.zero_grad()
        outputs = model(embeddings, src_key_padding_mask=padding_mask, seq_lengths=lengths) # Pass lengths to model
        outputs = outputs.squeeze(-1) # Shape: [B, SeqLen]
        active_mask = (labels != -1) # Mask is True for non-padded positions
        if active_mask.sum() == 0:
            continue # Skip batch if no active labels
        active_logits = outputs[active_mask]
        active_labels = labels[active_mask]
        if active_labels.numel() == 0:
            continue
        loss = criterion(active_logits, active_labels)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: NaN or Inf loss encountered in epoch {epoch+1}. Skipping batch.")
            optimizer.zero_grad() # Prevent propagation of invalid gradients
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # preventing exploding gradients
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1 # Increment batch count only if processed
        progress_bar.set_postfix(loss=loss.item())
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss


# --- Evaluation Function ---      
def evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    total_loss = 0
    all_preds_prob = []
    all_labels_list = []
    num_batches = len(dataloader)
    with torch.no_grad():
        progress_bar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1} Evaluating", leave=False, ncols=100)
        for batch in progress_bar:
            embeddings = batch['embeddings'].to(device)
            labels = batch['labels'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            lengths = batch['lengths'].to(device)  # Get the sequence lengths
            
            outputs = model(embeddings, src_key_padding_mask=padding_mask, seq_lengths=lengths).squeeze(-1)
            active_mask = (labels != -1)
            if active_mask.sum() == 0: continue
            active_logits = outputs[active_mask]
            active_labels = labels[active_mask]
            # Ensure there are elements to compute loss on
            if active_labels.numel() == 0:
                continue
            loss = criterion(active_logits, active_labels)
             # Check for NaN/inf loss during validation as well
            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1
                probs = torch.sigmoid(active_logits).cpu().numpy()
                all_preds_prob.extend(probs)
                all_labels_list.extend(active_labels.cpu().numpy())
            else:
                 print(f"Warning: NaN or Inf loss encountered during evaluation epoch {epoch+1}. Skipping batch metrics.")

    # --- Metric Calculation (inside eval function) ---    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    precision, recall, f1, auc_roc, auc_pr, auc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # +++ Initialize confusion matrix variable +++
    cm = np.zeros((2, 2), dtype=int) # Default to zeros

    if len(all_labels_list) > 0:
        all_labels_np = np.array(all_labels_list).astype(int) # Ensure labels are integers for CM
        all_preds_prob_np = np.array(all_preds_prob)

        # Ensure probabilities are valid before metric calculation
        if np.any(np.isnan(all_preds_prob_np)) or np.any(np.isinf(all_preds_prob_np)):
            print("Warning: NaNs or Infs found in predicted probabilities. Metrics calculation skipped.")
        else:
            # Calculate binary predictions using threshold
            all_preds_binary = (all_preds_prob_np >= threshold).astype(int)

            # +++ Calculate Confusion Matrix +++
            try:
                # Ensure labels=[0, 1] to get TN, FP, FN, TP order consistently
                cm = confusion_matrix(all_labels_np, all_preds_binary, labels=[0, 1])
            except ValueError as e:
                # This might happen if only one class is predicted AND present
                print(f"Warning: Could not compute confusion matrix: {e}. Using default zero matrix.")
                # Try to compute precision/recall/f1 anyway, they might handle single class better
                # but the CM itself might fail.
        
            # --- Calculate other metrics (Precision, Recall, F1, AUCs) ---
            # Check for single class presence in TRUE labels for AUC calculation
            if len(np.unique(all_labels_np)) < 2:
                print(f"Warning: Only one class ({np.unique(all_labels_np)}) present in TRUE labels for epoch {epoch+1}. AUC ROC/AUC10 set to 0.0.")

            precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds_binary, average='binary', zero_division=0, labels=[0, 1])
            auc_roc = 0.0
            auc10 = 0.0
            try:
                auc_pr = average_precision_score(all_labels_np, all_preds_prob_np)
            except ValueError:
                auc_pr = 0.0
            else:
                all_preds_binary = (all_preds_prob_np >=threshold).astype(int)
                precision, recall, f1, _ = precision_recall_fscore_support(all_labels_np, all_preds_binary, average='binary', zero_division=0)
                try:
                    auc_pr = average_precision_score(all_labels_np, all_preds_prob_np)
                except ValueError as e:
                    print(f"Warning: ValueError during AUC-PR calculation: {e}")
                    auc_pr = 0.0
                try:
                    auc_roc = roc_auc_score(all_labels_np, all_preds_prob_np)
                    # Calculate AUC10 safely
                    try:
                        # max_fpr=0.1 might still raise ValueError if not enough points/classes
                        auc10 = roc_auc_score(all_labels_np, all_preds_prob_np, max_fpr=0.1)
                    except ValueError:
                        print("Warning: ValueError during AUC10 calculation (max_fpr=0.1). Setting AUC10 to 0.0.")
                        auc10 = 0.0 # Set to 0 if AUC10 calculation fails
                except ValueError as e:
                    print(f"Warning: ValueError during AUC-ROC calculation: {e}")
                    auc_roc = 0.0
                    auc10 = 0.0 # Also set AUC10 to 0 if AUC-ROC fails

    # --- Print Evaluation Results ---
    print(f"Epoch {epoch+1} Eval Results: ({trial_num}/{N_SEARCH_ITERATIONS})")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, AUC10: {auc10:.4f}")
    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    # +++ Print Confusion Matrix Nicely +++
    print("\n  Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"        Predicted 0    Predicted 1")
    print(f"Actual 0 | {cm[0, 0]:<12} | {cm[0, 1]:<12} | (TN={cm[0, 0]}, FP={cm[0, 1]})")
    print(f"Actual 1 | {cm[1, 0]:<12} | {cm[1, 1]:<12} | (FN={cm[1, 0]}, TP={cm[1, 1]})")
    
    # ++++++++++++++++++++++++++++++++++++++

    # nedenfor kan egentlig godt slettes
    # if writer:
    #     writer.add_scalar('Loss/val', avg_loss, epoch)
    #     writer.add_scalar('Metrics/F1', f1, epoch)
    #     writer.add_scalar('Metrics/AUC-PR', auc_pr, epoch)
    #     writer.add_scalar('Metrics/AUC-ROC', auc_roc, epoch)
    #     writer.add_scalar('Metrics/AUC10', auc10, epoch)
    #     writer.add_scalar('Metrics/Precision', precision, epoch)
    #     writer.add_scalar('Metrics/Recall', recall, epoch)
    
    return avg_loss, precision, recall, f1, auc_roc, auc_pr, auc10

cuda = input("Which GPU to use? [0, 1, 2, 3]: ")
threshold = float(input("Which threshold value? "))
# --- Configuration ---
# MAPPING
## 1. Define configurations for each type, including target GPU
EMBEDDING_CONFIG = {
    'esm2': {
        'h5_path': 'esm2_protein_embeddings.h5', # 
        'embed_dim': 1280,
        'target_gpu_index': 2  
    },
    'esmc': {
        'h5_path': 'esmc_protein_embeddings.h5', 
        'embed_dim': 960, 
        'target_gpu_index': 3  
    }
}

## 2. Ask the user for input
EMBEDDING_TYPE = ""
print("-----------------------------------------------------")
while EMBEDDING_TYPE not in EMBEDDING_CONFIG:
    EMBEDDING_TYPE = input(f"Select embedding type ({'/'.join(EMBEDDING_CONFIG.keys())}): ").lower().strip()
    if EMBEDDING_TYPE not in EMBEDDING_CONFIG:
        print("Invalid choice. Please try again.")
print("-----------------------------------------------------")


# PATHS AND DATA
config = EMBEDDING_CONFIG[EMBEDDING_TYPE]
H5_FILE_PATH = config['h5_path']
EMBED_DIM = config['embed_dim']
target_gpu = config['target_gpu_index']
early_stopping_counter = 0

# Use a timestamp for the main run directory to avoid overwriting previous searches
run_timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M")
BASE_RUNS_DIR = f'{EMBEDDING_TYPE}_transformer_runs_{run_timestamp}' # Base directory for this specific execution
RUN_TYPE_DIR = os.path.join(BASE_RUNS_DIR, EMBEDDING_TYPE) # subfolder in dir
MODEL_SAVE_DIR = os.path.join(RUN_TYPE_DIR, 'saved models') #subfolder with saved models
TENSORBOARD_LOG_DIR = os.path.join(RUN_TYPE_DIR, 'tensorboard_logs') 
RESULTS_FILE = os.path.join(RUN_TYPE_DIR, f'{EMBEDDING_TYPE}_hyperparam_search_results.json')

# --- Creating Directories ---
os.makedirs(RUN_TYPE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
print(f"Created directories. Models will be saved in: {MODEL_SAVE_DIR}")
print(f"TensorBoard logs will be saved in: {TENSORBOARD_LOG_DIR}")
print(f"Results JSON will be saved at: {RESULTS_FILE}")

# Fixed Training Settings
N_SPLITS = 4 # number of K-Fold splits
N_EPOCHS = 15 # Max epochs per fold
BATCH_SIZE = 8 # ændret fra 16 --> 8 for esm2, da GPU ikke kunne klare det. Fredag 18/04, kl. 11:51.
RANDOM_SEED = 21 #bc 21 is a cool number 8-)
EARLY_STOPPING_PATIENCE = 4 # Number of epochs to wait for improvement before stopping
# --- Random Search Space ---
param_dist = {
    'learning_rate': loguniform(1e-6, 5e-5), # !!!! come back to this
    'dropout': uniform(0.05, 0.25),
    'num_encoder_layers': [2, 3, 4 ,5], # they had 8 in Attention is all you need
    'nhead': [4, 6, 8, 10],
    'ffn_hidden_dim_factor': [2, 3 ,4],
    'weight_decay': loguniform(1e-3, 1e-1)
}
N_SEARCH_ITERATIONS = 20 # number of random combinations to try. Gemini recommends 30


# # --- Device Setup ---
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(f'--- Running Experiment for_ {EMBEDDING_TYPE} ---')
# print(f'Using device: {DEVICE}')
# print(f'Embeddings Path: {H5_FILE_PATH}')
# print(f'Embeddings Dimensin: {EMBED_DIM}')
# print(f'Output Directory: {RUN_TYPE_DIR}')

## 4. Determine the device string (cuda:X or cpu)
device_str = "cpu" # Default to CPU
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"CUDA is available. Found {num_gpus} GPU(s).")
    if target_gpu < num_gpus:
        # Use the target GPU if it exists
        device_str = f"cuda:{target_gpu}"
    elif num_gpus > 0:
        # Target GPU doesn't exist, fallback to GPU 0 if any GPUs are present
        print(f"Warning: Target GPU {target_gpu} not available ({num_gpus} GPUs). Using cuda:0.")
        device_str = "cuda:0"
    else:
        print("Warning: CUDA reports available but no devices found? Using CPU.")
else:
    print("CUDA not available. Using CPU.")

DEVICE = torch.device(device_str)


# 7. Print confirmation (uses variables set above)
print("-----------------------------------------------------")
print(f"--- Running Experiment for: {EMBEDDING_TYPE} ---")
print(f"Using device: {DEVICE}")
print(f"Embeddings Path: {H5_FILE_PATH}")
print(f"Embeddings Dimension: {EMBED_DIM}")
print(f"Output Directory: {RUN_TYPE_DIR}")
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


#### Main execution
# 1. Loading all protein keys (from hfd5 file)
try:
    with h5py.File(H5_FILE_PATH, 'r') as f:
        if 'embeddings_folder' not in f: raise KeyError("Not found")
        all_protein_keys = list(f['embeddings_folder'].keys())
        all_protein_keys.sort()
        all_protein_keys = np.array(all_protein_keys)
    print(f"Loaded {len(all_protein_keys)} keys from {H5_FILE_PATH}")

except FileNotFoundError:
    print("File not found")
    exit()

except KeyError as e:
    print(f"Error loading from HDF {e}")
except Exception as e:
    print(f"Unexpected error loading keys; {e}")
    exit()

# Hyperparameter Search Initialization
sampler = ParameterSampler(param_dist, n_iter=N_SEARCH_ITERATIONS, random_state=RANDOM_SEED)
all_trial_results = []
best_overall_avg_auc_pr = -1 # Track best average AUC-PR across all trials
best_trial_config = None # Store the config of the best trial


print(f"\n--- Starting Hyperparameter Search ({N_SEARCH_ITERATIONS} trials) for {EMBEDDING_TYPE} ---")


# --- Setup ONE TensorBoard Writer for the entire hyperparameter search ---
# This writer will log aggregated results for each trial using add_hparams
# All logs for this run will be under TENSORBOARD_LOG_DIR
main_tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
print(f"TensorBoard HParam logs will be stored in: {TENSORBOARD_LOG_DIR}")


# --- Trial Loop ---
for trial_num, params in enumerate(sampler, 1):
    print(f"\n----- Trial {trial_num}/{N_SEARCH_ITERATIONS} ({EMBEDDING_TYPE}) -----")
    print(f"Parameters: {params}")

    # Extract params for this trial
    LEARNING_RATE = params['learning_rate']
    DROPOUT = params['dropout']
    NUM_ENCODER_LAYERS = params['num_encoder_layers']
    N_HEAD = params['nhead']
    WEIGHT_DECAY = params['weight_decay']
    FFN_HIDDEN_DIM_FACTOR = params['ffn_hidden_dim_factor']
    FFN_HIDDEN_DIM = int(EMBED_DIM * FFN_HIDDEN_DIM_FACTOR) # Calculate actual hidden dim

      # --- Parameter Sanity Check ---
    if EMBED_DIM % N_HEAD != 0:
        print(f"Skipping trial: embed_dim ({EMBED_DIM}) is not divisible by nhead ({N_HEAD}).")
        # Log skipped trial info (optional)
        trial_summary = {
            'trial_num': trial_num, 'params': params, 'status': 'skipped',
            'reason': f'embed_dim ({EMBED_DIM}) % nhead ({N_HEAD}) != 0',
            'avg_metrics': {}, 'std_metrics': {}, 'individual_fold_metrics': []
            }
        all_trial_results.append(trial_summary)
        continue # Move to the next trial

    # 3. K-Fold Cross-Validation Loop for this set of parameters
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(all_protein_keys)):
        print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
        set_seed(RANDOM_SEED + fold) # Set seed for this specific fold for reproducibility within the fold

        train_keys = all_protein_keys[train_idx]
        val_keys = all_protein_keys[val_idx]

        # Create datasets and dataloaders
        train_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=list(train_keys))
        val_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=list(val_keys))
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True if DEVICE == torch.device("cuda") else False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True if DEVICE == torch.device("cuda") else False)

        # Calculate pos_weight
        print("Calculating pos_weight...")
        num_pos, num_neg = 0, 0
        # Safer iteration to calculate pos_weight
        temp_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        for batch in tqdm.tqdm(temp_loader, desc="Calculating pos_weight", leave=False, ncols=80):
            labels = batch['labels'] # Labels are already tensors
            active_mask = (labels != -1)
            active_labels = labels[active_mask].numpy() # Calculate on active labels
            num_pos += np.sum(active_labels == 1)
            num_neg += np.sum(active_labels == 0)
        del temp_loader # Free memory

        if num_pos == 0 or num_neg == 0:
            print(f"Warning: Fold {fold+1} - num_pos={num_pos}, num_neg={num_neg}. Using pos_weight=1.0")
            pos_weight = 1.0
        else:
            pos_weight = num_neg / num_pos
        print(f"Fold {fold+1} - pos_weight: {pos_weight:.2f}")
        pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
        # BCEWithLogitsLoss with pos_weight for imbalance
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        # Define Warmup Epochs
        N_WARMUP_EPOCHS = 3 


        # Instantiate NEW model and optimizer
        model = EpitopeTransformer(
            embed_dim=EMBED_DIM,
            nhead=N_HEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            ffn_hidden_dim=FFN_HIDDEN_DIM,
            dropout=DROPOUT,
            pre_norm=True, # True to use pre-LN
            use_seq_length=True
            ).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

        # Learning Rate Warmup
        ## Scheduler 1: Linear Warmup
        scheduler_warmup = LinearLR(optimizer,
                                    start_factor=0.1, # Start LR close to 0
                                    end_factor=1.0,    # End LR at the LEARNING_RATE set in optimizer
                                    total_iters=N_WARMUP_EPOCHS) # Duration of warmup in epochs

        ## Scheduler 2: Cosine Decay
        ## T_max is the number of steps for the decay phase
        scheduler_decay = CosineAnnealingLR(optimizer,
                                            T_max=(N_EPOCHS - N_WARMUP_EPOCHS), # Number of epochs AFTER warmup
                                            eta_min=LEARNING_RATE * 0.01) # Minimum LR, e.g., 1% of peak LR

        ## Combine schedulers: Apply warmup first, then decay
        scheduler = SequentialLR(optimizer, 
                                 schedulers=[scheduler_warmup, 
                                 scheduler_decay], 
                                 milestones=[N_WARMUP_EPOCHS])

        best_val_f1 = -1
        best_fold_metrics = {}
        model_saved_for_fold = False # Flag to check if a model was saved
        
        # Initialize Early Stopping Counter
        epochs_no_improve = 0

        # 4. Epoch Loop for the Fold
        print(f"Starting training for max {N_EPOCHS} epochs (Warmup_ {N_WARMUP_EPOCHS}, Patience: {EARLY_STOPPING_PATIENCE})...")
        # --- Run Training for One Epoch ---
        for epoch in range(N_EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
            current_lr = optimizer.param_groups[0]['lr'] # to show the current LR of the epoch
            print(f"Epoch {epoch+1}/{N_EPOCHS} - Train Loss: {train_loss:.4f} | LR Used: {current_lr:.2e}") # Changed print label
            val_loss, precision, recall, f1, auc_roc, auc_pr, auc10 = evaluate(model, val_loader, criterion, DEVICE, epoch)

            # --- Early Stopping Logic & Model Saving ---
            # Check validation metrics and saves if best model so far
            if f1 > best_val_f1:
                print(f"Epoch {epoch+1}: Val F1 improved ({best_val_f1:.4f} -> {f1:.4f}). Saving model & resetting patience.")
                best_val_f1 = f1
                best_fold_metrics = {
                    'loss': val_loss, 'precision': precision, 'recall': recall,
                    'f1': f1, 'auc_roc': auc_roc, 'auc_pr': auc_pr, 'auc10': auc10,'epoch': epoch+1
                }
                # Save model to the type-specific directory
                model_save_path = os.path.join(MODEL_SAVE_DIR, f'{EMBEDDING_TYPE}_trial_{trial_num}_fold_{fold+1}_best.pth')
                #model_save_path = os.path.join(MODEL_SAVE_DIR, f'trial_{trial_num}_fold_{fold+1}_best.pth') # denne indentede linje er blevet erstattet med den foroven.
                torch.save(model.state_dict(), model_save_path)

                try:
                    torch.save(model.state_dict(), model_save_path)
                    model_saved_for_fold = True # You might not need this flag anymore
                    print(f"Model saved to {model_save_path}") # Optional print
                    print(f"------------------------------------------")
                except Exception as e:
                    print(f"Error saving model for trial {trial_num}, fold {fold+1}: {e}")

                epochs_no_improve = 0 # Reset counter because performance improved
            else:
                epochs_no_improve += 1
                print(f"Epoch {epoch+1}: Val F1 ({f1:.4f}) did not improve from best ({best_val_f1:.4f}). Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
                print(f"------------------------------------------")
                if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    early_stopping_counter +=1 
                    print(early_stopping_counter)
                    print(f"\n--- Early stopping triggered for Fold {fold+1} at epoch {epoch+1} ---")
                    print(f"------------------------------------------")
                    break # Exit the epoch loop for this fold

            scheduler.step() #updates LR in optimzer (warmup or decay). New LR is in next epoch.

        # --- End of EPOCH Loop ---

        if best_fold_metrics: # Check if any improvement was found during epochs
            print(f"\nBest Validation Metrics for Trial {trial_num}, Fold {fold + 1} (Found at Epoch {best_fold_metrics.get('epoch', 'N/A')}):")
            # Print metrics clearly
            for key, val in best_fold_metrics.items():
                if key != 'epoch': print(f"  {key.upper()}: {val:.4f}", end=" | ")
            print() # Newline
            fold_results.append(best_fold_metrics) # Add the best metrics for this fold to the trial's list
        else:
            print(f"\nFold {fold+1} did not yield improving metrics based on AUC-PR score. No model saved for this fold.")
            print(f"------------------------------------------")


        # Garbage collector
        del model, optimizer, train_loader, val_loader, train_dataset, val_dataset, criterion, pos_weight_tensor
        gc.collect()
        if DEVICE == torch.device("cuda"): torch.cuda.empty_cache()
    
    # --- End of Fold Loop ---

        # --- Summarize Fold Results & Log to TensorBoard for the Current Trial ---
    trial_avg_metrics = {}
    trial_std_metrics = {}
    if fold_results: # Check if any folds completed successfully and yielded metrics
        # Calculate average and std dev metrics across successful folds for this trial
        # Ensure all dicts in fold_results have the same keys before averaging
        metric_keys = fold_results[0].keys() if fold_results else []
        valid_metric_keys = [k for k in metric_keys if k != 'epoch'] # Exclude 'epoch' from averaging

        trial_avg_metrics = {key: np.mean([fold[key] for fold in fold_results])
                             for key in valid_metric_keys}
        trial_std_metrics = {key: np.std([fold[key] for fold in fold_results])
                             for key in valid_metric_keys}

        print(f"\n----- Trial {trial_num} ({EMBEDDING_TYPE}) Cross-Validation Summary -----")
        print("Average Metrics across folds:")
        for key, value in trial_avg_metrics.items():
            print(f"  Avg {key.upper()}: {value:.4f} (+/- {trial_std_metrics[key]:.4f})")

        # --- Log Hyperparameters and Aggregated Metrics to TensorBoard ---
        # Prepare hyperparameters dictionary for logging (ensure values are simple types)
        hparams_for_log = {k: v for k, v in params.items()}

        # Prepare metrics dictionary for logging (use the calculated averages)
        # Make keys more descriptive for TensorBoard HParams view
        metrics_for_log = {f'metrics/avg_{key}': value for key, value in trial_avg_metrics.items()}
        # Optionally add standard deviations too
        # metrics_for_log.update({f'metrics/std_{key}': value for key, value in trial_std_metrics.items()})

        try:
            # Use trial_num as a unique identifier for this run within TensorBoard HParams
            # Pass hparams_for_log and metrics_for_log
            main_tb_writer.add_hparams(hparams_for_log, metrics_for_log, run_name=f'trial_{trial_num}_results')
            print(f"Logged Trial {trial_num} summary to TensorBoard.")
        except Exception as e:
            print(f"Error logging Trial {trial_num} to TensorBoard: {e}")


        # Check if this trial is the best one so far based on average F1
        current_avg_auc_pr = trial_avg_metrics.get('auc_pr', -1)
        current_time = time.time()
        elapsed_seconds = current_time - script_start_time
        elapsed_time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
        if current_avg_auc_pr > best_overall_avg_auc_pr:
            print(f"--- *** New Best Trial Found (Trial {trial_num}) *** ---")
            print(f"--- *** Average AUC-PR improved from {best_overall_avg_auc_pr:.4f} to {current_avg_auc_pr:.4f} *** ---")
            best_overall_avg_auc_pr = current_avg_auc_pr
            best_trial_config = {'trial_num': trial_num, 'params': params, 'avg_metrics': trial_avg_metrics}

        trial_summary = {
            'trial_num': trial_num, 'params': params, 'elapsed_time': elapsed_seconds, 'status': 'completed',
            'avg_metrics': trial_avg_metrics, 'std_metrics': trial_std_metrics,
            'individual_fold_metrics': fold_results # Keep individual fold results if needed
            }
    else:
        # Handle case where no folds completed successfully for this trial
        print(f"----- Trial {trial_num} ({EMBEDDING_TYPE}) had no successful folds -----")
        trial_summary = {
            'trial_num': trial_num, 'params': params, 'status': 'failed',
            'reason': 'No folds completed successfully or yielded improving metrics.',
            'avg_metrics': {}, 'std_metrics': {}, 'individual_fold_metrics': []
            }

    all_trial_results.append(trial_summary)

    # --- Save results incrementally after each trial ---
    try:
        # Use a helper function to convert numpy types for JSON compatibility
        def default_serializer(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                                np.int16, np.int32, np.int64, np.uint8,
                                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)): # Handle arrays if needed, e.g., convert to list
                 return obj.tolist()
            # Add other numpy types if necessary
            elif isinstance(obj, (np.bool_)):
                return bool(obj)
            elif isinstance(obj, (np.void)): # Handle numpy void type if it appears
                return None # Or some other representation
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(RESULTS_FILE, 'w') as f:
            json.dump(all_trial_results, f, indent=4, default=default_serializer)
        print(f"Trial {trial_num} results updated in {RESULTS_FILE}")
    except IOError as e:
        print(f"Error saving results to {RESULTS_FILE}: {e}")
    except TypeError as e:
        print(f"Error serializing results to JSON: {e}. Check data types in results.")
        # You might need to inspect `all_trial_results` to find the problematic type
        # print(all_trial_results[-1]) # Print the last trial's data to debug

# --- End of Trial Loop ---

# Close the main TensorBoard writer
main_tb_writer.close()

# --- Final Hyperparameter Search Summary ---
print(f"\n--- Hyperparameter Search Complete for {EMBEDDING_TYPE} ---")
if all_trial_results:
    # Filter out trials with no avg_metrics before finding max
    valid_trials = [t for t in all_trial_results if t.get('avg_metrics')]
    if valid_trials:
        best_trial = max(valid_trials, key=lambda x: x.get('avg_metrics', {}).get('auc_pr', -1))
        print("\nBest Trial Found:")
        print(f"  Trial Number: {best_trial['trial_num']}")
        print(f"  Parameters: {best_trial['params']}")
        print(f"  Avg F1 Score: {best_trial.get('avg_metrics', {}).get('f1', 'N/A'):.4f}")
        print(f"  Avg AUC-PR: {best_trial.get('avg_metrics', {}).get('auc_pr', 'N/A'):.4f}")
    else:
        print("No trials yielded valid average metrics.")

    print(f"\nFull results saved in: {RESULTS_FILE}")
    print(f"TensorBoard logs in: {TENSORBOARD_LOG_DIR}")
    print(f"Best model checkpoints saved in: {MODEL_SAVE_DIR}")
else:
    print("No successful trials were completed.")

# --- Calculate and Print Elapsed Time ---
script_end_time = time.time()
total_elapsed_seconds = script_end_time - script_start_time
# Format into H:M:S
elapsed_time_str = str(datetime.timedelta(seconds=total_elapsed_seconds))
print("\n-----------------------------------------------------")
print(f"Script Finished.")
print(f"Total Execution Time: {elapsed_time_str}")
print(f"Amount of Early Stoppings {early_stopping_counter}")
print("-----------------------------------------------------")