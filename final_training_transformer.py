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
import smtplib # for email
from email.mime.text import MIMEText # for email 
from email.mime.multipart import MIMEMultipart # for email
from scipy.stats import loguniform, uniform
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR # For warm up LR
from sklearn.model_selection import train_test_split 

import torch
import torch.nn as nn
import torch.nn.functional as F # For Swish if needed
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix

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
            self.output_layer = nn.Linear(embed_dim + 1, 1) # transformed to 1 dimension and adding sequence lenght
        else:
            self.output_layer = nn.Linear(embed_dim , 1) # transformed to 1 dimension
        self.init_weights() # optional, but good practice

    def init_weights(self):
        initrange = 0.1
        nn.init.zeros_(self.output_layer.bias)
        nn.init.uniform_(self.output_layer.weight, -initrange, initrange) # initialize bias with zeros, weights with uniform

    def forward(self, src, src_key_padding_mask=None, seq_lengths = None): #padding mask=None as we're only in encoder layer
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
            output_logits = self.output_layer(output)
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
        lengths = batch['lengths'].to(device)

        optimizer.zero_grad()
        outputs = model(embeddings, src_key_padding_mask=padding_mask, seq_lengths = lengths) # Shape: [B, SeqLen, 1]
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
            lengths = batch['lengths'].to(device)
            outputs = model(embeddings,src_key_padding_mask=padding_mask, seq_lengths = lengths).squeeze(-1)
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
    
    if len(all_preds_prob) > 0 and len(all_labels_list) > 0:
        all_labels_np = np.array(all_labels_list).astype(int) # Ensure labels are integers for CM
        all_preds_prob_np = np.array(all_preds_prob)
        # Ensure probabilities are valid before metric calculation
        if np.any(np.isnan(all_preds_prob_np)) or np.any(np.isinf(all_preds_prob_np)):
            print("Warning: NaNs or Infs found in predicted probabilities. Metrics calculation skipped.")
        else:
            # Calculate binary predictions using fixed 0.5 threshold
            all_preds_binary = (all_preds_prob_np >= 0.5).astype(int)
            # +++ Calculate Confusion Matrix +++
            try:
                # Ensure labels=[0, 1] to get TN, FP, FN, TP order consistently
                cm = confusion_matrix(all_labels_np, all_preds_binary, labels=[0, 1])
            except ValueError as e:
                # This might happen if only one class is predicted AND present
                print(f"Warning: Could not compute confusion matrix: {e}. Using default zero matrix.")
        
            # --- Calculate other metrics (Precision, Recall, F1, AUCs) ---
            # Check for single class presence in TRUE labels for AUC calculation
            if len(np.unique(all_labels_np)) < 2:
                print(f"Warning: Only one class ({np.unique(all_labels_np)}) present in TRUE labels for epoch {epoch+1}. AUC ROC/AUC10 set to 0.0.")
            precision, recall, f1, support = precision_recall_fscore_support(all_labels_np, all_preds_binary, average='binary', zero_division=0, labels=[0, 1])
            auc_roc = 0.0
            auc10 = 0.0
            try:
                auc_pr = average_precision_score(all_labels_np, all_preds_prob_np)
            except ValueError:
                auc_pr = 0.0
            else:
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
    print(f"Epoch {epoch+1})")
    print(f"  Loss: {avg_loss:.4f}")
    print(f"  AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, AUC10: {auc10:.4f}")
    print(f"  F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    # +++ Print Confusion Matrix Nicely +++
    print("\n  Confusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"        Predicted 0    Predicted 1")
    print(f"Actual 0 | {cm[0, 0]:<12} | {cm[0, 1]:<12} | (TN={cm[0, 0]}, FP={cm[0, 1]})")
    print(f"Actual 1 | {cm[1, 0]:<12} | {cm[1, 1]:<12} | (FN={cm[1, 0]}, TP={cm[1, 1]})")
    print(f"------------------------------------------")
    
    return avg_loss, precision, recall, f1, auc_roc, auc_pr, auc10

cuda = int(input("Which GPU to use? [0, 1, 2, 3]: "))
# --- Configuration ---
# MAPPING
## 1. Define configurations for each type, including target GPU
EMBEDDING_CONFIG = {
    'esm2': {
        'h5_path': 'esm2_protein_embeddings.h5', 
        'h5_test_path': 'esm2_test_protein_embeddings.h5',
        'embed_dim': 1280,
        'target_gpu_index': cuda  
    },
    'esmc': {
        'h5_path': 'esmc_protein_embeddings.h5', 
        'h5_test_path': 'esmc_test_protein_embeddings.h5',
        'embed_dim': 960, 
        'target_gpu_index': cuda 
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
H5_TEST_PATH = config['h5_test_path']
EMBED_DIM = config['embed_dim']
target_gpu = config['target_gpu_index']

# Use a timestamp for the main run directory to avoid overwriting previous searches
run_timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M")
BASE_RUNS_DIR = f'{EMBEDDING_TYPE}_best_transformer_runs_{run_timestamp}' # Base directory for this specific execution
RUN_TYPE_DIR = os.path.join(BASE_RUNS_DIR, EMBEDDING_TYPE) # subfolder in dir
MODEL_SAVE_DIR = os.path.join(RUN_TYPE_DIR, 'saved models') #subfolder with saved models
TENSORBOARD_LOG_DIR = os.path.join(RUN_TYPE_DIR, 'tensorboard_logs') 
RESULTS_FILE = os.path.join(RUN_TYPE_DIR, f'{EMBEDDING_TYPE}_best_model_results.json')

# --- Creating Directories ---
os.makedirs(RUN_TYPE_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# Fixed Training Settings
N_EPOCHS = 100 # Max epochs 
BATCH_SIZE = 8 # !!!maybe adjust for GPU memory!!!!
RANDOM_SEED = 32 #bc 32 is a cool number 8-)
EARLY_STOPPING_PATIENCE = 10 # Number of epochs to wait for improvement before stopping
N_SPLITS = 5 # k fold validation

PATH_TO_JSON =""
while not PATH_TO_JSON.lower().endswith('json'):
    PATH_TO_JSON = input("Give path to JSON file with best performance: ")
with open(PATH_TO_JSON, 'r') as f:
    data = json.load(f)

METRIC_TYPE = ""
while METRIC_TYPE.lower() not in ["f1", "auc_pr", "recall", "precision", "auc-roc", "auc10"]:
    METRIC_TYPE = input("Which metric do you want to use when evaluating the best model? F1, auc_pr, recall, precision, roc-auc, or auc10: ").lower()
    if METRIC_TYPE not in ["f1", "auc_pr", "recall", "precision", "auc-roc", "auc10"]:
        print("Try again")

max_val_metric = 0
best_run = -1
for run_index, run_data in enumerate(data):
    params = run_data.get('params')
    if not params:
        continue
    avg_metrics = run_data.get('avg_metrics')
    if not isinstance(avg_metrics, dict) or not avg_metrics: # Checks if it's None, not a dict, or an empty dict
        continue # Skip this iteration
    metric = avg_metrics.get(METRIC_TYPE) # Use .get() which returns None if key is absent
    if metric is None:
        # Optional: Log this
        # print(f"Info: Skipping run index {run_index} because metric '{METRIC_TYPE}' not found in 'avg_metrics'.")
        continue
    if metric > max_val_metric:
        best_run = run_index
        max_val_metric = metric
        
# Add a check to ensure we found at least one valid run
if best_run == -1:
    print("No valid runs found with the specified metric. Exiting.")
    sys.exit(1)
    
best_params = data[best_run]['params']
print("-----------------------------------------------------")
print(f"Best trial number found with random search: {best_run+1}")
print("-----------------------------------------------------")

LEARNING_RATE = best_params['learning_rate']
DROPOUT = best_params['dropout']
FFN_HIDDEN_DIM_FACTOR = best_params['ffn_hidden_dim_factor']
NUM_ENCODER_LAYERS = best_params['num_encoder_layers']
N_HEAD = best_params['nhead']
FFN_HIDDEN_DIM = int(EMBED_DIM * FFN_HIDDEN_DIM_FACTOR)  # Calculate actual hidden dim
WEIGHT_DECAY = best_params['weight_decay']

# --- Setup TensorBoard Writer ---
tb_writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)

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
print(f"Model concfig: Dropout {DROPOUT}, nheads {N_HEAD}, ffn {FFN_HIDDEN_DIM_FACTOR}, lr {LEARNING_RATE}, num encoder {NUM_ENCODER_LAYERS}")
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

#!!!!!!!!Training start!!!!!!!!!!!!!!
kf = KFold(n_splits=4, shuffle=True, random_state=RANDOM_SEED)
fold_results = []
final_best_model_path = None

for fold, (train_idx, val_idx) in enumerate(kf.split(all_protein_keys)):
    print(f"\n--- Fold {fold + 1}/{N_SPLITS} ---")
    set_seed(RANDOM_SEED + fold)  # Set seed for reproducibility

    train_keys = all_protein_keys[train_idx]
    val_keys = all_protein_keys[val_idx]

    # Create datasets and dataloaders
    train_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=list(train_keys))
    val_dataset = Embedding_retriever(H5_FILE_PATH, protein_keys=list(val_keys))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, 
                             num_workers=2, pin_memory=True if DEVICE == torch.device("cuda") else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, 
                           num_workers=2, pin_memory=True if DEVICE == torch.device("cuda") else False)

    print("Calculating pos_weight...")
    num_pos, num_neg = 0, 0
    temp_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    for batch in tqdm.tqdm(temp_loader, desc="Calculating pos_weight", leave=False, ncols=80):
        labels = batch['labels']
        active_mask = (labels != -1)
        active_labels = labels[active_mask].numpy()
        num_pos += np.sum(active_labels == 1)
        num_neg += np.sum(active_labels == 0)
    del temp_loader  # Free memory

    if num_pos == 0 or num_neg == 0:
        print(f"Warning: Fold {fold+1} - num_pos={num_pos}, num_neg={num_neg}. Using pos_weight=1.0")
        pos_weight = 1.0
    else:
        pos_weight = num_neg / num_pos * 0.7
    print(f"Fold {fold+1} - pos_weight: {pos_weight:.2f}")
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Define Warmup Epochs
    N_WARMUP_EPOCHS = 3 # Eksempel: Warmup over the first epoch. Adjust if needed.

    model = EpitopeTransformer(
        embed_dim=EMBED_DIM,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        ffn_hidden_dim=FFN_HIDDEN_DIM,
        dropout=DROPOUT,
        pre_norm=True, # True to use pre-LN
        use_seq_length = True
        ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Learning Rate Warmup
    ## Scheduler 1: Linear Warmup
    scheduler_warmup = LinearLR(optimizer,
                                start_factor=1e-6, # Start LR close to 0
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

    best_val_f1= -1
    best_fold_metrics = {}
    epochs_no_improve = 0

    # Epoch Loop for the Fold
    print(f"Starting training for max {N_EPOCHS} epochs (Patience: {EARLY_STOPPING_PATIENCE})...")
    for epoch in range(N_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE, epoch)
        current_lr = optimizer.param_groups[0]['lr'] # to show the current LR of the epoch
        print(f"Epoch {epoch+1}/{N_EPOCHS} - Train Loss: {train_loss:.4f} | LR Used: {current_lr:.2e}") # Changed print label
        val_loss, precision, recall, f1, auc_roc, auc_pr, auc10= evaluate(model, val_loader, criterion, DEVICE, epoch)
        
        # Log metrics to TensorBoard
        tb_writer.add_scalar('loss/train', train_loss, global_step=epoch)
        tb_writer.add_scalar('loss/val', val_loss, global_step=epoch)
        tb_writer.add_scalar('metrics/val_precision', precision, global_step=epoch)
        tb_writer.add_scalar('metrics/val_recall', recall, global_step=epoch)
        tb_writer.add_scalar('metrics/val_f1', f1, global_step=epoch)
        tb_writer.add_scalar('metrics/val_auc_roc', auc_roc, global_step=epoch)
        tb_writer.add_scalar('metrics/val_auc_pr', auc_pr, global_step=epoch)
        tb_writer.add_scalar('metrics/val_auc10', auc10, global_step=epoch)

        if f1 > best_val_f1:
            print(f"Epoch {epoch+1}: Val F1 improved ({best_val_f1:.4f} -> {f1:.4f}). Saving model & resetting patience.")
            best_val_f1 = f1
            best_fold_metrics = {
                'loss': val_loss, 'precision': precision, 'recall': recall,
                'f1': f1, 'auc_roc': auc_roc, 'auc_pr': auc_pr, 'auc10': auc10, 'epoch': epoch+1
            }
            
            # Save model
            model_save_path = os.path.join(MODEL_SAVE_DIR, f'{EMBEDDING_TYPE}_best_model_fold_{fold+1}.pth')
            try:
                torch.save(model.state_dict(), model_save_path)
                print(f"Model saved to {model_save_path}")
            except Exception as e:
                print(f"Error saving model for fold {fold+1}: {e}")

            epochs_no_improve = 0  # Reset counter
        else:
            epochs_no_improve += 1
            print(f"Epoch {epoch+1}: Val F1 ({f1:.4f}) did not improve from best ({best_val_f1:.4f}). "
                  f"Patience: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"\n--- Early stopping triggered for Fold {fold+1} at epoch {epoch+1} ---")
                break  # Exit the epoch loop
        scheduler.step() #updates LR in optimzer (warmup or decay). New LR is in next epoch.

    # End of Epoch Loop
    if best_fold_metrics:
        print(f"\nBest Validation Metrics for Fold {fold + 1} (Found at Epoch {best_fold_metrics.get('epoch', 'N/A')}):")
        for key, val in best_fold_metrics.items():
            if key != 'epoch': 
                print(f"  {key.upper()}: {val:.4f}", end=" | ")
        print()  # Newline
        fold_results.append(best_fold_metrics)
    else:
        print(f"\nFold {fold+1} did not yield improving metrics based on AUC-PR score. No model saved for this fold.")

    # Clean up memory
    del model, optimizer, train_loader, val_loader, train_dataset, val_dataset, criterion, pos_weight_tensor
    gc.collect()
    if DEVICE == torch.device("cuda"): 
        torch.cuda.empty_cache()

# ----- End of K-Fold Cross-Validation Loop -----
# Calculate and summarize fold results
if fold_results:
    # Calculate average metrics across folds
    metric_keys = fold_results[0].keys() if fold_results else []
    valid_metric_keys = [k for k in metric_keys if k != 'epoch']

    avg_metrics = {key: np.mean([fold[key] for fold in fold_results]) for key in valid_metric_keys}
    std_metrics = {key: np.std([fold[key] for fold in fold_results]) for key in valid_metric_keys}

    print("\n----- Cross-Validation Summary -----")
    print("Average Metrics across folds:")
    for key, value in avg_metrics.items():
        print(f"  Avg {key.upper()}: {value:.4f} (+/- {std_metrics[key]:.4f})")
    
    # Find the best fold based on the chosen metric
    best_fold_index = max(range(len(fold_results)), 
                          key=lambda i: fold_results[i][METRIC_TYPE])
    best_fold_num = best_fold_index + 1
    best_fold_metric_value = fold_results[best_fold_index][METRIC_TYPE]
    
    # Save the best model with a clear name
    best_model_source_path = os.path.join(MODEL_SAVE_DIR, f'{EMBEDDING_TYPE}_best_model_fold_{best_fold_num}.pth')
    final_best_model_path = os.path.join(MODEL_SAVE_DIR, f'{EMBEDDING_TYPE}_FINAL_BEST_MODEL.pth')
    
    if os.path.exists(best_model_source_path):
        try:
            import shutil
            shutil.copy2(best_model_source_path, final_best_model_path)
            print(f"\n----- BEST MODEL SAVED -----")
            # ... rest of print statements ...
        except Exception as e:
            print(f"Error saving final best model: {e}")
            final_best_model_path = None # Indicate failure
    else:
        print(f"Error: Source model path for best fold ({best_fold_num}) not found: {best_model_source_path}")
        final_best_model_path = None # Indicate failure

    # Save results to JSON file
    results_summary = {
        'hyperparameters': {
            'learning_rate': LEARNING_RATE,
            'dropout': DROPOUT,
            'nhead': N_HEAD,
            'num_encoder_layers': NUM_ENCODER_LAYERS,
            'ffn_hidden_dim_factor': FFN_HIDDEN_DIM_FACTOR
        },
        'fold_results': fold_results,
        'avg_metrics': avg_metrics,
        'std_metrics': std_metrics,
        'embedding_type': EMBEDDING_TYPE,
        'best_fold': best_fold_num,
        'best_model_path': final_best_model_path
    }
    # Close TensorBoard writer
    tb_writer.close()
    # Function to handle JSON serialization of numpy types
    def default_serializer(obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")
    
    try:
        with open(RESULTS_FILE, 'w') as f:
            json.dump(results_summary, f, indent=4, default=default_serializer)
        print(f"Results saved to {RESULTS_FILE}")
    except Exception as e:
        print(f"Error saving results to {RESULTS_FILE}: {e}")
else:
    print("No successful folds were completed.")



# Calculate and Print Elapsed Time
script_end_time = time.time()
total_elapsed_seconds = script_end_time - script_start_time
elapsed_time_str = str(datetime.timedelta(seconds=int(total_elapsed_seconds)))
print("\n-----------------------------------------------------")
print(f"Training Finished.")
print(f"Total Execution Time: {elapsed_time_str}")
print("-----------------------------------------------------")

print("-----------------------------------------------------")
print("Evaluation on test set")
print("-----------------------------------------------------")

test_metrics = {} # Dictionary to store test results
if final_best_model_path and H5_TEST_PATH and os.path.exists(H5_TEST_PATH):
    print("\n----- Evaluating Best Model on Hold-Out Test Set -----")
    print(f"Test Data HDF5: {H5_TEST_PATH}")

    # 1. Create Test Dataset and DataLoader
    try:
        print("Loading test data...")
        test_dataset = Embedding_retriever(H5_TEST_PATH) # Load all keys from the test HDF5
        if len(test_dataset) == 0:
             print("Warning: Test dataset is empty. Skipping test evaluation.")
             test_metrics = {"status": "Skipped - Test dataset empty"}
             dataset_creation_ok = False
        else:
            print(f"Test dataset loaded with {len(test_dataset)} proteins.")
            test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                                     num_workers=2, pin_memory=(DEVICE.type == 'cuda'))
            dataset_creation_ok = True
    except FileNotFoundError:
        print(f"Error: Test HDF5 file not found at {H5_TEST_PATH}. Skipping test evaluation.")
        test_metrics = {"error": f"Test HDF5 not found: {H5_TEST_PATH}"}
        dataset_creation_ok = False
    except KeyError as e:
         print(f"Error: Structure issue (e.g., missing 'embeddings_folder') in test HDF5 {H5_TEST_PATH}: {e}. Skipping test evaluation.")
         test_metrics = {"error": f"KeyError in test HDF5: {e}"}
         dataset_creation_ok = False
    except Exception as e:
         print(f"Error creating test dataset/loader: {e}. Skipping test evaluation.")
         test_metrics = {"error": f"Test dataset creation failed: {e}"}
         dataset_creation_ok = False

    # --- Proceed only if dataset and loader are okay ---
    if dataset_creation_ok:
        # 2. Load the best model architecture and weights
        print(f"Loading best model weights from: {final_best_model_path}")
        model_test = EpitopeTransformer(
            embed_dim=EMBED_DIM,
            nhead=N_HEAD,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            ffn_hidden_dim=FFN_HIDDEN_DIM,
            dropout=DROPOUT,
            pre_norm=True,
            use_seq_length=True
        ).to(DEVICE)

        try:
            # Load the state dict onto the correct device
            checkpoint = torch.load(final_best_model_path, map_location=DEVICE)
            model_test.load_state_dict(checkpoint['model_state_dict'])

            model_test.eval() # Set model to evaluation mode (important!)

            # 3. Perform Evaluation on Test Set
            all_test_preds_prob = []
            all_test_labels_list = []
            print("Running inference on test set...")
            with torch.no_grad(): # Disable gradient calculations
                test_progress_bar = tqdm.tqdm(test_loader, desc="Test Set Evaluation", leave=False, ncols=100)
                for batch in test_progress_bar:
                    if batch is None: continue 

                    embeddings = batch['embeddings'].to(DEVICE)
                    labels = batch['labels'].to(DEVICE)
                    padding_mask = batch['padding_mask'].to(DEVICE)

                    # Get model output (logits)
                    outputs = model_test(embeddings, src_key_padding_mask=padding_mask).squeeze(-1)

                    # Mask for active (non-padded) residues
                    active_mask = (labels != -1)

                    if active_mask.sum() > 0:
                        active_logits = outputs[active_mask]
                        active_labels = labels[active_mask]

                        if active_labels.numel() > 0:
                            # Check for NaNs/Infs in logits *before* sigmoid
                            if torch.isnan(active_logits).any() or torch.isinf(active_logits).any():
                                print("Warning: NaN or Inf detected in test logits. Skipping batch metrics.")
                                continue # Skip metrics for this batch

                            # Calculate probabilities
                            probs = torch.sigmoid(active_logits).cpu().numpy()
                            all_test_preds_prob.extend(probs)
                            all_test_labels_list.extend(active_labels.cpu().numpy())

            # 4. Calculate Test Metrics (if predictions were generated)
            if len(all_test_labels_list) > 0 and len(all_test_preds_prob) > 0:
                print("Calculating final test metrics...")
                all_test_labels_np = np.array(all_test_labels_list).astype(int)
                all_test_preds_prob_np = np.array(all_test_preds_prob)

                test_precision, test_recall, test_f1, test_auc_roc, test_auc_pr, test_auc10 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                test_cm = np.zeros((2, 2), dtype=int)

                # Final check for NaNs/Infs in probabilities
                if np.any(np.isnan(all_test_preds_prob_np)) or np.any(np.isinf(all_test_preds_prob_np)):
                    print("Warning: NaNs or Infs found in final predicted test probabilities. Metrics calculation may be inaccurate.")
                    test_metrics = {"error": "NaNs/Infs in test predictions"}
                elif all_test_labels_np.size == 0:
                     print("Warning: No valid labels collected for test set. Test metrics calculation skipped.")
                     test_metrics = {"error": "No valid test labels collected"}
                else:
                    all_test_preds_binary = (all_test_preds_prob_np >= 0.5).astype(int)

                    # Calculate CM
                    try:
                        unique_labels_test = np.unique(all_test_labels_np)
                        labels_for_cm = [0, 1]
                        if len(unique_labels_test) < 2:
                             print(f"Warning: Only one class ({unique_labels_test}) present in true test labels for CM calculation.")
                        test_cm = confusion_matrix(all_test_labels_np, all_test_preds_binary, labels=labels_for_cm)
                    except ValueError as e:
                        print(f"Warning: Could not compute test confusion matrix: {e}.")

                    # Calculate other metrics
                    unique_labels_true_test = np.unique(all_test_labels_np)
                    if len(unique_labels_true_test) < 2:
                        print(f"Warning: Only one class ({unique_labels_true_test}) present in TRUE test labels. Test AUC ROC/AUC10 set to 0.0. AUC-PR might be NaN or 0.")
                        test_auc_roc = 0.0
                        test_auc10 = 0.0
                        # AUC PR is ill-defined with one class, set to 0.0 or handle as appropriate
                        try:
                             test_auc_pr = average_precision_score(all_test_labels_np, all_test_preds_prob_np)
                             if np.isnan(test_auc_pr): test_auc_pr = 0.0 # Handle NaN case
                        except ValueError: test_auc_pr = 0.0 # Handle error case
                        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_test_labels_np, all_test_preds_binary, average='binary', zero_division=0, labels=[0, 1])
                    else: # Both classes are present
                        test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_test_labels_np, all_test_preds_binary, average='binary', zero_division=0, labels=[0, 1])
                        try:
                            test_auc_pr = average_precision_score(all_test_labels_np, all_test_preds_prob_np)
                        except ValueError as e: print(f"Warning: Test AUC-PR Error: {e}"); test_auc_pr = 0.0
                        try:
                             test_auc_roc = roc_auc_score(all_test_labels_np, all_test_preds_prob_np)
                             # Calculate AUC10 safely only if AUC-ROC is valid
                             try:
                                 # Ensure enough samples exist for max_fpr calculation
                                 if len(all_test_labels_np) > 10: # Heuristic threshold
                                     test_auc10 = roc_auc_score(all_test_labels_np, all_test_preds_prob_np, max_fpr=0.1)
                                 else:
                                      print("Warning: Test AUC10 - Not enough samples for reliable calculation. Setting to 0.0.")
                                      test_auc10 = 0.0
                             except ValueError:
                                 print("Warning: Test AUC10 calculation failed (ValueError). Setting to 0.0.")
                                 test_auc10 = 0.0
                        except ValueError as e:
                             print(f"Warning: Test AUC-ROC calculation failed: {e}. Setting AUC-ROC & AUC10 to 0.0.")
                             test_auc_roc = 0.0; test_auc10 = 0.0

                    # Store metrics
                    test_metrics = {
                        'test_precision': test_precision, 'test_recall': test_recall, 'test_f1': test_f1,
                        'test_auc_roc': test_auc_roc, 'test_auc_pr': test_auc_pr, 'test_auc10': test_auc10,
                        'test_confusion_matrix': test_cm.tolist() # Convert CM to list for JSON
                    }

                    # Print Test Metrics
                    print("\n--- Test Set Performance ---")
                    print(f"  AUC-PR: {test_auc_pr:.4f}, AUC-ROC: {test_auc_roc:.4f}, AUC10: {test_auc10:.4f}")
                    print(f"  F1: {test_f1:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
                    print("\n  Test Confusion Matrix (Rows: Actual, Cols: Predicted):")
                    print(f"        Predicted 0    Predicted 1")
                    print(f"Actual 0 | {test_cm[0, 0]:<12} | {test_cm[0, 1]:<12} | (TN={test_cm[0, 0]}, FP={test_cm[0, 1]})")
                    print(f"Actual 1 | {test_cm[1, 0]:<12} | {test_cm[1, 1]:<12} | (FN={test_cm[1, 0]}, TP={test_cm[1, 1]})")
                    print(f"------------------------------------------")

            else: # Case where no predictions/labels were collected
                print("Warning: No predictions or labels were collected from the test set. Cannot calculate metrics.")
                test_metrics = {"status": "No predictions/labels collected"}

        except FileNotFoundError:
            print(f"Error: Best model file not found at {final_best_model_path}. Cannot perform test evaluation.")
            test_metrics = {"error": "Best model file not found"}
        except RuntimeError as e:
             print(f"Error loading model state dict (possible architecture mismatch?): {e}")
             test_metrics = {"error": f"Model state_dict loading error: {e}"}
        except Exception as e:
            print(f"An unexpected error occurred during test set evaluation: {e}")
            test_metrics = {"error": f"Unexpected test evaluation error: {e}"}

        # Clean up test model and data loader
        finally: # Ensure cleanup happens even if errors occurred
             if 'model_test' in locals(): del model_test
             if 'test_loader' in locals(): del test_loader
             if 'test_dataset' in locals(): del test_dataset
             gc.collect()
             if DEVICE.type == 'cuda':
                 torch.cuda.empty_cache()

elif not final_best_model_path:
    print("\nSkipping test set evaluation because no best model was successfully saved from K-Fold.")
    test_metrics = {"status": "Skipped - No best model available"}
elif not H5_TEST_PATH or not os.path.exists(H5_TEST_PATH):
     print(f"\nSkipping test set evaluation because the test HDF5 path is invalid or file doesn't exist: {H5_TEST_PATH}")
     test_metrics = {"status": f"Skipped - Invalid test HDF5 path: {H5_TEST_PATH}"}

# --- Save final results ---
results_summary = {
    'hyperparameters': { # Save the hyperparameters used for this run
        'learning_rate': LEARNING_RATE, 'dropout': DROPOUT, 'nhead': N_HEAD,
        'num_encoder_layers': NUM_ENCODER_LAYERS, 'ffn_hidden_dim_factor': FFN_HIDDEN_DIM_FACTOR,
        'weight_decay': WEIGHT_DECAY, 'batch_size': BATCH_SIZE, 'epochs_max': N_EPOCHS,
        'early_stopping_patience': EARLY_STOPPING_PATIENCE, 'k_folds': N_SPLITS,
        'random_seed': RANDOM_SEED,
        'best_params_source_json': PATH_TO_JSON,
    },
    'data_setup': {
        'embedding_type': EMBEDDING_TYPE,
        'train_validation_h5_file': H5_FILE_PATH, # Path used for K-Fold
        'test_h5_file': H5_TEST_PATH             # Path used for final test
    },
    'kfold_validation_results': {
        'fold_results': fold_results, # List of dicts, one per fold
        'avg_val_metrics': avg_metrics if fold_results else {}, # Avg over folds
        'std_val_metrics': std_metrics if fold_results else {}, # Std over folds
        'best_fold_num_by_val_metric': best_fold_num if fold_results else None,
        'metric_for_best_fold': METRIC_TYPE if fold_results else None,
        'best_model_path': final_best_model_path, # Path to the final .pth file
    },
    # <<< ADDED: Include the test metrics dictionary here >>>
    'final_test_set_results': test_metrics
}