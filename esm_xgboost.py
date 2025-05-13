# Key difference for building the XGBoost is preparing the data in tabular format, as the model expects.

# Goal: build a XGBoost model on protein embeddings where each amino acid residue is treated as an independent data point.

# Idea: Transform the (Sequence Length x Embedding Dimension) data for each protein 
#       into one large table (Total Residues x Embedding Dimension).

''' ------------------------------------------------------------------------ '''
#pip install -r requirements.txt


import sys
import xgboost

print("--- Python Executable ---")
print(sys.executable)
print("\n--- XGBoost Info ---")
print(f"XGBoost Version: {xgboost.__version__}")
print(f"XGBoost Path: {xgboost.__file__}")
print("\n--- sys.path ---")
import pprint
pprint.pprint(sys.path)
print("----------------------\n")
# --- import required modules ---
import os
import sys
import datetime
import time
import h5py
import numpy as np
import pandas as pd
import random
import gc
import json
import tqdm  # For progress bars
import logging

from sklearn.model_selection import KFold, ParameterSampler
from scipy.stats import uniform, randint, loguniform
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score, confusion_matrix, auc
import xgboost as xgb # ligner at den ik virker, men burde
from xgboost.callback import EarlyStopping

# --- Start Timer ---
script_start_time = time.time()

# --- Configuration ---
EMBEDDING_CONFIG = {
    'esm2': {
        'h5_path': 'esm2_protein_embeddings.h5',
        'embed_dim': 1280,
        'target_gpu_index': 2
    },
    'esmc': {
        'h5_path': 'esmc_protein_embeddings.h5',
        'embed_dim': 960,
        'target_gpu_index': 1
    }
}

# Ask user for embedding type
EMBEDDING_TYPE = ""
print("-----------------------------------------------------")
while EMBEDDING_TYPE not in EMBEDDING_CONFIG:
    EMBEDDING_TYPE = input(f"Select embedding type ({'/'.join(EMBEDDING_CONFIG.keys())}): ").lower().strip()
    if EMBEDDING_TYPE not in EMBEDDING_CONFIG:
        print("Invalid choice. Please try again.")
print("-----------------------------------------------------")

threshold = float(input("Which threshold do you want to use?"))

# Load config
config = EMBEDDING_CONFIG[EMBEDDING_TYPE]
H5_FILE_PATH = config['h5_path']
EMBED_DIM = config['embed_dim']
# target_gpu = config['target_gpu_index'] # Note: Used for config consistency, XGBoost CPU version doesn't use this. Its so far for logging

# --- Creating Folders for Tracking ---
ALL_RUNS_DIR = 'xgboost_runs'
os.makedirs(ALL_RUNS_DIR, exist_ok=True)

# Count existing runs
existing_runs = [
    name for name in os.listdir(ALL_RUNS_DIR)
    if os.path.isdir(os.path.join(ALL_RUNS_DIR, name)) and 
    name.split('_')[1] == EMBEDDING_TYPE and  # Check embedding type in second position
    '_xgboost_run_' in name  # Check the run identifier
]
run_number = len(existing_runs) + 1

# Timestamp and naming
run_timestamp = datetime.datetime.now().strftime("%m-%d_%H:%M")
BASE_RUN_DIR = os.path.join(
    ALL_RUNS_DIR,
    f'{run_number:02d}_{EMBEDDING_TYPE}_xgboost_run_{run_timestamp}'
)

# Subdirectories
MODEL_SAVE_DIR = os.path.join(BASE_RUN_DIR, 'saved_models')
LOG_DIR = os.path.join(BASE_RUN_DIR, 'logs')
RESULTS_FILE = os.path.join(BASE_RUN_DIR, f'{EMBEDDING_TYPE}_hyperparam_search_results.json')

# Create folders
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Final output paths confirmation
print(f"Created run directory: {BASE_RUN_DIR}")
print(f"Models will be saved in: {MODEL_SAVE_DIR}")
print(f"Logs will be saved in: {LOG_DIR}")
print(f"Results JSON will be saved at: {RESULTS_FILE}")


# ===============================
# --- Training Settings ---
# ===============================
N_SPLITS = 4 # number of K-Fold splits
RANDOM_SEED = 17 #bc 17 is a cool number 8-)

xgb_param_dist = {
    'learning_rate': loguniform(0.005, 0.3),     # Broader range
    'max_depth': randint(3, 12),                 # Extended upper bound
    'subsample': uniform(0.5, 0.5),              # Range [0.5, 1.0]
    'colsample_bytree': uniform(0.5, 0.5),       # Range [0.5, 1.0]
    'colsample_bylevel': uniform(0.5, 0.5),      # 
    'gamma': loguniform(0.001, 5.0),             # Log scale for better range coverage
    'lambda': loguniform(0.1, 10.0),             # Log scale L2 reg
    'alpha': loguniform(0.001, 10.0),            # Log scale L1 reg
    'min_child_weight': randint(1, 10),          # Extended range
    'max_delta_step': randint(0, 10)             # New parameter, useful for imbalanced classes
}

N_SEARCH_ITERATIONS = 30 # Set the number of parameter combinations to try

# --- Fixed XGBoost settings (used alongside tuned params) --
XGB_N_ESTIMATORS = 1500 # Set high, early stopping will choose optimal
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_OBJECTIVE = 'binary:logistic'
XGB_EVAL_METRIC = 'aucpr'

# --- SEEDING ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(RANDOM_SEED) # Set the initial global seed

# --- Loading HFD5 files ---
print(f"\n--- Loading Protein Keys from {H5_FILE_PATH} ---")
try:
    with h5py.File(H5_FILE_PATH, 'r') as f:
        if 'embeddings_folder' not in f:
            raise KeyError("Group 'embeddings_folder' not found in HDF5 file.")
        all_protein_keys = list(f['embeddings_folder'].keys())
        all_protein_keys.sort() # sort alphabetically (good practice)
        all_protein_keys = np.array(all_protein_keys) # convert to numpy array
    print(f"Successfully loaded {len(all_protein_keys)} protein keys.")

except FileNotFoundError:
    print(f"Error: HDF5 file not found at {H5_FILE_PATH}")
    exit()

except KeyError as e:
    print(f"Error: Problem accessing data within HDF5 file. {e}")
    exit()

except Exception as e:
    print(f"An unexpected error occurred while loading keys: {e}")
    exit()


# --- Initialize Hyperparameter Search ---
print(f"\n--- Starting Hyperparameter Search ({N_SEARCH_ITERATIONS} trials) ---")
sampler = ParameterSampler(xgb_param_dist, n_iter=N_SEARCH_ITERATIONS, random_state=RANDOM_SEED)
all_trial_results = [] # List to store results dictionaries from each trial
best_overall_avg_metric = -1 # Track best average metric (e.g., F1 or AUC-PR)
best_trial_config = None # Store the config of the best trial


# # Create an early stopping callback (now removed as we try with native API)
# early_stop = EarlyStopping(
#     rounds=XGB_EARLY_STOPPING_ROUNDS,
#     metric_name=XGB_EVAL_METRIC,
#     save_best=True
# )

# ==============================================================================
# --- Outer Trial Loop --- 
# ==============================================================================
for trial_num, params in enumerate(sampler, 1):
    
    print(f"\n**************** Trial {trial_num}/{N_SEARCH_ITERATIONS} ****************")
    print(f"Parameters: {params}")


    # ==============================================================================
    # --- Step 2: Set up the K-Fold Loop ---
    # ==============================================================================
    print(f"\n--- Setting up {N_SPLITS}-Fold Cross-Validation ---")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    fold_results = [] # Initialize list to store results from each fold

    # Start the K-Fold loop
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_protein_keys)):
        print(f"\n===== Fold {fold + 1}/{N_SPLITS} =====")
        fold_seed = RANDOM_SEED + fold
        set_seed(fold_seed) # Seed specifically for this fold
        print(f"Fold {fold + 1} using seed: {fold_seed}")

        train_keys = all_protein_keys[train_idx]
        val_keys = all_protein_keys[val_idx]
        print(f"Fold {fold + 1}: {len(train_keys)} train proteins, {len(val_keys)} validation proteins.")


        # --------------------------------------------------------------------------
        # --- Step 3: Load and Aggregate Data for the Current Fold ---
        # --------------------------------------------------------------------------
        print("\nStep 3: Loading and Aggregating Data...")
        train_embeddings_list = []
        train_labels_list = []
        val_embeddings_list = []
        val_labels_list = []

        # Load Training Data
        print(f"--> Loading training data ({len(train_keys)} proteins)...")
        try:
            # Open HDF5 file specifically for training data
            with h5py.File(H5_FILE_PATH, 'r') as f_train: # Use f_train or reuse f
                embeddings_group_train = f_train['embeddings_folder']
                for protein_key in tqdm.tqdm(train_keys, desc="Loading Train Embeddings"):
                    protein_group = embeddings_group_train[protein_key]
                    embeddings = protein_group['embeddings'][:]
                    labels = protein_group['labels'][:]
                    labels = labels.astype(np.int32)
                    train_embeddings_list.append(embeddings)
                    train_labels_list.append(labels)
            # File f_train is automatically closed here
        except Exception as e:
            print(f"Error loading training data for fold {fold+1}, trial {trial_num}: {e}")
            
            continue # Skip to the next fold if loading fails

        # Load Validation Data
        print(f"--> Loading validation data ({len(val_keys)} proteins)...")
        try:
            # +++ Re-open HDF5 file specifically for validation data +++
            with h5py.File(H5_FILE_PATH, 'r') as f_val: # Use f_val or reuse f
                embeddings_group_val = f_val['embeddings_folder']
                # +++ Corrected loop start +++
                for protein_key in tqdm.tqdm(val_keys, desc="Loading Val Embeddings"):
                    protein_group = embeddings_group_val[protein_key] # Use group from f_val
                    embeddings = protein_group['embeddings'][:]
                    labels = protein_group['labels'][:]
                    labels = labels.astype(np.int32)
                    val_embeddings_list.append(embeddings)
                    val_labels_list.append(labels)
            # File f_val is automatically closed here
        except Exception as e:
            print(f"Error loading validation data for fold {fold+1}, trial {trial_num}: {e}")
            continue # Skip to the next fold

        # Concatenate into large NumPy arrays
        try:
            X_train = np.concatenate(train_embeddings_list, axis=0)
            y_train = np.concatenate(train_labels_list, axis=0)
            X_val = np.concatenate(val_embeddings_list, axis=0)
            y_val = np.concatenate(val_labels_list, axis=0)
        except ValueError as e:
            print(f"Error during concatenation for fold {fold+1}: {e}. Likely empty list.")
            print(f"  Train list lengths: Emb={len(train_embeddings_list)}, Labels={len(train_labels_list)}")
            print(f"  Val list lengths: Emb={len(val_embeddings_list)}, Labels={len(val_labels_list)}")
            continue # Skip fold if concatenation fails
        
        # Print shapes and clean up lists
        print(f"Fold {fold+1} Data Shapes:")
        print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}, y_val: {y_val.shape}")
        del train_embeddings_list, train_labels_list, val_embeddings_list, val_labels_list
        
        gc.collect() # GET THAT GARBAGE OUTTA HERE
        print("Step 3: Data Aggregation Complete.")

        # --------------------------------------------------------------------------
        # --- Step 4: XGBoost Model Definition & Training (Native API) ---
        # --------------------------------------------------------------------------
        print("\nStep 4: Defining and Training XGBoost Model (Native API)...")

        # Calculate scale_pos_weight for handling class imbalance
        num_neg = np.sum(y_train == 0)
        num_pos = np.sum(y_train == 1)
        if num_pos == 0:
            print(f"Warning: Fold {fold+1} - No positive samples found in training data! Skipping fold.")
            continue  # Cannot train reasonably without positive samples
        scale_pos_weight = num_neg / num_pos
        print(f"Fold {fold+1} - scale_pos_weight: {scale_pos_weight:.2f} ({num_neg} neg / {num_pos} pos)")

        # Convert training and validation data to DMatrix format
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

       
        # Set up parameters dictionary for native API
        xgb_params = {
            'objective': XGB_OBJECTIVE,
            'eval_metric': XGB_EVAL_METRIC,
            'scale_pos_weight': scale_pos_weight,
            'learning_rate': params['learning_rate'],
            'max_depth': params['max_depth'],
            'subsample': params['subsample'],
            'colsample_bytree': params['colsample_bytree'],
            'colsample_bylevel': params.get('colsample_bylevel', params['colsample_bytree']),  # Use if available
            'gamma': params['gamma'],
            'lambda': params['lambda'],
            'alpha': params['alpha'],
            'min_child_weight': params['min_child_weight'],
            'max_delta_step': params.get('max_delta_step', 0),  # Use if available
            'seed': fold_seed,
            'nthread': -1  # Use all available threads
        }

        # Define evaluation list - you can include both train and val or just val
        evals = [(dtrain, 'train'), (dval, 'val')] # - remember to remove dtrain for final model evaluation
        evals_result = {}  # This will store evaluation results

        # Train the model using native API
        print(f"--> Training XGBoost with early stopping (rounds={XGB_EARLY_STOPPING_ROUNDS})...")
        try:
            xgb_model = xgb.train(
                xgb_params,
                dtrain,
                num_boost_round=XGB_N_ESTIMATORS,
                evals=evals,
                early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
                evals_result=evals_result,
                verbose_eval=100
            )
            
            # Get best iteration
            best_iteration = xgb_model.best_iteration
            print(f"Training complete. Best iteration: {best_iteration}")
            
        except Exception as e:
            print(f"Error during XGBoost training for fold {fold+1}: {e}")
            continue  # Skip fold if training fails

        # Save the trained model for this fold
        model_filename = os.path.join(MODEL_SAVE_DIR, f'{EMBEDDING_TYPE}_trial_{trial_num}_fold_{fold+1}_model.json')
        try:
            xgb_model.save_model(model_filename)
            # print(f"Saved model for trial {trial_num}, fold {fold+1}") # Optional
        except Exception as e:
            print(f"Error saving model for trial {trial_num}, fold {fold+1}: {e}")

        print("Step 4: Model Training Complete.")


        # --------------------------------------------------------------------------
        # --- Step 5: Prediction and Evaluation ---
        # --------------------------------------------------------------------------
        print("\nStep 5: Predicting and Evaluating...")

        try:
            # Predict probabilities on the validation set using native API
            dval = xgb.DMatrix(X_val)  # Convert to DMatrix if not already done
            y_pred_prob = xgb_model.predict(dval)  # Returns probabilities directly

            # Calculate metrics that use probabilities directly
            auc_pr = average_precision_score(y_val, y_pred_prob)

            # Handle potential error if only one class present in y_val for ROC AUC
            try:
                auc_roc = roc_auc_score(y_val, y_pred_prob)
            except ValueError:
                print("Warning: Only one class present in y_val. AUC ROC calculation skipped (set to 0.0).")
                auc_roc = 0.0

            # Handle potential error for partial AUC (AUC10)
            try:
                auc10 = roc_auc_score(y_val, y_pred_prob, max_fpr=0.1)
            except ValueError:
                print("Warning: Could not calculate AUC10 (max_fpr=0.1). Skipped (set to 0.0).")
                auc10 = 0.0

            # Calculate metrics requiring a threshold (using 0.5)
            y_pred_binary = (y_pred_prob >= threshold).astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_val, y_pred_binary, average='binary', zero_division=0
            )

            # Calculate confusion matrix
            cm = confusion_matrix(y_val, y_pred_binary, labels=[0, 1]) # Ensure consistent label order
            true_neg, false_pos, false_neg, true_pos = cm.ravel()

            # Store metrics for this fold
            current_fold_metrics = {
                'fold': fold + 1,
                'f1': f1,
                'auc_pr': auc_pr,
                'auc_roc': auc_roc,
                'auc10': auc10,
                'precision': precision,
                'recall': recall,
                'true_neg': int(true_neg), # Convert to standard int for JSON
                'false_pos': int(false_pos),
                'false_neg': int(false_neg),
                'true_pos': int(true_pos),
                'num_val_samples': len(y_val),
                'num_pos_val': int(np.sum(y_val == 1)),
                'best_iteration': best_iteration
            }
            fold_results.append(current_fold_metrics)

            # Print fold summary
            print(f"Fold {fold+1} Eval Results:")
            print(f"  F1: {f1:.4f}, AUC-PR: {auc_pr:.4f}, AUC-ROC: {auc_roc:.4f}, AUC10: {auc10:.4f}")
            print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(f"  Confusion Matrix (TrueNeg, FalsePos, FalseNeg, TruePos): ({true_neg}, {false_pos}, {false_neg}, {true_pos})")

        except Exception as e:
            print(f"Error during prediction/evaluation for trial {trial_num}, fold {fold+1}: {e}")

        print("Step 5: Evaluation Complete.")


        # --- Garbage Collection per fold ---
        del X_train, y_train, X_val, y_val, xgb_model
        gc.collect()
        print("Fold garbage collected.")

    # --- End of K-Fold Loop ---
    print(f"\n===== Cross-Validation Complete =====")


    # --------------------------------------------------------------------------
    # --- Step 6: Aggregate Results for the CURRENT TRIAL ---
    # --------------------------------------------------------------------------
    print(f"\n----- Aggregating Results for Trial {trial_num} -----")
    trial_summary = { # Prepare summary structure for this trial
        'trial_num': trial_num,
        'params': params,
        'status': 'failed', # Default status
        'avg_metrics': {},
        'std_metrics': {},
        'individual_fold_results': fold_results # Store detailed fold results
    }

    if not fold_results:
        print(f"Trial {trial_num}: No folds completed successfully.")
    else:
        results_df = pd.DataFrame(fold_results)
        metric_cols = ['f1', 'auc_pr', 'auc_roc', 'auc10', 'precision', 'recall', 'true_neg', 'false_pos', 'false_neg', 'true_pos']
        # Filter out potential non-numeric columns if necessary before calculating mean/std
        numeric_results_df = results_df[metric_cols].apply(pd.to_numeric, errors='coerce')

        trial_avg_metrics = numeric_results_df.mean().to_dict()
        trial_std_metrics = numeric_results_df.std().to_dict()

        trial_summary['status'] = 'completed'
        trial_summary['avg_metrics'] = trial_avg_metrics
        trial_summary['std_metrics'] = trial_std_metrics

        print("Average Metrics across folds:")
        for key, value in trial_avg_metrics.items():
            std_dev = trial_std_metrics.get(key, 0)
            print(f"  Avg {key.upper()}: {value:.4f} (+/- {std_dev:.4f})")

        # --- Check if this trial is the best one so far ---
        # Choose metric to optimize, e.g., 'f1' or 'auc_pr'
        current_avg_metric = trial_avg_metrics.get('f1', -1) # Default to -1 if metric not found
        if current_avg_metric > best_overall_avg_metric:
             print(f"--- *** New Best Trial Found (Trial {trial_num}) *** ---")
             print(f"--- *** Average F1 improved from {best_overall_avg_metric:.4f} to {current_avg_metric:.4f} *** ---")
             best_overall_avg_metric = current_avg_metric
             # Make a copy of params and metrics for the best trial
             best_trial_config = {
                 'trial_num': trial_num,
                 'params': params.copy(),
                 'avg_metrics': trial_avg_metrics.copy()
                 }

    # Append the summary for this trial to the overall list
    all_trial_results.append(trial_summary)

    # --- Save results incrementally after each trial --- # +++ ADDED/MOVED +++
    print(f"--- Saving intermediate results after Trial {trial_num} ---")
    try:
        # Helper function to convert numpy types for JSON compatibility
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
            # Check for pandas NaNs which are often float NaNs
            elif isinstance(obj, float) and np.isnan(obj):
                return None # Represent NaN as null in JSON
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        with open(RESULTS_FILE, 'w') as f:
            # Use the serializer when dumping
            json.dump(all_trial_results, f, indent=4, default=default_serializer)
        print(f"Trial {trial_num} results updated in {RESULTS_FILE}")
    except IOError as e:
        print(f"Error saving results to {RESULTS_FILE}: {e}")
    except TypeError as e:
        print(f"Error serializing results to JSON: {e}. Check data types.")
        # print(all_trial_results[-1]) # Uncomment to debug last trial data


# --- End of Outer Trial Loop ---
print(f"\n**************** Hyperparameter Search Complete ****************")


# --------------------------------------------------------------------------
# --- Step 6: Final Summary Reporting ---
# --------------------------------------------------------------------------
print("\nStep 6: Final Summary Reporting...")

if not all_trial_results:
    print("No trials were run or completed successfully.")
elif best_trial_config:
    print("\n--- Best Trial Found ---")
    print(f"  Trial Number: {best_trial_config['trial_num']}")
    print(f"  Best Avg F1 Score: {best_trial_config['avg_metrics'].get('f1', 'N/A'):.4f}")
    print(f"  Corresponding Parameters: {best_trial_config['params']}")
    print("\n  Corresponding Average Metrics:")
    for key, value in best_trial_config['avg_metrics'].items():
        # Look up std dev if needed from all_trial_results
        std_dev = next((t['std_metrics'].get(key, 0) for t in all_trial_results if t['trial_num'] == best_trial_config['trial_num']), 0)
        print(f"    Avg {key.upper()}: {value:.4f} (+/- {std_dev:.4f})")
else:
    print("\nNo successful trials completed, could not determine best trial.")

print(f"\nFull results for all trials saved in: {RESULTS_FILE}")
print(f"Best model checkpoints saved in subfolders within: {MODEL_SAVE_DIR}")

# --- End of script ---
script_end_time = time.time()
total_elapsed_seconds = script_end_time - script_start_time
elapsed_time_str = str(datetime.timedelta(seconds=int(total_elapsed_seconds))) # Format H:M:S
print("\n-----------------------------------------------------")
print(f"XGBoost Script Finished.")
print(f"Total Execution Time: {elapsed_time_str}")
print("-----------------------------------------------------")

# --- END OF FILE esm_XGBoost.py ---

