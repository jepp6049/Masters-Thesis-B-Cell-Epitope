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

from sklearn.model_selection import train_test_split # Needed for validation split
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score,
                             average_precision_score, confusion_matrix, auc)
import xgboost as xgb

# --- Start Timer ---
script_start_time = time.time()

# --- Configuration ---
EMBEDDING_CONFIG = {
    'esm2': {
        'h5_path': 'esm2_protein_embeddings.h5',    # Training data
        'embed_dim': 1280,
        'test_path': 'esm2_test_protein_embeddings.h5' # Test data path
    },
    'esmc': {
        'h5_path': 'esmc_protein_embeddings.h5',    # Training data
        'embed_dim': 960,
        'test_path': 'esmc_test_protein_embeddings.h5' # Test data path
    }
}

# --- User Inputs ---
# Ask user for embedding type
EMBEDDING_TYPE = ""
print("-----------------------------------------------------")
while EMBEDDING_TYPE not in EMBEDDING_CONFIG:
    EMBEDDING_TYPE = input(f"Select embedding type ({'/'.join(EMBEDDING_CONFIG.keys())}): ").lower().strip()
    if EMBEDDING_TYPE not in EMBEDDING_CONFIG:
        print("Invalid choice. Please try again.")
print("-----------------------------------------------------")

# Ask for threshold
threshold = float(input("Enter the classification threshold to use (e.g., 0.5): "))
print("-----------------------------------------------------")

# Ask for the results JSON file path
json_file_path = ""
while not os.path.isfile(json_file_path) or not json_file_path.lower().endswith('.json'):
    json_file_path = input("Enter the full path to the hyperparameter search results JSON file: ")
    if not os.path.isfile(json_file_path):
        print(f"Error: File not found at '{json_file_path}'. Please check the path.")
    elif not json_file_path.lower().endswith('.json'):
        print("Error: File must be a .json file.")
print("-----------------------------------------------------")


# --- Load Configuration and Best Parameters ---
config = EMBEDDING_CONFIG[EMBEDDING_TYPE]
H5_TRAIN_PATH = config['h5_path']
H5_TEST_PATH = config['test_path']
EMBED_DIM = config['embed_dim']

print(f"Using Training Data: {H5_TRAIN_PATH}")
print(f"Using Test Data:     {H5_TEST_PATH}")

# Load and find best parameters from JSON
try:
    with open(json_file_path, 'r') as f:
        all_trial_results = json.load(f)

    # Filter for completed trials and find the one with the best average F1 score
    best_trial = None
    best_metric_score = -1
    metric_to_optimize = 'f1' # Or 'auc_pr', adjust if needed

    for trial in all_trial_results:
        if trial.get('status') == 'completed' and trial.get('avg_metrics'):
            current_score = trial['avg_metrics'].get(metric_to_optimize, -1)
            if current_score > best_metric_score:
                best_metric_score = current_score
                best_trial = trial

    if best_trial:
        best_params = best_trial['params']
        best_trial_num = best_trial['trial_num']
        print(f"Found best trial (Trial #{best_trial_num}) based on avg {metric_to_optimize.upper()} score: {best_metric_score:.4f}")
        print("Best Parameters:")
        print(json.dumps(best_params, indent=2))
    else:
        print(f"Error: Could not find a 'completed' trial with avg '{metric_to_optimize}' metric in {json_file_path}")
        exit()

except FileNotFoundError:
    print(f"Error: JSON results file not found at {json_file_path}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {json_file_path}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while processing the JSON file: {e}")
    exit()
print("-----------------------------------------------------")


# --- Paths and Directories Setup for Final XGBoost Training ---
print("\n--- Setting up Directories for Final Training Run ---")
FINAL_RUNS_PARENT_DIR = 'xgboost_final_runs'
os.makedirs(FINAL_RUNS_PARENT_DIR, exist_ok=True)
run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") # Use YYYYMMDD for better sorting
run_name = f'{EMBEDDING_TYPE}_final_train_{run_timestamp}_trial_{best_trial_num}' # Include best trial num
FINAL_RUN_DIR = os.path.join(FINAL_RUNS_PARENT_DIR, run_name)
FINAL_MODEL_SAVE_PATH = os.path.join(FINAL_RUN_DIR, f'{EMBEDDING_TYPE}_xgb_final_model.json') # Standard XGBoost model format
FINAL_RESULTS_FILE = os.path.join(FINAL_RUN_DIR, f'{EMBEDDING_TYPE}_xgb_final_results_summary.json')
os.makedirs(FINAL_RUN_DIR, exist_ok=True)
print(f"Created final run directory: {FINAL_RUN_DIR}")
print(f"Final model will be saved as: {FINAL_MODEL_SAVE_PATH}")
print(f"Final run summary will be saved at: {FINAL_RESULTS_FILE}")
print("-----------------------------------------------------")

# ===============================
# --- Final Training Settings ---
# ===============================
print("\n--- Defining Final Training Settings ---")
RANDOM_SEED = 1989 # Use a fixed seed for reproducibility of the final run
VALIDATION_SPLIT_RATIO = 0.15 # Reserve 15% OF TRAINING data for early stopping validation
XGB_N_ESTIMATORS = 2000 # Max rounds, early stopping will optimize
XGB_EARLY_STOPPING_ROUNDS = 50
XGB_OBJECTIVE = 'binary:logistic'
XGB_EVAL_METRIC = 'aucpr' # Metric for early stopping evaluation

print(f"Random Seed for final run: {RANDOM_SEED}")
print(f"Validation split ratio (from training data): {VALIDATION_SPLIT_RATIO}")
print(f"XGBoost training will use '{XGB_EVAL_METRIC}' for evaluation and early stopping.")
print(f"Early stopping patience: {XGB_EARLY_STOPPING_ROUNDS} rounds")
print(f"Max boosting rounds: {XGB_N_ESTIMATORS}")
print("-----------------------------------------------------")

# --- SEEDING ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
set_seed(RANDOM_SEED)

# --- Helper Function to Load and Aggregate Data ---
def load_and_aggregate_data(h5_path, description="Data"):
    """Loads embeddings and labels from HDF5 and aggregates them."""
    print(f"\n--- Loading and Aggregating {description} from {h5_path} ---")
    embeddings_list = []
    labels_list = []
    protein_keys = []

    try:
        with h5py.File(h5_path, 'r') as f:
            if 'embeddings_folder' not in f:
                raise KeyError("Group 'embeddings_folder' not found in HDF5 file.")
            keys_in_file = list(f['embeddings_folder'].keys())
            protein_keys.extend(keys_in_file) # Store keys if needed later

            print(f"--> Found {len(keys_in_file)} protein keys in {description} file.")
            embeddings_group = f['embeddings_folder']
            for protein_key in tqdm.tqdm(keys_in_file, desc=f"Loading {description} Embeddings"):
                try:
                    protein_group = embeddings_group[protein_key]
                    embeddings = protein_group['embeddings'][:]
                    labels = protein_group['labels'][:]
                    labels = labels.astype(np.int32) # Ensure integer type for labels
                    embeddings_list.append(embeddings)
                    labels_list.append(labels)
                except KeyError:
                    print(f"Warning: Protein key '{protein_key}' found in list but not accessible in HDF5 group. Skipping.")
                except Exception as load_err:
                    print(f"Warning: Error loading data for protein '{protein_key}'. Skipping. Error: {load_err}")


        if not embeddings_list:
             print(f"Error: No embeddings were successfully loaded for {description}. Exiting.")
             exit()

        # Concatenate into large NumPy arrays
        X_data = np.concatenate(embeddings_list, axis=0)
        y_data = np.concatenate(labels_list, axis=0)

        print(f"{description} Data Shapes:")
        print(f"  X_{description.lower()}: {X_data.shape}, y_{description.lower()}: {y_data.shape}")

        # Cleanup
        del embeddings_list, labels_list
        gc.collect()
        print(f"--- {description} Aggregation Complete ---")
        return X_data, y_data, protein_keys

    except FileNotFoundError:
        print(f"Error: HDF5 file not found at {h5_path}")
        exit()
    except KeyError as e:
        print(f"Error: Problem accessing data within HDF5 file '{h5_path}'. {e}")
        exit()
    except ValueError as e:
         print(f"Error during concatenation for {description}: {e}. Likely empty list or shape mismatch.")
         exit()
    except Exception as e:
        print(f"An unexpected error occurred while loading {description} data: {e}")
        exit()

# --- Load Full Training Data ---
X_train_full, y_train_full, train_keys = load_and_aggregate_data(H5_TRAIN_PATH, "Training Data")

# --- Load Test Data ---
X_test, y_test, test_keys = load_and_aggregate_data(H5_TEST_PATH, "Test Data")

# --- Split Training Data for Validation (for Early Stopping) ---
print(f"\n--- Splitting Training Data (Ratio: {VALIDATION_SPLIT_RATIO}) for Early Stopping Validation ---")
X_train_final, X_val_early_stop, y_train_final, y_val_early_stop = train_test_split(
    X_train_full, y_train_full,
    test_size=VALIDATION_SPLIT_RATIO,
    random_state=RANDOM_SEED,
    stratify=y_train_full # Stratify to maintain class balance in split
)
print("Data Split Shapes:")
print(f"  X_train_final: {X_train_final.shape}, y_train_final: {y_train_final.shape}")
print(f"  X_val_early_stop: {X_val_early_stop.shape}, y_val_early_stop: {y_val_early_stop.shape}")
del X_train_full, y_train_full # Free up memory
gc.collect()
print("-----------------------------------------------------")

# --- Prepare Data for XGBoost (DMatrix) ---
print("\n--- Preparing Data for XGBoost (DMatrix format) ---")

# Calculate scale_pos_weight using the FINAL training split
num_neg_train = np.sum(y_train_final == 0)
num_pos_train = np.sum(y_train_final == 1)
if num_pos_train == 0:
    print("Error: No positive samples found in the final training split! Cannot train.")
    exit()
scale_pos_weight = num_neg_train / num_pos_train
print(f"Calculated scale_pos_weight for training: {scale_pos_weight:.2f} ({num_neg_train} neg / {num_pos_train} pos)")

# Create DMatrix objects
dtrain_final = xgb.DMatrix(X_train_final, label=y_train_final)
dval_early_stop = xgb.DMatrix(X_val_early_stop, label=y_val_early_stop)
dtest = xgb.DMatrix(X_test, label=y_test) # Include labels for potential evaluation within XGBoost if needed, though we evaluate externally

print("DMatrix objects created.")
print("-----------------------------------------------------")

# --- Define Final Model Parameters ---
print("\n--- Configuring Final XGBoost Model Parameters ---")
final_xgb_params = {
    'objective': XGB_OBJECTIVE,
    'eval_metric': XGB_EVAL_METRIC,
    'scale_pos_weight': scale_pos_weight,
    'seed': RANDOM_SEED,
    'nthread': -1, # Use all available threads
    # Add the best parameters found during search
    **best_params # Unpacks the best_params dictionary here
}
print("Final Parameters for Training:")
print(json.dumps(final_xgb_params, indent=2))
print("-----------------------------------------------------")

# --- Train the Final XGBoost Model ---
print("\n--- Training Final XGBoost Model ---")
evals = [(dtrain_final, 'train'), (dval_early_stop, 'val')] # Use validation split for early stopping
evals_result = {} # To store evaluation results during training

try:
    start_train_time = time.time()
    final_xgb_model = xgb.train(
        final_xgb_params,
        dtrain_final,
        num_boost_round=XGB_N_ESTIMATORS,
        evals=evals,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        evals_result=evals_result,
        verbose_eval=100 # Print progress every 100 rounds
    )
    end_train_time = time.time()
    training_duration_sec = end_train_time - start_train_time
    training_duration_str = str(datetime.timedelta(seconds=int(training_duration_sec)))

    best_iteration = final_xgb_model.best_iteration
    print(f"\nTraining Complete.")
    print(f"  Best Iteration found via early stopping: {best_iteration}")
    print(f"  Training Duration: {training_duration_str}")

except Exception as e:
    print(f"\nError during Final XGBoost training: {e}")
    exit()

# --- Save the Final Model ---
print(f"\n--- Saving Final Trained Model to {FINAL_MODEL_SAVE_PATH} ---")
try:
    final_xgb_model.save_model(FINAL_MODEL_SAVE_PATH)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving final model: {e}")
print("-----------------------------------------------------")

# --- Evaluate on the Held-Out Test Set ---
print("\n--- Evaluating Final Model on the Test Set ---")

test_results = {}
try:
    # Predict probabilities on the TEST set
    y_pred_prob_test = final_xgb_model.predict(dtest)

    # Calculate metrics using probabilities
    test_auc_pr = average_precision_score(y_test, y_pred_prob_test)
    test_results['auc_pr'] = test_auc_pr

    try:
        test_auc_roc = roc_auc_score(y_test, y_pred_prob_test)
        test_results['auc_roc'] = test_auc_roc
    except ValueError:
        print("Warning: Only one class present in y_test. AUC ROC calculation skipped (set to 0.0).")
        test_auc_roc = 0.0
        test_results['auc_roc'] = test_auc_roc

    try:
        test_auc10 = roc_auc_score(y_test, y_pred_prob_test, max_fpr=0.1)
        test_results['auc10'] = test_auc10
    except ValueError:
        print("Warning: Could not calculate AUC10 on test set. Skipped (set to 0.0).")
        test_auc10 = 0.0
        test_results['auc10'] = test_auc10

    # Calculate metrics using the user-defined threshold
    y_pred_binary_test = (y_pred_prob_test >= threshold).astype(int)

    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        y_test, y_pred_binary_test, average='binary', zero_division=0
    )
    test_results['f1'] = test_f1
    test_results['precision'] = test_precision
    test_results['recall'] = test_recall
    test_results['threshold_used'] = threshold

    # Calculate confusion matrix for the test set
    test_cm = confusion_matrix(y_test, y_pred_binary_test, labels=[0, 1])
    tn, fp, fn, tp = test_cm.ravel()
    test_results['true_neg'] = int(tn)
    test_results['false_pos'] = int(fp)
    test_results['false_neg'] = int(fn)
    test_results['true_pos'] = int(tp)
    test_results['num_test_samples'] = len(y_test)
    test_results['num_pos_test'] = int(np.sum(y_test == 1))

    # Print Test Set Evaluation Summary
    print("\nTest Set Evaluation Results:")
    print(f"  Threshold Used: {threshold}")
    print(f"  F1: {test_f1:.4f}, AUC-PR: {test_auc_pr:.4f}, AUC-ROC: {test_auc_roc:.4f}, AUC10: {test_auc10:.4f}")
    print(f"  Precision: {test_precision:.4f}, Recall: {test_recall:.4f}")
    print(f"  Confusion Matrix (TN, FP, FN, TP): ({tn}, {fp}, {fn}, {tp})")

except Exception as e:
    print(f"\nError during Test Set evaluation: {e}")

print("-----------------------------------------------------")

# --- Save Final Run Summary ---
print(f"\n--- Saving Final Run Summary to {FINAL_RESULTS_FILE} ---")

final_summary = {
    'embedding_type': EMBEDDING_TYPE,
    'best_trial_source_file': json_file_path,
    'best_trial_num_from_search': best_trial_num,
    'final_model_path': FINAL_MODEL_SAVE_PATH,
    'final_training_settings': {
        'random_seed': RANDOM_SEED,
        'validation_split_ratio': VALIDATION_SPLIT_RATIO,
        'n_estimators_max': XGB_N_ESTIMATORS,
        'early_stopping_rounds': XGB_EARLY_STOPPING_ROUNDS,
        'objective': XGB_OBJECTIVE,
        'eval_metric': XGB_EVAL_METRIC,
        'scale_pos_weight': scale_pos_weight,
        'best_iteration_found': best_iteration,
        'training_duration_seconds': training_duration_sec
    },
    'best_hyperparameters_used': best_params,
    'test_set_evaluation': test_results if test_results else "Evaluation failed"
}

# Helper function to convert numpy types for JSON compatibility
def default_serializer(obj):
    if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8,
                        np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        # Handle NaN specifically before converting to float
        if np.isnan(obj):
            return None # Represent NaN as null in JSON
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

try:
    with open(FINAL_RESULTS_FILE, 'w') as f:
        json.dump(final_summary, f, indent=4, default=default_serializer)
    print("Final summary saved successfully.")
except IOError as e:
    print(f"Error saving final summary to {FINAL_RESULTS_FILE}: {e}")
except TypeError as e:
    print(f"Error serializing final summary to JSON: {e}. Check data types.")
    # print(final_summary) # Uncomment to debug the dictionary causing issues
print("-----------------------------------------------------")


# --- End of script ---
script_end_time = time.time()
total_elapsed_seconds = script_end_time - script_start_time
elapsed_time_str = str(datetime.timedelta(seconds=int(total_elapsed_seconds))) # Format H:M:S
print("\n-----------------------------------------------------")
print(f"Final XGBoost Training Script Finished.")
print(f"Total Execution Time: {elapsed_time_str}")
print("-----------------------------------------------------")