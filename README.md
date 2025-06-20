# B-cell Epitope Prediction Using ESM-2 and ESM-C Embeddings

This repository accompanies our MSc thesis at Copenhagen Business School in collaboration with Evaxion Biotech. The project investigates how different protein language model embeddings (ESM-2 and ESM-C) affect downstream B-cell epitope prediction performance across various machine learning architectures.

## Project Overview

We explore the use of large protein language models for predicting B-cell epitopes based solely on amino acid sequences. Our primary goals are:

- Compare ESM-2 and ESM-C embeddings for epitope prediction  
- Benchmark across models: MLP, XGBoost and Transformer  
- Evaluate performance using standard metrics (ROC AUC, F1, Precision, Recall)  
- Analyze trade-offs between accuracy, model size, and deployment feasibility  

## Repository Structure

| File/Directory              | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `Data/`                    | FASTA files containing protein sequences and epitope annotations (BepiPred-3.0) |
| `final_training/`          | Trained models, performance metrics, and evaluation results                 |
| `final_training_scripts/`  | Scripts for training models with optimal hyperparameters                    |
| `random_search/`           | Output files and results from hyperparameter optimization experiments       |
| `random_search_scripts/`   | Python scripts for hyperparameter tuning via random search                  |
| `analysis.ipynb`           | Jupyter notebook for model performance analysis                             |
| `h5 demo visualisering.txt`| HDF5 database structure                                                     |
| `load_and_vis_class.ipynb` | Notebook for data and HDF5 loading and visualization                        |
| `requirements.txt`         | Python dependencies 

## Reproducibility

To set up the environment:

```pip install -r requirements.txt```

All relevant FASTA input files and labels are stored in /Data/.
You will also need access to the pretrained ESM-2 and ESM-C models (see Meta's [GitHub](https://github.com/facebookresearch/esm)).

## Acknowledgements
This project was developed as part of our master's thesis in Business Administration and Data Science at CBS. We thank Evaxion Biotech for their domain guidance and collaboration.