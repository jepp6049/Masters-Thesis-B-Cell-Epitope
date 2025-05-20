# B-cell Epitope Prediction Using ESM-2 and ESM-C Embeddings

This repository accompanies our MSc thesis at Copenhagen Business School in collaboration with Evaxion Biotech. The project investigates how different protein language model embeddings (ESM-2 and ESM-C) affect downstream B-cell epitope prediction performance across various machine learning architectures.

## Project Overview

We explore the use of large protein language models (PLMs) for predicting B-cell epitopes based solely on amino acid sequences. Our primary goals are:

- Compare ESM-2 and ESM-C embeddings for epitope prediction  
- Benchmark across models: MLP, XGBoost and Transformer  
- Evaluate performance using standard metrics (ROC AUC, F1, Precision, Recall)  
- Analyze trade-offs between accuracy, model size, and deployment feasibility  

## Repository Structure

.
├── Data/ # FASTA input files and label data
├── final_training/ # Final trained models and logs
├── final_training_scripts/ # Scripts for final training runs
├── random_search/ # Model outputs from random search experiments
├── random_search_scripts/ # Scripts for random hyperparameter searches
├── analysis.ipynb # Performance analysis and plots
├── load_and_vis_class.ipynb # Class distribution & embedding visualization
├── h5 demo visualisering.txt # Notes from embedding visualization tests
├── requirements.txt # Python dependencies
└── README.md # You are here


## Reproducibility

To set up the environment:

```bash```
pip install -r requirements.txt

All relevant FASTA input files and labels are stored in /Data/.
You will also need access to the pretrained ESM-2 and ESM-C models (see Meta's GitHub).

# Acknowledgements
This project was developed as part of our master's thesis in Business Administration and Data Science at CBS. We thank Evaxion Biotech for their domain guidance and collaboration.