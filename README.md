# Anomaly Detection Using Deep Learning Models

This project implements anomaly detection for spacecraft telemetry data using deep learning models including a Multi-Layer Perceptron (MLP) and a Transformer.

## Project Overview

The project aims to identify anomalies in spacecraft telemetry data by training supervised learning models on labeled time series data. It leverages PyTorch for modeling and includes both traditional MLP and attention-based Transformer architectures to detect deviations in sensor readings.

## Key Functionalities

- **Data Loading & Labeling**: Reads telemetry data (`.npy` files) and labels them using a `labeled_anomalies.csv` reference.
- **Visualization**: Includes seaborn and matplotlib-based plots for data exploration.
- **Preprocessing**: Formats the data into windows suitable for input to deep learning models.
- **Model Architectures**: Implements both an MLP and a Transformer model using PyTorch.
- **Training & Evaluation**: Trains the models, computes standard evaluation metrics (accuracy, precision, recall, F1 score), and visualizes performance.

## Setup Instructions

### Prerequisites

Install required Python packages using pip:

```bash
pip install numpy pandas matplotlib seaborn torch scikit-learn
```

### Data Setup
 Link to the Dataset : [text](https://www.kaggle.com/datasets/patrickfleith/nasa-anomaly-detection-dataset-smap-msl/data)

1. Download the above dataset and place the following files in a `./data/` directory:
   - `labeled_anomalies.csv`: Contains metadata and labels for telemetry channels.
   - `.npy` files: Each represents a time series from a different channel.

2. Ensure the directory structure is:

```
.
├── anomaly_detection.ipynb
└── data
    ├── labeled_anomalies.csv
    ├── train
         └── chan_id.npy
         └── chan_id2.npy
         └── ...
    ├── test
         └── chan_id.npy
         └── chan_id2.npy
         └── ...
```

## Running the Notebook

1. Launch Jupyter Notebook:

```bash
jupyter notebook anomaly_detection.ipynb
```

2. Execute cells sequentially:
   - Data loading and labeling
   - Visualizations
   - Model definitions (MLP and Transformer)
   - Training and evaluation

## Reproducing Results

To reproduce the results:
- Use the provided data files and maintain consistent preprocessing steps.
- You can adjust model hyperparameters or dataset splits for experimentation.
- Performance is reported via accuracy, precision, recall, and F1 score.
