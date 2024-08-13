# FinGraphDL

Overview
This repository presents a comprehensive framework for training and evaluating Graph Neural Network (GNN) models for regression tasks. It includes preprocessing steps, model definitions, cross validation, evaluation, and prediction visualization. The framework is designed to be modular and configurable through YAML files, making it easy to test different models, optimizers, and other parameters.

## Prerequisites

Ensure you have Python 3.8+ and the following libraries installed:

- torch
- torch-geometric
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- pyyaml

Use the following command to install the necessary libraries:

```bash
pip install torch torch-geometric scikit-learn pandas numpy matplotlib seaborn pyyaml
```

## Configuration
The pipeline uses two YAML files for configuration:

`training_config.yaml`: Configures models, data paths, training parameters, and visualization settings.

Make sure to configure this file in the `config` directory according to your needs.

## Project Structure
The project is structured as follows:

```bash
project/
├── src/
│   ├── __init__.py
│   ├── preprocessing.py  # Graph creation and data preparation
│   ├── models.py         # GNN model definitions
│   ├── loss.py           # Custom loss functions
│   ├── training_utils.py # Utilities for initializing models and training components
│   ├── training.py       # Training and evaluation routines
│   └── evaluation_utils.py # Evaluation metrics and visualization tools
├── main.py               # Main script for running experiments
├── config/
│   └── training_config.yaml # Configuration file for experiments
└── requirements.txt      # Required Python packages
```

- `preprocessing.py`: Contains functions for creating graph data from raw data and for data preprocessing steps.
- `models.py`: Contains definitions for various GNN models.
- `training.py`: Contains functions and classes for training GNN models and evaluating their performance.
- `training_utils.py`:Provides utility functions for initializing optimizers, criteria, and schedulers based on configuration.
- `evalation_utils.py`: Contains utility functions for model evaluation and result visualization.
- `main.py`: The main script that executes the training or evaluation pipeline based on GNN models.


## Usage
To run the pipeline, navigate to the project's root directory and execute:

```bash
python main.py
```

## Output
The pipeline outputs model evaluation results to a CSV file specified in `training_config.yaml`. It also generates and saves visualization plots, if enabled, in directories named after each model within the output directory.

## Customization
Customize the pipeline by editing the YAML configuration files for different models, hyperparameters, training settings, and visualization preferences. The Python scripts can also be modified for additional preprocessing steps, evaluation metrics, or to support other models.

This flexible framework supports quick experimentation and comprehensive analysis of GNN models' performance.