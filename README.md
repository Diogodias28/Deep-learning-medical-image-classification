# AP-TP2

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![torchvision](https://img.shields.io/badge/torchvision-Models%20%26%20Transforms-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Metrics-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Weights & Biases](https://img.shields.io/badge/Weights%20%26%20Biases-Experiment%20Tracking-FFBE00?logo=weightsandbiases&logoColor=black)](https://wandb.ai/)

## Overview

This repository contains the group project for the AP course unit, focused on medical image classification with deep neural networks, architecture comparison, and ensemble model combination.

The project studies the impact of different models and training strategies on the identification of four classes:

- `Biliary_Leaks`
- `Lithiasis`
- `Normal`
- `Stricture`

## Repository Structure

### Main pipeline

- `Treino.py` — main training script for the base model.
- `Dataloader.py` — data loading and preprocessing for the main pipeline.
- `evaluate.py` — evaluation of previously trained checkpoints.
- `Attention_maps.py` — generation of attention maps for visual inspection.
- `demo.ipynb` — interactive demonstration with execution examples and visualisations.
- `checkpoints/` — trained models and checkpoints saved during the experiments.

### Variants, specialists, and tests

- `TreinoBinarySpecialist.py` — training of a binary specialist model.
- `Dataloader_inicial.py` — initial data-loading version used in earlier stages.
- `DataLoaderEnsemble.py` — dataloader adapted for ensemble scenarios.
- `TreinoEnsemble.py` — training of ensemble variants.
- `Ensemble3.py` — combination of three models.
- `ensembleCascade.py` — approach using a specialist model.
- `ensembleTest.py` — final ensemble test without training.
- `ComparacaoDataloader.py` — comparison of data-loading and preprocessing pipelines.
- `Confusion_matrix_v6.py` — generation and analysis of confusion matrices.

## Data

The expected directory layout is:

```text
dataset/
  train/
    Biliary_Leaks/
    Lithiasis/
    Normal/
    Stricture/
  val/
    Biliary_Leaks/
    Lithiasis/
    Normal/
    Stricture/
  test/
    Biliary_Leaks/
    Lithiasis/
    Normal/
    Stricture/
```

The training and evaluation scripts assume this organisation when calling `get_dataloaders(...)`.

## Requirements

Recommended:

- Python 3.8 or newer
- A dedicated virtual environment
- CUDA-compatible GPU

Typical installation on Windows:

```bash
python -m venv .venv
\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r Requirements.txt
```

## Usage

### 1. Train the main model

```bash
python Treino.py
```

### 2. Train the specialist model

```bash
python TreinoBinarySpecialist.py
```

### 3. Evaluate a checkpoint

```bash
python evaluate.py --checkpoint checkpoints/best_model_v6.pth --data-path dataset
```

### 4. Test the ensemble

```bash
python ensembleTest.py
```

### 5. Generate attention maps

```bash
python Attention_maps.py
```

### 6. Run the demonstration notebook

Open `demo.ipynb` in Jupyter or VS Code and execute the cells in order.

## Checkpoints

The `checkpoints/` directory contains the weights of the trained models. The most relevant files include:

- `best_model_v6.pth` — the best-performing model.
- `convnext-base-flips.pth`
- `best_convnext_v4.pth`
- Other checkpoints used in tests and comparisons.

These files are required to reproduce the results without training the models from scratch.

## Comparison and Ensemble Scripts

The following scripts were used to compare approaches or test model combinations:

- `ensembleTest.py` — probability blending between two previously trained models.
- `TreinoEnsemble.py` — training of specific ensemble strategies.
- `Ensemble3.py` — comparison of three models.
- `ensembleCascade.py` — ensemble using a specialist model.
- `ComparacaoDataloader.py` — analysis of the impact of dataloader changes.
- `Confusion_matrix_v6.py` — generation of the confusion matrix for model v6.

## Results and Outputs

Depending on the script being executed, results may be saved to:

- `checkpoints/` — model weights and best epochs.
- `results/` — figures, metrics, and auxiliary outputs, when applicable.
- `wandb/` — experiment logs and metrics, when Weights & Biases is enabled.

## Usage Notes

- The project was developed through several training and testing iterations; therefore, some scripts have similar functions but different purposes.
- Some files were retained to support historical comparisons between architectures and dataloaders.
- To run on a GPU, ensure that PyTorch is installed with CUDA support and that the device is detected correctly.

## Quick Start

To reproduce the main workflow:

1. Prepare the data using the expected directory structure.
2. Train with `Treino.py` or use one of the existing checkpoints.
3. Evaluate with `evaluate.py`.
4. Use `demo.ipynb` and `Attention_maps.py` for visualisation.

## Authorship Note

This project was developed collaboratively for the **Deep Learning** course unit of the **Master's Degree in Artificial Intelligence at the University of Minho (UMinho)**.
