# Nudity Detection and Prevention using Faster R-CNN

## Overview

This project aims to develop a nudity detection and prevention system using the Faster R-CNN model. The model is trained to identify nudity in images and apply appropriate measures to prevent it. This repository includes all necessary scripts for data collection, model training, evaluation, and visualization.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Visualization](#visualization)
- [Usage](#usage)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Directory Structure

The repository is organized as follows:
Nudity_Detection_and_Prevention_using_Faster_R-CNN/ │ ├── dataset/ │ ├── train/ │ ├── validate/ │ ├── test/ │ ├── models/ │ └── nudity_prevention_model.pth │ ├── scripts/ │ ├── data_collection.py │ ├── train_model.py │ ├── validate_model.py │ ├── blackout_nudes.py │ ├── generate_synthetic_labels.py │ ├── verify_clean_data.py │ ├── precision_recall_curve.py │ ├── roc_curve.py │ └── confusion_matrix.py │ ├── visualize/ │ ├── precision_recall_curve.py │ ├── roc_curve.py │ └── confusion_matrix.py │ └── requirements.txt└── README.md

## Data Collection

Data collection is a crucial step in this project. We used web scraping techniques to collect images for training, validation, and testing. The `data_collection.py` script in the `scripts/` directory contains the necessary code for web scraping.

### Data Directory Structure

- `dataset/train`: Contains training images.
- `dataset/validate`: Contains validation images.
- `dataset/test`: Contains test images.

## Model Training

The model is trained using the Faster R-CNN architecture. The `train_model.py` script in the `scripts/` directory is responsible for training the model. The pre-trained weights are saved in the `models/` directory.

## Evaluation

The model is evaluated using various metrics, and the predictions are saved in the `validation_predictions.csv` file. The `validate_model.py` script in the `scripts/` directory handles the validation process.

## Visualization

Visualization is important to understand the model's performance. The following scripts are provided for visualization:

- `precision_recall_curve.py`: Generates and plots the Precision-Recall curve.
- `roc_curve.py`: Generates and plots the ROC curve.
- `confusion_matrix.py`: Generates and plots the Confusion Matrix.

### Visualization Results

(Please add your visualization results here)

## Usage

### Running the Model

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/uv-goswami/Nudity_Detection_and_Prevention_using_Faster_R-CNN.git
   cd Nudity_Detection_and_Prevention_using_Faster_R-CNN
