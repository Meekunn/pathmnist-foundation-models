# Foundation Models for Medical Image Classification (PathMNIST)

A comprehensive exploration of deep learning models for histopathology image classification using the PathMNIST dataset. This project compares Convolutional Neural Networks (CNNs), Residual Networks (ResNet), and Vision Transformers (ViTs) as foundation models for medical image analysis.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Findings](#key-findings)
- [Technologies](#technologies)

## Overview

This project investigates the effectiveness of different deep learning architectures for medical image classification, specifically focusing on histopathology images. We implement and compare three model architectures:

- **Custom CNN**: A baseline convolutional neural network
- **ResNet18**: A pretrained residual network leveraging transfer learning
- **Vision Transformer (ViT-B/16)**: A transformer-based architecture for image classification

## Dataset

**PathMNIST** is a histopathology dataset from the MedMNIST collection. It contains:

- **Image size**: 28×28 RGB image tiles
- **Source**: Colorectal cancer tissue samples
- **Classes**: 9 different tissue categories
- **Purpose**: Classification of histopathological tissue types

The dataset is automatically downloaded when running the notebooks via the `medmnist` package.

## Models

### 1. Custom CNN (Baseline)

A lightweight convolutional neural network designed as a baseline model to establish performance benchmarks.

### 2. ResNet18 (Transfer Learning)

A pretrained ResNet18 model fine-tuned on ImageNet weights. This model leverages transfer learning to improve performance on the medical imaging task.

### 3. Vision Transformer (ViT-B/16)

A Vision Transformer model with base architecture (ViT-B/16) that uses self-attention mechanisms to capture global patterns in histopathology images.

## Project Structure

```
pathmnist-foundation-models/
├── notebooks/
│   ├── explore_data.ipynb      # Data exploration and visualization
│   ├── cnn_baseline.ipynb      # Custom CNN implementation
│   ├── resnet18.ipynb          # ResNet18 model training
│   └── vit.ipynb               # Vision Transformer implementation
├── results/
│   ├── cnn_accuracy.png        # CNN training accuracy curves
│   ├── cnn_loss.png            # CNN training loss curves
│   ├── resnet_confusion.png    # ResNet18 confusion matrix
│   └── vit_confusion.png       # ViT confusion matrix
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/Meekunn/pathmnist-foundation-models.git
cd pathmnist-foundation-models
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The project is organized as Jupyter notebooks. Run them in the following order:

1. **Data Exploration**: Start with `notebooks/explore_data.ipynb` to understand the dataset
2. **Baseline Model**: Run `notebooks/cnn_baseline.ipynb` to train the custom CNN
3. **ResNet18**: Execute `notebooks/resnet18.ipynb` for transfer learning approach
4. **Vision Transformer**: Run `notebooks/vit.ipynb` to train the ViT model

Each notebook is self-contained and can be run independently, though following the suggested order provides better context.

## Results

Model performance metrics and visualizations are stored in the `results/` directory:

- **Training Curves**: Accuracy and loss plots for the CNN baseline
- **Confusion Matrices**: Classification performance for ResNet18 and ViT models

These visualizations help compare model performance and identify class-specific classification challenges.

## Key Findings

- **Transfer Learning Advantage**: Pretrained models (ResNet18) significantly outperform baseline CNNs on small medical datasets
- **Global Pattern Recognition**: Vision Transformers excel at capturing global tissue patterns compared to shallow CNNs
- **Class-Specific Challenges**: Confusion matrices reveal specific tissue types that are more difficult to classify accurately
- **Architecture Comparison**: Different architectures show varying strengths in histopathology image classification

## Technologies

- **PyTorch**: Deep learning framework
- **TorchVision**: Pretrained models and image transformations
- **MedMNIST**: Medical imaging dataset collection
- **NumPy**: Numerical computations
- **Matplotlib**: Visualization and plotting
- **scikit-learn**: Evaluation metrics and utilities
