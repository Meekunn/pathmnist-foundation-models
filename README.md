# Foundation Models for Medical Image Classification (PathMNIST)

This project explores the use of Convolutional Neural Networks (CNNs), Residual Networks (ResNet) and vision transformers (ViTs)
as foundation models for medical image classification.

## Dataset

PathMNIST is a histopathology dataset containing 28×28 RGB image tiles
from colorectal cancer tissue and is categorized into 9 classes.

## Models Implemented

- Custom CNN (baseline)
- ResNet18 (pretrained on ImageNet)
- Vision Transformer (ViT-B/16)

## Key Findings

- Transfer learning significantly improves performance on small medical datasets.
- Vision Transformers capture global tissue patterns better than shallow CNNs.
- Confusion matrices reveal class-specific challenges in histopathology.

## Tools

- PyTorch
- TorchVision
- MedMNIST
