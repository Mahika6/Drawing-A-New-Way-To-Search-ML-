# Drawing-A-New-Way-To-Search
ML Mini Project 
# Drawing: A New Way To Search

**Course:** UE23CS352A Machine Learning  
**Project Duration:** September 29 - October 13, 2025  
**Team Project:** Hand-drawn Image Recognition System

## Problem Statement

Develop an efficient system that recognizes labels of hand-drawn images based on Google's QuickDraw dataset. Using words can be limited when communicating across cultures and literacy levels. Images are a shared medium of communication that can beneficially bridge those divides.

## Project Overview

This project implements multiple machine learning approaches to classify hand-drawn doodles from Google's QuickDraw dataset - the world's largest doodling dataset consisting of hand-drawn images from over 15 million people worldwide.

## Dataset

**Google QuickDraw Dataset**
- World's largest doodling dataset
- Hand-drawn images from 15+ million users globally
- Uses .npy bitmap version of the data
- Raw pixel inputs with values from 0-255
- Binary images (black and white) for efficient processing

**Sample Classes:**
- Banana
- Hockey stick
- Squirrel  
- Watermelon
- Bathtub

## Implemented Models

### 1. Logistic Regression (Baseline)
- Simple and fast training model
- Uses raw pixel inputs from numpy bitmaps
- **Results on 10 classes:** 64.64% accuracy, 122s training time
- **Results on 50 classes:** 43.89% accuracy, 1089s training time

### 2. Support Vector Machine (SVM)
- Implemented with four different kernels:
  - Linear Kernel: 22.22% accuracy, 1831s training
  - RBF Kernel: 61.01% accuracy, 2842s training  
  - Polynomial Kernel: 50.89% accuracy, 6673s training
  - Sigmoid Kernel: 11.72% accuracy, 6971s training
- Parameters: polynomial degree=5, RBF coefficient=1, RBF gamma=1, sigmoid coefficient=1

### 3. Convolutional Neural Network (CNN)
- **Best performing model overall**
- Progressive simplification analysis:
  - v1 (Full CNN): 86.59% accuracy (10 classes), 82.12% (50 classes)
  - v2 (Remove 2nd conv layer): 85.5% accuracy (10 classes), 76.1% (50 classes)
  - v3 (Remove 1st max pool): 86.55% accuracy, faster training
  - v4 (Dense layer 64 units): 85.6% accuracy

### 4. Transfer Learning
- Tested pre-trained ImageNet models:
  - Inception v3: 48.77% accuracy (3 classes)
  - MobileNet: 75.4% accuracy, 10290s training
  - ResNet50: 62.72% accuracy, 15232s training
  - VGG: Memory issues encountered

## Installation and Setup

### Prerequisites
```bash
pip install tensorflow
pip install numpy
pip install matplotlib  
pip install scikit-learn
pip install pandas
```

### Dataset Download
1. Download Google QuickDraw dataset from: https://github.com/googlecreativelab/quickdraw-dataset
2. Use the .npy format files for bitmap data
3. Place dataset files in `data/` directory

### Project Structure
```
drawing-search/
â”œâ”€â”€ data/              # QuickDraw dataset files
â”œâ”€â”€ models/            # Trained model files
â”œâ”€â”€ src/               # Source code
â”‚   â”œâ”€â”€ logistic_regression.py
â”‚   â”œâ”€â”€ svm_models.py  
â”‚   â”œâ”€â”€ cnn_models.py
â”‚   â”œâ”€â”€ transfer_learning.py
â”‚   â””â”€â”€ utils.py       # Helper functions
â”œâ”€â”€ notebooks/         # Jupyter notebooks for experimentation  
â”œâ”€â”€ results/           # Model outputs and evaluation results
â”œâ”€â”€ README.md
â””â”€â”€ requirements.txt
```

## Running the Code

### Training Models
```bash
# Logistic Regression
python src/logistic_regression.py --classes 10

# SVM with different kernels  
python src/svm_models.py --kernel rbf --classes 10

# CNN variations
python src/cnn_models.py --version 1 --classes 10 --binarized False

# Transfer Learning
python src/transfer_learning.py --model mobilenet --classes 3
```

### Evaluation
```bash
python src/evaluate.py --model cnn --version 3 --classes 10
```

## Key Findings

### Model Performance Comparison
- **Best Overall**: CNN v3 (simplified) - 86.55% accuracy with significantly reduced training time
- **Fastest**: Logistic Regression - ~30 minutes training time  
- **Most Resource Intensive**: Transfer Learning models - hours of training
- **Surprising**: SVMs performed worse than linear regression, likely due to insufficient parameter tuning

### Training Efficiency
- **Logistic Regression**: Fastest training (~0.5 hours)
- **CNN**: Best accuracy/time tradeoff (~4+ hours)  
- **Transfer Learning**: Most expensive due to deep model complexity

### Common Misclassifications
- Banana often confused with hockey stick
- Indicates need for more sophisticated models to handle drawing quality variations

## Technical Implementation Notes

### Data Preprocessing
- Binarization of pixel values for efficiency
- Normalization of raw pixel inputs (0-255 range)
- Image resizing for transfer learning models

### CNN Architecture Optimization  
- Progressive layer removal analysis
- Impact study of max pooling layers
- Dense layer size optimization (128 vs 64 units)

### Transfer Learning Challenges
- Image preprocessing for pre-trained model compatibility
- Memory constraints with larger models (VGG)
- Limited performance improvement over custom CNN

## Future Work

1. **CNN Enhancement**: More extensive experiments to determine optimal layer configurations
2. **Alternative Transfer Learning**: Use pre-trained models as fixed feature extractors for logistic regression  
3. **Data Efficiency**: Develop efficiency metrics for smaller dataset scenarios
4. **Multi-cultural Analysis**: Evaluate model performance across different cultural drawing styles

## Deliverables

- [x] Source code in GitHub repository
- [x] README.md with setup instructions  
- [ ] One-page PDF write-up
- [ ] Presentation slides for demo
- [ ] Live demonstration preparation

## Team Information

This project is part of the UE23CS352A Machine Learning course mini-project requirement, implementing a 2-week hands-on machine learning solution.