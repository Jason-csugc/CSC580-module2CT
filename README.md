# CSC580-module2CT
CSC580 Module 2 Critical Thinking

## Product Overview

This repository contains a TensorFlow/Keras-based MNIST classifier implementation and analysis for CSC580 Module 2 (Critical Thinking). It includes utilities for loading and preprocessing the dataset, training/evaluating a multi-layer perceptron, exploring hyperparameter effects, and visualizing misclassified examples.

## Features

- `load_and_preprocess_mnist()`: load MNIST and normalize data
- `model_evaluation()`: train and evaluate a selectable MLP model
- `create_accuracy_tracker()`: track best accuracy across experiments
- Misclassified sample display and metric reporting
- Experimentation over hidden units, learning rates, and batch sizes

## Getting Started

1. Create and activate virtual environment:
   ```bash
   python3 -m venv mod2ct
   source mod2ct/bin/activate
   pip install -r requirements.txt
   ```
2. Run the project:
   ```bash
   python main.py
   ```

## Notes

- Designed for MNIST 28x28 digit classification.
- Provides structured evaluation for coursework questions.
- GUI / plotting uses `matplotlib` for misclassified examples.

## Outputs

1. Command line results

2. Images of misclassified characters

