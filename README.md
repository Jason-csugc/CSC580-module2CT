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
<img width="1241" height="1050" alt="image" src="https://github.com/user-attachments/assets/4d24efcf-62da-4b95-9489-64142152ef86" />

2. Images of misclassified characters
<img width="648" height="552" alt="image" src="https://github.com/user-attachments/assets/326b96b2-66d4-4b5b-bdbb-443c08b8a1f8" />
<img width="651" height="553" alt="image" src="https://github.com/user-attachments/assets/9c12dcda-3914-49f5-8ae9-5200f5723b62" />
<img width="652" height="555" alt="image" src="https://github.com/user-attachments/assets/8f6f6f7b-8b03-4146-bf50-232891ce5569" />
<img width="650" height="552" alt="image" src="https://github.com/user-attachments/assets/64ac7708-ada3-48d8-8a9a-557daa722c53" />
<img width="651" height="553" alt="image" src="https://github.com/user-attachments/assets/f95476db-1acb-4a9c-9eae-c97ea58b7af1" />
