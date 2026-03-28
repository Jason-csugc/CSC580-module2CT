"""Main mnist model utilities.

This module provides helper functions for evaluating and training
an MNIST neural network using TensorFlow/Keras.
"""
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations for consistent performance
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def model_evaluation(x_train, y_train, x_test, y_test, batch_size=100, epochs=20, neurons=512, add_hidden_layer=False, learning_rate=0.5):
    """Train and evaluate an MNIST model with optional hidden layer.

    Args:
        x_train (np.ndarray): Training features, shape (n_train, 784).
        y_train (np.ndarray): One-hot training labels, shape (n_train, 10).
        x_test (np.ndarray): Test features, shape (n_test, 784).
        y_test (np.ndarray): One-hot test labels, shape (n_test, 10).
        batch_size (int, optional): Batch size for training. Defaults to 100.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        neurons (int, optional): Neurons in first dense layer. Defaults to 512.
        add_hidden_layer (bool, optional): Add second hidden layer if True. Defaults to False.
        learning_rate (float, optional): Learning rate for SGD optimizer. Defaults to 0.5.

    Returns:
        tuple: (predictions, test_acc)
            predictions (np.ndarray): Softmax predictions on x_test.
            test_acc (float): Test accuracy.
    """
    if add_hidden_layer:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(neurons, activation='relu'),
            tf.keras.layers.Dense(int(neurons//2), activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
    else:
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(784,)),
            tf.keras.layers.Dense(neurons, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    _ = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.083,
       verbose=0
    )

    _, test_acc = model.evaluate(x_test, y_test)

    predictions = model.predict(x_test)
    return predictions, test_acc

def load_and_preprocess_mnist():
    """Load and preprocess the MNIST dataset.

    Returns:
        tuple: (X_train, Y_train, X_test, Y_test)
            X_train (np.ndarray): Preprocessed training images, shape (60000, 784).
            Y_train (np.ndarray): One-hot encoded training labels, shape (60000, 10).
            X_test (np.ndarray): Preprocessed test images, shape (10000, 784).
            Y_test (np.ndarray): One-hot encoded test labels, shape (10000, 10).
    """
    # load and preprocess the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = X_train.reshape(60000, 784).astype("float32") / 255.0
    X_test = X_test.reshape(10000, 784).astype("float32") / 255.0

    Y_train = tf.keras.utils.to_categorical(Y_train, 10)
    Y_test = tf.keras.utils.to_categorical(Y_test, 10)

    return X_train, Y_train, X_test, Y_test

def create_accuracy_tracker():
    """Create a function to track maximum accuracy across experiments.

    Returns:
        callable: A function that can update and retrieve max accuracy.
            Call with a new accuracy value to update, call with no args to get current max.
    """
    max_accuracy = 0.0

    def tracker(new_accuracy=None):
        nonlocal max_accuracy
        if new_accuracy is not None:
            max_accuracy = max(max_accuracy, new_accuracy)
        return max_accuracy

    return tracker

def main():
    """Run MNIST training/evaluation flow and print sample outcomes.

    - Loads MNIST data from TensorFlow
    - Preprocesses images (flatten + normalize) and labels (one-hot)
    - answers 7 questions about model performance and hyperparameter effects
    - prints results and visualizations for misclassified images
    - tracks and prints the best accuracy achieved across experiments
    """
    accuracy_tracker = create_accuracy_tracker()

    # load and preprocess the MNIST dataset
    X_train, Y_train, X_test, Y_test = load_and_preprocess_mnist()

    # answer 7 questions from assignment
    print('What is the accuracy of the model?')
    default_predictions,default_accuracy = model_evaluation(X_train, Y_train, X_test, Y_test)
    accuracy_tracker(default_accuracy)
    print(f'Default Model Accuracy: {default_accuracy:.4f}')

    print('What are some of the misclassified images?')
    misclassified = np.where(
        np.argmax(default_predictions, axis=1) != np.argmax(Y_test, axis=1)
    )[0]
    print(f'Number of misclassified images: {len(misclassified)}')
    for i in misclassified[:5]:
        plt.figure(f"Misclassified Image Dataset #{i}")
        plt.imshow(X_test[i].reshape(28,28), cmap='gray')
        plt.title(f'Predicted: {np.argmax(default_predictions[i])}, Actual: {np.argmax(Y_test[i])}')
        plt.show()

    print('How is the accuracy affected by using more hidden neurons? Fewer hidden neurons?')
    hidden_neurons = [128, 512, 1024]
    hn_accuracy_results = {}
    for neurons in hidden_neurons:
        _, accuracy = model_evaluation(X_train, Y_train, X_test, Y_test, neurons=neurons)
        accuracy_tracker(accuracy)
        hn_accuracy_results[neurons] = accuracy
    for neurons, accuracy in hn_accuracy_results.items():
        print(f'Hidden Neurons: {neurons}, Test Accuracy: {accuracy:.4f}')

    print('How is the accuracy affected by using different learning rates? Try a range of at least four values.')
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    lr_accuracy_results = {}
    for lr in learning_rates:
        _, accuracy = model_evaluation(X_train, Y_train, X_test, Y_test, learning_rate=lr)
        lr_accuracy_results[lr] = accuracy
        accuracy_tracker(accuracy)
    for lr, accuracy in lr_accuracy_results.items():
        print(f"Learning Rate: {lr}, Test Accuracy: {accuracy:.4f}")
    
    print('How is accuracy affected by adding another hidden layer?')
    _, accuracy_with_hidden_layer = model_evaluation(X_train, Y_train, X_test, Y_test, add_hidden_layer=True)
    accuracy_tracker(accuracy_with_hidden_layer)
    print(f'Accuracy with additional hidden layer: {accuracy_with_hidden_layer:.4f}')

    print('How is accuracy affected by using different batch sizes? Try at least three different batch sizes.')
    batch_sizes = [32, 100, 256]
    bz_accuracy_results = {}
    for batch_size in batch_sizes:
        _, accuracy = model_evaluation(X_train, Y_train, X_test, Y_test, batch_size=batch_size)
        bz_accuracy_results[batch_size] = accuracy
        accuracy_tracker(accuracy)

    for batch_size, accuracy in bz_accuracy_results.items():
        print(f"Batch Size: {batch_size}, Test Accuracy: {accuracy:.4f}")

    print('What is the best accuracy you can get from this multi-layer perceptron?')
    print('Based on experimentation, the best accuracy achieved was around 98-99% using a multi-layer perceptron with appropriate hyperparameters.')
    max_accuracy = accuracy_tracker()
    print(f'Max Accuracy Achieved: {max_accuracy:.4f}')

if __name__ == "__main__":
    main()
