
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow import keras

def plot_history(history):
    """
    This function plots the training and validation accuracy and loss.

    Input:
        history: the history of the model
    """

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], "bo", label="Training accuracy")
    plt.plot(history['val_accuracy'], "b", label="Validation accuracy")
    plt.title("Training and validation accuracy", fontsize = 16)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Accuracy", fontsize = 12)
    plt.xlim([0, len(history)])  # Valid x-range
    plt.xticks(range(len(history)))  # Set ticks for all epochs
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], "bo", label="Training loss")
    plt.plot(history['val_loss'], "b", label="Validation loss")
    plt.title("Training and validation loss", fontsize = 16)
    plt.xlabel("Epochs", fontsize = 12)
    plt.ylabel("Loss", fontsize = 12)
    plt.xlim([0, len(history)])  # Valid x-range
    plt.xticks(range(len(history)))  # Optional: tick at every epoch
    plt.legend()

    plt.show()

