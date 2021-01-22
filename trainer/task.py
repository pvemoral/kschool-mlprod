"""A simple main file to showcase the template."""

import logging.config
import argparse
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations

#importar datos
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics



def _download_data():
    train, test = datasets.mnist.load_data()
    
    x_train, y_train = train
    x_test, y_test = test

    return x_train, y_train, x_test, y_test


def train_and_evaluate(batch_size, epoch, job_dir, output_path):

    # Download data
    x_train, y_train, x_test, y_test = _download_data()

    # Processes the data

    # Build the model

    # Train the model

    # Evalutate de model
    pass

def main():
    """Entry point for your module."""
    # PARAMETROS QUE NECESITAMOS

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size for the training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for the training')
    parser.add_argument('--job-dir', default=None, required=False, help='Option for AI Platform')
    parser.add_argument('--model-output-path', help='Path to write the SaveModel format')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, job_dir, output_path)
    pass

if __name__ == "__main__":
    main()
