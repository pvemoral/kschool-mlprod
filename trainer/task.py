"""A simple main file to showcase the template."""

import logging.config
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations

#importar datos
from tensorflow.keras import datasets
from tensorflow.keras.utils import to_categorical

from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import utils

from tensorflow.keras import callbacks

from . import __version__

LOGGER = logging.getLogger()
VERSION = __version__

def _download_data():
    logging.info("Download data")
    train, test = datasets.mnist.load_data()
    
    x_train, y_train = train
    x_test, y_test = test

    return x_train, y_train, x_test, y_test


def _preprocess_data(x_train, y_train):
    logging.info("Preporcess data")
    x_train = x_train / 255.0
    y_train = utils.to_categorical(y_train)

    return x_train, y_train

def _build_model():
    logging.info("Create model")
    model = models.Sequential()

    model.add(layers.Input((28,28), name='my_input_layer'))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation=activations.relu))
    model.add(layers.Dense(64, activation=activations.relu))
    model.add(layers.Dense(32, activation=activations.relu))
    model.add(layers.Dense(10, activation=activations.softmax))

    return model

def _train_model(model, x_train, y_train, batch_size, epochs, validation_split=0.15):

    return model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    

def train_and_evaluate(batch_size, epoch, job_dir, model_output_path):

    # Download data
    x_train, y_train, x_test, y_test = _download_data()

    # Processes the data
    x_train_p, y_train_p = _preprocess_data(x_train, y_train)
    x_test_p, y_test_p = _preprocess_data(x_test, y_test)
    
    # Build the model

    model = _build_model()
    model.compile(loss=losses.categorical_crossentropy,
                  optimizer=optimizers.Adam(),
                  metrics=[metrics.categorical_accuracy])
    # Train the model
    logdir = os.path.join(job_dir,"logs/scalars/" + time.strftime("%Y%m%d-%H%M%S"))

    #Callback para tensorflow
    #Sirve para ver el proceso de entrenamiento fuera de los Logs

    tb_callback = callbacks.TensorBoard(log_dir=logdir)
    model.fit(x_train_p, 
              y_train_p, 
              batch_size=batch_size, 
              epochs=epoch,
              callbacks=[tb_callback])

    # Evalutate de model
    loss_value, accuracy = model.evaluate(x_test_p, y_test_p)
    logging.info("loss_value:{loss_value}, accuracy:{accuracy}".format(loss_value = loss_value, accuracy = accuracy))

    # Save model in TF SavedModel Format
    model_dir = os.path.join(model_output_path, VERSION)
    models.save_model(model, model_dir, save_format='tf')

    #return loss_value, accuracy

def main():
    """Entry point for your module."""
    # PARAMETROS QUE NECESITAMOS
    # los hiperparametros que queramos añadir al programa se tiene que pasar por linea de comandos. 
    # Cualquier parámetro que queramos añadir para que el optimizador de hiperparámetros los pueda modificar
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size for the training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for the training')
    # Estos parámetros son opciones de para AI platform
    # '--job-dir' obligatorio
    # '--model-output-path' path donde escribiremos el binario del modelo después de entrenarlo y que utilizaremos en producción.
    parser.add_argument('--job-dir', default=None, required=False, help='Option for AI Platform')
    parser.add_argument('--model-output-path', help='Path to write the SaveModel format')

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    job_dir = args.job_dir
    model_output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, job_dir, model_output_path)
    pass

if __name__ == "__main__":
    main()
