"""A simple main file to showcase the template."""

import argparse
import logging.config
import os
import time

import tensorflow as tf

from tensorflow.keras import datasets
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras import metrics
from tensorflow.keras import utils
from tensorflow.keras import callbacks

from . import __version__
from . import __version__

LOGGER = logging.getLogger()
VERSION = __version__

def _download_data():
    logging.info("Download data")
    train, test = datasets.mnist.load_data()
    
    x_train, y_train = train
    x_test, y_test = test

    return x_train, y_train, x_test, y_test

def _preprocess_convo_data(x_train, y_train):
    logging.info("Preporcess data CONVO")
    x_train = x_train / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = utils.to_categorical(y_train)

    return x_train, y_train

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

def _build_conv_model():
    logging.info("Create model")
    model = models.Sequential()

    model.add(layers.Input((28, 28, 1), name='my_input_layer'))
    model.add(layers.Conv2D(32, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(8, (3, 3), activation=activations.relu))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation=activations.softmax))

    return model

def _train_model(model, x_train, y_train, batch_size, epochs, validation_split=0.15):

    return model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
    

def train_and_evaluate(batch_size, epoch, is_hypertune, model_type, job_dir, model_output_path):

    # Download data
    x_train, y_train, x_test, y_test = _download_data()

    
    
    # Build the model
    if model_type == 'Dense':
        # Processes the data
        x_train_p, y_train_p = _preprocess_data(x_train, y_train)
        x_test_p, y_test_p = _preprocess_data(x_test, y_test)
        model = _build_model()
    else:
        # Processes the data
        x_train_p, y_train_p = _preprocess_convo_data(x_train, y_train)
        x_test_p, y_test_p = _preprocess_convo_data(x_test, y_test)
        model = _build_conv_model()

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

    if not is_hypertune:
        # Save model in TF SavedModel Format
        model_dir = os.path.join(model_output_path, VERSION)
        models.save_model(model, model_dir, save_format='tf')
    else:
        # communicate the result of the evaluate of the model to the cosole
        metric_tag = 'accuracy_live_class'
        # debe ser subdirectorio del job_dir a un subdirectorio del TAG
        eval_path = os.path.join(job_dir, metric_tag)
        writer = tf.summary.create_file_writer(eval_path)

        with writer.as_default():
            # escribiré la metrica i el valor de la métrica
            tf.summary.scalar(metric_tag, accuracy, step=epoch)
        writer.flush()

    #return loss_value, accuracy

def main():
    """Entry point for your module."""
    # PARAMETROS QUE NECESITAMOS
    # los hiperparametros que queramos añadir al programa se tiene que pasar por linea de comandos. 
    # Cualquier parámetro que queramos añadir para que el optimizador de hiperparámetros los pueda modificar
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, help='Batch size for the training')
    parser.add_argument('--epochs', type=int, help='Number of epochs for the training')
    # Para informar si vamos a hacer o no el hyperTuning
    parser.add_argument('--hypertune', action='store_true', help='This is a hypertunning job')
    parser.add_argument('--model-type', default='Dense', help='This is model type job')
    
    # Estos parámetros son opciones de para AI platform
    # '--job-dir' obligatorio
    # '--model-output-path' path donde escribiremos el binario del modelo después de entrenarlo y que utilizaremos en producción.
    parser.add_argument('--job-dir', default=None, required=False, help='Option for AI Platform')
    parser.add_argument('--model-output-path', help='Path to write the SaveModel format')

    args = parser.parse_args()
    batch_size = args.batch_size
    epochs = args.epochs
    is_hypertune = args.hypertune
    model_type = args.model_type
 
    job_dir = args.job_dir
    model_output_path = args.model_output_path

    train_and_evaluate(batch_size, epochs, is_hypertune, model_type, job_dir, model_output_path)
    pass

if __name__ == "__main__":
    main()
