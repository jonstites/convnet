#!/usr/bin/env python3

import argh
import datasets
import glob
import numpy as np
import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras import regularizers
import os



def find_model_file(model_dir, weights_filename):
    weights_regex = os.path.join(model_dir, "model-*.hdf5")
    weights_files = glob.glob(weights_regex)
    best_weights_file = None
    best_epoch = 0

    for weights_file in weights_files:
        base_name = os.path.basename(weights_file)
        epoch = int(base_name.split("-")[1])
        if epoch > best_epoch:
            best_epoch = epoch
            best_weights_file = weights_file
    return best_weights_file, best_epoch


def main(train_dir, test_dir, model_dir=".", batch_size=32, epochs=100, save_period=10):

    
    model_filepath = os.path.join(model_dir, "model-{epoch:02d}-{val_loss:.2f}.hdf5")
    last_model_file, last_epoch = find_model_file(model_dir, model_filepath)
    if last_model_file:
        model = load_model(last_model_file)
    else:
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=5, input_shape=(28, 28, 1), activation="relu", kernel_regularizer=regularizers.l2(0.0001), padding="same"))
        model.add(BatchNormalization())

        for _ in range(5):
            model.add(Conv2D(filters=256, kernel_size=3, activation="relu", strides=2, kernel_regularizer=regularizers.l2(0.0001),padding="same"))
            model.add(BatchNormalization())

        model.add(Flatten())
        model.add(Dense(10, kernel_regularizer=regularizers.l2(0.0001), activation="softmax"))
        model.compile(optimizer='Adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    print(model.summary())
    checkpointer = ModelCheckpoint(filepath=model_filepath, verbose=1, save_best_only=False, period=save_period)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    

    not_mnist = datasets.NotMNISTDataset(train_dir, test_dir)

    x_train = not_mnist.train.images
    y_train = not_mnist.train.labels
    x_val = not_mnist.validate.images
    y_val = not_mnist.validate.labels

    model.fit(
        x_train, y_train, batch_size=batch_size,
        validation_data=(x_val, y_val),
        epochs=epochs, callbacks=[checkpointer], initial_epoch=last_epoch)

    
                
if __name__ == "__main__":
    argh.dispatch_command(main)
