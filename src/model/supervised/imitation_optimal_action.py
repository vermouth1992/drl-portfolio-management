"""
Train a classifier given optimal action
"""
from utils.data import create_optimal_imitation_dataset, create_imitation_dataset

import numpy as np

from keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D, Conv2D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils


def create_network_given_future(nb_classes, weight_path='weights/optimal_3_stocks.h5'):
    model = Sequential()
    model.add(Dense(512, input_shape=(nb_classes,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    try:
        model.load_weights(weight_path)
        print('Model load successfully')
    except:
        print('Build model from scratch')
    return model


def train_optimal_action_given_future_obs(model, target_history, target_stocks,
                                          weight_path='weights/optimal_3_stocks.h5'):
    (X_train, y_train), (X_test, y_test) = create_optimal_imitation_dataset(target_history)
    nb_classes = len(target_stocks) + 1

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    continue_train = True
    while continue_train:
        model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_data=(X_test, Y_test), shuffle=True)
        save_weights = input('Type True to save weights\n')
        if save_weights:
            model.save(weight_path)
        continue_train = input('True to continue train, otherwise stop\n')


def create_network_give_past(nb_classes, window_length, weight_path='weights/imitation_3_stocks.h5'):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(1, 3), input_shape=(nb_classes, window_length, 1),
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32, kernel_size=(1, window_length - 2), input_shape=(nb_classes, window_length - 2, 1),
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten(input_shape=(window_length, nb_classes)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    try:
        model.load_weights(weight_path)
        print('Model load successfully')
    except:
        print('Build model from scratch')
    return model


def train_optimal_action_given_history_obs(model, target_history, target_stocks, window_length,
                                           weight_path='weights/imitation_3_stocks.h5'):
    nb_classes = len(target_stocks) + 1
    (X_train, y_train), (X_validation, y_validation) = create_imitation_dataset(target_history, window_length)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_validation = np_utils.to_categorical(y_validation, nb_classes)
    X_train = np.expand_dims(X_train, axis=-1)
    X_validation = np.expand_dims(X_validation, axis=-1)
    continue_train = True
    while continue_train:
        model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_validation, Y_validation),
                  shuffle=True)
        save_weights = input('Type True to save weights\n')
        if save_weights:
            model.save(weight_path)
        continue_train = input("True to continue train, otherwise stop training...\n")
