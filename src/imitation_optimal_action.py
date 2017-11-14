"""
Train a classifier given optimal action
"""
from utils.data import read_stock_history, create_optimal_imitation_dataset, create_imitation_dataset

import numpy as np

from keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import np_utils

# dataset
history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
history = history[:, :, :4]
target_stocks = ['AAPL', 'CMCSA', 'REGN']
target_history = np.empty(shape=(len(target_stocks), history.shape[1], history.shape[2]))
for i, stock in enumerate(target_stocks):
    target_history[i] = history[abbreviation.index(stock), :, :]


def create_network_given_future(nb_classes):
    model = Sequential()
    model.add(Dense(256, input_shape=(nb_classes,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return model


def train_optimal_action_given_future_obs(target_history, target_stocks):
    (X_train, y_train), (X_test, y_test) = create_optimal_imitation_dataset(target_history)

    nb_classes = len(target_stocks) + 1

    # normalize the input
    X_train = (X_train - 1) * 100
    X_test = (X_test - 1) * 100

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = create_network_given_future(nb_classes)
    model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_test, Y_test), shuffle=True)
    return model


def create_network_give_past(nb_classes, window_length):
    model = Sequential()
    model.add(Conv1D(16, 3, input_shape=(window_length, nb_classes), activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten(input_shape=(window_length, nb_classes)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    return model


def train_optimal_action_given_history_obs(target_history, target_stocks, window_length):
    nb_classes = len(target_stocks) + 1
    (X_train, y_train), (X_test, y_test) = create_imitation_dataset(target_history, window_length)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    model = create_network_give_past(nb_classes, window_length)
    X_train = np.transpose(X_train, (0, 2, 1))
    X_test = np.transpose(X_test, (0, 2, 1))
    model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_test, Y_test), shuffle=True)
    return model

if __name__ == '__main__':
    optimal_given_future = train_optimal_action_given_future_obs(target_history, target_stocks)
    optimal_given_history = train_optimal_action_given_history_obs(target_history, target_stocks, 7)