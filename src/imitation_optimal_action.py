"""
Train a classifier given optimal action
"""
from utils.data import read_stock_history, create_optimal_imitation_dataset, create_imitation_dataset

import numpy as np

from keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D, Conv2D
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

# test on 3 never seen stocks
test_stocks = ['GOOGL', 'DISH', 'ILMN']
test_history = np.empty(shape=(len(test_stocks), history.shape[1], history.shape[2]))
for i, stock in enumerate(test_stocks):
    test_history[i] = history[abbreviation.index(stock), :, :]


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
    return model


def train_optimal_action_given_history_obs(target_history, target_stocks, window_length):
    nb_classes = len(target_stocks) + 1
    (X_train, y_train), (X_validation, y_validation) = create_imitation_dataset(target_history, window_length)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_validation = np_utils.to_categorical(y_validation, nb_classes)
    model = create_network_give_past(nb_classes, window_length)
    X_train = np.expand_dims(X_train, axis=-1)
    X_validation = np.expand_dims(X_validation, axis=-1)
    continue_train = True
    while continue_train:
        model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_validation, Y_validation),
                  shuffle=True)
        continue_train = input("True to continue train, otherwise stop training...\n")
    return model


if __name__ == '__main__':
    # optimal_given_future = train_optimal_action_given_future_obs(target_history, target_stocks)
    optimal_given_history_model = train_optimal_action_given_history_obs(target_history, target_stocks, 3)
    (X_test, y_test), (_, _) = create_imitation_dataset(test_history, 3)
    Y_test = np_utils.to_categorical(y_test, 4)
    X_test = np.expand_dims(X_test, axis=-1)
    loss, acc = optimal_given_history_model.evaluate(X_test, Y_test)
    print('Testing result: loss - {}, accuracy - {}'.format(loss, acc))