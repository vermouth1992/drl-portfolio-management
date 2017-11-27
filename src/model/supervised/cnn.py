"""
Train a supervised CNN model using optimal stock as label
"""
import numpy as np
from keras import backend as K
import os

assert K.backend() == 'theano'
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import load_model
from keras.optimizers import Adam
from ..base_model import BaseModel
from utils.data import normalize

class StockCNN(BaseModel):
    def __init__(self, nb_classes, window_length, weights_file='weights/cnn.h5'):
        self.model = None
        self.weights_file = weights_file
        self.nb_classes = nb_classes
        self.window_length = window_length

    def build_model(self, load_weights=True):
        """ Load training history from path

        Args:
            load_weights (Bool): True to resume training from file or just deploying.
                                 Otherwise, training from scratch.

        Returns:

        """
        if load_weights:
            self.model = load_model(self.weights_file)
            print('Successfully loaded model')
        else:
            self.model = Sequential()

            self.model.add(
                Conv2D(filters=32, kernel_size=(1, 3), input_shape=(1, self.nb_classes - 1, self.window_length),
                       activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(filters=32, kernel_size=(1, self.window_length - 2), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Flatten())
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.nb_classes, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(lr=1e-3),
                               metrics=['accuracy'])
            print('Built model from scratch')

    def train(self, X_train, Y_train, verbose=True):
        self.model.fit(X_train, Y_train, batch_size=128, epochs=100, verbose=verbose)
        self.model.save(self.weights_file)
        print('Finish.')

    def evaluate(self, X_test, Y_test, verbose=False):
        return self.model.evaluate(X_test, Y_test, verbose=verbose)

    def predict(self, X_test, verbose=False):
        return self.model.predict(X_test, verbose=verbose)

    def predict_single(self, observation):
        """ Predict the action of a single observation

        Args:
            observation: (num_stocks + 1, window_length)

        Returns: a single action array with shape (num_stocks + 1,)

        """
        obsX = observation[1:, -self.window_length:, 3] / observation[1:, -self.window_length:, 0]
        obsX = normalize(obsX)
        obsX = np.flip(obsX, axis=1)
        obsX = np.expand_dims(np.expand_dims(obsX, axis=0), axis=0)
        return np.squeeze(self.model.predict(obsX), axis=0)
