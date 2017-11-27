"""
Train a supervised CNN model using optimal stock as label
"""
import numpy as np
import os
from keras import backend as K
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.models import load_model
from ..base_model import BaseModel

class LSTM(BaseModel):
    def __init__(self, num_classes, window_length, weights_file='weights/lstm.h5'):
        self.model = None
        self.weights_file = weights_file
        self.num_classes = num_classes
        self.window_length = window_length
    def build_model(self, load_weights=True):
        """ Load training history from path

        Args:
            load_weights (Bool): True to resume training from file or just deploying.
                                 Otherwise, training from scratch.

        Returns:

        """
        if load_weights:
            model = keras.models.load_model(self.weights_file)
            print('Successfully loaded model')
        else:
            self.model = Sequential()
            self.model.add(keras.layers.LSTM(20, input_shape=(self.num_classes-1, self.window_length), dropout=0.5))
            self.model.add(Dense(self.num_classes, activation='softmax'))
            
            self.model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
            print('Built model from scratch')
            

    def train(self, X_train, Y_train, verbose=True):
        self.model.fit(X_train, Y_train, batch_size=32, epochs=7, verbose=verbose)
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
        obsX = observation[1:, -(self.window_length):, 3]/observation[1:, :, 0]
        obsX = np.flip(obsX, axis = 1)
        obsX = np.expand_dims(obsX, axis = 0)
        return np.squeeze(self.model.predict(obsX), axis = 0)