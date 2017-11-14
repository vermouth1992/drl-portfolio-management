"""
Train a supervised CNN model using optimal stock as label
"""
import numpy as np
from keras import backend as K
import os
os.environ['KERAS_BACKEND'] = "theano"
reload(K)
assert K.backend() == "theano", "Couldn't load theano as keras backend"
K.set_image_dim_ordering('th')
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import np_utils
from keras.models import load_model

class CNN(object):
    def __init__(self, env, weights_file='weights/cnn.h5'):
        self.env = env
        self.model = None
        self.weights_file = weights_file
    def build_model(self, load_weights=True):
        """ Load training history from path

        Args:
            load_weights (Bool): True to resume training from file or just deploying.
                                 Otherwise, training from scratch.

        Returns:

        """
        if load_weights:
            model = load_model(self.weights_file)
            print('Successfully loaded model')
        else:
            self.model = Sequential()
            
            self.model.add(Conv2D(128, (1, 1), activation='relu', input_shape=(4,self.env.window_length+1,self.env.num_stocks)))
            self.model.add(BatchNormalization())
            self.model.add(Conv2D(64, (1, 1), activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(MaxPooling2D(pool_size=(1,1)))
            self.model.add(Dropout(0.25))

            self.model.add(Flatten())
            self.model.add(Dense(1024, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.4))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(BatchNormalization())
            self.model.add(Dropout(0.4))
            self.model.add(Dense(self.env.num_stocks+1, activation='softmax'))
            self.model.add(BatchNormalization())
            
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            print('Built model from scratch')
            

    def train(self, X_train, Y_train, verbose=True):
        self.model.fit(X_train, Y_train, batch_size=32, epochs=10, verbose=verbose)
        self.model.save(self.weights_file)
        print('Finish.')
        
    def evaluate(self, X_test, Y_test, verbose=False):
        return self.model.evaluate(X_test, Y_test, verbose=verbose)
    
    def predict(self, X_test, verbose=False):
        return self.model.predict(X_test, verbose=verbose)