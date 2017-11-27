"""
Train imitation learning model
"""

from utils.data import read_stock_history, normalize, create_imitation_dataset
from model.supervised.cnn import StockCNN
from model.supervised.lstm import StockLSTM
from keras.utils import np_utils

import numpy as np
import argparse
import pprint


def get_model_name(predictor_type, window_length):
    return 'imit_{}%3A window = {}'.format(predictor_type.upper(), window_length)


if __name__ == '__main__':
    # dataset for 16 stocks by splitting timestamp
    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]

    # 16 stocks are all involved. We choose first 3 years as training data
    num_training_time = 1095
    target_stocks = abbreviation
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))

    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

    nb_classes = len(target_stocks) + 1

    parser = argparse.ArgumentParser(description='Provide arguments for training different imitation learning models')

    parser.add_argument('--predictor_type', '-p', help='cnn or lstm predictor', required=True)
    parser.add_argument('--window_length', '-w', help='observation window length', required=True)

    args = vars(parser.parse_args())

    pprint.pprint(args)

    assert args['predictor_type'] in ['CNN', 'LSTM']

    window_length = int(args['window_length'])

    model_path = 'weights/{}.h5'.format(get_model_name(args['predictor_type'], window_length))

    if args['predictor_type'] == 'CNN':
        model = StockCNN(nb_classes, window_length, model_path)
    else:
        model = StockLSTM(nb_classes, window_length, model_path)

    model.build_model(load_weights=False)
    (X_train, y_train), (X_validation, y_validation) = create_imitation_dataset(target_history, window_length)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_validation = np_utils.to_categorical(y_validation, nb_classes)

    if args['predictor_type'] == 'CNN':
        X_train = np.expand_dims(X_train, axis=-1)
        X_validation = np.expand_dims(X_validation, axis=-1)

    model.train(X_train, Y_train, X_validation, Y_validation)
