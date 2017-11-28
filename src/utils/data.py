"""
Contains a set of utility function to process data
"""

from __future__ import print_function

import csv
import datetime
import numpy as np
import h5py

start_date = '2012-08-13'
end_date = '2017-08-11'
date_format = '%Y-%m-%d'
start_datetime = datetime.datetime.strptime(start_date, date_format)
end_datetime = datetime.datetime.strptime(end_date, date_format)
number_datetime = (end_datetime - start_datetime).days + 1

exclude_set = {'AGN', 'PNR', 'UA', 'AAL', 'EVHC', 'CHTR', 'CCI', 'WBA', 'ETN', 'NLSN', 'ALLE', 'AVGO', 'XL', 'NWS',
               'MNST', 'AON', 'MYL', 'KHC', 'MDT', 'BHGE', 'FTV', 'NAVI', 'PYPL', 'WRK', 'ICE', 'COTY', 'CSRA', 'IRM',
               'FTI', 'JCI', 'HPE', 'SYF', 'INFO', 'EQIX', 'ABBV', 'PRGO', 'CFG', 'HLT', 'BHF', 'ZTS', 'NWSA', 'QRVO',
               'DXC'}

target_list = ['AAPL', 'ATVI', 'CMCSA', 'COST', 'CSX', 'DISH', 'EA', 'EBAY', 'FB', 'GOOGL', 'HAS', 'ILMN', 'INTC',
               'MAR', 'REGN', 'SBUX']

target_list_2 = ['FOX', 'FISV', 'EXPE', 'FAST', 'ESRX', 'DLTR', 'CTSH', 'CSCO', 'QCOM', 'PCLN', 'CELG',
                 'AMGN', 'WFM', 'WDC', 'NVDA', 'STX']


def normalize(x):
    """ Create a universal normalization function across close/open ratio

    Args:
        x: input of any shape

    Returns: normalized data

    """
    return (x - 1) * 100

def create_dataset(filepath):
    """ create the raw dataset from all_stock_5yr.csv. The data is Open,High,Low,Close,Volume

    Args:
        path: path of all_stocks_5yr.csv

    Returns:
        history: numpy array of size (N, number_day, 5),
        abbreviation: a list of company abbreviation where index map to name

    """
    history = np.empty(shape=(460, number_datetime, 5), dtype=np.float)
    abbreviation = []
    with open(filepath, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        first_row = reader.next()
        current_company = None
        current_company_index = -1
        current_date = None
        current_date_index = None
        previous_day_data = None
        for row in reader:
            if row[6] in exclude_set:
                continue
            if row[6] != current_company:
                current_company_index += 1
                # initialize
                if current_date != None and (current_date - end_datetime).days != 1:
                    print(row[6])
                    print(current_date)
                assert current_date is None or (current_date - end_datetime).days == 1, \
                    'Previous end date is not 2017-08-11'
                current_date = start_datetime
                current_date_index = 0
                date = datetime.datetime.strptime(row[0], date_format)
                if (date - start_datetime).days != 0:
                    print(row[6])
                    print(current_date)
                    exclude_set.add(row[6])
                    current_date = end_datetime + datetime.timedelta(days=1)
                    continue
                assert (date - start_datetime).days == 0, 'Start date is not 2012-08-13'
                try:
                    if row[5] == '':
                        row[5] = 0
                    data = np.array(map(float, row[1:6]))
                except:
                    print(row[6])
                    assert False
                history[current_company_index][current_date_index] = data
                previous_day_data = data

                current_company = row[6]
                abbreviation.append(current_company)
            else:
                date = datetime.datetime.strptime(row[0], date_format)
                # missing date, loop to the date difference is 0
                while (date - current_date).days != 0:
                    history[current_company_index][current_date_index] = previous_day_data.copy()
                    current_date += datetime.timedelta(days=1)
                    current_date_index += 1
                # miss data
                try:
                    data = np.array(map(float, row[1:6]))
                except:
                    data = previous_day_data.copy()
                history[current_company_index][current_date_index] = data
                previous_day_data = data

            current_date += datetime.timedelta(days=1)
            current_date_index += 1
    write_to_h5py(history, abbreviation)


def write_to_h5py(history, abbreviation, filepath='datasets/stocks_history_2.h5'):
    """ Write a numpy array history and a list of string to h5py

    Args:
        history: (N, timestamp, 5)
        abbreviation: a list of stock abbreviations

    Returns:

    """
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('history', data=history)
        abbr_array = np.array(abbreviation, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        f.create_dataset("abbreviation", data=abbr_array, dtype=string_dt)


def create_target_dataset(target_list, filepath='datasets/stocks_history_target.h5'):
    """ Create 16 company history datasets

    Args:
        target_list:
        filepath:

    Returns:

    """
    history_all, abbreviation_all = read_stock_history()
    history = None
    for target in target_list:
        data = np.expand_dims(history_all[abbreviation_all.index(target)], axis=0)
        if history is None:
            history = data
        else:
            history = np.concatenate((history, data), axis=0)
    write_to_h5py(history, target_list, filepath=filepath)


def read_stock_history(filepath='datasets/stocks_history.h5'):
    """ Read data from extracted h5

    Args:
        filepath: path of file

    Returns:
        history:
        abbreviation:

    """
    with h5py.File(filepath, 'r') as f:
        history = f['history'][:]
        abbreviation = f['abbreviation'][:].tolist()
        abbreviation = [abbr.decode('utf-8') for abbr in abbreviation]
    return history, abbreviation


def index_to_date(index):
    """

    Args:
        index: the date from start-date (2012-08-13)

    Returns:

    """
    return (start_datetime + datetime.timedelta(index)).strftime(date_format)


def date_to_index(date_string):
    """

    Args:
        date_string: in format of '2012-08-13'

    Returns: the days from start_date: '2012-08-13'

    >>> date_to_index('2012-08-13')
    0
    >>> date_to_index('2012-08-12')
    -1
    >>> date_to_index('2012-08-15')
    2
    """
    return (datetime.datetime.strptime(date_string, date_format) - start_datetime).days


def create_optimal_imitation_dataset(history, training_data_ratio=0.8, is_normalize=True):
    """ Create dataset for imitation optimal action given future observations

    Args:
        history: size of (num_stocks, T, num_features) contains (open, high, low, close)
        training_data_ratio: the ratio of training data

    Returns: un-normalized close/open ratio with size (T, num_stocks), labels: (T,)
             split the data according to training_data_ratio

    """
    num_stocks, T, num_features = history.shape
    cash_history = np.ones((1, T, num_features))
    history = np.concatenate((cash_history, history), axis=0)
    close_open_ratio = np.transpose(history[:, :, 3] / history[:, :, 0])
    if is_normalize:
        close_open_ratio = normalize(close_open_ratio)
    labels = np.argmax(close_open_ratio, axis=1)
    num_training_sample = int(T * training_data_ratio)
    return (close_open_ratio[:num_training_sample], labels[:num_training_sample]), \
           (close_open_ratio[num_training_sample:], labels[num_training_sample:])


def create_imitation_dataset(history, window_length, training_data_ratio=0.8, is_normalize=True):
    """ Create dataset for imitation optimal action given past observations

    Args:
        history: size of (num_stocks, T, num_features) contains (open, high, low, close)
        window_length: length of window as feature
        training_data_ratio: for splitting training data and validation data
        is_normalize: whether to normalize the data

    Returns: close/open ratio of size (num_samples, num_stocks, window_length)

    """
    num_stocks, T, num_features = history.shape
    cash_history = np.ones((1, T, num_features))
    history = np.concatenate((cash_history, history), axis=0)
    close_open_ratio = history[:, :, 3] / history[:, :, 0]
    if is_normalize:
        close_open_ratio = normalize(close_open_ratio)
    Xs = []
    Ys = []
    for i in range(window_length, T):
        obs = close_open_ratio[:, i - window_length:i]
        label = np.argmax(close_open_ratio[:, i:i+1], axis=0)
        Xs.append(obs)
        Ys.append(label)
    Xs = np.stack(Xs)
    Ys = np.concatenate(Ys)
    num_training_sample = int(T * training_data_ratio)
    return (Xs[:num_training_sample], Ys[:num_training_sample]), \
           (Xs[num_training_sample:], Ys[num_training_sample:])
