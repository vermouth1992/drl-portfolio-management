"""
Critic Network definition, the input is (o, a_{t-1}, a_t) since (o, a_{t-1}) is the state.
Basically, it evaluates the value of current action given previous action and observation
"""

import tensorflow as tf

class CriticNetwork(object):
    def __init__(self, sess, window_length, num_stocks, feature_size=4, tau=1e-3, learning_rate=1e-4):
        """

        Args:
            sess: a tensorflow session
            window_length: window length
            num_stocks: number of stocks to trade, not include cash
            feature_size: open, high, low, close
            tau: target network update parameter
            learning_rate: learning rate
        """
        pass


