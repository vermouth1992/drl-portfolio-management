"""
Modified from https://github.com/yanpanlau/DDPG-Keras-Torcs/blob/master/OU.py
"""

import numpy as np


class OrnsteinUhlenbeck(object):
    @classmethod
    def function(self, x, mu, theta, sigma):
        return theta * (mu - x) + sigma * np.random.randn(1)
