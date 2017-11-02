"""
Modified from https://github.com/wassname/rl-portfolio-management/blob/master/src/environments/portfolio.py
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pprint import pprint

import gym
import gym.spaces

from utils.data import date_to_index

eps = 1e-7


def random_shift(x, fraction):
    """Apply a random shift to a pandas series."""
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def normalize(x):
    """Normalize to a pandas series."""
    x = (x - x.mean()) / (x.std() + eps)
    return x


def scale_to_start(x):
    """Scale pandas series so that it starts at one."""
    x = (x + eps) / (x[0] + eps)
    return x


def sharpe(returns, freq=30, rfr=0):
    """Given a set of returns, calculates naive (rfr=0) sharpe (eq 28)."""
    return (np.sqrt(freq) * np.mean(returns - rfr)) / np.std(returns - rfr)


def max_drawdown(returns):
    """Max drawdown."""
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / trough


class DataGenerator(object):
    """Acts as data provider for each new episode."""

    def __init__(self, history, abbreviation, steps=730, window_length=50, start_date=None):
        """

        Args:
            history: (N, timestamp, 5) open, high, low, close, volume
            abbreviation: a list of length N with assets name
            steps: the total number of steps to simulate, default is 2 years
            window_length: observation window, must be less than 50
            start_date: the date to start. Default is None and random pick one.
                        It should be a string e.g. '2012-08-13'
        """
        import copy

        self.steps = steps + 1
        self.window_length = window_length
        self.start_date = start_date

        # make immutable class
        self._data = history.copy() # all data
        self.asset_names = copy.copy(abbreviation)

        self.reset()

    def _step(self):
        # get observation matrix from history, exclude volume, maybe volume is useful as it
        # indicates how market total investment changes.
        obs = self.data[:, self.step:self.step + self.window_length, :4].copy()

        self.step += 1
        done = self.step >= self.steps
        return obs, done

    def reset(self):
        self.step = 0

        # get data for this episode, each episode might be different.
        if self.start_date is None:
            self.idx = np.random.randint(
                low=self.window_length, high=self._data.shape[1] - self.steps)
        else:
            # compute index corresponding to start_date for repeatable sequence
            self.idx = date_to_index(self.start_date)
            assert self.idx >= self.window_length and self.idx <= self._data.shape[1] - self.steps, \
                'Invalid start date, must be window_length day after start date and simulation steps day before end date'
        data = self._data[self.idx - self.window_length:self.idx + self.steps + 1]

        # apply augmentation?
        self.data = data


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    Based of [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    def __init__(self, asset_names=list(), steps=730, trading_cost=0.0025, time_cost=0.0):
        self.asset_names = asset_names
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.reset()

    def _step(self, w1, y1):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9,0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        Numbered equations are from https://arxiv.org/abs/1706.10059
        """
        assert w1.shape == y1.shape, 'w1 and y1 must have the same shape'
        assert y1[0] == 1.0, 'y1[0] must be 1'

        w0 = self.w0
        p0 = self.p0

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # (eq7) weights evolve into

        mu1 = self.cost * (np.abs(dw1 - w1)).sum()  # (eq16) cost to change portfolio

        p1 = p0 * (1 - mu1) * np.dot(y1, w0)  # (eq11) final portfolio value

        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding

        p1 = np.clip(p1, 0, np.inf)

        rho1 = p1 / p0 - 1  # rate of returns
        r1 = np.log((p1 + eps) / (p0 + eps))  # log rate of return
        reward = r1 / self.steps  # (22) average logarithmic accumulated return
        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done (losing all the money)
        done = p1 == 0

        info = {
            "reward": reward,
            "log_return": r1,
            "portfolio_value": p1,
            "return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": mu1,
        }
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1.0] + [0.0] * len(self.asset_names))
        self.p0 = 1.0


class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self,
                 history,
                 abbreviation,
                 steps=730,  # 2 years
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 ):
        """
        An environment for financial portfolio management.
        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
            steps - steps in episode
            scale - scale data and each episode (except return)
            augment - fraction to randomly shift data by
            trading_cost - cost of trade as a fraction
            time_cost - cost of holding as a fraction
            window_length - how many past observations to return
        """
        self.src = DataGenerator(history, abbreviation, steps=steps, window_length=window_length)

        self.sim = PortfolioSim(
            asset_names=abbreviation,
            trading_cost=trading_cost,
            time_cost=time_cost,
            steps=steps)

        # openai gym attributes
        # action will be the portfolio weights from 0 to 1 for each asset
        self.action_space = gym.spaces.Box(
            0, 1, shape=len(self.src.asset_names) + 1)  # include cash

        # get the observation space from the data min and max
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(len(abbreviation), window_length,
                                                                      history.shape[-1]))
        self._reset()

    def _step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight from 0 to 1. The first is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(
            action.shape,
            (len(self.sim.asset_names) + 1,)
        )

        # normalise just in case
        action = np.clip(action, 0, 1)

        weights = action  # np.array([cash_bias] + list(action))  # [w0, w1...]
        weights /= (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)  # so if weights are all zeros we normalise to [1,0...]

        assert ((action >= 0) * (action <= 1)).all(), 'all action values should be between 0 and 1. Not %s' % action
        np.testing.assert_almost_equal(
            np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        observation, done1 = self.src._step()

        # relative price vector of last observation day (close/open)
        close_price_vector = observation[:, -1, 3]
        open_price_vector = observation[:, -1, 0]
        y1 = np.insert(close_price_vector / open_price_vector, 0, 1.0)
        reward, info, done2 = self.sim._step(weights, y1)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["return"] for inf in self.infos + [info]])[-1]
        # add dates
        # info['date'] = self.src.data.index[self.src.step].timestamp()
        info['steps'] = self.src.step

        self.infos.append(info)

        return observation, reward, done1 or done2, info

    def _reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
        return observation

    def _render(self, mode='ansi', close=False):
        if close:
            return
        if mode == 'ansi':
            pprint(self.infos[-1])
        # elif mode == 'human':
        #     self.plot()

    # def plot(self):
    #     # show a plot of portfolio vs mean market performance
    #     df_info = pd.DataFrame(self.infos)
    #     df_info.index = pd.to_datetime(df_info["date"], unit='s')
    #     del df_info['date']
    #
    #     mdd = max_drawdown(df_info.rate_of_return + 1)
    #     sharpe_ratio = sharpe(df_info.rate_of_return)
    #     title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
    #
    #     df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf())


if __name__ == '__main__':
    env = PortfolioEnv(np.random.rand(2, 1000, 4), ['a', 'b'])