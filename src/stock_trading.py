"""
Use DDPG to train a stock trader based on a window of history price
"""

from __future__ import print_function, division

from model.ddpg.actor import ActorNetwork
from model.ddpg.critic import CriticNetwork
from model.ddpg.ddpg import DDPG
from model.ddpg.ornstein_uhlenbeck import OrnsteinUhlenbeckActionNoise

import numpy as np
import tflearn
import tensorflow as tf
import gym

from environment.portfolio import PortfolioEnv
from utils.data import read_stock_history, normalize


class StockActor(ActorNetwork):
    def create_actor_network(self):
        """
        self.s_dim: a list specifies shape
        """
        nb_classes, window_length = self.s_dim
        assert nb_classes == self.a_dim[0]
        assert window_length > 2, 'This architecture only support window length larger than 2.'
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1], name='input')
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid', activation='relu')
        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid', activation='relu')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 64)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 64)
        # net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out


class StockCritic(CriticNetwork):
    def create_critic_network(self):
        nb_classes, window_length = self.s_dim
        inputs = tflearn.input_data(shape=[None] + self.s_dim + [1])
        action = tflearn.input_data(shape=[None] + self.a_dim)
        net = tflearn.conv_2d(inputs, 32, (1, 3), padding='valid', activation='relu')
        net = tflearn.conv_2d(net, 32, (1, window_length - 2), padding='valid', activation='relu')
        net = tflearn.flatten(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 64)
        t2 = tflearn.fully_connected(action, 64)

        # net = tflearn.activation(
        #     tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        net = tf.add(t1, t2)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out


def obs_normalizer(observation):
    """ Preprocess observation obtained by environment

    Args:
        observation: (nb_classes, window_length, nun_features) or with info

    Returns: normalized

    """
    if isinstance(observation, tuple):
        observation = observation[0]
    # directly use close/open ratio as feature
    observation = observation[:, :, 3:4] / observation[:, :, 0:1]
    observation = normalize(observation)
    return observation


if __name__ == '__main__':
    history, abbreviation = read_stock_history(filepath='utils/datasets/stocks_history_target.h5')
    history = history[:, :, :4]
    target_stocks = ['AAPL', 'COST', 'DISH']
    num_training_time = history.shape[1]
    window_length = 3
    nb_classes = len(target_stocks) + 1

    # get target history
    target_history = np.empty(shape=(len(target_stocks), num_training_time, history.shape[2]))
    for i, stock in enumerate(target_stocks):
        target_history[i] = history[abbreviation.index(stock), :num_training_time, :]

    # setup environment
    env = PortfolioEnv(target_history, target_stocks, steps=1500, window_length=window_length)

    sess = tf.Session()
    action_dim = [nb_classes]
    state_dim = [nb_classes, window_length]
    batch_size = 64
    action_bound = 1.
    tau = 1e-3
    actor = StockActor(sess, state_dim, action_dim, action_bound, 1e-4, tau, batch_size)
    critic = StockCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                         learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars())
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    ddpg_model = DDPG(env, sess, actor, critic, actor_noise, obs_normalizer=obs_normalizer,
                      config_file='config/stock.json', model_save_path='weights/stock/checkpoint.ckpt',
                      summary_path='results/stock/')
    ddpg_model.initialize(load_weights=False)
    ddpg_model.train()
