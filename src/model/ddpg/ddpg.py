"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""
from __future__ import print_function

import json
import numpy as np
import tensorflow as tf
import sys

from .actor import ActorNetwork
from .critic import CriticNetwork
from .replay_buffer import ReplayBuffer

class DDPG(object):
    def __init__(self, env=None, config_file='config/default.json',
                 actor_weights_path='weights/actor_default.h5', critic_weights_path='weights/critic_default.h5'):
        with open(config_file) as f:
            self.config = json.load(f)
        assert self.config != None, "Can't load config file"
        self.actor_weights_path = actor_weights_path
        self.critic_weights_path = critic_weights_path
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        from keras import backend as K
        K.set_session(self.sess)

        # if env is None, then DDPG just predicts
        self.env = env

    def build_model(self, load_weights=True):
        """ Build model and load from previous trained weights if any

        """
        self.actor = ActorNetwork(self.sess, self.env.window_length, self.env.num_stocks, tau=self.config['tau'],
                                  learning_rate=self.config['actor learning rate'])
        self.critic = CriticNetwork(self.sess, self.env.window_length, self.env.num_stocks, tau=self.config['tau'],
                                    learning_rate=self.config['critic learning rate'])
        if load_weights:
            try:
                self.actor.model.load_weights(self.actor_weights_path)
                self.actor.target_model.load_weights(self.actor_weights_path)
                self.critic.model.load_weights(self.critic_weights_path)
                self.critic.model.load_weights(self.critic_weights_path)
                print('Model load successfully')
            except:
                print('Build model from scratch')
        else:
            print('Build model from scratch')
        self.buffer = ReplayBuffer(self.config['buffer size'])
        self.sess.run(tf.global_variables_initializer())

    def train(self, save_every_episode=5, print_every_step=365, verbose=True, debug=False):
        np.random.seed(self.config['seed'])
        num_episode = self.config['episode']
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        # main training loop
        for i in range(num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation, previous_action = self.env.reset()
            total_reward = 0
            done = False
            # keeps sampling until done
            while not done:
                loss = 0
                action = self.predict(previous_observation, previous_action).squeeze(axis=0)
                # add noise
                sigma = np.std(action, axis=0) * 100
                # noise = OrnsteinUhlenbeck.function(action, 1.0 / self.env.num_stocks, 1.0, 0.1)
                if verbose and self.env.src.step % print_every_step == 0 and debug:
                    print("Episode: {}, Action before: {}".format(i, action))
                noise = np.random.randn(*action.shape) * sigma
                if verbose and self.env.src.step % print_every_step == 0 and debug:
                    print("Episode: {}, Noise: {}".format(i, noise))
                action = action + noise
                action = np.clip(action, 0.0, 1.0)
                # if action is 0, assign one of them to 1.0 in case zero division
                if np.sum(action) == 0.0:
                    idx = np.random.randint(len(action))
                    action[idx] = 1.0
                action /= np.sum(action)
                if verbose and self.env.src.step % print_every_step == 0 and debug:
                    print("Episode: {}, Action after: {}".format(i, action))

                if debug:
                    if sys.version_info.major == 3:
                        input_method = input
                    elif sys.version_info.major == 2:
                        input_method = raw_input
                    else:
                        raise ValueError('Unknown Python Version')
                    input_method('Press any key to continue...')

                # step forward
                observation, reward, done, _ = self.env.step(action)
                # add to buffer
                self.buffer.add((previous_observation, previous_action), action, reward, observation, done)
                # batch update
                batch = self.buffer.getBatch(batch_size)
                old_observations = np.asarray([e[0][0] for e in batch])
                old_actions = np.asarray([e[0][1] for e in batch])
                new_actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_observations = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])
                y_t = np.empty_like(rewards)

                target_q_values = self.evaluate_q(new_observations, new_actions,
                                                  self.predict(new_observations, new_actions, model='target'))

                for k in range(len(batch)):
                    if dones[k]:
                        y_t[k] = rewards[k]
                    else:
                        y_t[k] = rewards[k] + gamma * target_q_values[k]

                loss += self.critic.model.train_on_batch([old_observations, old_actions, new_actions], y_t)
                a_for_grad = self.predict(old_observations, old_actions)
                grads = self.critic.gradients((old_observations, old_actions), a_for_grad)
                self.actor.train((old_observations, old_actions), grads)
                self.actor.target_train()
                self.critic.target_train()

                total_reward += reward
                previous_observation, previous_action = observation, action

                if verbose and self.env.src.step % print_every_step == 0 and debug:
                    print("Episode:", i, "Step:", self.env.src.step, "Reward:", reward, "Loss:", loss)

            # save weights after every # of episodes
            if i % save_every_episode == 0:
                self.actor.model.save_weights(self.actor_weights_path)
                self.critic.model.save_weights(self.critic_weights_path)

            print("Total Reward @ {}-th Episode: {}".format(i, total_reward))

        print('Finish.')

    def predict(self, observation, previous_action, model='actor'):
        """ predict the next action using actor model

        Args:
            observation: (batch_size, num_stocks + 1, window_length, 4)
            previous_action: (batch_size, num_stocks + 1)
            model: actor or target

        Returns: next_action: (batch_size, num_stocks + 1)

        """
        if observation.ndim == 3:
            observation = np.expand_dims(observation, axis=0)
        if previous_action.ndim == 1:
            previous_action = np.expand_dims(previous_action, axis=0)
        # batch_size, num_stocks, window_length, feature_size = observation.shape
        # comment out to accelerate
        # assert batch_size <= self.config['batch size']
        # assert num_stocks == self.env.num_stocks + 1
        # assert window_length == self.env.window_length
        # assert feature_size == 4
        if model == 'actor':
            return self.actor.model.predict([observation, previous_action])
        elif model == 'target':
            return self.actor.target_model.predict([observation, previous_action])
        else:
            raise ValueError('Unknown model')

    def evaluate_q(self, observation, previous_action, next_action):
        """ only call on target critic network

        Args:
            observation: (batch_size, num_stocks + 1, window_length, 4)
            previous_action: (batch_size, num_stocks + 1)
            next_action: (batch_size, num_stocks + 1)

        Returns: a scalar evaluation of the current Q state

        """
        # assert observation.ndim == 4 and previous_action.ndim == 2 and next_action.ndim == 2
        # batch_size, num_stocks, window_length, feature_size = observation.shape
        # assert num_stocks == self.env.num_stocks + 1
        return self.critic.target_model.predict([observation, previous_action, next_action])
