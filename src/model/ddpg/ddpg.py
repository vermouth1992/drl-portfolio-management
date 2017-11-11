"""
The deep deterministic policy gradient model. Contains main training loop and deployment
"""
from __future__ import print_function
from past.builtins import raw_input

import json
import numpy as np
import tensorflow as tf
import sys

from .actor import ActorNetwork
from .critic import CriticNetwork
from .replay_buffer import ReplayBuffer

class DDPG(object):
    def __init__(self, env=None, config_file='config/default.json',
                 actor_path='weights/actor_default.h5',
                 critic_path='weights/critic_default.h5',
                 actor_target_path='weights/actor_target_default.h5',
                 critic_target_path='weights/critic_target_default.h5'):
        with open(config_file) as f:
            self.config = json.load(f)
        assert self.config != None, "Can't load config file"
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.actor_target_path = actor_target_path
        self.critic_target_path = critic_target_path
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tf_config)
        from keras import backend as K
        K.set_session(self.sess)

        # if env is None, then DDPG just predicts
        self.env = env

    def build_model(self, load_weights=True):
        """ Load training history from path

        Args:
            load_weights (Bool): True to resume training from file or just deploying.
                                 Otherwise, training from scratch.

        Returns:

        """
        self.actor = ActorNetwork(self.sess, self.env.window_length, self.env.num_stocks, feature_size=1,
                                  tau=self.config['tau'],
                                  learning_rate=self.config['actor learning rate'])
        self.critic = CriticNetwork(self.sess, self.env.window_length, self.env.num_stocks, feature_size=1,
                                    tau=self.config['tau'], learning_rate=self.config['critic learning rate'])
        self.sess.run(tf.global_variables_initializer())
        if load_weights:
            try:
                self.actor.model.load_weights(self.actor_path)
                self.critic.model.load_weights(self.critic_path)
                self.actor.target_model.load_weights(self.actor_path)
                self.critic.target_model.load_weights(self.critic_path)
                print('Model load successfully')
            except:
                print('Build model from scratch')
        else:
            print('Build model from scratch')
        self.buffer = ReplayBuffer(self.config['buffer size'])

    def train(self, save_every_episode=1, print_every_step=365, verbose=True, debug=False):
        self.total_reward_stat = []
        np.random.seed(self.config['seed'])
        num_episode = self.config['episode']
        batch_size = self.config['batch size']
        gamma = self.config['gamma']
        # use fixed exploration rate
        exploration_rate = 0.3
        # main training loop
        for i in range(num_episode):
            if verbose and debug:
                print("Episode: " + str(i) + " Replay Buffer " + str(self.buffer.count()))

            previous_observation, previous_action = self.env.reset()

            # observation is close/open
            previous_observation = previous_observation[:, :, 3] / previous_observation[:, :, 1]

            previous_observation = np.expand_dims(previous_observation, axis=-1)

            total_reward = 0
            done = False
            # keeps sampling until done
            while not done:
                loss = 0
                explore = np.random.random_sample() < exploration_rate
                if explore:
                    action = self.env.action_space.sample()
                    action /= np.sum(action)
                else:
                    action = self.predict(previous_observation).squeeze(axis=0)
                # # add noise
                # sigma = np.std(action, axis=0) * 100
                # # noise = OrnsteinUhlenbeck.function(action, 1.0 / self.env.num_stocks, 1.0, 0.1)
                # if verbose and self.env.src.step % print_every_step == 0 and debug:
                #     print("Episode: {}, Action before: {}".format(i, action))
                # noise = np.random.randn(*action.shape) * sigma
                # if verbose and self.env.src.step % print_every_step == 0 and debug:
                #     print("Episode: {}, Noise: {}".format(i, noise))
                # action = action + noise
                # action = np.clip(action, 0.0, 1.0)
                # # if action is 0, assign one of them to 1.0 in case zero division
                # if np.sum(action) == 0.0:
                #     idx = np.random.randint(len(action))
                #     action[idx] = 1.0
                # action /= np.sum(action)
                if verbose and self.env.src.step % print_every_step == 0 and debug:
                    print("Episode: {}, Action: {}, Explore: {}".format(i, action, explore))

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

                observation = observation[:, :, 3] / observation[:, :, 0]

                observation = np.expand_dims(observation, axis=-1)

                # add to buffer
                self.buffer.add(previous_observation, previous_action, reward, observation, done)
                # batch update
                batch = self.buffer.getBatch(batch_size)
                old_observations = np.asarray([e[0] for e in batch])
                old_actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_observations = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])

                target_q_values = self.evaluate_q(new_observations,
                                                  self.predict(new_observations, model='target'))
                target_q_values = np.squeeze(target_q_values, axis=1)
                y_t = rewards + gamma * target_q_values * dones

                loss += self.critic.model.train_on_batch([old_observations, old_actions], y_t)
                a_for_grad = self.predict(old_observations)
                grads = self.critic.gradients(old_observations, a_for_grad)
                self.actor.train(old_observations, grads)
                self.actor.target_train()
                self.critic.target_train()

                total_reward += reward
                previous_observation, previous_action = observation, action

                if verbose and self.env.src.step % print_every_step == 0 and debug:
                    print("Episode:", i, "Step:", self.env.src.step, "Reward:", reward, "Loss:", loss)

            # save weights after every # of episodes
            if i % save_every_episode == 0:
                self.save_model()

            print("Total Reward @ {}-th Episode: {}".format(i, total_reward))
            self.total_reward_stat.append(total_reward)

        self.save_model()
        print('Finish.')

    def predict(self, observation, model='actor'):
        """ predict the next action using actor model

        Args:
            observation: (batch_size, num_stocks + 1, window_length, 4)
            previous_action: (batch_size, num_stocks + 1)
            model: actor or target

        Returns: next_action: (batch_size, num_stocks + 1)

        """
        if observation.ndim == 3:
            observation = np.expand_dims(observation, axis=0)
        if model == 'actor':
            return self.actor.model.predict(observation)
        elif model == 'target':
            return self.actor.target_model.predict(observation)
        else:
            raise ValueError('Unknown model')

    def evaluate_q(self, observation, next_action):
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
        return self.critic.target_model.predict([observation, next_action])

    def save_model(self):
        self.actor.model.save_weights(self.actor_path)
        self.critic.model.save_weights(self.critic_path)
        self.actor.target_model.save_weights(self.actor_target_path)
        self.critic.target_model.save_weights(self.critic_target_path)
