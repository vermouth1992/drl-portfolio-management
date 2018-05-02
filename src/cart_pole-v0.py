"""
Use DDPG to solve CartPole-v0
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

class CartPoleActor(ActorNetwork):
    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None] + self.s_dim, name='input')
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim[0], activation='softmax', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out


class CartPoleCritic(CriticNetwork):
    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None] + self.s_dim)
        action = tflearn.input_data(shape=[None] + self.a_dim)
        net = tflearn.fully_connected(inputs, 256)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 256)
        t2 = tflearn.fully_connected(action, 256)

        # net = tflearn.activation(
        #     tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        net = tf.add(t1, t2)
        net = tflearn.activations.relu(net)

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out


def test_model(env, model, num_test):
    total_reward = 0.0
    for num_episode in range(num_test):
        current_reward = 0
        observation = env.reset()
        for i in range(1000):
            env.render()
            action = model.predict_single(observation)
            observation, reward, done, info = env.step(action)
            current_reward += reward
            if done:
                break
        print("Episode: {}, Reward: {}".format(num_episode, current_reward))
        total_reward += current_reward
    print('Average Reward: {}'.format(total_reward / num_test))


if __name__ == '__main__':
    sess = tf.Session()
    env = gym.make('CartPole-v0')
    action_dim = [2]
    state_dim = [4]
    batch_size = 64
    tau = 1e-3
    actor = CartPoleActor(sess, state_dim, action_dim, 1., 1e-4, tau, batch_size)
    critic = CartPoleCritic(sess=sess, state_dim=state_dim, action_dim=action_dim, tau=1e-3,
                            learning_rate=1e-3, num_actor_vars=actor.get_num_trainable_vars())
    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    ddpg_model = DDPG(env, sess, actor, critic, actor_noise, action_processor=np.argmax,
                          model_save_path='weights/cartpole/checkpoint.ckpt', summary_path='results/cartpole/')
    ddpg_model.initialize(load_weights=False)
    ddpg_model.train()
    test_model(env, ddpg_model, 10)
