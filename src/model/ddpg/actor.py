"""
Actor Network definition, The CNN architecture follows the one in this paper
https://arxiv.org/abs/1706.10059
"""

import tensorflow as tf

from keras.layers import Input, Conv2D, Activation, Lambda
from keras.models import Model

import keras
import keras.backend as K


class ActorNetwork(object):
    def __init__(self, sess, window_length, num_stocks, feature_size=4, tau=1e-3, learning_rate=1e-4):
        """

        Args:
            sess: tensorflow session
            window_length: length of observation window
            num_stocks: number of stocks, not include cash
            feature_size: (open, high, low, close)
            tau: target network update parameter
            learning_rate: learning rate
        """
        self.sess = sess
        self.window_length = window_length
        self.num_stocks = num_stocks
        self.feature_size = feature_size
        self.tau = tau
        self.learning_rate = learning_rate

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network()
        self.target_model, self.target_weights, self.target_state = self.create_actor_network()
        self.action_gradient = tf.placeholder(tf.float32, [None, self.num_stocks + 1])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(learning_rate).apply_gradients(grads)

    def create_actor_network(self):
        """ Create actor network.
        Topology: (N, 17, 50, 4) -> (N, 17, 48, 10) -> (N, 17, 1, 20) -> (N, 17, 1, 21) -> (N, 17) -> softmax

        Returns: actor network

        """
        observation_in = Input(shape=(self.num_stocks + 1, self.window_length, self.feature_size),
                               dtype='float32', name='observation_input')

        conv1_output = Conv2D(10, (1, 3), strides=(1, 1), padding='valid', data_format='channels_last',
                              activation='relu', kernel_initializer='he_normal')(observation_in)

        conv2_output = Conv2D(20, (1, self.window_length - 2), strides=(1, 1), padding='valid',
                              data_format='channels_last',
                              activation='relu', kernel_initializer='he_normal')(conv1_output)

        previous_action_in = Input(shape=(self.num_stocks + 1,))

        ExpandLayer = Lambda(lambda x: K.expand_dims(x))
        # x is (N, 17, 1, 21)
        x = keras.layers.concatenate([conv2_output, ExpandLayer(ExpandLayer(previous_action_in))])

        # output is (N, 17, 1, 1)
        output = Conv2D(1, (1, 1), strides=(1, 1), padding='valid', data_format='channels_last',
                        activation=None, kernel_initializer='he_normal')(x)
        # output is (N, 17)
        SqueezeLayer = Lambda(lambda x: K.squeeze(x, axis=-1))
        output = SqueezeLayer(output)
        output = SqueezeLayer(output)

        output = Activation('softmax')(output)

        model = Model(inputs=[observation_in, previous_action_in], outputs=output)

        return model, model.trainable_weights, (observation_in, previous_action_in)

    def train(self, states, action_grads):
        """

        Args:
            states: (observation, previous_action),
                    observation is (N, num_stocks, window_length, 4), previous_action is (N, 17)
            action_grads: (N, 17)
        """
        self.sess.run(self.optimize, feed_dict={
            self.state: states,
            self.action_gradient: action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)


if __name__ == '__main__':
    actor = ActorNetwork(tf.Session(), 50, 16)
