"""
Critic Network definition, the input is (o, a_{t-1}, a_t) since (o, a_{t-1}) is the state.
Basically, it evaluates the value of (current action, previous action and observation) pair
"""

import tensorflow as tf

from keras.layers import Input, Conv2D, Lambda, Dense
from keras.models import Model
from keras.optimizers import Adam

import keras
import keras.backend as K


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
        self.sess = sess
        self.window_length = window_length
        self.num_stocks = num_stocks
        self.feature_size = feature_size
        self.tau = tau
        self.learning_rate = learning_rate

        K.set_session(sess)

        # create model
        self.model, self.action, self.state = self.create_critic_network()
        self.target_model, self.target_action, self.target_state = self.create_critic_network()
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update

    def create_critic_network(self):
        """
        Topology: Similar to actor network. Instead just taking previous_action_in, it also takes
                  current_action_in.

        Returns:

        """
        observation_in = Input(shape=(self.num_stocks + 1, self.window_length, self.feature_size),
                               dtype='float32', name='observation_input')

        conv1_output = Conv2D(10, (1, 3), strides=(1, 1), padding='valid', data_format='channels_last',
                              activation='relu', kernel_initializer='he_normal')(observation_in)

        conv2_output = Conv2D(20, (1, self.window_length - 2), strides=(1, 1), padding='valid',
                              data_format='channels_last',
                              activation='relu', kernel_initializer='he_normal')(conv1_output)

        previous_action_in = Input(shape=(self.num_stocks + 1,))
        current_action_in = Input(shape=(self.num_stocks + 1,))

        ExpandLayer = Lambda(lambda x: K.expand_dims(x))
        # x is (N, num_stocks, 1, 22)
        expanded_previous_action_in = ExpandLayer(ExpandLayer(previous_action_in))
        expanded_current_action_in = ExpandLayer(ExpandLayer(current_action_in))
        x = keras.layers.concatenate([conv2_output, expanded_previous_action_in, expanded_current_action_in])

        # output is (N, num_stocks, 1, 1)
        output = Conv2D(1, (1, 1), strides=(1, 1), padding='valid', data_format='channels_last',
                        activation='relu', kernel_initializer='he_normal')(x)
        # output is (N, 17)
        SqueezeLayer = Lambda(lambda x: K.squeeze(x, axis=-1))
        output = SqueezeLayer(output)
        output = SqueezeLayer(output)

        # the final fc layer put all stocks feature together
        output = Dense(1, activation='linear')(output)

        model = Model(input=[observation_in, previous_action_in, current_action_in], output=output)

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model, current_action_in, (observation_in, previous_action_in)

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)
