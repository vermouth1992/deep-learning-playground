"""
Train a policy network on CartPole-v0 using deep deterministic policy gradient (DDPG)
"""

from __future__ import division, print_function

from collections import deque
from keras.models import Model
from keras.layers import Dense, Input, Add, BatchNormalization, Activation, Lambda
from keras.optimizers import Adam
from keras.initializers import RandomUniform, TruncatedNormal, Constant
from keras.regularizers import l2

import keras.backend as K
import tensorflow as tf
import random
import numpy as np
import gym


class ActorNetwork():
    def __init__(self, sess, feature_size, action_size, tau=1e-3, learning_rate=1e-4):
        self.feature_size = feature_size
        self.action_size = action_size
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate

        K.set_session(sess)

        # Now create the model
        self.model, self.weights, self.state = self.create_actor_network()
        self.target_model, self.target_weights, self.target_state = self.create_actor_network()
        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size])
        self.unnormalized_actor_gradients = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, 64), self.unnormalized_actor_gradients))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
            zip(self.actor_gradients, self.weights))
        self.sess.run(tf.global_variables_initializer())

    def create_actor_network(self):
        observation_in = Input(shape=(self.feature_size,))
        output = Dense(400, kernel_initializer=TruncatedNormal(stddev=0.01), kernel_regularizer=l2(0.001))(observation_in)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dense(300, kernel_initializer=TruncatedNormal(stddev=0.01), kernel_regularizer=l2(0.001))(output)
        output = BatchNormalization()(output)
        output = Activation('relu')(output)
        output = Dense(self.action_size, activation='tanh', kernel_initializer=RandomUniform(-0.003, 0.003), kernel_regularizer=l2(0.001))(output)
        output = Lambda(lambda x: x * 2.)(output)
        model = Model(inputs=observation_in, outputs=output)
        return model, model.trainable_weights, observation_in

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
            actor_target_weights[i] = self.tau * actor_weights[i] + (1. - self.tau) * actor_target_weights[i]
        self.target_model.set_weights(actor_target_weights)


class CriticNetwork():
    def __init__(self, sess, feature_size, action_size, tau=1e-3, learning_rate=1e-4):
        self.feature_size = feature_size
        self.action_size = action_size
        self.sess = sess
        self.tau = tau
        self.learning_rate = learning_rate

        K.set_session(sess)

        # create model
        self.model, self.action, self.state = self.create_critic_network()
        self.target_model, self.target_action, self.target_state = self.create_critic_network()
        self.action_grads = tf.gradients(self.model.output, self.action)  # GRADIENTS for policy update
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])
        # Define loss and optimization Op
        self.loss = tf.reduce_mean(tf.square(self.predicted_q_value - self.model.output))
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)
        self.sess.run(tf.global_variables_initializer())

    def create_critic_network(self):
        observation_in = Input(shape=(self.feature_size,))
        observation_output = Dense(256, kernel_initializer=TruncatedNormal(stddev=0.02), kernel_regularizer=l2(0.001))(observation_in)
        observation_output = BatchNormalization()(observation_output)
        observation_output = Activation('relu')(observation_output)
        observation_output = Dense(256, activation='linear', kernel_initializer=TruncatedNormal(stddev=0.02), kernel_regularizer=l2(0.001))(observation_output)

        action_in = Input(shape=(self.action_size,))
        action_output = Dense(256, activation='linear', kernel_initializer=TruncatedNormal(stddev=0.02), kernel_regularizer=l2(0.001))(action_in)

        output = Add()([observation_output, action_output])
        # output = Activation('relu')(output)
        output = Dense(1, activation='linear', kernel_initializer=RandomUniform(-0.003, 0.003))(output)
        model = Model(inputs=[observation_in, action_in], outputs=output)
        model.compile(loss='mean_squared_error', optimizer=Adam(self.learning_rate))
        return model, action_in, observation_in

    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action: actions
        })[0]

    def train(self, state, action, predicted_q_value):
        return self.sess.run([self.model.output, self.optimize], feed_dict={
            self.state: state,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + (1. - self.tau) * critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)


class ReplayBuffer(object):
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.num_experiences = 0
        self.buffer = deque()

    def getBatch(self, batch_size):
        # Randomly sample batch_size examples
        if self.num_experiences < batch_size:
            return random.sample(self.buffer, self.num_experiences)
        else:
            return random.sample(self.buffer, batch_size)

    def size(self):
        return self.buffer_size

    def add(self, state, action, reward, new_state, done):
        experience = (state, action, reward, new_state, done)
        if self.num_experiences < self.buffer_size:
            self.buffer.append(experience)
            self.num_experiences += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def count(self):
        # if buffer is full, return buffer size
        # otherwise, return experience counter
        return self.num_experiences

    def erase(self):
        self.buffer = deque()
        self.num_experiences = 0


class RandomProcess(object):
    def reset_states(self):
        pass


class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def test_model(env):
    # deploy
    observation = env.reset()
    total_reward = 0
    for i in range(1000):
        action = actor.model.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)
        print(observation, action)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print("Total Reward: {}".format(total_reward))


if __name__ == '__main__':
    # hyperparameters here
    gamma = 0.99
    batch_size = 64
    theta = 0.15
    sigma = 0.3
    num_episode = 200
    num_steps = 1000

    sess = tf.Session()
    K.set_session(sess)
    K.set_learning_phase(1)
    actor = ActorNetwork(sess, feature_size=3, action_size=1, learning_rate=1e-4)
    critic = CriticNetwork(sess, feature_size=3, action_size=1, learning_rate=1e-3)
    # set weights
    actor.target_model.set_weights(actor.model.get_weights())
    critic.target_model.set_weights(critic.model.get_weights())
    total_reward_stat = []

    buffer = ReplayBuffer(100000)
    env = gym.make('Pendulum-v0')
    OU = OrnsteinUhlenbeckActionNoise(mu=np.zeros(shape=(1,)))

    verbose = False

    for i in range(num_episode):
        observation = env.reset()
        total_reward = 0
        ep_ave_max_q = 0
        running_steps = num_steps
        for j in range(num_steps):
            action = actor.model.predict(np.expand_dims(observation, axis=0)).squeeze(axis=0)

            if verbose:
                print('Action before: {}'.format(action))
            noise = OU()
            if verbose:
                print('Noise: {}'.format(noise))
            # noise = 0
            action += noise
            if verbose:
                print('Action after: {}'.format(action))

            if verbose:
                input("Press...\n")

            next_observation, reward, done, info = env.step(action)
            buffer.add(observation, action, reward, next_observation, done)

            if buffer.count() >= batch_size:
                # batch update
                batch = buffer.getBatch(batch_size)
                old_observations = np.asarray([e[0] for e in batch])
                old_actions = np.asarray([e[1] for e in batch])
                rewards = np.asarray([e[2] for e in batch])
                new_observations = np.asarray([e[3] for e in batch])
                dones = np.asarray([e[4] for e in batch])

                target_q_values = critic.target_model.predict([new_observations,
                                                               actor.target_model.predict(new_observations)])
                # print(old_observations.shape, old_actions.shape, rewards.shape, new_observations.shape, dones.shape, target_q_values.shape)
                target_q_values = np.squeeze(target_q_values, axis=1)

                y_t = []
                for k in range(64):
                    if dones[k]:
                        y_t.append(rewards[k])
                    else:
                        y_t.append(rewards[k] + gamma * target_q_values[k])
                y_t = np.array(y_t)
                #y_t = rewards + gamma * target_q_values * dones

                q_values, _ = critic.train(old_observations, old_actions, np.expand_dims(y_t, axis=1))
                # critic.model.train_on_batch([old_observations, old_actions], y_t)
                # record Q value
                # q_values = critic.model.predict([old_observations, old_actions])
                ep_ave_max_q += np.amax(q_values)

                a_for_grad = actor.model.predict(old_observations)
                grads = critic.gradients(old_observations, a_for_grad)
                actor.train(old_observations, grads)
                actor.target_train()
                critic.target_train()

            total_reward += reward
            observation = next_observation
            if done:
                running_steps = j + 1
                break

        print("Episode: {}, Reward: {}, Qmax: {}".format(i, total_reward, ep_ave_max_q / float(running_steps)))
        total_reward_stat.append(total_reward)
