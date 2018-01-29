"""
PyTorch implementation of deep deterministic policy gradient
"""

import argparse
import pprint as pp

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from gym import wrappers
from torch.nn.modules.loss import MSELoss

from replay_buffer import ReplayBuffer


class ActorModule(nn.Module):
    def __init__(self, state_dim, action_dim, action_bound):
        super(ActorModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        torch.nn.init.uniform(self.fc3.weight.data, -3e-3, 3e-3)
        self.action_bound = action_bound

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        x = x * self.action_bound
        return x


class ActorNetwork(object):
    def __init__(self, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.tau = tau

        self.actor_network = self.create_actor_network()
        self.target_actor_network = self.create_actor_network()

        self.optimizer = optim.Adam(self.actor_network.parameters(), lr=learning_rate)

    def create_actor_network(self):
        return ActorModule(self.s_dim, self.a_dim, self.action_bound)

    def train(self, inputs, a_gradient):
        inputs = Variable(torch.FloatTensor(inputs))
        a_gradient = Variable(torch.FloatTensor(a_gradient))
        self.optimizer.zero_grad()
        actions = self.actor_network(inputs)
        actions.backward(a_gradient)
        self.optimizer.step()

    def predict(self, inputs):
        inputs = Variable(torch.FloatTensor(inputs))
        return self.actor_network(inputs).data.numpy()

    def predict_target(self, inputs):
        inputs = Variable(torch.FloatTensor(inputs))
        return self.target_actor_network(inputs).data.numpy()

    def update_target_network(self):
        source = self.actor_network
        target = self.target_actor_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


class CriticModule(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticModule, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_action = nn.Linear(action_dim, 256)
        self.fc3 = nn.Linear(256, 1)
        torch.nn.init.uniform(self.fc3.weight.data, -3e-3, 3e-3)

    def forward(self, state, action):
        x = self.fc1(state)
        x = F.relu(x)
        x_state = self.fc2(x)
        x_action = self.fc_action(action)
        x = torch.add(x_state, x_action)
        x = F.relu(x)
        x = self.fc3(x)
        return x


class CriticNetwork(object):
    def __init__(self, state_dim, action_dim, learning_rate, tau, gamma):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma

        self.critic_network = self.create_critic_network()
        self.target_critic_network = self.create_critic_network()

        self.optimizer = optim.Adam(self.critic_network.parameters(), lr=learning_rate)
        self.loss = MSELoss()

    def create_critic_network(self):
        return CriticModule(self.state_dim, self.action_dim)

    def train(self, inputs, action, predicted_q_value):
        inputs = Variable(torch.FloatTensor(inputs))
        action = Variable(torch.FloatTensor(action))
        predicted_q_value = Variable(torch.FloatTensor(predicted_q_value))

        self.optimizer.zero_grad()
        q_value = self.critic_network(inputs, action)
        output = self.loss(q_value, predicted_q_value)
        output.backward()
        self.optimizer.step()
        return q_value.data.numpy(), None

    def predict(self, inputs, action):
        inputs = Variable(torch.FloatTensor(inputs))
        action = Variable(torch.FloatTensor(action))
        return self.critic_network(inputs, action).data.numpy()

    def predict_target(self, inputs, action):
        inputs = Variable(torch.FloatTensor(inputs))
        action = Variable(torch.FloatTensor(action))
        return self.target_critic_network(inputs, action).data.numpy()

    def action_gradients(self, inputs, actions):
        inputs = Variable(torch.FloatTensor(inputs))
        actions = Variable(torch.FloatTensor(actions), requires_grad=True)
        q_value = self.critic_network(inputs, actions)
        q_value = torch.sum(q_value)
        return torch.autograd.grad(q_value, actions)[0].data.numpy(), None

    def update_target_network(self):
        source = self.critic_network
        target = self.target_critic_network
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )


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


# ===========================
#   Agent Training
# ===========================

def train(env, args, actor, critic, actor_noise):
    # Initialize target network weights
    actor.update_target_network()
    critic.update_target_network()

    # Initialize replay memory
    replay_buffer = ReplayBuffer(int(args['buffer_size']), int(args['random_seed']))

    for i in range(int(args['max_episodes'])):

        s = env.reset()

        ep_reward = 0
        ep_ave_max_q = 0

        for j in range(int(args['max_episode_len'])):

            if args['render_env']:
                env.render()

            # Added exploration noise
            # a = actor.predict(np.reshape(s, (1, 3))) + (1. / (1. + i))
            a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()

            s2, r, terminal, info = env.step(np.argmax(a[0]))

            replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                              terminal, np.reshape(s2, (actor.s_dim,)))

            # Keep adding experience to the memory until
            # there are at least minibatch size samples
            if replay_buffer.size() > int(args['minibatch_size']):
                s_batch, a_batch, r_batch, t_batch, s2_batch = \
                    replay_buffer.sample_batch(int(args['minibatch_size']))

                # Calculate targets
                target_q = critic.predict_target(
                    s2_batch, actor.predict_target(s2_batch))

                y_i = []
                for k in range(int(args['minibatch_size'])):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + critic.gamma * target_q[k])

                # Update the critic given the targets
                predicted_q_value, _ = critic.train(
                    s_batch, a_batch, np.reshape(y_i, (int(args['minibatch_size']), 1)))

                ep_ave_max_q += np.amax(predicted_q_value)

                # Update the actor policy using the sampled gradient
                a_outs = actor.predict(s_batch)
                grads = critic.action_gradients(s_batch, a_outs)
                actor.train(s_batch, grads[0])

                # Update target networks
                actor.update_target_network()
                critic.update_target_network()

            s = s2
            ep_reward += r

            if terminal:
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward),
                                                                             i, (ep_ave_max_q / float(j))))
                break


def main(args):
    env = gym.make(args['env'])
    np.random.seed(int(args['random_seed']))
    torch.manual_seed(seed=int(args['random_seed']))
    env.seed(int(args['random_seed']))

    state_dim = env.observation_space.shape[0]
    action_dim = 2
    action_bound = 1
    # Ensure action bound is symmetric
    # assert (env.action_space.high == -env.action_space.low)

    actor = ActorNetwork(state_dim, action_dim, action_bound,
                         float(args['actor_lr']), float(args['tau']),
                         int(args['minibatch_size']))

    critic = CriticNetwork(state_dim, action_dim,
                           float(args['critic_lr']), float(args['tau']),
                           float(args['gamma'])
                           )

    actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

    if args['use_gym_monitor']:
        if not args['render_env']:
            env = wrappers.Monitor(
                env, args['monitor_dir'], video_callable=False, force=True)
        else:
            env = wrappers.Monitor(env, args['monitor_dir'], force=True)

    train(env, args, actor, critic, actor_noise)

    if args['use_gym_monitor']:
        env.monitor.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for DDPG agent')

    # agent parameters
    parser.add_argument('--actor-lr', help='actor network learning rate', default=0.0001)
    parser.add_argument('--critic-lr', help='critic network learning rate', default=0.001)
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.001)
    parser.add_argument('--buffer-size', help='max size of the replay buffer', default=1000000)
    parser.add_argument('--minibatch-size', help='size of minibatch for minibatch-SGD', default=64)

    # run parameters
    parser.add_argument('--env', help='choose the gym env- tested on {Pendulum-v0}', default='CartPole-v0')
    parser.add_argument('--random-seed', help='random seed for repeatability', default=1234)
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=50000)
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)
    parser.add_argument('--render-env', help='render the gym env', action='store_true')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_true')
    parser.add_argument('--monitor-dir', help='directory for storing gym results', default='./results/gym_ddpg')
    parser.add_argument('--summary-dir', help='directory for storing tensorboard info', default='./results/tf_ddpg')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=True)

    args = vars(parser.parse_args())

    pp.pprint(args)

    main(args)
