try:
    import roboschool
except:
    pass
import gym
from gym import wrappers
import os

import sys

env_name = sys.argv[1]


env = gym.make(env_name)

for _ in range(100):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
