try:
    import roboschool
except:
    pass
import sys

import gym
import pybullet_envs.gym_locomotion_envs as e

env_name = sys.argv[1]

env = e.AntBulletEnv(render=True)

for i in range(100):
    done = False
    obs = env.reset()
    r = 0
    while not done:
        env.render()
        action = env.action_space.sample()
        ob, reward, done, _ = env.step(action)
        r += reward
    print('Episode {}, Reward {}'.format(i + 1, r))
