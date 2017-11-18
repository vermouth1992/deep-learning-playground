import gym

env = gym.make('Pendulum-v0')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print('Obs: {}'.format(observation))
        action = env.action_space.sample()
        print('Action: {}'.format(action))
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break
