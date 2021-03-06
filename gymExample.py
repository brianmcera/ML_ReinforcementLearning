import gym
env = gym.make('Ant-v2')

for episode in range(20):
    observation = env.reset()
    for t in range(10000):
        env.render()
        print(observation)
        action = env.action_space.sample();
        observation,reward,done,info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break  
env.close()
