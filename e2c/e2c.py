import numpy as np
import tensorflow as tf
import time
import gym
import pdb
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class policyGradient(Model):
    def __init__(self,num_outputs):
        super(myModel, self).__init__()
        self.conv1 = Conv2D(batch_size, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.d3 = Dense(num_outputs) 

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)

def sample_trajectories(num_traj, num_steps, env, controller, show_visual=False)
    trajectories = []
    for _ in range(num_traj)
        ob = env.reset()
        obs, next_obs, acs, rewards, returns = [], [], [], []
        steps = 0
        ret = 0
        while True:
            if(show_visual):
                env.render()
                time.sleep(0.02)
            obs.append(ob)
            ac = controller.predict(ob) 
            acs.append(ac)
            ob, reward, done, _ = env.step(action)
            next_obs.append(ob)
            rewards.append(reward)
            ret += reward
            returns.append(ret)
            if(steps%1000==0):
                print(return_val)
            steps += 1
            if done or steps>num_steps:
                print("Episode finished after {} timesteps".format(steps))
                break
        returns = np.array(rewards)
        traj = {'observations' : np.array(obs),
                'next_observations': np.array(next_obs),
                'rewards' : np.array(rewards),
                'actions' : np.array(acs),
                'returns' : np.array(returns)}
        trajectories.append(traj)
    return trajectories

def compute_normalized_data(data):
    obs_mean = np.mean(data['observations'],axis=0)
    obs_std = np.std(data['observations'],axis=0)
    acs_mean = np.mean(data['actions'],axis=0)
    acs_std = np.std(data['actions'],axis=0)
    return obs_mean, obs_std, acs_mean, acs_std

def main():
    
    # initialize environment and deep model
    env = gym.make('CarRacing-v0')
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    if discrete:
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossEntropy()


    train_loss = tf.keras.metrics.Mean(name = 'train_loss')
    test_loss = tf.keras.metrics.Mean(name = 'test_loss')

    controller = policyGradient(ac_dim)

    controller.compile(optimizer = tf.keras.optimizers.Adam(),
                    loss = loss_object,
                    metrics = [train_loss])

    # training loop
    training_epochs = 20
    for _ in range(training_epochs):
        # generate training data 
        num_traj = 20
        num_steps = 5000
        data = sample_trajectories(num_traj, num_steps, env, controller, show_visual=True)

        # output current training rewards
        print('The maximum return is {}'.format(np.amax(data['returns'])))
        print('The average reward is {}'.format(np.mean(data['rewards'])))

        # backwards pass to calculate reward-to-go
        reward_to_go = zeros(data['rewards'].shape)
        discount = 0.95
        for i in range(reward_to_go.shape[0]):
            reward_to_go[-(i+1)] = data['rewards'][-(i+1)] + discount*reward_to_go[-i] 
        # normalize advantages    
        reward_to_go = reward_to_go - np.mean(reward_to_go)
        reward_to_go = reward_to_go/(np.std(reward_to_go)+1e-6)
        
        obs_mean, obs_std, acs_mean, acs_std = compute_normalized_data(data)
        obs_data = (data['observations'] - obs_mean) / obs_std
        train_dataset = tf.data.Dataset.from_tensor_slices(
        



if __name__ == '__main__':
    main() 
