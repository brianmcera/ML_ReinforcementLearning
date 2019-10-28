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

def sampleTrajectories(num_traj, num_steps, env, controller, show_visual=False)
    trajectories = []
    for _ in range(num_traj)
        ob = env.reset()
        obs, next_obs, acs, rewards, returns = [], [], [], []
        steps = 0
        ret = 0
        while True:
            if(show_visual):
                env.render()
                time.sleep(0.05)
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

    optimizer = tf.keras.optimizers.Adam()

    train_loss = tf.keras.metrics.Mean(name = 'train_loss')
    test_loss = tf.keras.metrics.Mean(name = 'test_loss')

    
    @tf.function
    def train_step(input, output, advantage):
        with tf.GradientTape() as tape:
            predictions = model(input)
            loss = loss_object(input, output, sample_weight=advantage)
        train_loss(loss)
    
    @tf.function
    def test_step():
        pass


if __name__ == '__main__':
    main()
