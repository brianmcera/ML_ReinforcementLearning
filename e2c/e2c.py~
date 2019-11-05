import numpy as np
import tensorflow as tf
import time
import gym
import pdb
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class policyMu(Model):
    def __init__(self, num_outputs, ob_dim, ac_dim):
        super(policyGradient, self).__init__()
        self.conv1 = Conv2D(32, 5, activation='relu', input_shape=ob_dim)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(128, activation='relu')
        self.d3 = Dense(num_outputs) 
        self.sample = sample_gaussian_input(ac_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.sample(x)

def sample_gaussian_input(layers.Layer):
    def __init__(self, ac_dim):
        super(sampleMu, self).__init__()
        self.ac_dim = ac_dim

    def build(self,input_shape):
        self.logstd = np.identity(self.ac_dim) # debug as identity for now, make trainable later
    
    def call(self,inputs):
        return inputs + tf.matvec(tf.exp(self.logstd),tf.shape(inputs[0],self.ac_dim))

def loss_function(y_true,y_pred):
    return tf.math.reduce_sum(y_true-y_pred)

def sample_trajectories(num_traj, num_steps, env, controller, show_visual=False):
    trajectories = []
    for _ in range(num_traj):
        ob = env.reset()
        pdb.set_trace()
        obs, next_obs, acs, rewards, returns = [], [], [], [], []
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

        # backwards pass to calculate reward-to-go
        reward_to_go = zeros(data['rewards'].shape)
        discount = 0.95
        for i in range(reward_to_go.shape[0]):
            reward_to_go[-(i+1)] = data['rewards'][-(i+1)] + discount*reward_to_go[-i] 
        # normalize advantages    
        reward_to_go = reward_to_go - np.mean(reward_to_go)
        reward_to_go = reward_to_go/(np.std(reward_to_go)+1e-6)
        
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
    env = gym.make('CarRacing-v0') # define chosen environment here
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    if discrete:
        loss_object = tf.keras.losses.MeanSquaredError()
    else:
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    train_loss = tf.keras.metrics.Mean(name = 'train_loss')
    test_loss = tf.keras.metrics.Mean(name = 'test_loss')

    controller = policyGradient(ac_dim,ob_dim)

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

        obs_mean, obs_std, acs_mean, acs_std = compute_normalized_data(data)
        #obs_data = (data['observations'] - obs_mean) / obs_std
        #acs_data = (data['actions'] - acs_mean) / acs_std
        obs_data = data['observations']
        acs_data = data['actions']
        num_samples = data['observations'].shape[0]
        split_size = 8
        train_dataset = tf.data.Dataset.from_tensor_slices(obs_data[:-num_samples//split_size],
                acs_data[:-num_samples//split_size],reward_to_go[:-num_samples//split_size]).shuffle(10000).batch(32)
        validation_dataset = tf.data.Dataset.from_tensor_slices(obs_data[num_samples//split_size:],
                acs_data[num_samples//split_size:],reward_to_go[num_samples//split_size:]).shuffle(10000).batch(32)

        history = model.fit(train_dataset, epochs=5, validation_data=validation_dataset)


        


if __name__ == '__main__':
    main() 
