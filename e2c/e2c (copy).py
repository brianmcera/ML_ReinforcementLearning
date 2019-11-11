import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import gym
import pdb
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

class policyMu(Model):
    def __init__(self, ob_dim, ac_dim):
        super(policyMu, self).__init__()
        self.conv1 = Conv2D(5, 3, activation='relu', input_shape=ob_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.flatten = Flatten()
        self.d1 = Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.d2 = Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.d3 = Dense(ac_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.sample = sample_gaussian_input(ac_dim)
        #self.dropout = Dropout(rate=0.5)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        #x = self.dropout(x)
        x = self.d2(x)
        x = self.d3(x)
        x = self.sample(x)
        return x

class baselineNN(Model):
    def __init__(self, ob_dim):
        super(baselineNN, self).__init__()
        self.conv1 = Conv2D(5, 3, activation='relu', input_shape=ob_dim, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.flatten = Flatten()
        self.d1 = Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.d2 = Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.d3 = Dense(1) 

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        x = self.d3(x)
        return x
    
class sample_gaussian_input(layers.Layer):
    def __init__(self, ac_dim):
        super(sample_gaussian_input, self).__init__()
        self.ac_dim = ac_dim

    def build(self,input_shape):
        self.std = 1e-1*np.identity(self.ac_dim) # identity for now, make trainable later
    
    def call(self,inputs):
        #return inputs + tf.linalg.matvec(self.std,np.random.multi
        #return np.random.multivariate_normal(inputs,self.std)
        #mvn = tfp.distributions.MultivariateNormalDiag(loc=inputs, scale_diag=1e-1*tf.ones(self.ac_dim))
        #output = mvn.sample()
        #del mvn
        #return output
        check = tf.random.normal(inputs.shape[1:], mean=inputs, stddev=[1])#self.std)
        return check

def sample_trajectories(num_traj, num_steps, env, controller, norm_constants, show_visual=False, first_run=False):
    for traj_num in range(num_traj):
        ob = env.reset()
        obs, next_obs, acs, rewards, returns, reward_to_go = [], [], [], [], [], []
        steps = 0
        ret = 0
        while True:
            if(show_visual or traj_num==0):
                env.render()
            obs.append(ob)
            if(first_run):
                ac = env.action_space.sample() 
                if(ac[1]>0.5):
                    ac[1] = 1 # force initial trials to use gas
                else:
                    ac[1] = 0
                if(ac[0]<0): # force full steering for exploration
                    ac[0] = -1
                else:
                    ac[0] = 1
                if(ac[2]>0.5):
                    ac[2] = 1
                else:
                    ac[2] = 0
            else:
                inputs = tf.expand_dims(ob.astype(float),0) 
                inputs = np.divide(inputs - norm_constants[0],norm_constants[1])
                ac = controller(inputs)[0]
                ac = np.multiply(ac,norm_constants[3]) + norm_constants[2]
            ac = np.array(ac)
            if(np.any(np.isnan(ac))):
                print('nan error')
                pdb.set_trace()
            # discretize gas pedal
            #if(ac[1]>0.5):
            #    ac[1] = 1
            #else:
            #    ac[1] = 0
            # discretize steering
            #if(ac[0]>0):
            #    ac[0] = 1
            #else:
            #    ac[0] = -1
            if(not isinstance(env.action_space, gym.spaces.Discrete)):
                ac = np.minimum(ac,env.action_space.high)
                ac = np.maximum(ac,env.action_space.low)
            acs.append(ac)
            ob, reward, done, info = env.step(ac)
            next_obs.append(ob)
            rewards.append(reward)
            ret += reward
            returns.append(ret)
            if(steps%1000==0):
                print(ret)
            steps += 1
            # check if run is stuck
            lookback = 5
            if(steps>lookback and np.all(ob==obs[-lookback])):
                done = True
            if done or steps>num_steps:
                print("Episode finished after {} timesteps".format(steps))
                break

        # backwards pass to calculate reward-to-go
        reward_to_go = np.full(len(rewards),np.nan)
        discount = 0.9
        reward_to_go[-1] = rewards[-1]
        for i in range(2, reward_to_go.shape[0]+1):
            reward_to_go[-(i)] = rewards[-(i)] + discount*reward_to_go[-(i-1)] 
        
        if(traj_num==0):
            trajectories = {"observations" : np.array(obs),
                "next_observations": np.array(next_obs),
                "rewards" : np.array(rewards),
                "actions" : np.array(acs),
                "returns" : np.array(returns),
                "reward_to_go" : np.array(reward_to_go)}
        else:
            traj = {"observations" : np.array(obs),
                "next_observations": np.array(next_obs),
                "rewards" : np.array(rewards),
                "actions" : np.array(acs),
                "returns" : np.array(returns),
                "reward_to_go" : np.array(reward_to_go)}
            for k in traj:
                trajectories[k] = np.append(trajectories[k],traj[k],axis=0)
    return trajectories

def compute_normalized_data(data):
    obs_mean = np.mean(data['observations'],axis=0)
    obs_std = np.std(data['observations'],axis=0) + 1e-6*np.ones(data['observations'].shape[1:])
    acs_mean = np.mean(data['actions'],axis=0)
    acs_std = np.std(data['actions'],axis=0)  + 1e-6*np.ones(data['actions'].shape[1:])
    return (obs_mean, obs_std, acs_mean, acs_std)


def main():
    #enable dynamic GPU memory allocation
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
    # initialize environment and deep model
    env = gym.make('CarRacing-v0') # define chosen environment here
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    loss_object = tf.keras.losses.MeanSquaredError() 

    controller = policyMu(ob_dim, ac_dim)
    controller.compile(optimizer = tf.keras.optimizers.Adam(),
                    loss = loss_object,
                    metrics = [])

    baseline_loss_object = tf.keras.losses.MeanSquaredError()
    baseline = baselineNN(ob_dim)
    baseline.compile(optimizer = tf.keras.optimizers.Adam(),
                    loss = baseline_loss_object,
                    metrics = [])
    
    # training loop
    training_epochs = 20
    first_run = True
    norm_constants = ()
    for _ in range(training_epochs):
        # generate training data 
        num_steps = 1500
        if(first_run):
            num_traj = 10
        else:
            num_traj = 10
        data = sample_trajectories(num_traj, num_steps, env, controller, norm_constants, show_visual=False, first_run=first_run)
        first_run = False # toggle off after first run

        # output current training rewards
        print('The maximum return is {}'.format(np.amax(data['returns'])))
        print('The average return is {}'.format(np.mean(data['returns'])))
        print('The standard deviation of return is {}'.format(np.std(data['returns'])))

        norm_constants = compute_normalized_data(data)
        obs_data = np.divide(data['observations']-norm_constants[0], norm_constants[1])
        acs_data = np.divide(data['actions']-norm_constants[2], norm_constants[3])
        #obs_data = data['observations'].astype(float)
        #acs_data = data['actions']
        reward_to_go = data['reward_to_go']
        #rewards = data['rewards']
        num_samples = data['observations'].shape[0]


        # baseline neural network training
        print('Training baseline network...')
        split_size = 8
        baseline_dataset = tf.data.Dataset.from_tensor_slices((obs_data[:-num_samples//split_size], reward_to_go[:-num_samples//split_size])).shuffle(1000).skip(5).batch(16)
        baseline_validation = tf.data.Dataset.from_tensor_slices((obs_data[-num_samples//split_size:], reward_to_go[-num_samples//split_size:])).shuffle(1000).skip(5).batch(16)
        baseline.fit(baseline_dataset, epochs=5, validation_data=baseline_validation)
        print('')

        # control policy training
        print('Training policy network...')
        split_size = 8
        if(1): # baseline network normalization
            reward_to_go -= baseline(obs_data)[:,0]
            reward_to_go = np.divide(reward_to_go, np.std(reward_to_go)+1e-8)
            reward_to_go -= np.min(reward_to_go)
        elif(0):
            reward_to_go -= np.min(reward_to_go)
            #reward_to_go /= (np.std(reward_to_go) + 1e-8)
            reward_to_go = np.divide(reward_to_go,np.std(reward_to_go)+1e-8)

        #train_dataset = tf.data.Dataset.from_tensor_slices((obs_data[:-num_samples//split_size], acs_data[:-num_samples//split_size], reward_to_go[:-num_samples//split_size])).shuffle(1000).batch(32)
        #validation_dataset = tf.data.Dataset.from_tensor_slices((obs_data[-num_samples//split_size:], acs_data[-num_samples//split_size:], reward_to_go[-num_samples//split_size:])).shuffle(1000).batch(32)

        if(np.any(np.isnan(reward_to_go))):
            print('nan error')
            program_pause = raw_input("reward to go NaNs")
        if(np.any(np.isnan(obs_data))):
            print('nan error')
            program_pause = raw_input("observation NaNs")
        if(np.any(np.isnan(acs_data))):
            print('nan error')
            program_pause = raw_input("action NaNs")
        
        controller.fit(x=obs_data[:-num_samples//split_size], y=acs_data[:-num_samples//split_size], sample_weight=reward_to_go[:-num_samples//split_size], batch_size=128, epochs=5)
                #validation_data=(obs_data[-num_samples//split_size:],acs_data[-num_samples//split_size:],
                    #reward_to_go[-num_samples//split_size:]))
        #controller.fit(train_dataset, epochs=5, validation_data=validation_dataset)

        del obs_data, acs_data, reward_to_go, data

    controller.save('test_policy.h5')

if __name__ == "__main__":
    main()
        


