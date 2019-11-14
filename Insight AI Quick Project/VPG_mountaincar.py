import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import time
import gym
import pdb
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D

class policyMu(Model):
    def __init__(self, ob_dim, ac_dim):
        super(policyMu, self).__init__()
        #self.co1nv1 = Conv2D(10, 10, activation='relu', input_shape=ob_dim)
        #self.flatten = Flatten()
        self.d1 = Dense(3, activation='relu', input_shape = ob_dim)
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(ac_dim) 
        self.sample = sample_gaussian_input(ac_dim)

    def call(self, x):
        #x = self.conv1(x)
        #x = self.flatten(x)
        x = self.d1(x)
        #x = self.d2(x)
        x = self.d3(x)
        return self.sample(x)

    def model(self):
        # this function is used to probe the subclassed-model dimensions
        # Need to hardcode the input shape here (not used for actual computations)
        x = tf.keras.Input(shape=(96,96,3))
        return Model(inputs = x, outputs = self.call(x))

class baselineNN(Model):
    def __init__(self, ob_dim):
        super(baselineNN, self).__init__()
        self.d1 = Dense(10, activation='relu', input_shape = ob_dim)
        self.d2 = Dense(10, activation='relu')
        self.d3 = Dense(1) 

    def call(self, x):
        #x = self.conv1(x)
        #x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return self.d3(x)
    
class sample_gaussian_input(layers.Layer):
    def __init__(self, ac_dim):
        super(sample_gaussian_input, self).__init__()
        self.ac_dim = ac_dim

    def build(self,input_shape):
        self.std = 1e-1*np.identity(self.ac_dim) # identity for now, make trainable later
    
    def call(self,inputs):
        #return inputs + tf.linalg.matvec(self.std,np.random.multi
        #return np.random.multivariate_normal(inputs,self.std)
        #mvn = tfp.distributions.MultivariateNormalDiag(loc=inputs, scale_diag=1e-2*tf.ones(self.ac_dim))
        #output = mvn.sample()
        #del mvn
        #return output
        return tf.random.normal(inputs.shape[1:], mean=inputs, stddev=self.std)

def loss_function(y_true,y_pred):
    return tf.math.reduce_sum(y_true-y_pred,0)

def sample_trajectories(num_traj, num_steps, env, controller, norm_constants, show_visual=False, first_run=False):
    trajectories = {}
    for _ in range(num_traj):
        ob = env.reset()
        obs, next_obs, acs, rewards, returns, reward_to_go = [], [], [], [], [], []
        steps = 0
        ret = 0
        while True:
            if(show_visual):
                env.render()
            obs.append(ob)
            if(first_run):
                ac = env.action_space.sample() 
            else:
                inputs = tf.expand_dims(ob.astype(float),0) 
                if(not first_run): # normalize inputs
                    inputs = (inputs - norm_constants[0])/norm_constants[1]
                ac = controller.predict(inputs)[0]
                #print(ac)
                #if(not first_run):
                #    ac = ac*norm_constants[3] + norm_constants[2]
            if(not isinstance(env.action_space, gym.spaces.Discrete)):
                ac = np.minimum(ac,env.action_space.high)
                ac = np.maximum(ac,env.action_space.low)
            acs.append(ac)
            ob, reward, done, _ = env.step(ac)
            next_obs.append(ob)
            rewards.append(reward)
            ret += reward
            returns.append(ret)
            if(steps%1000==0):
                print(ret)
            steps += 1
            if done or steps>num_steps:
                print("Episode finished after {} timesteps".format(steps))
                break

        # backwards pass to calculate reward-to-go
        reward_to_go = np.zeros(len(rewards))
        discount = 0.9
        reward_to_go[-1] = rewards[-1]
        for i in range(1, reward_to_go.shape[0]):
            reward_to_go[-(i+1)] = rewards[-(i+1)] + discount*reward_to_go[-i] 
        
        traj = {"observations" : np.array(obs),
                "next_observations": np.array(next_obs),
                "rewards" : np.array(rewards),
                "actions" : np.array(acs),
                "returns" : np.array(returns),
                "reward_to_go" : np.array(reward_to_go)}
        trajectories.update(traj)

        del obs, next_obs, acs, rewards, returns, reward_to_go 

    # # normalize advantages    
    # trajectories['reward_to_go'] = trajectories['reward_to_go'] \
    #         - np.mean(trajectories['reward_to_go'])
    # trajectories['reward_to_go'] = trajectories['reward_to_go'] / \
    #          (np.std(trajectories['reward_to_go']))

    return trajectories

def compute_normalized_data(data):
    obs_mean = np.mean(data['observations'],axis=0)
    obs_std = np.std(data['observations'],axis=0) + 1e-6*np.ones(data['observations'].shape[1:])
    acs_mean = np.mean(data['actions'],axis=0)
    acs_std = np.std(data['actions'],axis=0)  + 1e-6*np.ones(data['actions'].shape[1:])

    return (obs_mean, obs_std, acs_mean, acs_std)


def main():
    # initialize environment and deep model
    env = gym.make('MountainCarContinuous-v0') # define chosen environment here
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    loss_object = tf.keras.losses.MeanSquaredError() # loss_function

    train_loss = tf.keras.metrics.Mean(name = 'train_loss')

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
        num_steps = 500
        if(first_run):
            num_traj = 20
        else:
            num_traj = 20
        data = sample_trajectories(num_traj, num_steps, env, controller, norm_constants, show_visual=True, first_run=first_run)
        first_run = False # toggle off after first run

        # output current training rewards
        print('The maximum return is {}'.format(np.amax(data['returns'])))
        print('The average return is {}'.format(np.mean(data['returns'])))
        print('The standard deviation of return is {}'.format(np.std(data['returns'])))

        norm_constants = compute_normalized_data(data)
        obs_data = (data['observations'] - norm_constants[0]) / norm_constants[1]
        #acs_data = (data['actions'] - norm_constants[2]) / norm_constants[3]
        #obs_data = data['observations']
        #acs_data = data['actions']
        #obs_data = tf.keras.utils.normalize(data['observations'])
        acs_data = data['actions']
        reward_to_go = data['reward_to_go']
        rewards = data['rewards']
        num_samples = data['observations'].shape[0]


        # baseline neural network training
        print('Training baseline network...')
        split_size = 8
        baseline_dataset = tf.data.Dataset.from_tensor_slices((obs_data[:-num_samples//split_size],
            reward_to_go[:-num_samples//split_size])).shuffle(10000).batch(32)
        baseline_validation = tf.data.Dataset.from_tensor_slices((obs_data[-num_samples//split_size:],
            reward_to_go[-num_samples//split_size:])).shuffle(10000).batch(32)
        baseline.fit(baseline_dataset, epochs=5, validation_data=baseline_validation)
        print('')

        # control policy training
        print('Training policy network...')
        split_size = 8
        if(1): # baseline network normalization
            reward_to_go -= baseline(obs_data)[:,0]
            reward_to_go /= np.std(reward_to_go)
        else:
            reward_to_go -= np.mean(reward_to_go)
            reward_to_go /= np.std(reward_to_go)
        train_dataset = tf.data.Dataset.from_tensor_slices((obs_data[:-num_samples//split_size],
                acs_data[:-num_samples//split_size],reward_to_go[:-num_samples//split_size])).shuffle(10000).batch(32)
        validation_dataset = tf.data.Dataset.from_tensor_slices((obs_data[-num_samples//split_size:],
                acs_data[-num_samples//split_size:],reward_to_go[-num_samples//split_size:])).shuffle(10000).batch(32)
        history = controller.fit(train_dataset, epochs=5, validation_data=validation_dataset)


        


if __name__ == '__main__':
    main() 
