import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
import datetime
import time
import gym
import pdb
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization

class policyMu(Model):
    def __init__(self, ob_dim, ac_dim):
        super(policyMu, self).__init__()
        self.conv1 = Conv2D(3, 5, activation='tanh', input_shape=ob_dim, kernel_regularizer=tf.keras.regularizers.l2(0.00))
        self.conv2 = Conv2D(2, 3, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.00))
        self.maxpool1 = MaxPool2D(pool_size=(2,2))
        self.maxpool2 = MaxPool2D(pool_size=(3,3))
        self.conv3 = Conv2D(5, 3, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.00))
        self.flatten = Flatten()
        self.d1 = Dense(10, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.00))
        self.d2 = Dense(5, activation='tanh', kernel_regularizer=tf.keras.regularizers.l2(0.00))
        self.d3 = Dense(ac_dim, kernel_regularizer=tf.keras.regularizers.l2(0.00))
        self.sample = sample_gaussian_input(ac_dim)
        self.dropout = Dropout(rate=0.5)
        self.batchnormalization = BatchNormalization()

    def call(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        #x = self.conv2(x)
        #x = self.maxpool2(x)
        #x = self.conv3(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.d1(x)
        x = self.batchnormalization(x)
        x = self.d2(x)
        x = self.d3(x)
        #x = self.sample(x)
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
        #check = tf.random.normal(inputs.shape[1:], mean=inputs, stddev=[1])#self.std)
        return inputs

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
                ac[1] = 1 # force initial trials to use gas
                if(ac[0]<0): # force full steering for exploration
                    ac[0] = -1
                else:
                    ac[0] = 1
                if(ac[2]>0.5):
                    ac[2] = 1
                else:
                    ac[2] = 0
            else:
                inputs = np.expand_dims(ob.astype(np.float32),0) 
                inputs = inputs + 1e-1*np.random.standard_normal(inputs.shape[1:])
                # normalize inputs
                inputs = np.divide(inputs - norm_constants[0],norm_constants[1])
                ac = controller(inputs)[0]
                # de-normalize outputs
                #ac = np.multiply(ac,norm_constants[3]) + norm_constants[2]
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
            if(ac[0]>0):
                ac[0] = 1
            else:
                ac[0] = -1
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
    obs_mean = tf.math.reduce_mean(tf.cast(data['observations'],tf.float32),axis=0)
    obs_std = tf.math.reduce_std(tf.cast(data['observations'],tf.float32),axis=0) + 1e-6*tf.ones(data['observations'].shape[1:])
    acs_mean = tf.math.reduce_mean(data['actions'],axis=0)
    acs_std = tf.math.reduce_std(data['actions'],axis=0)  + 1e-6*tf.ones(data['actions'].shape[1:])
    return (obs_mean, obs_std, acs_mean, acs_std)
# def compute_normalized_data(data):
#     obs_mean = np.mean(tf.cast(data['observations'],tf.float32),axis=0)
#     obs_std = np.std(tf.cast(data['observations'],tf.float32),axis=0) + 1e-6*tf.ones(data['observations'].shape[1:])
#     acs_mean = np.mean(data['actions'],axis=0)
#     acs_std = np.std(data['actions'],axis=0)  + 1e-6*tf.ones(data['actions'].shape[1:])
#     return (obs_mean, obs_std, acs_mean, acs_std)


def main():
    #enable dynamic GPU memory allocation
    #physical_devices = tf.config.experimental.list_physical_devices('GPU')
    #assert len(physical_devices) > 0
    #tf.config.experimental.set_memory_growth(physical_devices[0], True)
    #tf.keras.backend.set_floatx('float64')
    
    # set up tensorboard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # initialize environment and deep model
    env = gym.make('CarRacing-v0') # define chosen environment here
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.8)
    optimizer2 = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.8)

    loss_object = tf.keras.losses.MeanSquaredError() 
    baseline_loss_object = tf.keras.losses.MeanSquaredError()

    train_MSE_metric = tf.keras.metrics.MeanSquaredError()
    val_MSE_metric = tf.keras.metrics.MeanSquaredError()

    controller = policyMu(ob_dim, ac_dim)
    controller.compile(optimizer = optimizer2,
                    loss = loss_object,
                    metrics = [])

    baseline = baselineNN(ob_dim)
    baseline.compile(optimizer = optimizer1,
                    loss = baseline_loss_object,
                    metrics = [])

    # make graph read-only to prevent accidentally adding nodes per iteration
    graph = tf.compat.v1.get_default_graph()
    graph.finalize()
    
    # training loop
    training_epochs = 20
    first_run = True
    norm_constants = ()
    for _ in range(training_epochs):
        # generate training data 
        num_steps = 1000
        if(first_run):
            num_traj = 5
        else:
            num_traj = 10
        data = sample_trajectories(num_traj, num_steps, env, controller, norm_constants, show_visual=False, first_run=first_run)
        first_run = False # toggle off after first run

        # output current training rewards
        print('The maximum return is {}'.format(tf.math.reduce_max(data['returns'])))
        print('The average return is {}'.format(tf.math.reduce_mean(data['returns'])))
        print('The standard deviation of return is {}'.format(tf.math.reduce_std(data['returns'])))

        norm_constants = compute_normalized_data(data)
        obs_data = tf.math.divide(data['observations']-norm_constants[0], norm_constants[1])
        #acs_data = tf.math.divide(data['actions']-norm_constants[2], norm_constants[3])
        #obs_data = data['observations'].astype(np.float32)
        acs_data = data['actions']
        reward_to_go = data['reward_to_go']
        #rewards = data['rewards']
        num_samples = data['observations'].shape[0]

        # make sure no NaN data errors
        if(np.any(np.isnan(reward_to_go))):
            print('nan error')
            program_pause = raw_input("reward to go NaNs")
        if(np.any(np.isnan(obs_data))):
            print('nan error')
            program_pause = raw_input("observation NaNs")
        if(np.any(np.isnan(acs_data))):
            print('nan error')
            program_pause = raw_input("action NaNs")


        # baseline neural network training
        print('Training baseline network...')
        batch_size = 32
        split_size = 7
        baseline_dataset = tf.data.Dataset.from_tensor_slices((obs_data[:-num_samples//split_size], reward_to_go[:-num_samples//split_size])).shuffle(1024).batch(batch_size)
        baseline_validation = tf.data.Dataset.from_tensor_slices((obs_data[-num_samples//split_size:], reward_to_go[-num_samples//split_size:])).shuffle(1024).batch(batch_size)
        #baseline.fit(baseline_dataset, epochs=5, validation_data=baseline_validation)

        for epoch in range(5):
            print('Start of epoch %d' % (epoch,))

            # iterate over batches of dataset
            for step, (x_batch_train, y_batch_train) in enumerate(baseline_dataset):
                with tf.GradientTape() as tape:
                    model_output = baseline(x_batch_train)
                    loss_value = baseline_loss_object(y_batch_train,model_output)
                grads = tape.gradient(loss_value, baseline.trainable_weights)
                optimizer1.apply_gradients(zip(grads, baseline.trainable_weights))

                # update training metric
                train_MSE_metric(y_batch_train, model_output)

                # log every 20 batches
                if step % 20 == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step+1) * batch_size))

            # display training metrics at the end of each epoch
            train_MSE = train_MSE_metric.result()
            print('Training MSE over epoch: %s' % (float(train_MSE),))
            # reset training metrics at the end of each epoch
            train_MSE_metric.reset_states()

            # run a validation loop at the end of each epoch
            for x_batch_val, y_batch_val in baseline_validation:
                val_output = baseline(x_batch_val)
                #update val metrics
                val_MSE_metric(y_batch_val,val_output)
            val_MSE = val_MSE_metric.result()
            val_MSE_metric.reset_states()
            print('Validation MSE: %s' % (float(val_MSE),))

                    
        print('')
        # control policy training #######################################################
        #################################################################################
        print('Training policy network...')
        split_size = 8
        if(1): # baseline network normalization
            reward_to_go -= baseline(obs_data)[:,0]
            reward_to_go = np.divide(reward_to_go, np.std(reward_to_go)+1e-8)
            reward_to_go -= np.min(reward_to_go)
        elif(0):
            reward_to_go -= np.mean(reward_to_go)
            reward_to_go = np.divide(reward_to_go,np.std(reward_to_go)+1e-8)

        train_dataset = tf.data.Dataset.from_tensor_slices(
                (obs_data[:-num_samples//split_size], 
                    acs_data[:-num_samples//split_size], 
                    reward_to_go[:-num_samples//split_size])
                ).shuffle(1024).batch(batch_size)
        validation_dataset = tf.data.Dataset.from_tensor_slices(
                (obs_data[-num_samples//split_size:], 
                    acs_data[-num_samples//split_size:], 
                    reward_to_go[-num_samples//split_size:])
                ).shuffle(1024).batch(batch_size)

        
        for epoch in range(2):
            print('Start of epoch %d' % (epoch,))

            # iterate over batches of dataset
            for step, (x_batch_train, y_batch_train, reward_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    model_output = controller(x_batch_train)
                    loss_value = loss_object(y_batch_train, model_output, sample_weight=reward_batch_train)
                grads = tape.gradient(loss_value, controller.trainable_weights)
                #if step%20==0:
                #print(grads)
                optimizer2.apply_gradients(zip(grads, controller.trainable_weights))

                # update training metric
                train_MSE_metric(y_batch_train, model_output,
                        sample_weight=reward_batch_train)

                # log every 20 batches
                if step % 20 == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step+1) * batch_size))

            # display training metrics at the end of each epoch
            train_MSE = train_MSE_metric.result()
            print('Training MSE over epoch: %s' % (float(train_MSE),))
            # reset training metrics at the end of each epoch
            train_MSE_metric.reset_states()

            # run a validation loop at the end of each epoch
            for x_batch_val, y_batch_val, reward_batch_val in validation_dataset:
                val_output = baseline(x_batch_val)
                #update val metrics
                val_MSE_metric(y_batch_val, val_output,
                        sample_weight=reward_batch_val)
            val_MSE = val_MSE_metric.result()
            val_MSE_metric.reset_states()
            print('Validation MSE: %s' % (float(val_MSE),))

        del obs_data, acs_data, reward_to_go, data

    controller.save('test_policy.h5')

if __name__ == "__main__":
    main()
        


