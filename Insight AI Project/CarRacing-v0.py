import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from sklearn import preprocessing, decomposition
import matplotlib.pyplot as plt
import datetime
import time
import gym
import pdb
import random
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPool2D, BatchNormalization

class policyMu(Model):
    def __init__(self, ob_dim, ac_dim):
        super(policyMu, self).__init__()
        self.conv1 = Conv2D(10, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv2 = Conv2D(3, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv3 = Conv2D(3, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.maxpool1 = MaxPool2D(pool_size=(2,2), input_shape=ob_dim)
        self.maxpool2 = MaxPool2D(pool_size=(2,2))
        self.maxpool3 = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.d1 = Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d2 = Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d3 = Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d4 = Dense(ac_dim)
        self.dropout = Dropout(rate=0.5)
        self.batchnormalization1 = BatchNormalization()
        self.batchnormalization2 = BatchNormalization()

    def call(self, x):
        x = self.maxpool1(x)
        x = self.conv1(x)
        x = self.maxpool2(x)
        x = self.conv2(x)
        x = self.maxpool3(x)
        x = self.conv3(x)
        x = self.batchnormalization1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.batchnormalization2(x)
        x = self.d2(x)
        x = self.dropout(x)
        x = self.d3(x)
        return self.d4(x)

class baselineNN(Model):
    def __init__(self, ob_dim):
        super(baselineNN, self).__init__()
        self.conv1 = Conv2D(10, 5, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv2 = Conv2D(3, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.conv3 = Conv2D(3, 3, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.maxpool1 = MaxPool2D(pool_size=(2,2), input_shape=ob_dim)
        self.maxpool2 = MaxPool2D(pool_size=(2,2))
        self.maxpool3 = MaxPool2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.d1 = Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d2 = Dense(20, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d3 = Dense(10, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.d4 = Dense(1)
        self.dropout = Dropout(rate=0.5)
        self.batchnormalization1 = BatchNormalization()
        self.batchnormalization2 = BatchNormalization()

    def call(self, x):
        x = self.maxpool1(x)
        x = self.conv1(x)
        x = self.maxpool2(x)
        x = self.conv2(x)
        x = self.maxpool3(x)
        x = self.conv3(x)
        x = self.batchnormalization1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.batchnormalization2(x)
        x = self.d2(x)
        x = self.dropout(x)
        x = self.d3(x)
        return self.d4(x)
    
def sample_trajectories(num_traj, num_steps, env, controller, norm_constants, show_visual=False, first_run=False, normalize_inputs=True):
    return_vec = np.array([])
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
                    ac[1]= 1 # encourage gas use for initial training
            else:
                inputs = np.expand_dims(ob.astype(np.float32),0) 
                inputs = inputs + 1e-1*np.random.standard_normal(inputs.shape[1:])
                # normalize inputs
                if(normalize_inputs):
                    #inputs = np.divide(inputs-norm_constants[0], norm_constants[1])
                    inputs = tf.image.per_image_standardization(inputs)
                ac = controller(inputs)[0]
            ac = np.array(ac)
            if(ac[1]>0.5):
                ac[1]= 1 # promote gas pedal
            if(np.any(np.isnan(ac))):
                print('nan error')
                pdb.set_trace()
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

            #check to see if car is stuck
            lookback = 5
            if(steps>lookback and np.all(ob==obs[-lookback])):
                rewards[-1] -= 10
                returns[-1] -= 10
                done = True

            if done or steps>num_steps:
                print("Episode {} finished after {} timesteps".format(traj_num, steps))
                break

        # backwards pass to calculate reward-to-go
        reward_to_go = np.full(len(rewards),np.nan)
        discount = 0.90
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
        return_vec = np.append(return_vec, returns[-1])
    return trajectories, return_vec

def normalize_data(data):
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0) + 1e-6
    return data_mean, data_std


def main():
    random.seed(0)
    tf.random.set_seed(0)
    #enable dynamic GPU memory allocation
    # physical_devices = tf.config.experimental.list_physical_devices('GPU')
    # assert len(physical_devices) > 0
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # tf.keras.backend.set_floatx('float64')

    # set up tensorboard logging
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
    test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # initialize environment and deep model
    env = gym.make("procgen:procgen-coinrun-v0") # define chosen environment here
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    ob_dim = env.observation_space.shape
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]

    optimizer1 = tf.keras.optimizers.Adam(learning_rate=1e-3) # baseline optimizer
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=1e-4) # controller optimizer - smaller stepsize
    #optimizer1 = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9)
    #optimizer2 = tf.keras.optimizers.SGD(learning_rate=1e-1, momentum=0.9)

    loss_object = tf.keras.losses.MeanSquaredError() 
    baseline_loss_object = tf.keras.losses.MeanSquaredError()

    train_MSE_metric = tf.keras.metrics.MeanSquaredError()
    train_weightedMSE_metric = tf.keras.metrics.MeanSquaredError()
    val_MSE_metric = tf.keras.metrics.MeanSquaredError()
    val_weightedMSE_metric = tf.keras.metrics.MeanSquaredError()

    controller = policyMu(ob_dim, ac_dim)
    controller.compile(optimizer = optimizer2,
                    loss = loss_object,
                    metrics = [])

    baseline = baselineNN(ob_dim)
    baseline.compile(optimizer = optimizer1,
                    loss = baseline_loss_object,
                    metrics = [])

    #controller.load_weights('./model_checkpoints/controller/20191114-003700')

    # make graph read-only to prevent accidentally adding nodes per iteration
    graph = tf.compat.v1.get_default_graph()
    graph.finalize()
    
    # training loop
    training_epochs = 50
    first_run = True
    norm_constants = ()
    avg_return_vec = np.array([])
    std_return_vec = np.array([])
    max_return_vec = np.array([])
    plt.show()

    normalize_input = True # flag for toggling input normalization/kernel transformation
    for training_epoch in range(training_epochs):
        # generate training data 
        num_steps = 500
        if(first_run):
            num_traj = 1
        else:
            num_traj = 10
        data, return_vec = sample_trajectories(num_traj, num_steps, env, controller, 
                                                norm_constants, show_visual=True, 
                                                first_run=first_run, normalize_inputs=normalize_input)
        # output current training rewards
        print('The maximum return is {}'.format(tf.math.reduce_max(data['returns'])))
        print('The average return is {}'.format(tf.math.reduce_mean(data['returns'])))
        print('The standard deviation of return is {}'.format(tf.math.reduce_std(data['returns'])))
        avg_return_vec = np.append(avg_return_vec, np.mean(return_vec))
        std_return_vec = np.append(std_return_vec, np.std(return_vec))
        max_return_vec = np.append(max_return_vec, np.max(return_vec))
        print(avg_return_vec)
        plt.plot(avg_return_vec)
        plt.plot(avg_return_vec + std_return_vec, ':')
        plt.plot(avg_return_vec - std_return_vec, ':')
        plt.draw()
        plt.pause(1e-3)

        # find normalization statistics
        if(first_run):
            norm_constants = normalize_data(data['observations'])

        # collect data
        obs_data = data['observations'].astype(np.float32)
        if(normalize_input):
            #obs_data = np.divide(obs_data-norm_constants[0], norm_constants[1])
            obs_data = tf.image.per_image_standardization(obs_data)
        acs_data = data['actions']
        reward_to_go = data['reward_to_go']
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

        # baseline neural network training ################################
        ###################################################################
        print('Training baseline network...')
        batch_size = 16
        split_size = 8
        baseline_dataset = tf.data.Dataset.from_tensor_slices((obs_data[:-num_samples//split_size], reward_to_go[:-num_samples//split_size])).shuffle(1024).skip(2).batch(batch_size)
        baseline_validation = tf.data.Dataset.from_tensor_slices((obs_data[-num_samples//split_size:], reward_to_go[-num_samples//split_size:])).shuffle(1024).skip(2).batch(batch_size)

        baseline.fit(baseline_dataset, epochs=10, validation_data=baseline_validation)

        for epoch in []:#range(10):
            print('Start of epoch %d' % (epoch,))

            # iterate over batches of dataset
            for step, (x_batch_train, y_batch_train) in enumerate(baseline_dataset):
                with tf.GradientTape() as tape:
                    model_output = baseline(x_batch_train)
                    loss_value = baseline_loss_object(y_batch_train,model_output, sample_weight=1/y_batch_train.shape[0])
                grads = tape.gradient(loss_value, baseline.trainable_weights)
                optimizer1.apply_gradients(zip(grads, baseline.trainable_weights))

                # update training metric
                train_MSE_metric(y_batch_train, model_output, sample_weight=1/y_batch_train.shape[0])

                # log every 20 batches
                if step % 20 == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step+1) * batch_size))

            # display training metrics at the end of each epoch
            train_MSE = train_MSE_metric.result()
            print('Baseline - Training MSE over epoch: %s' % (float(train_MSE),))
            # reset training metrics at the end of each epoch
            train_MSE_metric.reset_states()

            # run a validation loop at the end of each epoch
            for x_batch_val, y_batch_val in baseline_validation:
                val_output = baseline(x_batch_val)
                #update val metrics
                val_MSE_metric(y_batch_val,val_output, sample_weight=1/y_batch_val.shape[0])
            val_MSE = val_MSE_metric.result()
            val_MSE_metric.reset_states()
            print('Baseline - Validation MSE: %s' % (float(val_MSE),))

                    
        print('~~~~~~~~~~~~~~')
        # control policy training #########################################
        ###################################################################
        print('Training policy network...')
        batch_size = 16
        split_size = 8
        if(1 and not first_run): # baseline network normalization
            reward_to_go -= baseline(obs_data)[:,0]
            reward_to_go = np.divide(reward_to_go, np.std(reward_to_go)+1e-8)
            #reward_to_go -= np.mean(reward_to_go)
        elif(1):
            reward_to_go -= np.mean(reward_to_go)
            reward_to_go = np.divide(reward_to_go,np.std(reward_to_go)+1e-8)

        train_dataset = tf.data.Dataset.from_tensor_slices(
                (obs_data[:-num_samples//split_size], 
                    acs_data[:-num_samples//split_size], 
                    reward_to_go[:-num_samples//split_size])
                ).shuffle(10000).skip(2).batch(batch_size)
        validation_dataset = tf.data.Dataset.from_tensor_slices(
                (obs_data[-num_samples//split_size:], 
                    acs_data[-num_samples//split_size:], 
                    reward_to_go[-num_samples//split_size:])
                ).shuffle(10000).skip(2).batch(batch_size)

        controller.fit(train_dataset, epochs=1, validation_data=validation_dataset)

        for epoch in []:#range(10):
            print('Start of epoch %d' % (epoch,))

            # iterate over batches of dataset
            for step, (x_batch_train, y_batch_train, reward_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    model_output = controller(x_batch_train)
                    loss_value = loss_object(y_batch_train, model_output, sample_weight = reward_batch_train/y_batch_train.shape[0])
                grads = tape.gradient(loss_value, controller.trainable_weights)
                optimizer2.apply_gradients(zip(grads, controller.trainable_weights))

                # update training metrics
                train_MSE_metric(y_batch_train, model_output, sample_weight=1/y_batch_train.shape[0])
                train_weightedMSE_metric(y_batch_train, model_output,
                        sample_weight=reward_batch_train/y_batch_train.shape[0])

                # log every 20 batches
                if step % 20 == 0:
                    print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                    print('Seen so far: %s samples' % ((step+1) * batch_size))

            # display training metrics at the end of each epoch
            train_MSE = train_MSE_metric.result()
            weighted_MSE = train_weightedMSE_metric.result()

            print('Controller - Training MSE over epoch: %s' % (float(train_MSE),))
            print('Controller - Weighted training MSE over epoch: %s' % (float(weighted_MSE),))
            # reset training metrics at the end of each epoch
            train_MSE_metric.reset_states()
            train_weightedMSE_metric.reset_states()

            # run a validation loop at the end of each epoch
            for x_batch_val, y_batch_val, reward_batch_val in validation_dataset:
                val_output = controller(x_batch_val)
                #update val metrics
                val_weightedMSE_metric(y_batch_val, val_output,
                        sample_weight=reward_batch_val/y_batch_val.shape[0])
                val_MSE_metric(y_batch_val, val_output, sample_weight=1/y_batch_val.shape[0])
            val_MSE = val_MSE_metric.result()
            val_weightedMSE = val_weightedMSE_metric.result()
            val_MSE_metric.reset_states()
            val_weightedMSE_metric.reset_states()
            print('Controller - Validation MSE: %s' % (float(val_MSE),))
            print('Controller - Weighted validation MSE: %s' % (float(val_weightedMSE),))

        
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        controller.save_weights('model_checkpoints/controller/' + current_time)
        baseline.save_weights('model_checkpoints/baseline/' + current_time)


        del obs_data, acs_data, reward_to_go, data
        first_run = False # toggle off after first run

    #controller.save('test_policy.h5')

if __name__ == "__main__":
    main()
        


