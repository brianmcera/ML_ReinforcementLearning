#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import matplotlib.pyplot as plt
import os

from tensorflow.examples.tutorials.mnist import input_data



def main():

    #parse input arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

    #run initial behavior cloning (cold-start)
    with tf.Session():
        tf_util.initialize()

        import gym
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(args.num_rollouts):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

    expert_mean = np.mean(returns)
    print('EXPERT')
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions)}

    print('Observations shape is: ' + repr(expert_data['observations'].shape))
    print('Actions shape is: ' + repr(expert_data['actions'].shape))

    hidden_size = 64
    #train initial policy, pi_0 based on BC
    train_policy(expert_data,hidden_size)

    #initialize and store BC results
    num_DAGGER_iter = 3
    DAGGER_means = np.zeros(num_DAGGER_iter+1)
    DAGGER_stds = np.zeros(num_DAGGER_iter+1)
    DAGGER_returns = np.zeros((10,num_DAGGER_iter+1))
    DAGGER_means[0],DAGGER_stds[0],DAGGER_returns[:,0] = run_BC(args,hidden_size)

    
    for i in range(num_DAGGER_iter):
        print(i+1)
        expert_data,DAGGER_means[i+1],DAGGER_stds[i+1],DAGGER_returns[:,i+1] = run_DAGGER(args,hidden_size,expert_data)
        train_policy(expert_data,hidden_size)

    plt.boxplot(DAGGER_returns)
    plt.axhline(y=expert_mean)
    plt.axhline(y=DAGGER_means[0], color = 'r')
    plt.show()
    print(DAGGER_means)
    print(DAGGER_stds)

    
def run_DAGGER(args,hidden_size,expert_data):
    import gym
    env = gym.make(args.envname)
    obs_size = env.reset().shape[0]
    action_size = env.action_space.shape[0]
    
    tf.reset_default_graph()

    dir = os.path.dirname(os.path.realpath(__file__))
    save_path = dir + '/data-all.cpkt'
    saver = tf.train.import_meta_graph(save_path+'.meta')
    graph = tf.get_default_graph()

    output = tf.get_collection('output')[0]

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')

            
    with tf.Session() as sess:
        saver.restore(sess,'data-all.cpkt')
        sample_mean = tf.get_collection('sample_mean')[0].eval()
        sample_std = tf.get_collection('sample_std')[0].eval()

        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        beta = 0.7
        for i in range(5):
            print('DAGGER iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs_normalized = np.divide((obs-sample_mean),sample_std)
                action = sess.run(output,feed_dict = {'x:0':np.reshape(obs_normalized,(-1,obs_size))})
                expert_action = policy_fn(obs[None,:])
                observations.append(obs)
                actions.append(expert_action) #append expert action
                if(np.random.uniform()<beta):
                    obs, r, done, _ = env.step(expert_action)
                else:
                    obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if 0:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('DAGGER')
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        trial_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        expert_data['observations'] = np.concatenate((expert_data['observations'],trial_data['observations']),axis=0)
        expert_data['actions'] = np.concatenate((expert_data['actions'],trial_data['actions']),axis=0)

    return expert_data,np.mean(returns),np.std(returns),np.transpose(returns)
                            
def run_BC(args,hidden_size):
    import gym
    env = gym.make(args.envname)
    obs_size = env.reset().shape[0]
    action_size = env.action_space.shape[0]
    
    tf.reset_default_graph()

    dir = os.path.dirname(os.path.realpath(__file__))
    save_path = dir + '/data-all.cpkt'
    saver = tf.train.import_meta_graph(save_path+'.meta')
    graph = tf.get_default_graph()

    output = tf.get_collection('output')[0]
            
    with tf.Session() as sess:
        saver.restore(sess,'data-all.cpkt')
        sample_mean = tf.get_collection('sample_mean')[0].eval()
        sample_std = tf.get_collection('sample_std')[0].eval()

        max_steps = args.max_timesteps or env.spec.timestep_limit

        returns = []
        observations = []
        actions = []
        for i in range(10):
            print('iter', i)
            obs = env.reset()
            done = False
            totalr = 0.
            steps = 0
            while not done:
                obs_normalized = np.divide((obs-sample_mean),sample_std)
                action = sess.run(output,feed_dict = {'x:0':np.reshape(obs_normalized,(-1,obs_size))})
                #action = sess.run(output,feed_dict = {'x:0':np.reshape(obs,(-1,obs_size))})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if args.render:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, 1000))
                if steps >= 500: #same number of steps so all test trials are the same
                    break
            returns.append(totalr)
        print('BEHAVIOR_CLONING')
        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        trial_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        return np.mean(returns),np.std(returns),np.transpose(returns)
        
    
def weight_variable(shape,namestring):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name = namestring)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def train_policy(expert_data,hidden_size):

    tf.reset_default_graph()
    
    sample_mean = tf.constant([np.mean(expert_data['observations'].astype(np.float),axis=0)][0])
    sample_std = tf.constant(np.maximum(1e-6,[np.std(expert_data['observations'].astype(np.float),axis=0)][0]))
    tf.add_to_collection('sample_mean',sample_mean)
    tf.add_to_collection('sample_std',sample_std)
    
    sess = tf.InteractiveSession()

    expert_data['observations'] = expert_data['observations']-sample_mean.eval()
    expert_data['observations'] = np.divide(expert_data['observations'],sample_std.eval())
    
    x = tf.placeholder(tf.float32, shape=[None,expert_data['observations'].shape[1]],name='x')
    y_ = tf.placeholder(tf.float32, shape=[None, expert_data['actions'].shape[2]],name='y_')

    W_h1 = weight_variable([expert_data['observations'].shape[1],hidden_size],'W_h1')
    b_h1 = bias_variable([hidden_size])
    h1 = tf.nn.relu(tf.matmul(x,W_h1) + b_h1)

    W_o = weight_variable([hidden_size,expert_data['actions'].shape[2]],'W_o')
    b_o = bias_variable([expert_data['actions'].shape[2]])
    output = tf.matmul(h1,W_o) + b_o

    mse_loss = tf.losses.mean_squared_error(y_,output)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mse_loss)

    sess.run(tf.global_variables_initializer())

    num_iter = 2000
    mse_results = np.zeros(num_iter)
    num_samples = expert_data['observations'].shape[0]
    for grad_iter in range(num_iter):
        rand_idx = np.arange(expert_data['observations'].shape[0])
        np.random.shuffle(rand_idx)
        expert_data['observations'] = expert_data['observations'][rand_idx,:]
        expert_data['actions'] = expert_data['actions'][rand_idx,:,:]
        batchsize = 2000#expert_data['observations'].shape[0]
        
        for i in range(np.ceil(float(num_samples)/batchsize).astype(int)):
            sess.run([train_step,mse_loss], feed_dict = {x:expert_data['observations'].astype(np.float)[i*batchsize:(i+1)*np.minimum(batchsize-1,num_samples-1),:], y_:expert_data['actions'].astype(np.float)[i*batchsize:(i+1)*np.minimum(batchsize-1,num_samples-1),0,:]})
            mse_results[grad_iter] = sess.run([mse_loss],{x:expert_data['observations'].astype(np.float)[i*batchsize:(i+1)*np.minimum(batchsize-1,num_samples-1),:], y_:expert_data['actions'].astype(np.float)[i*batchsize:(i+1)*np.minimum(batchsize-1,num_samples-1),0,:]})[0]
            
        if grad_iter%100==0:
            print('Current MSE Loss:' + repr(mse_results[grad_iter]))


    tf.add_to_collection('output',output)
    tf.add_to_collection('mse_loss',mse_loss)
    
    allSaver = tf.train.Saver()
    dir = os.path.dirname(os.path.realpath(__file__))
    save_path = dir + '/data-all.cpkt'
    allSaver.save(sess,save_path)
    print('Model saved in file: %s' % save_path)


if __name__ == '__main__':
    main()



    
