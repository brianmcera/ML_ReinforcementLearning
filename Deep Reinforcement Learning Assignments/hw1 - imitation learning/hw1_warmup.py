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

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        print('Observations shape is: ' + repr(expert_data['observations'].shape))
        print('Actions shape is: ' + repr(expert_data['actions'].shape))

    hidden_size = 64
    train_behavior_cloning(expert_data,hidden_size)
    run_behavior_cloning(args,hidden_size)

    
def run_behavior_cloning(args,hidden_size):

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
        #tf_util.initialize()

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
                action = sess.run(output,feed_dict = {'x:0':np.reshape(obs,(-1,obs_size))})
                observations.append(obs)
                actions.append(action)
                obs, r, done, _ = env.step(action)
                totalr += r
                steps += 1
                if 1:
                    env.render()
                if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                if steps >= max_steps:
                    break
            returns.append(totalr)

        print('returns', returns)
        print('mean return', np.mean(returns))
        print('std of return', np.std(returns))

        trial_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}
        
    
def weight_variable(shape,namestring):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial,name = namestring)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def train_behavior_cloning(expert_data,hidden_size):
    print((expert_data['observations'].astype(np.float32)).dtype)
    print(expert_data['actions'][:,0,:].shape)

    sess = tf.InteractiveSession()
    
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

    num_iter = 10000;
    mse_results = np.zeros(num_iter)
    for i in range(num_iter):
        rand_idx = np.random.randint(0,expert_data['observations'].shape[0],size=1000)
        sess.run([train_step,mse_loss], feed_dict = {x:expert_data['observations'].astype(np.float)[rand_idx,:], y_:expert_data['actions'].astype(np.float)[rand_idx,0,:]})

        mse_results[i] = sess.run([mse_loss],{x:expert_data['observations'].astype(np.float)[rand_idx,:], y_:expert_data['actions'].astype(np.float)[rand_idx,0,:]})[0]
        if i%100==0:
            print('Current MSE Loss:' + repr(mse_results[i]))

    
    plt.plot(range(num_iter),mse_results)
    plt.ylabel('Mean Squared Error Loss')
    plt.xlabel('Iteration')
    plt.title('MSE Loss vs. Batch Gradient Descent Iteration')
    plt.show()

    tf.add_to_collection('output',output)
    tf.add_to_collection('mse_loss',mse_loss)
    
    allSaver = tf.train.Saver()
    dir = os.path.dirname(os.path.realpath(__file__))
    save_path = dir + '/data-all.cpkt'
    allSaver.save(sess,save_path)
    print('Model saved in file: %s' % save_path)


if __name__ == '__main__':
    main()



    
