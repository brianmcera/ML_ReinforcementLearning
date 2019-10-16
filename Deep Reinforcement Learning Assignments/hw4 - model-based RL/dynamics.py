import tensorflow as tf
import numpy as np

# Predefined function to build a feedforward neural network
def build_mlp(input_placeholder, 
              output_size,
              scope, 
              n_layers=2, 
              size=500, 
              activation=tf.tanh,
              output_activation=None
              ):
    out = input_placeholder
    with tf.variable_scope(scope):
        for _ in range(n_layers):
            out = tf.layers.dense(out, size, activation=activation)
        out = tf.layers.dense(out, output_size, activation=output_activation)
    return out

class NNDynamicsModel():
    def __init__(self, 
                 env, 
                 n_layers,
                 size, 
                 activation, 
                 output_activation, 
                 normalization,
                 batch_size,
                 iterations,
                 learning_rate,
                 sess
                 ):
        """ YOUR CODE HERE """
        """ Note: Be careful about normalization """
        self.normalization = normalization
        self.batch_size = batch_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.sess = sess

        self.input_normalized = tf.placeholder(tf.float64,shape=(None,env.observation_space.shape[0]+env.action_space.shape[0]))
        self.model = build_mlp(self.input_normalized,env.observation_space.shape[0],'DynModel',n_layers,size,activation,output_activation)
        
        self.deltas = tf.placeholder(tf.float64,shape=(None,env.observation_space.shape[0]))
        self.loss = tf.losses.mean_squared_error(self.deltas,self.normalization[2]+tf.multiply(self.normalization[3],self.model))

        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

    def fit(self, data):
        """
        Write a function to take in a dataset of (unnormalized)states, (unnormalized)actions, (unnormalized)next_states and fit the dynamics model going from normalized states, normalized actions to normalized state differences (s_t+1 - s_t)
        """

        """YOUR CODE HERE """
        #shuffle data
        num_samples = data['observations'].shape[0]
        rand_idx = np.random.permutation(num_samples)
        observations = data['observations'][rand_idx]
        actions = data['actions'][rand_idx]
        deltas = data['next_observations'][rand_idx] - data['observations'][rand_idx]
        
        for _ in range(self.iterations):
            batch_num = 0
            while((batch_num+1)*self.batch_size<num_samples):
                #split into batches
                observations_batch = observations[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
                actions_batch = actions[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
                deltas_batch = deltas[batch_num*self.batch_size:(batch_num+1)*self.batch_size]
                #train
                self.sess.run(self.train_op,feed_dict={self.input_normalized : np.hstack(((observations_batch-self.normalization[0])/(self.normalization[1]+1e-6),(actions_batch-self.normalization[4])/(self.normalization[5]+1e-6))), self.deltas : deltas_batch})
                batch_num = batch_num + 1

    def predict(self, states, actions):
        """ Write a function to take in a batch of (unnormalized) states and (unnormalized) actions and return the (unnormalized) next states as predicted by using the model """
        """ YOUR CODE HERE """
        pred = self.sess.run(self.model,feed_dict={self.input_normalized : np.hstack(((states-self.normalization[0])/(self.normalization[1]+1e-6),(actions-self.normalization[4])/(self.normalization[5]+1e-6)))})
        pred_states_plus1 = states + (self.normalization[2]+np.multiply(self.normalization[3],pred))
        return pred_states_plus1
