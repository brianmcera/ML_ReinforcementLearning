import numpy as np
from cost_functions import trajectory_cost_fn
import time

class Controller():
	def __init__(self):
		pass

	# Get the appropriate action(s) for this state(s)
	def get_action(self, state):
		pass


class RandomController(Controller):
	def __init__(self, env):
		""" YOUR CODE HERE """
                self.env = env
		pass

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Your code should randomly sample an action uniformly from the action space """
                upper_bound = self.env.action_space.high
                lower_bound = self.env.action_space.low
                return np.random.uniform(lower_bound,upper_bound)


class MPCcontroller(Controller):
	""" Controller built using the MPC method outlined in https://arxiv.org/abs/1708.02596 """
	def __init__(self, 
				 env, 
				 dyn_model, 
				 horizon=5, 
				 cost_fn=None, 
				 num_simulated_paths=10,
				 ):
		self.env = env
		self.dyn_model = dyn_model
		self.horizon = horizon
		self.cost_fn = cost_fn
		self.num_simulated_paths = num_simulated_paths

	def get_action(self, state):
		""" YOUR CODE HERE """
		""" Note: be careful to batch your simulations through the model for speed """
                s = np.ndarray(shape=(self.horizon,self.num_simulated_paths,self.env.observation_space.shape[0]))
                s[0,:,:] = state
                a = np.ndarray(shape=(self.horizon-1,self.num_simulated_paths,self.env.action_space.shape[0]))
                upper_bound = self.env.action_space.high
                lower_bound = self.env.action_space.low
                for i in range(self.horizon-1):
                        a[i,:,:] = np.random.uniform(lower_bound,upper_bound,(self.num_simulated_paths,self.env.action_space.shape[0]))
                        s[i+1,:,:] = self.dyn_model.predict(s[i,:,:],a[i,:,:])
                traj_cost = trajectory_cost_fn(self.cost_fn, s[0:-1], a, s[1:])
                best_path = np.argmin(traj_cost)
                return a[0,best_path,:]
        
                        
                
                

                

                        
                

