import numpy as np
import matplotlib.pyplot as plt
import sys

import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
from torch.autograd import grad
from rewardfunctions import *

import pdb
from scipy import linalg


#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def count_parameters(model):
	'''Count total number of trainable parameters in argument
	Args:
		model: the function whose parameters to count

	Returns:
		int: number of trainable parameters in model
	'''
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def non_decreasing(L):
	''' Determine whether a list of numbers is in non-decreasing order
	Args:
		L: list of floats 

	Returns:
		bool: if list is in non-decreasing order
	'''
	return all(x>=y for x, y in zip(L, L[1:]))


def discount_rewards(list_of_rewards, discount, center=True, batch_wise=False):
	'''Return 
	'''
	if isinstance(list_of_rewards, list) or isinstance(list_of_rewards, np.ndarray):
		list_of_rewards = np.asarray(list_of_rewards, dtype=np.float32)
		r = np.zeros_like(list_of_rewards)

		for i in range(len(list_of_rewards)):
			r = r + discount**i * np.pad(list_of_rewards,(0,i),'constant')[i:]

		if center:
			return torch.DoubleTensor((r - list_of_rewards.mean())/(list_of_rewards.std()+ 1e-5))
		else:
			return torch.DoubleTensor(r.copy())

	elif isinstance(list_of_rewards, torch.Tensor):
		r = torch.zeros_like(list_of_rewards)
		if batch_wise:
			lim_range = list_of_rewards.shape[1]
		else:
			lim_range = list_of_rewards.shape[0]

		for i in range(lim_range):
			r = r + discount**i * shift(list_of_rewards,i, dir='up')

		if center:
			return (r - torch.mean(list_of_rewards))/(torch.std(list_of_rewards) + 1e-5)
		else:
			return r
			
class StableNoise(object):
	def __init__(self, states_dim, salient_states_dim, param, init=1):
		self.states_dim = states_dim
		self.salient_states_dim = salient_states_dim
		self.extra_dim = self.states_dim - self.salient_states_dim
		self.param = param
		self.random_initial = 2*init*np.random.random(size = (self.extra_dim,)) - init #I THINK THIS NOISE SHOULD BE MORE AGGRESSIVE

	def get_obs(self, obs, t=0):
		if self.extra_dim == 0:
			return obs
		extra_obs = self.random_initial * self.param**t * np.random.random_sample()
		# split_idx = self.salient_states_dim + int(np.floor(self.extra_dim/3))
		new_obs = np.hstack([obs, extra_obs])
		return new_obs

#very stupidly written function, too lazy for now to make better 
def generate_data(env, states_dim, dataset, val_dataset, actor, train_starting_states, val_starting_states, max_actions,noise, epsilon, epsilon_decay, num_action_repeats, temp_data=False, discount=0.995, all_rewards=[], use_pixels=False, reset=True, start_state=None, start_noise=None):
	# dataset = ReplayMemory(1000000)
	noise_decay = 1.0 - epsilon_decay #0.99999
	if env.spec.id != 'lin-dyn-v0':
		salient_states_dim = env.observation_space.shape[0]
	else:
		salient_states_dim = 2

	stablenoise = StableNoise(states_dim, salient_states_dim, 0.992)

	for ep in range(train_starting_states):
		if reset or start_state is None:
			state = env.reset()
		else:
			state = start_state #have to find a way to set env to this state ... otherwise this is doing nothing
		# full_state = env.env.state_vector().copy()

		if env.spec.id != 'lin-dyn-v0' and reset:
			state = stablenoise.get_obs(state, 0)

		if use_pixels:
			obs = env.render(mode='pixels')
			observations = [preprocess(obs)]

		if reset or start_noise is None:
			noise.reset()
		else:
			noise = start_noise

		states = [state]
		actions = []
		rewards = []
		get_new_action = True
		for timestep in range(max_actions):
			if get_new_action:
				with torch.no_grad():
					action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
					# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()#
					# action += noise()*max(0, epsilon) #try without noise
					# action = np.clip(action, -1., 1.)
					action = noise.get_action(action, timestep, multiplier=1.0)
					get_new_action = False
					action_counter = 1

			state_prime, reward, done, _ = env.step(action)
			#UNCOMMENT THESE FOR EXTRA DIMS AND REMOVE THE FULL_STATE NONSENSE!
			if env.spec.id != 'lin-dyn-v0':
				state_prime = stablenoise.get_obs(state_prime, timestep+1)
			if use_pixels:
				obs_prime = env.render(mode='pixels')
				observations.append(preprocess(obs_prime))
				
			if reward is not None: #WHY IS REWARD NONE sometimes?!
				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)

				# dataset.push(full_state, state, state_prime, action, reward)
				if temp_data:
					dataset.temp_push(state, state_prime, action, reward)
				else:
					dataset.push(state, state_prime, action, reward)
			state = state_prime
			# full_state = env.env.state_vector().copy()

			get_new_action = True if action_counter == num_action_repeats else False
			action_counter += 1
		#returns = discount_rewards(rewards, discount, center=True, batch_wise=False)

		# for x, x_next, u, r, ret in zip(states[:-1], states[1:], actions, rewards, returns):
		# 	dataset.push(x, x_next, u, r, ret)
		# all_rewards.append(sum(rewards))
	# print('Average rewards on true dynamics: {:.3f}'.format(sum(all_rewards)/len(all_rewards)))

	# val_dataset = None
	if val_starting_states is not None:
		# val_dataset = ReplayMemory(100000)
		for ep in range(val_starting_states):
			state = env.reset()
			# full_state = env.env.state_vector().copy()
			if env.spec.id != 'lin-dyn-v0':
				state = stablenoise.get_obs(state, 0)
			states = [state]
			actions = []
			rewards = []

			for timestep in range(max_actions):
				with torch.no_grad():
					action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
					# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()
					action = noise.get_action(action, timestep+1, multiplier=1.0)
					
				state_prime, reward, done, _ = env.step(action)
				if env.spec.id != 'lin-dyn-v0':
					state_prime = stablenoise.get_obs(state_prime, timestep+1)
				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)
				# val_dataset.push(full_state, state, state_prime, action, reward)
				val_dataset.push(state, state_prime, action, reward)
				state = state_prime
				# full_state = env.env.state_vector().copy()

	return state, noise, epsilon


def lin_dyn(A, steps, policy, all_rewards, x=None, extra_dim=0, discount=0.9):
	#B = np.eye(2)
	#print linalg.eig(A)[0], np.abs(linalg.eig(A)[0])

	#This should be changed if it's actually used anywhere, to enable use with multiple x dimensions
	# if x is None:        
	# 	x = np.array([1.,0.])
	x = add_irrelevant_features(x, extra_dim=extra_dim, noise_level=0.4)
		
	EYE = np.eye(x.shape[0])

	x_list = [x]
	u_list = []
	r_list = []
	for m in range(steps):
		with torch.no_grad():
			u = policy.sample_action(torch.DoubleTensor(x).to(device)).cpu().numpy()
		
		u_list.append(u)

		r = -(np.dot(x.T, x) + np.dot(u.T,u))
		#r = -(np.dot(x[:-extra_dim].T, x[:-extra_dim]) + np.dot(u.T,u))
		if extra_dim > 0:
			x_next = np.asarray(A.dot(x[:-extra_dim])) #in case only one dimension is relevant
		else:
			x_next = A.dot(x)

		x_next = add_irrelevant_features(x_next, extra_dim=extra_dim, noise_level=0.4)
		x_next = x_next + u

		x_list.append(x_next)
		r_list.append(r)
		x = x_next
	
	x_list = np.array(x_list)
	u_list = np.array(u_list)
	r_list = np.array(r_list)

	x_curr = x_list[:-1,:]
	x_next = x_list[1:,:]
#    r = x_curr[:,0]
	# Quadratic reward

	#change reward:
	#r_list = -np.clip(x_curr[:,0]**2, 0., 1.0)

	all_rewards.append(sum(r_list))

	##WHY DIDN"T WORK WITH DISCOUNT + 0????? center was set to true ... 
	#returns1 = discount_rewards(r_list, discount, center=True)
	#returns1 = torch.from_numpy(r_list)

	#returns2 = discount_rewards(r2, discount, center=True)

#    r = float32(x_curr[:,0] > 0.1)
	
#    return x_list, r
	return x_curr, x_next, u_list, returns1, r_list


def add_irrelevant_features(x, extra_dim, noise_level = 0.4):
#    x_irrel= np.random.random((x.shape[0], extra_dim))
	if isinstance(x, np.ndarray):
		x_irrel= noise_level*np.random.randn(1, extra_dim).reshape(-1,)
	#    x_irrel_next = x_irrel**2 + 1.0
	#    x_irrel_next = x_irrel**2
	#    x_irrel_next = 0.1*np.random.random((x.shape[0], extra_dim))
	#	x_irrel_next = noise_level*np.random.randn(x.shape[0], extra_dim)
	#    x_irrel_next = x_irrel**2 + noise_level*np.random.randn(x.shape[0], extra_dim)    
	#    x_irrel_next = x_irrel**2 + np.random.random((x.shape[0], extra_dim))
		return np.hstack([x, x_irrel])#, np.hstack([x_next, x_irrel_next])

	elif isinstance(x, torch.Tensor):
		x_irrel= noise_level*torch.randn(x.shape[0],x.shape[1],extra_dim).double().to(device)
		#x_irrel_next = noise_level*torch.randn(x.shape[0],x.shape[1],extra_dim).double()
		
		return torch.cat((x, x_irrel),2)#, torch.cat((x_next, x_irrel_next),2)


def convert_one_hot(a, dim):
	if dim == 2: #binary value, no need to do one-hot encoding
		return a

	if a.shape:
		retval = torch.zeros(list(a.shape)+[dim])
		retval[list(np.indices(retval.shape[:-1]))+[a]] = 1
	else: #single value tensor
		retval = torch.zeros(dim)
		retval[int(a)] = 1
	return retval



def roll_1(x, n):  
	return torch.cat((x[:, -n:], x[:, :-n]), dim=1)



def shift(x, step, dir='up'):
	#up works, not tested down
	if step == 0:
		return x

	if len(x.shape) == 3: #batch_wise
		if dir=='down':
			return torch.cat((torch.zeros((x.shape[0], step, x.shape[2])).double().to(device),x),dim=1)[:,:-step]
		elif dir=='up':
			return torch.cat((x,torch.zeros((x.shape[0], step, x.shape[2])).double().to(device)),dim=1)[:,step:]

	elif len(x.shape) == 2: 
		if dir=='down':
				return torch.cat((torch.zeros((step, x.shape[1])).double().to(device),x),dim=0)[:-step]
		elif dir=='up':
			return torch.cat((x,torch.zeros((step, x.shape[1])).double().to(device)),dim=0)[step:]

	else:
		raise NotImplementedError('shape {shape_x} of input not corrent or implemented'.format(shape_x=x.shape))



def roll_left(x, n):  
	#return torch.cat((x[-n:], x[:-n]))
	return torch.cat((x[n:], x[:n]))


def calc_actual_state_values(target_critic, rewards, states, actions, discount):
	R = []
	rewards.reverse()

	# # If we happen to end the set on a terminal state, set next return to zero
	# if dones[-1] == True: next_return = 0
	    
	# If not terminal state, bootstrap v(s) using our critic
	# TODO: don't need to estimate again, just take from last value of v(s) estimates
	# s = torch.from_numpy(states[-1]).double().unsqueeze(0)
	# a = torch.from_numpy(actions[-1]).double().unsqueeze(0)
	s= states
	a = actions
	next_return = target_critic(torch.cat((s,a),dim=1)).data[0][0]

	# Backup from last state to calculate "true" returns for each state in the set
	R.append(next_return)
	# dones.reverse()
	for r in range(1, len(rewards)):
		# if not dones[r]: this_return = rewards[r] + next_return * discount
		this_return = torch.from_numpy(rewards[r]) + next_return * discount
		# else: this_return = 0
		R.append(this_return)
		next_return = this_return

	R.reverse()
	state_values_true = torch.DoubleTensor(R).unsqueeze(1)

	return state_values_true


def compute_returns(self, obs, action, reward, next_obs, done):
	
	with torch.no_grad():
		values, dist = self.ac_net(obs)
		if not done[-1]:
			next_value, _ = self.ac_net(next_obs[-1:])
			values = torch.cat([values, next_value], dim=0)
		else:
			values = torch.cat([values, values.new_tensor(np.zeros((1, 1)))], dim=0)

		returns = reward.new_tensor(np.zeros((len(reward), 1)))
		gae = 0.0
		for step in reversed(range(len(reward))):
			delta = reward[step] + self.gamma * values[step + 1] - values[step]
			gae = delta + self.gamma * self.lmbda * gae
			returns[step] = gae + values[step]

		values = values[:-1]  # remove the added step to compute returns

	return returns, log_probs, values


def get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor):

	action_probs = policy_estimator.get_action_probs(states_tensor)
	if not policy_estimator.continuous:
		c = Categorical(action_probs[0])
		selected_log_probs = c.log_prob(actions_tensor)

	else:
		n = Normal(*action_probs)
		selected_log_probs = n.log_prob(actions_tensor)

	return selected_log_probs

