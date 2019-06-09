import numpy as np
import matplotlib.pyplot as plt
# import gym
import sys

import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import pdb

from utils import *
from collections import namedtuple
import random


device='cpu'

def init_weights(m):
	if isinstance(m, nn.Linear):
		nn.init.normal_(m.weight, mean=0., std=0.1)
		nn.init.constant_(m.bias, 0.1)
		

Transition = namedtuple('Transition',('state', 'next_state', 'action', 'reward'))

# Using PyTorch's Tutorial
class ReplayMemory(object):

	def __init__(self, size):
		self.size = size
		self.memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		
		if len(self.memory) < self.size:
			self.memory.append(None)

		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.size

	def clear(self):
		self.memory = []
		self.position = 0

	def sample(self, batch_size, structured=False, max_actions=10, num_episodes_per_start=10, num_starting_states=5, start_at=None):
		if structured:
			#fix this : batch_size and num_episodes_per_start should be evenly divisible .. enforce it 
			#batch_size = number of episodes to return
			#max_actions = constant that is the length of the episode
			batch = np.empty(batch_size, object)
			num_starts_per_batch = int(batch_size/num_episodes_per_start)

			if start_at:
				starting_state = np.linspace(start_at, start_at + num_starts_per_batch, num=num_starts_per_batch)
			else:
				#this is the problem, as long as order is fixed, the loss is smooth, when order changes, everything gets messed up
				#starting_state = np.random.choice(range(num_starting_states), num_starts_per_batch, replace=False)
				# print(starting_state)
				starting_state = range(num_starting_states)
				#starting_state = np.array([8, 5, 0, 2, 1, 9, 7, 3, 6, 4])
				#starting_state = np.array([6, 7, 8, 9, 0, 1, 2, 3, 4, 5])
			#starting_state is a list now
			
			ep = np.zeros((num_starts_per_batch, num_episodes_per_start))
			start_id = np.zeros((num_starts_per_batch, num_episodes_per_start))

			for start in range(num_starts_per_batch):
				#pdb.set_trace()
				#ep[start] = np.random.choice(range(num_episodes_per_start), num_episodes_per_start, replace=False)
				ep[start] = range(num_episodes_per_start)
				start_id[start] = ep[start] * max_actions + starting_state[start]*num_episodes_per_start * max_actions

			start_id = start_id.reshape(batch_size).astype(int)
			for b in range(batch_size):
				batch[b] = self.memory[start_id[b]:start_id[b]+max_actions]
				if batch[b] == []:
					print('empty batch')
					pdb.set_trace()

			return batch

		else:
			return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)


class Policy(nn.Module):
	def __init__(self, in_dim, out_dim, continuous=False, std=-0.8, max_torque=1., small=False):#-0.8 GOOD
		super(Policy, self).__init__()
		self.n_actions = out_dim
		self.continuous = continuous
		self.max_torque = max_torque
		self.small = small
		if not small:
			self.lin1 = nn.Linear(in_dim, 64)
			self.relu = nn.ReLU()
			#self.lin2 = nn.Linear(32, 16)
			#self.theta = nn.Linear(4, out_dim)
			self.theta = nn.Linear(64, 64)

			#self.value_head = nn.Linear(64, 1)
			self.action_head = nn.Linear(64,out_dim)
			torch.nn.init.xavier_uniform_(self.theta.weight)
		else:
			self.lin1 = nn.Linear(in_dim, 8)
			self.relu = nn.ReLU()
			self.action_head = nn.Linear(8,out_dim)

		torch.nn.init.xavier_uniform_(self.lin1.weight)
		
		if continuous:
			if not small:
				self.log_std = nn.Linear(64, out_dim)
			else:
				self.log_std = nn.Linear(8, out_dim)
			# self.log_std = nn.Parameter(torch.ones(out_dim) * std, requires_grad=True)
			#self.log_std = (torch.ones(out_dim) * std).type(torch.DoubleTensor)

	def forward(self, x):
		x = self.relu(self.lin1(x))
		if not self.small:
			x = self.relu(self.theta(x))
		return x

	def sample_action(self, x):
		action_probs = self.get_action_probs(x)

		if not self.continuous:
			c = Categorical(action_probs[0])
			a = c.sample() 
			
			#a = convert_one_hot(a.double(), self.n_actions)#.unsqueeze(2)
			#the dim of this could be wrong due to change to batch_size. NOT TESTED
		else:
			c = Normal(*action_probs)
			#a = torch.clamp(c.rsample(), min=-self.max_torque, max=self.max_torque)
			a = c.rsample()#self.action_multiplier * c.rsample()

		return a#, values

	def get_action_probs(self, x):
		if self.continuous:
			mu = nn.Tanh()(self.action_head(self.forward(x)))*self.max_torque
			sigma_sq = F.softplus(self.log_std(self.forward(x))) + 0.1
			self.sigma_sq = sigma_sq.mean(dim=0).detach()
			# sigma_sq = self.log_std.exp().expand_as(mu)

			return (mu,sigma_sq)
		else:
			return (nn.Softmax(dim=-1)(self.action_head(self.forward(x))),0)


class DeterministicPolicy(nn.Module):
	def __init__(self, in_dim, out_dim, max_action):#-0.8 GOOD
		super(DeterministicPolicy, self).__init__()
		self.n_actions = out_dim
		self.max_action = max_action

		self.lin1 = nn.Linear(in_dim, 64)
		self.relu = nn.ReLU()

		self.theta = nn.Linear(64, 64)
		self.action_head = nn.Linear(64,out_dim)

		torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.theta.weight)

	def forward(self, x):
		x = self.relu(self.lin1(x))
		x = self.relu(self.theta(x))
		x = nn.Tanh()(self.action_head(x))
		return x

	def sample_action(self, x):
		action = self.forward(x) * self.max_action
		return action


class Value(nn.Module):
	def __init__(self, states_dim, actions_dim):
		super(Value, self).__init__()
		self.lin1 = nn.Linear(states_dim+actions_dim, 64)
		self.lin2 = nn.Linear(64, 64)
		self.relu = nn.ReLU()
		self.theta = nn.Linear(64, 1)

		torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.lin2.weight)
		torch.nn.init.xavier_uniform_(self.theta.weight)

	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x = self.relu(self.lin1(xu))
		values = self.theta(self.relu(self.lin2(x)))
		return values


class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, action_dim)
		
		self.max_action = max_action


	def forward(self, x):
		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = self.max_action * torch.tanh(self.l3(x)) 
		return x


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		# Q1 architecture
		self.l1 = nn.Linear(state_dim + action_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, 1)

		# Q2 architecture
		self.l4 = nn.Linear(state_dim + action_dim, 64)
		self.l5 = nn.Linear(64, 64)
		self.l6 = nn.Linear(64, 1)


	def forward(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)

		x2 = F.relu(self.l4(xu))
		x2 = F.relu(self.l5(x2))
		x2 = self.l6(x2)
		return x1, x2


	def Q1(self, x, u):
		xu = torch.cat([x, u], 1)

		x1 = F.relu(self.l1(xu))
		x1 = F.relu(self.l2(x1))
		x1 = self.l3(x1)
		return x1 


# Ornstein-Ulhenbeck Process
# Taken from #https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()
        
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
        
    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state
    
    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

# class OrnsteinUhlenbeckActionNoise:
# 	def __init__(self, mu=0, sigma=0.2, theta=.15, dt=1e-2, x0=None):
# 		self.theta = theta
# 		self.mu = mu
# 		self.sigma = sigma
# 		self.dt = dt
# 		self.x0 = x0
# 		self.reset()

# 	def __call__(self):
# 		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
# 		self.x_prev = x
# 		return x

# 	def reset(self):
# 		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

# 	def __repr__(self):
# 		return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

