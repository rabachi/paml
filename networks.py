import numpy as np
import matplotlib.pyplot as plt
import sys
import os

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
import gym


device='cpu'


class AddExtraDims(gym.ObservationWrapper):
    """ Wrap action """
    def __init__(self, env, extra_dims):
        super(AddExtraDims, self).__init__(env)

        self.extra_dims = extra_dims

    def observation(self, observation):
        new_observation = self.add_irrelevant_features(observation, self.extra_dims, noise_level=0.4)
        return new_observation

    def add_irrelevant_features(self, x, extra_dim, noise_level = 0.4):
        x_irrel= noise_level*np.random.randn(1, extra_dim).reshape(-1,)
        return np.hstack([x, x_irrel])


# https://github.com/openai/gym/blob/master/gym/core.py
class NormalizedEnv(gym.ActionWrapper):
    """ Wrap action """

    def _action(self, action):
        act_k = (self.action_space.high - self.action_space.low)/ 2.
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k * action + act_b

    def _reverse_action(self, action):
        act_k_inv = 2./(self.action_space.high - self.action_space.low)
        act_b = (self.action_space.high + self.action_space.low)/ 2.
        return act_k_inv * (action - act_b)


# Transition = namedtuple('Transition',('full_state','state', 'next_state', 'action', 'reward'))
Transition = namedtuple('Transition',('state', 'next_state', 'action', 'reward'))

# Using PyTorch's Tutorial
class ReplayMemory(object):

	def __init__(self, size):
		self.size = size
		self.memory = []
		self.temp_memory = []
		self.position = 0

	def push(self, *args):
		"""Saves a transition."""
		if len(self.memory) < self.size:
			self.memory.append(None)

		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.size

	def temp_push(self, *args):
		'''saves a transition temporarily'''
		self.temp_memory.append(Transition(*args))

	def clear_temp(self):
		'''clears temporarily saved transitions'''
		self.temp_memory = []

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
			return random.sample(self.memory + self.temp_memory, batch_size)
			# if len(self.temp_memory) >= batch_size:
			# 	return random.sample(self.temp_memory, batch_size)
			# else:
			# 	return random.sample(self.memory + self.temp_memory, batch_size)

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
	def __init__(self, in_dim, out_dim, max_action): 
		super(DeterministicPolicy, self).__init__()
		self.n_actions = out_dim
		self.max_action = max_action

		if type(in_dim) == tuple:
			self.p_type = 'cnn'
			in_channels, w, h = in_dim
			self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=5, stride=1)
			self.bn1 = nn.BatchNorm2d(16)
			self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1)
			self.bn2 = nn.BatchNorm2d(32)
			# self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
			# self.bn3 = nn.BatchNorm2d(32)
			# Number of Linear input connections depends on output of conv2d layers
			# and therefore the input image size, so compute it.
			def conv2d_size_out(size, kernel_size = 5, stride = 1):
				return (size - (kernel_size - 1) - 1) // stride  + 1
			convw = conv2d_size_out(conv2d_size_out(w))
			convh = conv2d_size_out(conv2d_size_out(h))
			linear_input_size = convw * convh * 32
		else:
			self.p_type = 'nn'
			self.lin1 = nn.Linear(in_dim, 30)
			# self.lin1 = nn.Linear(in_dim, out_dim)
			self.relu = nn.ReLU()

			self.theta = nn.Linear(30, 30)
			self.action_head = nn.Linear(30,out_dim)

			torch.nn.init.xavier_uniform_(self.lin1.weight)
			torch.nn.init.xavier_uniform_(self.theta.weight)
			torch.nn.init.xavier_uniform_(self.action_head.weight)

	def forward(self, x):
		if self.p_type == 'cnn':
			x = F.relu(self.bn1(self.conv1(x)))
			x = F.relu(self.bn2(self.conv2(x)))
			# x = F.relu(self.bn3(self.conv3(x)))
			pdb.set_trace()
			return self.head(x.view(x.size(0), -1))

		else:
			x = self.relu(self.lin1(x))
			x = self.relu(self.theta(x))
			x = nn.Tanh()(self.action_head(x))
			# x = nn.Tanh()(self.lin1(x))
			return x

	def sample_action(self, x):
		action = self.forward(x) * self.max_action
		return action


class Value(nn.Module):
	def __init__(self, states_dim, actions_dim, pretrain_val_lr=1e-4, pretrain_value_lr_schedule=None):
		super(Value, self).__init__()
		self.lin1 = nn.Linear(states_dim+actions_dim, 64)
		self.lin2 = nn.Linear(64, 64)
		self.relu = nn.ReLU()
		self.theta = nn.Linear(64, 1)

		torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.lin2.weight)
		torch.nn.init.xavier_uniform_(self.theta.weight)

		self.q_optimizer = optim.Adam(self.parameters(), lr=pretrain_val_lr)
		self.pretrain_value_lr_schedule = None
		self.states_dim = states_dim
		self.actions_dim = actions_dim

	def reset_weights(self):
		torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.lin2.weight)
		torch.nn.init.xavier_uniform_(self.theta.weight)

	def forward(self, x, u):
		xu = torch.cat([x, u], 1)
		x = self.relu(self.lin1(xu))
		values = self.theta(self.relu(self.lin2(x)))
		return values

	def pre_train(self, actor, dataset, epochs_value, discount, batch_size, salient_states_dim, file_location, file_id, model_type, env_name, max_actions, verbose=100):
		MSE = nn.MSELoss()
		TAU=0.001
		target_critic = Value(self.states_dim, self.actions_dim).double()

		for target_param, param in zip(target_critic.parameters(), self.parameters()):
			target_param.data.copy_(param.data)

		train_losses = []
		for i in range(epochs_value):
			batch = dataset.sample(batch_size)
			states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
			actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

			# actions_next = actor.sample_action(states_next[:,:salient_states_dim])
			actions_next = actor.sample_action(states_next)#[:,:salient_states_dim])
			# target_q = target_critic(states_next[:,:salient_states_dim], actions_next)
			target_q = target_critic(states_next, actions_next)
			y = rewards_tensor + discount * target_q.detach() #detach to avoid backprop target
			# q = critic(states_prev[:,:salient_states_dim], actions_tensor)
			q = self(states_prev, actions_tensor)

			self.q_optimizer.zero_grad()
			loss = MSE(y, q)
			train_losses.append(loss.detach().data)

			if i % verbose == 0:
				print('Epoch: {:4d} | LR: {:.4f} | Value estimator loss: {:.5f}'.format(i, self.q_optimizer.param_groups[0]['lr'], loss.detach().cpu()))
				torch.save(self.state_dict(), os.path.join(file_location,'critic_policy_{}_state{}_salient{}_checkpoint_{}_traj{}_{}.pth'.format(model_type, self.states_dim, salient_states_dim, env_name, max_actions + 1, file_id)))

			loss.backward()
			nn.utils.clip_grad_value_(self.parameters(), 100.0)
			self.q_optimizer.step()
			if self.pretrain_value_lr_schedule is not None:
				self.pretrain_value_lr_schedule.step()

			#soft update the target critic
			for target_param, param in zip(target_critic.parameters(), self.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)
			
			# if non_decreasing(train_losses):
			# 	self.q_optimizer = optim.Adam(self.parameters(), lr=self.q_optimizer.param_groups[0]['lr']/2)
		del target_critic


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
    
    def get_action(self, action, t=0, multiplier=1.0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + multiplier*ou_state, self.low, self.high)

