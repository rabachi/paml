import numpy as np
import matplotlib.pyplot as plt
import gym
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
from torchviz import make_dot


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
				starting_state = np.random.choice(range(num_starting_states), num_starts_per_batch, replace=False)
			#starting_state is a list now
			
			ep = np.zeros((num_starts_per_batch, num_episodes_per_start))
			start_id = np.zeros((num_starts_per_batch, num_episodes_per_start))

			for start in range(num_starts_per_batch):
				ep[start] = np.random.choice(range(num_episodes_per_start), num_episodes_per_start, replace=False)
				start_id[start] = ep[start]*max_actions + starting_state[start]*num_episodes_per_start*max_actions

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
	def __init__(self, in_dim, out_dim, continuous=False, std=-0.8, max_torque=1. ):#-0.8 GOOD
		super(Policy, self).__init__()
		self.n_actions = out_dim
		self.continuous = continuous
		self.max_torque = max_torque
		#self.lin1 = nn.Linear(in_dim, 4)
		#self.relu = nn.ReLU()
		#self.theta = nn.Linear(4, out_dim)
		self.theta = nn.Linear(in_dim, out_dim)

		#torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.theta.weight)

		if continuous:
			#self.log_std = nn.Linear(16, out_dim)
			self.log_std = nn.Parameter(torch.ones(out_dim) * std, requires_grad=True)
			#self.log_std = (torch.ones(out_dim) * std).type(torch.DoubleTensor)

	def forward(self, x):
		#phi = self.relu(self.lin1(x))

		if not self.continuous:
			y = self.theta(phi)
			out = nn.Softmax(dim=-1)(y)
			out = torch.clamp(out, min=-1e-4, max=100)
			return out, 0

		else:
			mu = nn.Tanh()(self.theta(x))
			#sigma = torch.exp(self.log_std(phi))
			sigma = self.log_std.exp().expand_as(mu)
			return mu, sigma


	def sample_action(self, x):
		action_probs = self.forward(x)

		if not self.continuous:
			c = Categorical(action_probs[0])
			a = c.sample() 
			a = convert_one_hot(a.double(), n_actions).unsqueeze(2)
			pdb.set_trace()
			#the dim of this could be wrong due to change to batch_size. NOT TESTED
		else:
			c = Normal(*action_probs)
			a = torch.clamp(c.rsample(), min=-self.max_torque, max=self.max_torque)

		return a



	# def get_fisher(self, x_prime, num_samples):
	# 	action_probs = self.forward(x_prime)

	# 	with torch.no_grad():
	# 		c = Normal(*actions_probs)
	# 		actions_tensor = torch.zeros((num_samples, x_prime.shape[0], self.n_actions))

	# 		for s in range(num_samples):
	# 			actions_tensor[i] = torch.clip(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)

	# 	log_probs = get_selected_log_probabilities(self, x_prime, actions_tensor)
	# 	grad(log_probs)
	# 	return



class PCPModel(torch.nn.Module):

	LINK_LENGTH_1 = 1.  # [m]
	LINK_LENGTH_2 = 1.  # [m]
	LINK_MASS_1 = 1.  #: [kg] mass of link 1
	LINK_MASS_2 = 1.  #: [kg] mass of link 2
	LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
	LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
	LINK_MOI = 1.  #: moments of inertia for both links

	MAX_VEL_1 = 4 * np.pi
	MAX_VEL_2 = 9 * np.pi

	AVAIL_TORQUE = [-1., 0., +1]

	def __init__(
		self, 
		state_dim, 
		env, 
		R, 
		n_neurons=1000,
		n_layers=9, 
		activation_fn=nn.SELU(),
		init_w=1e-4
		):
		super(PCPModel, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.range = R
		self.input_layer = nn.Linear(state_dim + action_dim*R, n_neurons)
		self.input_layer.weight.data.uniform_(-init_w, init_w)
		self.input_layer.bias.data.uniform_(-init_w, init_w)

		self.hidden_layers = nn.ModuleList()

		for layer_i in range(n_layers):
			fc_layer = nn.Linear(n_neurons, n_neurons)

			fc_layer.weight.data.uniform_(-init_w, init_w)
			fc_layer.bias.data.uniform_(-init_w, init_w)

			self.hidden_layers.append(fc_layer)

		self.out_layer = nn.Linear(n_neurons, state_dim)
		self.out_layer.weight.data.uniform_(-init_w, init_w)
		self.out_layer.bias.data.uniform_(-init_w, init_w)

		self.n_layers = n_layers 
		self.activation = activation_fn

		self.state = None
		self.viewer = None

	def reset(self):
		probs = torch.distributions.Uniform(low=-0.1, high=0.1)
		self.state = probs.sample(torch.zeros(self.states_dim).size())
		return self.state

	def forward(self, x):
		h = self.activation(self.input_layer(x))

		for layer in self.hidden_layers:
			h = layer(h)
			h = self.activation(h)

		out = self.out_layer(h)
		self.state = out

		return out


	# def render(self, mode='human'):
	# 	from gym.envs.classic_control import rendering

	# 	s = self.state.detach().numpy()[0,-1]

	# 	if self.viewer is None:
	# 		self.viewer = rendering.Viewer(500,500)
	# 		bound = self.LINK_LENGTH_1 + self.LINK_LENGTH_2 + 0.2  # 2.2 for default
	# 		self.viewer.set_bounds(-bound,bound,-bound,bound)

	# 	if s is None: return None

	# 	p1 = [-self.LINK_LENGTH_1 *
	# 			np.cos(s[0]), self.LINK_LENGTH_1 * np.sin(s[0])]

	# 	p2 = [p1[0] - self.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
	# 		p1[1] + self.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

	# 	xys = np.array([[0,0], p1, p2])[:,::-1]
	# 	thetas = [s[0]-np.pi/2, s[0]+s[1]-np.pi/2]
	# 	link_lengths = [self.LINK_LENGTH_1, self.LINK_LENGTH_2]

	# 	self.viewer.draw_line((-2.2, 1), (2.2, 1))
	# 	for ((x,y),th,llen) in zip(xys, thetas, link_lengths):
	# 		l,r,t,b = 0, llen, .1, -.1
	# 		jtransform = rendering.Transform(rotation=th, translation=(x,y))
	# 		link = self.viewer.draw_polygon([(l,b), (l,t), (r,t), (r,b)])
	# 		link.add_attr(jtransform)
	# 		link.set_color(0,.8, .8)
	# 		circ = self.viewer.draw_circle(.1)
	# 		circ.set_color(.8, .8, 0)
	# 		circ.add_attr(jtransform)

	# 	return self.viewer.render(return_rgb_array = mode=='rgb_array')


class ACPModel(torch.nn.Module):
	def __init__(
		self, 
		state_dim, 
		action_dim, 
		R_range,
		n_neurons=128,
		n_layers=3,
		activation_fn=nn.ReLU(),
		init_w=1e-4,
		clip_output=True
		):
		super(ACPModel, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.max_torque = 8.0
		self.range = R_range

		if action_dim == 2:
			self.input_layer = nn.Linear(state_dim + 1, n_neurons)
		else:
			self.input_layer = nn.Linear(state_dim + action_dim, n_neurons)
		self.input_layer.weight.data.uniform_(-init_w, init_w)
		self.input_layer.bias.data.uniform_(-init_w, init_w)

		self.hidden_layers = nn.ModuleList()

		for layer_i in range(n_layers):
			fc_layer = nn.Linear(n_neurons, n_neurons)

			fc_layer.weight.data.uniform_(-init_w, init_w)
			fc_layer.bias.data.uniform_(-init_w, init_w)

			self.hidden_layers.append(fc_layer)

		self.out_layer = nn.Linear(n_neurons, state_dim)
		self.out_layer.weight.data.uniform_(-init_w, init_w)
		self.out_layer.bias.data.uniform_(-init_w, init_w)

		self.n_layers = n_layers 
		self.activation = activation_fn
		self.clip_output = clip_output

	def forward(self, x):
		h = self.activation(self.input_layer(x))

		for layer in self.hidden_layers:
			h = layer(h)
			h = self.activation(h)

		#out = x[:,:,:self.state_dim] + self.out_layer(h)
		out = self.out_layer(h) 
		
		if self.clip_output:
			output = torch.zeros_like(out)
			#if len(x.shape) == 3:
			output[:,:,0:1] = torch.clamp(out[:,:,0:1], min=-1.0, max=1.0)
			output[:,:,2] = torch.clamp(out[:,:,2], min=-self.max_torque, max=self.max_torque)
		
			return output

		return out

	def reset(self):
		probs = torch.distributions.Uniform(low=-.05, high=0.05, size=(4,))
		self.state = probs.sample(torch.zeros(self.states_dim).size())
		return self.state



