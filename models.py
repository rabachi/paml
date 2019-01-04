import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
import math
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import pdb


class Policy(nn.Module):
	def __init__(self, in_dim, out_dim, continuous=False, std=1.0):
		super(Policy, self).__init__()
		self.continuous = continuous
		self.lin1 = nn.Linear(in_dim, 16)
		self.relu = nn.ReLU()
		self.theta = nn.Linear(16, out_dim)

		torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.theta.weight)

		if continuous:
			#self.sigma = nn.Linear(16, out_dim)
			self.log_std = nn.Parameter(torch.ones(out_dim) * std)


	def forward(self, x):
		phi = self.relu(self.lin1(x))

		if not self.continuous:
			y = self.theta(phi)
			out = nn.Softmax(dim=-1)(y)

			return out, 0

		else:
			mu = nn.Tanh()(self.theta(phi))
			#sigma2 = torch.exp(self.sigma(phi))
			sigma = self.log_std.exp().expand_as(mu)

			return mu, sigma

	def get_phi(self, x):
		return self.relu(self.lin1(x))


class StateActionFeatures(nn.Module):
	def __init__(self, N_STATES, N_ACTIONS, hidden_dim):
		super(Featured_Policy, self).__init__()
		self.fc1 = nn.Linear(N_STATES + N_ACTIONS, 128)
		self.fc2 = nn.Linear(128, hidden_dim)
		self.relu = nn.ReLU()

	def forward(self, x):
		return self.fc2(self.relu(self.fc1(x)))



class PolicyTorque(nn.Module):
	def __init__(self, in_dim, out_dim):
		super(PolicyTorque, self).__init__()
		self.lin1 = nn.Linear(in_dim, 128)
		self.relu = nn.ReLU()
		#self.lin2 = nn.Linear(128, 128)

		self.theta_mu = nn.Linear(128, out_dim)
		self.theta_log_sigma = nn.Linear(128, out_dim)
		# for param in self.lin1.parameters():
		# 	param.requires_grad = False
		# initialize layers
		torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.theta_mu.weight)
		torch.nn.init.xavier_uniform_(self.theta_log_sigma.weight)

	def forward(self, x):
		phi = self.relu(self.lin1(x))
		#phi = self.relu(self.lin2(x))

		mu = nn.Tanh()(self.theta_mu(phi))
		sigma = torch.exp(self.theta_log_sigma(phi)/2.0)

		return mu, sigma

	def get_phi(self, x):
		return self.relu(self.lin1(x))


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
		action_dim, 
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
		self.state = probs.sample(torch.zeros(self.n_states).size())
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
		n_neurons=128,
		n_layers=3,
		activation_fn=nn.ReLU(),
		init_w=1e-4,
		clip_output=True
		):
		super(ACPModel, self).__init__()
		self.state_dim = state_dim
		self.action_dim = action_dim
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
			output[:,:,2] = torch.clamp(out[:,:,2], min=-8.0, max=8.0)
		
			return output

		return out

	def reset(self):
		probs = torch.distributions.Uniform(low=-.05, high=0.05, size=(4,))
		self.state = probs.sample(torch.zeros(self.n_states).size())
		return self.state



class DirectEnvModel(torch.nn.Module):
	def __init__(self, N_STATES, N_ACTIONS):
		super(DirectEnvModel, self).__init__()
		# build network layers
		self.n_states = N_STATES
		self.n_actions = N_ACTIONS
		self.fc1 = nn.Linear(N_STATES + N_ACTIONS, 256)
		self.fc2 = nn.Linear(256, 128)
		self.fc3 = nn.Linear(128, 128)

		self._enc_mu = torch.nn.Linear(128, N_STATES)
		#self._enc_log_sigma = torch.nn.Linear(128, 4)

		#self.statePrime = nn.Linear(16, 4)
		self.reward = nn.Linear(N_STATES, 1)
		#self.done = nn.Linear(N_STATES, 1)

		# initialize layers
		torch.nn.init.xavier_uniform_(self.fc1.weight)
		torch.nn.init.xavier_uniform_(self.fc2.weight)

		torch.nn.init.xavier_uniform_(self._enc_mu.weight)
		#torch.nn.init.xavier_uniform_(self._enc_log_sigma.weight)

		#torch.nn.init.xavier_uniform_(self.statePrime.weight)
		torch.nn.init.xavier_uniform_(self.reward.weight)
		#torch.nn.init.xavier_uniform_(self.done.weight)

	def reset(self):
		probs = torch.distributions.Uniform(low=-0.1, high=0.1)
		self.state = probs.sample(torch.zeros(self.n_states).size())
		self.steps_beyond_done = None
		return self.state

	def forward(self, x):
		x = self.fc1(x)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		# x = self.fc3(x)
		# x = nn.ReLU()(x)

		mu = self._enc_mu(x)

		statePrime_value = mu
		reward_value = self.reward(mu)
		#done_value = nn.Sigmoid()(self.done(mu))


		#log_sigma = self._enc_log_sigma(x)
		#statePrime_value = mu.detach()#self.statePrime(x)

		# with torch.no_grad():
		# 	try:
		# 		x = statePrime_value[:,0]
		# 		theta = statePrime_value[:,2]
		# 	except:
		# 		x = statePrime_value[0]
		# 		theta = statePrime_value[2]

		# 	#don't train this value
		# 	done_value = (x < -self.x_threshold) | (x > self.x_threshold) | (theta < -self.theta_threshold_radians) | (theta > self.theta_threshold_radians)
		# 	reward_value = torch.ones_like(done_value)

		return mu#, reward_value, 0