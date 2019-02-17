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

from utils import *
from collections import namedtuple
import random
from torchviz import make_dot

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

	def sample(self, batch_size, structured=False, max_actions=10, num_episodes=10):
		if structured:
			#batch_size = number of episodes to return
			#max_actions = constant that is the length of the episode
			batch = np.empty(batch_size, object)
			for b in range(batch_size):
				ep = np.random.choice(range(num_episodes))
				batch[b] = self.memory[ep:ep+max_actions]

			return batch
			
		else:
			return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)



class Policy(nn.Module):
	def __init__(self, in_dim, out_dim, continuous=False, std=-0.8):#-0.8 GOOD
		super(Policy, self).__init__()
		self.n_actions = out_dim
		self.continuous = continuous
		self.lin1 = nn.Linear(in_dim, 16)
		self.relu = nn.ReLU()
		self.theta = nn.Linear(16, out_dim)

		torch.nn.init.xavier_uniform_(self.lin1.weight)
		torch.nn.init.xavier_uniform_(self.theta.weight)

		if continuous:
			#self.log_std = nn.Linear(16, out_dim)
			self.log_std = nn.Parameter(torch.ones(out_dim) * std, requires_grad=True)
			#self.log_std = (torch.ones(out_dim) * std).type(torch.DoubleTensor)

	def forward(self, x):
		phi = self.relu(self.lin1(x))

		if not self.continuous:
			y = self.theta(phi)
			out = nn.Softmax(dim=-1)(y)
			out = torch.clamp(out, min=-1e-4, max=100)
			return out, 0

		else:
			mu = nn.Tanh()(self.theta(phi))
			#sigma = torch.exp(self.log_std(phi))
			sigma = self.log_std.exp().expand_as(mu)

			return mu, sigma

	def get_phi(self, x):
		return self.relu(self.lin1(x))



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
		self.state = probs.sample(torch.zeros(self.n_states).size())
		return self.state





class DirectEnvModel(torch.nn.Module):
	def __init__(self, N_STATES, N_ACTIONS, MAX_TORQUE):
		super(DirectEnvModel, self).__init__()
		# build network layers
		self.n_states = N_STATES
		self.n_actions = N_ACTIONS
		self.max_torque = MAX_TORQUE

		self.fc1 = nn.Linear(N_STATES + N_ACTIONS, 32)
		self.fc2 = nn.Linear(32, 16)
		#self.fc3 = nn.Linear(32, 16)

		self._enc_mu = torch.nn.Linear(16, N_STATES)
		#self._enc_log_sigma = torch.nn.Linear(128, 4)

		#self.statePrime = nn.Linear(16, 4)
		#self.reward = nn.Linear(N_STATES, 1)
		#self.done = nn.Linear(N_STATES, 1)

		# initialize layers
		torch.nn.init.xavier_uniform_(self.fc1.weight)
		torch.nn.init.xavier_uniform_(self.fc2.weight)

		torch.nn.init.xavier_uniform_(self._enc_mu.weight)
		#torch.nn.init.xavier_uniform_(self._enc_log_sigma.weight)

		#torch.nn.init.xavier_uniform_(self.statePrime.weight)
		#torch.nn.init.xavier_uniform_(self.reward.weight)
		#torch.nn.init.xavier_uniform_(self.done.weight)

	def reset(self):
		probs = torch.distributions.Uniform(low=-0.1, high=0.1)
		self.state = probs.sample(torch.zeros(self.n_states).size())
		self.steps_beyond_done = None
		return self.state.type(torch.DoubleTensor)

	def forward(self, x):
		x = self.fc1(x)
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		# x = self.fc3(x)
		# x = nn.ReLU()(x)

		mu = self._enc_mu(x)

		#statePrime_value = mu
		#reward_value = self.reward(mu)
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


	def unroll(self, state_action, policy, n_states, steps_to_unroll=2, continuous_actionspace=True):
		if state_action is None:
			return -1	

		max_actions = state_action.shape[0]
		x_list = [state_action[:,:n_states]]
		a_list = [state_action[:,n_states:]]
		r_list = []

		for s in range(steps_to_unroll):
			if torch.isnan(torch.sum(state_action)):
					print('found nan in state')
					print(state_action)
					pdb.set_trace()
			x_next = self.forward(state_action)
			action_taken = state_action[:,n_states:]
			r = get_reward_fn('lin_dyn', x_next, action_taken)

			with torch.no_grad():
				action_probs = policy(torch.DoubleTensor(x_next))

				if not continuous_actionspace:
					c = Categorical(action_probs[0])
					a = c.sample() 
					next_state_action = torch.cat((x_next, convert_one_hot(a.double(), n_actions).unsqueeze(2)),dim=2)
				else:
					c = Normal(*action_probs)
					a = torch.clamp(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)
					next_state_action = torch.cat((x_next, a),dim=1)

			#since we're unrolling the last s states past the termination, we have to set the rewards for those predicted states to 0 because we have no data to learn from those
			if s > 0:
				r[-s:] = 0 
			a_list.append(a)
			r_list.append(r.unsqueeze(1))
			x_list.append(x_next)

			state_action = next_state_action

		#x_list = torch.stack(x_list)
		x_list = torch.cat(x_list, 1).view(max_actions, steps_to_unroll+1, n_states)
		a_list = torch.cat(a_list, 1).view(max_actions, steps_to_unroll +1, a.shape[1])
		r_list = torch.cat(r_list, 1).view(max_actions, steps_to_unroll, 1)
	
		x_curr = x_list[:,:-1,:]
		x_next = x_list[:,1:,:]
		a_used = a_list[:,:-1]

		return x_curr, x_next, a_used, r_list

		# #only need rewards from model for the steps we've unrolled, the rest is assumed to be equal to the environment's
		# model_rewards[:, step] = get_reward_fn('lin_dyn', shortened, a)
		# model_log_probs = get_selected_log_probabilities(policy, shortened, a)
		#don't need states, just the log probabilities 

	def train_paml(self, env, pe, states_prev, states_next, actions_tensor, rewards_tensor, epochs, max_actions, R_range, discount, opt, continuous_actionspace, losses, device='cpu'):

		best_loss = 20
		batch_size = states_next.shape[0]

		state_actions = torch.cat((states_prev,actions_tensor), dim=2)

		#pamlize the real trajectory (states_next)
		pe.zero_grad()
		multiplier = torch.arange(max_actions,0,-1).repeat(batch_size,1).unsqueeze(2).type(torch.FloatTensor).to(device)

		true_log_probs_t = torch.sum(
							get_selected_log_probabilities(
								pe, 
								states_prev, 
								actions_tensor, 
								range(actions_tensor.shape[0])
								) * rewards_tensor #* multiplier
							, dim=1)

		true_log_probs = torch.mean(true_log_probs_t, dim=0)
		true_pe_grads = grad(true_log_probs, pe.parameters(), create_graph=True)


		for i in range(epochs):
			opt.zero_grad()
			step_state_action = state_actions.to(device)
			k_step_log_probs = torch.zeros((R_range, batch_size, self.n_actions))
			pe.zero_grad()

			for step in range(R_range - 1):
				next_step_state = self.forward(step_state_action)
				#print('states_mean:', torch.mean(next_step_state))
				shortened = next_step_state[:,:-1,:]

				with torch.no_grad():
					print(step)
					action_probs = pe(torch.FloatTensor(shortened))
					
					if not continuous_actionspace:
						c = Categorical(action_probs)
						actions_t_l = c.sample() 
						step_state_action = torch.cat((shortened,convert_one_hot(actions_t_l, self.n_actions)),dim=2)
					else:
						c = Normal(*action_probs)
						actions_t_l = torch.clamp(c.rsample(), min=-2.,max=2.)
						step_state_action = torch.cat((shortened, actions_t_l),dim=2)


				model_rewards_t = get_reward_fn(env, shortened, actions_t_l) #need gradients of this because rewards are from the states
				discounted_model_rewards = discount_rewards(model_rewards_t, discount)
				model_log_probs = torch.sum(
									get_selected_log_probabilities(
										pe, 
										shortened,
										actions_t_l, range(actions_t_l.shape[0])) * 
										model_rewards_t
									, dim=1)
				k_step_log_probs[step] = model_log_probs#.squeeze()

			model_log_probs = torch.mean(torch.sum(k_step_log_probs, dim=0))

			model_pe_grads = grad(model_log_probs, pe.parameters(), create_graph=True)
			#total_log_probs.backward(retain_graph=True)
			#model_pe_grads = [x.grad for x in pe.parameters()]

			grad_diffs = torch.zeros((len(true_pe_grads)))
	
			for i in range(len(true_pe_grads)):
				grad_diffs[i] = torch.sqrt(torch.sum((model_pe_grads[i] - true_pe_grads[i])**2))

			model_loss = torch.mean(grad_diffs)
			model_loss.backward(retain_graph=True)

			# print('model_pe_grads[0]:', model_pe_grads[0])
			# print('true_pe_grads[0]:',true_pe_grads[0])

			# print('model_log_probs:', torch.mean(model_log_probs))
			# print('model_rewards_t:', torch.mean(model_rewards_t))
			# print('actions_t_l:', torch.mean(actions_t_l))

			# pdb.set_trace()

			print("i: {}, paml loss = {:.7f}".format(i, model_loss.data.cpu()))
			if model_loss < best_loss:
				torch.save(self.state_dict(), 
					env.spec.id+'_paml_trained_model.pth')
				#torch.save(P_hat, 'mle_trained_model.pth')
				best_loss = model_loss 

			opt.step()
			losses.append(model_loss.data.cpu())

		return best_loss


	def train_paml_skip(self, env, pe, states_prev, states_next, actions_tensor, rewards_tensor, epochs, max_actions, R_range, discount, opt, continuous_actionspace, losses, device='cpu'):

		best_loss = 20
		batch_size = states_next.shape[0]
		R_range = 1

		state_actions = torch.cat((states_prev,actions_tensor), dim=2)

		#pamlize the real trajectory (states_next)
		pe.zero_grad()
		multiplier = torch.arange(max_actions,0,-1).repeat(batch_size,1).unsqueeze(2).type(torch.FloatTensor).to(device)

		true_log_probs_t = torch.sum(
							get_selected_log_probabilities(
								pe, 
								states_prev, 
								actions_tensor, 
								range(actions_tensor.shape[0])
								) * rewards_tensor #* multiplier
							, dim=1)

		true_log_probs = torch.mean(true_log_probs_t, dim=0)
		true_pe_grads = grad(true_log_probs, pe.parameters(), create_graph=True)


		for i in range(epochs):
			opt.zero_grad()
			step_state_action = state_actions.to(device)
			k_step_log_probs = torch.zeros((R_range, batch_size, self.n_actions))
			pe.zero_grad()

			for step in range(R_range):
				next_step_state = self.forward(step_state_action)
				#print('states_mean:', torch.mean(next_step_state))
				shortened = next_step_state[:,:-1,:]

				with torch.no_grad():
					action_probs = pe(torch.FloatTensor(shortened))
					
					if not continuous_actionspace:
						c = Categorical(action_probs)
						actions_t_l = c.sample() 
						step_state_action = torch.cat((shortened,convert_one_hot(actions_t_l, self.n_action)),dim=2)
					else:
						c = Normal(*action_probs)
						actions_t_l = torch.clamp(c.rsample(), min=-2.,max=2.)
						step_state_action = torch.cat((shortened, actions_t_l),dim=2)


				model_rewards_t = get_reward_fn(env, shortened, actions_t_l) #need gradients of this because rewards are from the states
				discounted_model_rewards = discount_rewards(model_rewards_t[:,:R_range], discount)

				model_log_probs = get_selected_log_probabilities(
										pe, 
										shortened,
										actions_t_l, 
										range(actions_t_l.shape[0])
										)

				k_step_log_probs[step] = model_log_probs  
											#* (discounted_model_rewards_t + 


			model_log_probs = torch.mean(torch.sum(k_step_log_probs, dim=0))

			model_pe_grads = grad(model_log_probs, pe.parameters(), create_graph=True)
			#total_log_probs.backward(retain_graph=True)
			#model_pe_grads = [x.grad for x in pe.parameters()]

			grad_diffs = torch.zeros((len(true_pe_grads)))
	
			for i in range(len(true_pe_grads)):
				grad_diffs[i] = torch.sqrt(torch.sum((model_pe_grads[i] - true_pe_grads[i])**2))

			model_loss = torch.mean(grad_diffs)
			model_loss.backward(retain_graph=True)

			# print('model_pe_grads[0]:', model_pe_grads[0])
			# print('true_pe_grads[0]:',true_pe_grads[0])

			# print('model_log_probs:', torch.mean(model_log_probs))
			# print('model_rewards_t:', torch.mean(model_rewards_t))
			# print('actions_t_l:', torch.mean(actions_t_l))

			# pdb.set_trace()

			print("i: {}, paml loss = {:.7f}".format(i, model_loss.data.cpu()))
			if model_loss < best_loss:
				torch.save(P_hat.state_dict(), 
					env.spec.id+'_paml_trained_model.pth')
				#torch.save(P_hat, 'mle_trained_model.pth')
				best_loss = model_loss 

			opt.step()
			losses.append(model_loss.data.cpu())

		return best_loss


	def train_mle(self, state_actions, states_next, epochs, max_actions, R_range, opt, env_name, continuous_actionspace, losses):
		best_loss = 20
		for i in range(epochs):
			opt.zero_grad()

			squared_errors = torch.zeros_like(states_next)
			step_state = state_actions.to(device)

			for step in range(R_range - 1):
				next_step_state = self.forward(step_state)

				squared_errors += shift_down((states_next[:,step:,:] - next_step_state)**2, step, max_actions)

				#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)

				shortened = next_step_state[:,:-1,:]
				action_probs = pe(torch.FloatTensor(shortened))
				
				if not continuous_actionspace:
					c = Categorical(action_probs)
					a = c.sample() 
					step_state = torch.cat((shortened,convert_one_hot(a, self.n_actions)),dim=2)
				else:
					c = Normal(*action_probs)
					a = torch.clamp(c.rsample(), min=-self.max_torque, max=self.max_torque)
					step_state = torch.cat((shortened,a),dim=2)

			state_loss = torch.mean(squared_errors, dim=1)#torch.mean((states_next - rolled_out_states_sums)**2)

			model_loss = torch.mean(state_loss) #+ reward_loss)# + done_loss)
			print("i: {}, negloglik  = {:.7f}".format(i, model_loss.data.cpu()))

			if model_loss < best_loss:
				torch.save(P_hat.state_dict(), env_name+'_mle_trained_model.pth')
				#torch.save(P_hat, 'mle_trained_model.pth')
				best_loss = model_loss  

				model_loss.backward()
				opt.step()
				losses.append(model_loss.data.cpu())
		return best_loss