import numpy as np
import matplotlib.pyplot as plt
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
import os


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class DirectEnvModel(torch.nn.Module):
	def __init__(self, states_dim, N_ACTIONS, MAX_TORQUE, mult=0.1, model_size='nn', hidden_size=2):
		super(DirectEnvModel, self).__init__()
		# build network layers
		self.states_dim = states_dim
		self.n_actions = N_ACTIONS
		self.max_torque = MAX_TORQUE
		self.mult = mult
		self.model_size = model_size

		if self.model_size == 'small':
			self.fc1 = nn.Linear(states_dim + N_ACTIONS, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)

		elif self.model_size == 'constrained':
			self. fc1 = nn.Linear(states_dim + N_ACTIONS, hidden_size)
			self._enc_mu = torch.nn.Linear(hidden_size, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
			torch.nn.init.xavier_uniform_(self._enc_mu.weight)
		else:
			self.fc1 = nn.Linear(states_dim + N_ACTIONS, 64)
			self.fc2 = nn.Linear(64, 64)
			self._enc_mu = torch.nn.Linear(64, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
			torch.nn.init.xavier_uniform_(self.fc2.weight)
			torch.nn.init.xavier_uniform_(self._enc_mu.weight)


	def forward(self, x):
		if self.model_size == 'small':
			x = self.fc1(x)
			# mu = nn.Tanh()(x) * 8.0
			mu = x
		elif self.model_size == 'constrained':
			x = self._enc_mu(self.fc1(x))
			mu = x#nn.Tanh()(x) * 360.0
		else:
			x = nn.ReLU()(self.fc1(x))
			x = self.fc2(x)
			x = nn.ReLU()(x)
			# x = self.fc3(x)
			# x = nn.ReLU()(x)
			mu = nn.Tanh()(self._enc_mu(x)) * 360.0#* 3.0
		return mu


	def unroll(self, state_action, policy, states_dim, A_numpy, steps_to_unroll=2, continuous_actionspace=True, use_model=True, policy_states_dim=2, salient_states_dim=2, extra_dims_stable=False, noise=None, epsilon=None, epsilon_decay=None,env='lin_dyn'):
		if state_action is None:
			return -1	
		batch_size = state_action.shape[0]
		max_actions = state_action.shape[1]
		actions_dim = state_action[0,0,states_dim:].shape[0]
		
		if A_numpy is not None:
			A = torch.from_numpy(
				np.tile(
					A_numpy,
					(state_action[0,:,0].shape[0],1,1) #repeat along trajectory size, for every state
					)
				).to(device)
			extra_dim = states_dim - salient_states_dim
			B = torch.from_numpy(
				np.block([
						 [0.1*np.eye(salient_states_dim),            np.zeros((salient_states_dim, extra_dim))],
						 [np.zeros((extra_dim, salient_states_dim)), np.zeros((extra_dim,extra_dim))          ]
						])
				).to(device)
		elif A_numpy is None and not use_model:
			raise NotImplementedError

		x0 = state_action[:,:,:states_dim]
		with torch.no_grad():
			a0 = policy.sample_action(x0[:,:,:states_dim])
			if isinstance(policy, DeterministicPolicy):
				a0 = torch.from_numpy(noise.get_action(a0.numpy(), 0))

		state_action = torch.cat((x0, a0),dim=2)
		if use_model:	
			x_t_1 = self.forward(state_action.squeeze())
			if len(x_t_1.shape) < 3:
				x_t_1 = x_t_1.unsqueeze(1)
		else:
			x_t_1  = torch.einsum('jik,ljk->lji',[A,x0[:,:,:policy_states_dim]])
			if not extra_dims_stable:
				x_t_1 = add_irrelevant_features(x_t_1, states_dim-policy_states_dim, noise_level=0.4)
				x_t_1 = x_t_1 + 0.1*a0
			else:
				x_t_1 = x_t_1 + torch.einsum('ijk,kk->ijk',[a0,B])

		x_list = [x0, x_t_1]

		with torch.no_grad():
			a1 = policy.sample_action(x_list[1][:,:,:states_dim])
			if isinstance(policy, DeterministicPolicy):
				a1 = torch.from_numpy(noise.get_action(a1.numpy(), 1))

		a_list = [a0, a1]
		r_list = [get_reward_fn(env, x_list[0][:,:,:policy_states_dim], a0).unsqueeze(2), get_reward_fn(env, x_t_1[:,:,:policy_states_dim], a1).unsqueeze(2)]

		state_action = torch.cat((x_t_1, a1),dim=2)

		for s in range(steps_to_unroll - 1):
			if torch.isnan(torch.sum(state_action)):
					print('found nan in state')
					print(state_action)
					pdb.set_trace()

			if use_model:
				x_next = self.forward(state_action.squeeze())#.unsqueeze(1)
				if len(x_next.shape) < 3:		
					x_next = x_next.unsqueeze(1)
			else:
				x_next  = torch.einsum('jik,ljk->lji',[A,state_action[:,:,:policy_states_dim]])
				if not extra_dims_stable:
					x_next = add_irrelevant_features(x_next, states_dim-policy_states_dim, noise_level=0.4)
					x_next = x_next + 0.1*state_action[:,:,states_dim:]
				else:
					x_next = x_next + torch.einsum('ijk,kk->ijk',[state_action[:,:,states_dim:],B])

			with torch.no_grad():
				a = policy.sample_action(x_next[:,:,:states_dim])
				if isinstance(policy, DeterministicPolicy):
					a = torch.from_numpy(noise.get_action(a.numpy(), s))

			r = get_reward_fn(env, x_next[:,:,:policy_states_dim], a).unsqueeze(2) #this is where improvement happens when R_range = 1
			next_state_action = torch.cat((x_next, a),dim=2)
			
			############# for discrete, needs testing #######################
			#the dim of this could be wrong due to change to batch_size. NOT TESTED
			#next_state_action = torch.cat((x_next, convert_one_hot(a.double(), n_actions).unsqueeze(2)),dim=2)
			#################################################################
			#x_list[:10,0,-2:] all same
			a_list.append(a)
			r_list.append(r)
			x_list.append(x_next)

			state_action = next_state_action
	
		x_list = torch.cat(x_list, 2).view(batch_size, -1, steps_to_unroll+1, states_dim)
		a_list = torch.cat(a_list, 2).view(batch_size, -1, steps_to_unroll+1, actions_dim)
		r_list = torch.cat(r_list, 2).view(batch_size, -1, steps_to_unroll+1, 1)

		x_curr = x_list[:,:,:-1,:]
		x_next = x_list[:,:,1:,:]
		#print(x_next.contiguous().view(-1,2))
		a_used = a_list[:,:,:-1,:]
		a_prime = a_list[:,:,1:,:]
		r_used = r_list#[:,:,:-1,:]

		return x_curr, x_next, a_used, r_used, a_prime 

		# #only need rewards from model for the steps we've unrolled, the rest is assumed to be equal to the environment's
		# model_rewards[:, step] = get_reward_fn('lin_dyn', shortened, a)
		# model_log_probs = get_selected_log_probabilities(policy, shortened, a)
		#don't need states, just the log probabilities 

	
	def train_mle(self, pe, state_actions, states_next, epochs, max_actions, R_range, opt, env_name, losses, states_dim, salient_states_dim, file_location, file_id, save_checkpoints=False, verbose=10):
		best_loss = 1000

		for i in range(epochs):
			opt.zero_grad()

			squared_errors = torch.zeros_like(states_next)
			step_state = state_actions.to(device)

			for step in range(R_range - 1):
				next_step_state = self.forward(step_state)

				squared_errors += F.pad(input=(states_next[:,step:,:] - next_step_state)**2, pad=(0,0,step,0,0,0), mode='constant', value=0)

				shortened = next_step_state[:,:-1,:]
				# a = pe.sample_action(torch.DoubleTensor(shortened[:,:,:states_dim]))	
				a = state_actions[:, step+1:, states_dim:]
				step_state = torch.cat((shortened,a),dim=2)

			model_loss = torch.mean(squared_errors) #+ reward_loss)# + done_loss)

			if model_loss.detach().data.cpu() < best_loss and save_checkpoints:
				torch.save(self.state_dict(), os.path.join(file_location,'model_mle_checkpoint_state{}_salient{}_reinforce_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))
				best_loss = model_loss.detach().data.cpu()


			if i % verbose == 0:
				print("Epoch: {}, negloglik  = {:.7f}".format(i, model_loss.data.cpu()))

			model_loss.backward()
			opt.step()
			losses.append(model_loss.data.cpu())
		return model_loss


	def general_train_mle(self, pe, dataset, states_dim, salient_states_dim, epochs, max_actions, opt, env_name, losses, batch_size, file_location, file_id, save_checkpoints=False, verbose=20, lr_schedule=None):
		best_loss = 1000

		for i in range(epochs):
			#sample from dataset 
			batch = dataset.sample(batch_size)
			states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)
			
			step_state = torch.cat((states_prev, actions_tensor), dim=1).to(device)

			model_next_state = self.forward(step_state)
			squared_errors = (states_next - model_next_state)**2

			#step_state = torch.cat((next_step_state,a),dim=1)
			model_loss = torch.mean(squared_errors)

			if model_loss.detach().data.cpu() < best_loss and save_checkpoints:
				torch.save(self.state_dict(), os.path.join(file_location,'model_mle_checkpoint_state{}_salient{}_actorcritic_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, env_name, 1, max_actions + 1, file_id)))
				best_loss = model_loss.detach().data.cpu()

			if (i % verbose == 0) or (i == epochs - 1):
				print("R_range: {}, negloglik  = {:.7f}".format(1, model_loss.data.cpu()))

			opt.zero_grad()
			model_loss.backward()
			opt.step()
			if lr_schedule:
				lr_schedule.step()
			losses.append(model_loss.data.cpu())
		return model_loss