import numpy as np
import matplotlib.pyplot as plt
#import gym
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
# from get_data import *


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
		self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actions = (0, 1, 0, 1, 0, 1)
		if self.model_size == 'small':
			self.fc1 = nn.Linear(states_dim + N_ACTIONS, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
		elif self.model_size == 'constrained':
			# hidden_size = int(np.ceil(states_dim/2))
			self. fc1 = nn.Linear(states_dim + N_ACTIONS, hidden_size)
			self._enc_mu = torch.nn.Linear(hidden_size, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
			torch.nn.init.xavier_uniform_(self._enc_mu.weight)
		elif self.model_size == 'nn':
			self.fc1 = nn.Linear(states_dim + N_ACTIONS, 500)
			self.fc2 = nn.Linear(500, 500)
			self._enc_mu = torch.nn.Linear(500, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
			torch.nn.init.xavier_uniform_(self.fc2.weight)
			torch.nn.init.xavier_uniform_(self._enc_mu.weight)
		elif self.model_size == 'cnn':
			in_channels, w, h = states_dim
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
			# self.head = nn.Conv2d(32, h, )
		else:
			raise NotImplementedError

	def forward(self, x):
		if self.model_size == 'small':
			x = self.fc1(x)
			# mu = nn.Tanh()(x) * 8.0
			mu = x
		elif self.model_size == 'constrained':
			x = self._enc_mu(self.fc1(x))
			mu = x#nn.Tanh()(x) * 360.0
		elif self.model_size == 'nn':
			x = nn.ReLU()(self.fc1(x))
			x = self.fc2(x)
			x = nn.ReLU()(x)
			# x = self.fc3(x)
			# x = nn.ReLU()(x)
			mu = nn.Tanh()(self._enc_mu(x)) * 5.0 #100.0 #*3.0
		elif self.model_size == 'cnn':
			x = F.relu(self.bn1(self.conv1(x)))
			x = F.relu(self.bn2(self.conv2(x)))
			# x = F.relu(self.bn3(self.conv3(x)))
			pdb.set_trace()
			return self.head(x.view(x.size(0), -1))
		else:
			raise NotImplementedError
		return mu

	def unroll_small(self, env, state_action, policy, steps_to_unroll=1, salient_states_dim=0, noise=None):
		if state_action is None:
			return -1	
		batch_size = state_action.shape[0]
		max_actions = state_action.shape[1]
		actions_dim = state_action[0,0,self.states_dim:].shape[0]

		x0 = state_action[:,:,:states_dim]
		with torch.no_grad():
			a0 = policy.sample_action(x0[:,:,:policy_states_dim])
			if isinstance(policy, DeterministicPolicy):
				a0 = torch.from_numpy(noise.get_action(a0.numpy(), 0))

		pdb.set_trace()
		x0_norm = (x0 - self.mean_states) / (self.std_states + 1e-7)
		a0_norm = (a0 - self.mean_actions) / (self.std_actions + 1e-7)
		state_action_norm = torch.cat((x0_norm, a0_norm),dim=2)
		state_action = torch.cat((x0, a0),dim=2)

		x_list = [x0]
		a_list = [a0]

		for s in range(steps_to_unroll - 1):
			if torch.isnan(torch.sum(state_action)):
				print('found nan in state')
				print(state_action)
				pdb.set_trace()

			pdb.set_trace() #there might be a problem with dim in the function below given that I'm only applying it to one state and action in this case but function was written for tensors
			x_next = predict(state_action[:,:,:states_dim], state_action[:,:,states_dim:])#state_action_norm[:,:,:states_dim] + self.forward(state_action_norm.squeeze())#.unsqueeze(1)
			if len(x_next.shape) < 3:		
				x_next = x_next.unsqueeze(1)

			with torch.no_grad():
				a = policy.sample_action(x_next[:,:,:policy_states_dim])
				if isinstance(policy, DeterministicPolicy):
					a = torch.from_numpy(noise.get_action(a.numpy(), s))

			next_state_action = torch.cat((x_next, a),dim=2)

			a_list.append(a)
			x_list.append(x_next)
			state_action = next_state_action

		x_list = torch.cat(x_list, 2).view(batch_size, -1, steps_to_unroll+1, states_dim)
		a_list = torch.cat(a_list, 2).view(batch_size, -1, steps_to_unroll+1, actions_dim)

		x_curr = x_list[:,:,:-1,:]
		x_next = x_list[:,:,1:,:]
		a_used = a_list[:,:,:-1,:]
		a_prime = a_list[:,:,1:,:]
		
		return x_curr, x_next, a_used, a_prime 

	def unroll(self, state_action, policy, states_dim, A_numpy, steps_to_unroll=2, continuous_actionspace=True, use_model=True, policy_states_dim=2, extra_dims_stable=False, noise=None, epsilon=None, epsilon_decay=None,env='lin_dyn'):
		need_rewards = False
		if state_action is None:
			return -1	
		batch_size = state_action.shape[0]
		max_actions = state_action.shape[1]
		actions_dim = state_action[0,0,states_dim:].shape[0]
		
		if A_numpy is not None:
			A = torch.from_numpy(
				np.tile(
					#np.array([[0.9, 0.4], [-0.4, 0.9]]),
					A_numpy,
					(state_action[0,:,0].shape[0],1,1) #repeat along trajectory size, for every state
					)
				).to(device)
			extra_dim = states_dim - policy_states_dim
			B = torch.from_numpy(
				np.block([
						 [0.1*np.eye(policy_states_dim),            np.zeros((policy_states_dim, extra_dim))],
						 [np.zeros((extra_dim, policy_states_dim)), np.zeros((extra_dim,extra_dim))          ]
						])
				).to(device)
		elif A_numpy is None and not use_model:
			raise NotImplementedError

		x0 = state_action[:,:,:states_dim]
		with torch.no_grad():
			a0 = policy.sample_action(x0[:,:,:policy_states_dim])
			if isinstance(policy, DeterministicPolicy):
				# a0 += torch.from_numpy(noise()*max(0, epsilon)) #try without noise
				# a0 = torch.clamp(a0, min=-1.0, max=1.0)
				a0 = torch.from_numpy(noise.get_action(a0.numpy(), 0))

		state_action = torch.cat((x0, a0),dim=2)
		if use_model:	
			x_t_1 = self.forward(state_action.squeeze())#.unsqueeze(1)
			if len(x_t_1.shape) < 3:
				x_t_1 = x_t_1.unsqueeze(1)
		else:
			x_t_1  = torch.einsum('jik,ljk->lji',[A,x0[:,:,:policy_states_dim]])
			if not extra_dims_stable:
				x_t_1 = add_irrelevant_features(x_t_1, states_dim-policy_states_dim, noise_level=0.4)
				x_t_1 = x_t_1 + 0.1*a0#torch.einsum('ijk,kk->ijk',[a0,B])
			else:
				x_t_1 = x_t_1 + torch.einsum('ijk,kk->ijk',[a0,B])

		x_list = [x0, x_t_1]

		with torch.no_grad():
			a1 = policy.sample_action(x_list[1][:,:,:policy_states_dim])
			if isinstance(policy, DeterministicPolicy):
				# epsilon -= epsilon_decay
				# a1 += torch.from_numpy(noise()*max(0, epsilon)) #try without noise
				# a1 = torch.clamp(a1, min=-1.0, max=1.0)
				a1 = torch.from_numpy(noise.get_action(a1.numpy(), 1))

		a_list = [a0, a1]
		# r_list = [get_reward_fn('lin_dyn',x_list[0][:,:,:policy_states_dim], a0).unsqueeze(2), get_reward_fn('lin_dyn',x_t_1[:,:,:policy_states_dim], a1).unsqueeze(2)]
		if need_rewards:
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
					x_next = x_next + 0.1*state_action[:,:,states_dim:]#torch.einsum('ijk,kk->ijk',[state_action[:,:,states_dim:],B])#0.1*state_action[:,:,states_dim:]
				else:
					x_next = x_next + torch.einsum('ijk,kk->ijk',[state_action[:,:,states_dim:],B])#0.1*state_action[:,:,states_dim:]

			#action_taken = state_action[:,:,states_dim:]
			with torch.no_grad():
				a = policy.sample_action(x_next[:,:,:policy_states_dim])
				if isinstance(policy, DeterministicPolicy):
					# epsilon -= epsilon_decay
					# a += torch.from_numpy(noise()*max(0, epsilon)) #try without noise
					# a = torch.clamp(a, min=-1.0, max=1.0)
					a = torch.from_numpy(noise.get_action(a.numpy(), s))


			# r = get_reward_fn('lin_dyn', x_next[:,:,:policy_states_dim], a).unsqueeze(2) 
			if need_rewards:
				r = get_reward_fn(env, x_next[:,:,:policy_states_dim], a).unsqueeze(2) #this is where improvement happens when R_range = 1
			#r = get_reward_fn('lin_dyn', state_action[:,:,:states_dim], a)
			next_state_action = torch.cat((x_next, a),dim=2)
			
			############# for discrete, needs testing #######################
			#the dim of this could be wrong due to change to batch_size. NOT TESTED
			#next_state_action = torch.cat((x_next, convert_one_hot(a.double(), n_actions).unsqueeze(2)),dim=2)
			#################################################################
			#x_list[:10,0,-2:] all same
			a_list.append(a)
			if need_rewards:
				r_list.append(r)
			# if s == 5:
			# 	pdb.set_trace()
			x_list.append(x_next)

			state_action = next_state_action
		#x_list = torch.stack(x_list)		
		x_list = torch.cat(x_list, 2).view(batch_size, -1, steps_to_unroll+1, states_dim)
		a_list = torch.cat(a_list, 2).view(batch_size, -1, steps_to_unroll+1, actions_dim)
		if need_rewards:
			r_list = torch.cat(r_list, 2).view(batch_size, -1, steps_to_unroll+1, 1)

		#NOT true: there is no gradient for P_hat from here, with R_range = 1, the first state is from true dynamics so doesn't have grad
		x_curr = x_list[:,:,:-1,:]
		x_next = x_list[:,:,1:,:]
		#print(x_next.contiguous().view(-1,2))
		a_used = a_list[:,:,:-1,:]
		a_prime = a_list[:,:,1:,:]
		if need_rewards:
			r_used = r_list#[:,:,:-1,:]
		else:
			r_used = None
		return x_curr, x_next, a_used, r_used, a_prime 

		# #only need rewards from model for the steps we've unrolled, the rest is assumed to be equal to the environment's
		# model_rewards[:, step] = get_reward_fn('lin_dyn', shortened, a)
		# model_log_probs = get_selected_log_probabilities(policy, shortened, a)
		#don't need states, just the log probabilities 

	def actor_critic_paml_train(
							self, 
							actor, 
							critic,
							env,
							noise,
							epsilon,
							# target_critic,
							# q_optimizer,
							opt, 
							num_episodes, 
							num_starting_states, 
							batch_size,
							states_dim, 
							salient_states_dim,
							actions_dim, 
							use_model, 
							discount, 
							max_actions, 
							planning_horizon,
							train, 
							lr,
							num_iters,
							losses,
							dataset,
							verbose,
							save_checkpoints,
							file_location,
							file_id
							):
		best_loss = 15000
		env_name = env.spec.id
		# batch_size = 64
		R_range = planning_horizon
		unroll_num = max_actions
		end_of_trajectory = 1
		num_iters = num_iters if train else 1
		MSE = nn.MSELoss()
		noise.reset()
		# noise = OUNoise(env.action_space)
		# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))	
		model_opt = optim.SGD(self.parameters(), lr=lr)#, momentum=0.90, nesterov=True)

		# self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actions = compute_normalization(dataset)
		# epsilon = 1
		# epsilon_decay = 1./100000
		ell = 0
		saved = False
		for i in range(num_iters):
			batch = dataset.sample(batch_size)

			true_states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			true_states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			true_rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
			true_actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

			#calculate true gradients
			actor.zero_grad()

			#compute loss for actor
			true_policy_loss = -critic(true_states_next, actor.sample_action(true_states_next[:,:salient_states_dim]))
			true_term = true_policy_loss.mean()

			# save_stats(None, true_rewards_tensor, None, true_states_next, value=None, prefix='true_actorcritic_')
			true_pe_grads = torch.DoubleTensor()
			true_pe_grads_attached = grad(true_term.mean(), actor.parameters(), create_graph=True)
			for g in range(len(true_pe_grads_attached)):
				true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

			#probably don't need these ... .grad is not accumulated with the grad() function
			actor.zero_grad()
			step_state = torch.cat((true_states_prev, true_actions_tensor),dim=1).unsqueeze(1)
		
			if use_model:
				# model_x_curr, model_x_next, model_a_list, model_r_list, _ = P_hat.unroll(step_state, actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=salient_states_dim, noise=noise, epsilon=epsilon, epsilon_decay=None, env=env)
				model_x_curr, model_x_next, model_a_list, _, _ = self.unroll(step_state, actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=salient_states_dim, noise=noise, epsilon=epsilon, epsilon_decay=None, env=env)

				#can remove these after unroll is fixed
				model_x_curr = model_x_curr.squeeze(1).squeeze(1)
				model_x_next = model_x_next.squeeze(1).squeeze(1)
				#model_r_list = model_r_list.squeeze()[:,1:]
				model_a_list = model_a_list.squeeze(1).squeeze(1)
				##### DO ACTIONS ABOVE MESS WITH P_HAT GRADIENTS?!
			else:
				model_batch = dataset.sample(batch_size)
				model_x_curr = torch.tensor([samp.state for samp in model_batch]).double().to(device)
				model_x_next = torch.tensor([samp.next_state for samp in model_batch]).double().to(device)
				# model_r_list = torch.tensor([samp.reward for samp in model_batch]).double().to(device).unsqueeze(1)
				model_a_list = torch.tensor([samp.action for samp in model_batch]).double().to(device)

			model_policy_loss = -critic(model_x_next, actor.sample_action(model_x_next[:,:salient_states_dim]))
			model_term = model_policy_loss.mean()
			# save_stats(None, model_r_list, model_a_list, model_x_next, value=None, prefix='model_actorcritic_')
			model_pe_grads = torch.DoubleTensor()
			model_pe_grads_split = grad(model_term.mean(),actor.parameters(), create_graph=True)
			for g in range(len(model_pe_grads_split)):
				model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))
			# cos = nn.CosineSimilarity(dim=0, eps=1e-6)
			# loss = 1-cos(true_pe_grads,model_pe_grads)
			loss = torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum())#/num_starting_states)


			if loss.detach().cpu() < best_loss and use_model:
				#Save model and losses so far
				#if save_checkpoints:
				torch.save(self.state_dict(), os.path.join(file_location,'act_model_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)))
				saved = True
				if save_checkpoints:
					np.save(os.path.join(file_location,'act_loss_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), np.asarray(losses))

				best_loss = loss.detach().cpu()

			#update model
			if train: 
				opt.zero_grad()

				loss.backward()
				nn.utils.clip_grad_value_(self.parameters(), 5.0)
				opt.step()

				if torch.isnan(torch.sum(self.fc1.weight.data)):
					print('weight turned to nan, check gradients')
					pdb.set_trace()

			losses.append(loss.data.cpu())
			# lr_schedule.step()

			if train and i < 1: #and j < 1 
				initial_loss = losses[0]#.data.cpu()
				print('initial_loss',initial_loss)

			if train:
				if (i % verbose == 0) or (i == num_iters - 1):
					print("LR: {:.5f} | batch_num: {:5d} | critic ex val: {:.3f} | paml_loss: {:.5f}".format(lr, i, true_policy_loss.mean().data.cpu(), loss.data.cpu()))
			else: 
				print("-----------------------------------------------------------------------------------------------------")
				print("Validation loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}".format(use_model, R_range, i, loss.data.cpu()))
				print("-----------------------------------------------------------------------------------------------------")
		
		if train and saved:
			self.load_state_dict(torch.load(os.path.join(file_location,'act_model_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), map_location=device))

		return loss.data.cpu()

	#next two methods should be combined
	def mle_validation_loss(self, A_numpy, states_next, state_actions, policy, R_range, use_model=True, salient_dims=2):
		with torch.no_grad():
			continuous_actionspace = policy.continuous
			squared_errors = torch.zeros_like(states_next)
			step_state = state_actions.to(device)
			states_dim = states_next.shape[-1]

			for step in range(R_range - 1):
				
				if use_model:
					next_step_state = self.forward(step_state)
				else:
					A = torch.from_numpy(
						np.tile(
							A_numpy,
							(step_state[0,:,0].shape[0],1,1) #repeat along trajectory size, for every state
							)
						)
					next_step_state = torch.einsum('jik,ljk->lji',[A,step_state[:,:,:salient_dims]]) + step_state[:,:,states_dim:]
					#STILL NEED TO ADD IRRELEVANT FEATURES AND GET THESE RANDOM 5 and 2's out of here
				try:
					squared_errors += F.pad(input=(states_next[:,step:,:] - next_step_state)**2, pad=(0,0,step,0,0,0), mode='constant', value=0)
				except:
					pdb.set_trace()

				#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)

				shortened = next_step_state[:,:-1,:]
				if shortened.nelement() > 0:
					a = policy.sample_action(torch.DoubleTensor(shortened[:,:,:states_dim]))	
					step_state = torch.cat((shortened,a),dim=2)

			#state_loss = torch.mean(squared_errors)#torch.mean((states_next - rolled_out_states_sums)**2)
			model_loss = torch.mean(squared_errors) #+ reward_loss)# + done_loss)
			#print("R_range: {}, negloglik  = {:.7f}".format(R_range, model_loss.data.cpu()))
			
		return squared_errors.mean()
	
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


	def general_train_mle(self, pe, dataset, states_dim, salient_states_dim, epochs, max_actions, opt, env_name, losses, batch_size, file_location, file_id, save_checkpoints=False, verbose=20, lr_schedule=None, global_step=0):
		best_loss = 1000

		self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actions = compute_normalization(dataset)

		for i in range(epochs):
			#sample from dataset 
			batch = dataset.sample(batch_size)
			states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

			# normalize states and actions
			states_norm = (states_prev - self.mean_states) / (self.std_states + 1e-7)
			acts_norm = (actions_tensor - self.mean_actions) / (self.std_actions + 1e-7)
			# normalize the state differences
			deltas_states_norm = ((states_next - states_prev) - self.mean_deltas) / (self.std_deltas + 1e-7)
			step_state = torch.cat((states_norm, acts_norm), dim = 1).to(device)

			# step_state = torch.cat((states_prev, actions_tensor), dim=1).to(device)
			model_next_state_delta = self.forward(step_state)
			
			# squared_errors = ((states_next - states_prev) - model_next_state_delta)**2
			squared_errors = (deltas_states_norm - model_next_state_delta)**2
			# try:
			# save_stats(None, states_next, actions_tensor, states_prev, prefix='true_MLE_training_')
			# save_stats(None, model_next_state, actions_tensor, states_prev, prefix='model_MLE_training_')
			# except KeyboardInterrupt:
			# 	print("W: interrupt received, stopping…")
			# 	sys.exit(0)
			# with torch.no_grad():
			# 	a = pe.sample_action(torch.DoubleTensor(next_step_state))	
			# 	if isinstance(pe, DeterministicPolicy):
			# 		a += torch.from_numpy(noise()*max(0, epsilon)) #try without noise
			# 		a = torch.clamp(a, min=-1.0, max=1.0)

			#step_state = torch.cat((next_step_state,a),dim=1)
			model_loss = torch.mean(torch.sum(squared_errors,dim=1))

			if model_loss.detach().data.cpu() < best_loss and save_checkpoints:
				torch.save(self.state_dict(), os.path.join(file_location,'model_mle_checkpoint_state{}_salient{}_actorcritic_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, env_name, 1, max_actions + 1, file_id)))
				best_loss = model_loss.detach().data.cpu()

			if (i % verbose == 0) or (i == epochs - 1):
				print()	
				print("Iter: {} | LR: {:.7f} | salient_loss: {:.4f} | irrelevant_loss: {:.4f} | negloglik  = {:.7f}".format(i+epochs*global_step, opt.param_groups[0]['lr'], torch.mean(torch.sum(squared_errors[:,:salient_states_dim],dim=1)).detach().data.cpu(), torch.mean(torch.sum(squared_errors[:,salient_states_dim:],dim=1)).detach().data.cpu(), model_loss.data.cpu()))

			opt.zero_grad()
			model_loss.backward()
			opt.step()
			losses.append(model_loss.data.cpu())
		if lr_schedule is not None:
			lr_schedule.step()
		return model_loss

	def predict(self, states, actions):
		# normalize the states and actions
		states_norm = (states - self.mean_states) / (self.std_states + 1e-7)
		act_norm = (actions - self.mean_actions) / (self.std_actions + 1e-7)

		# concatenate normalized states and actions
		states_act_norm = torch.cat((states_norm, act_norm), dim=1)

		# predict the deltas between states and next states
		deltas = self.forward(states_act_norm)

		# calculate the next states using the predicted delta values and denormalize
		return deltas * self.std_deltas + self.mean_deltas + states