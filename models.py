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
from networks import *
# from get_data import *


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class DirectEnvModel(torch.nn.Module):
	def __init__(self, states_dim, N_ACTIONS, MAX_TORQUE, mult=0.1, model_size='nn', hidden_size=2, limit_output=False):
		super(DirectEnvModel, self).__init__()
		# build network layers
		self.states_dim = states_dim
		self.n_actions = N_ACTIONS
		self.max_torque = MAX_TORQUE
		self.mult = mult
		self.model_size = model_size
		self.limit_output = limit_output
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
			self.fc1 = nn.Linear(states_dim + N_ACTIONS, 512) #512 good for halfcheetah
			self.fc2 = nn.Linear(512, 512)
			self._enc_mu = torch.nn.Linear(512, states_dim)
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
			if self.limit_output:
				mu = nn.Tanh()(x) * 3.0
			else:
				mu = x

		elif self.model_size == 'constrained':
			mu = self._enc_mu(self.fc1(x))
			#mu = x#nn.Tanh()(x) * 360.0

		elif self.model_size == 'nn':
			x = nn.ReLU()(self.fc1(x))
			# x = self.fc2(x)
			x = nn.ReLU()(self.fc2(x))
			# x = self.fc3(x)
			# x = nn.ReLU()(x)
			if self.limit_output:
				mu = nn.Tanh()(self._enc_mu(x)) * 8.0 #100.0 #*3.0
			else:
				mu = self._enc_mu(x) #nn.Tanh()(self._enc_mu(x)) * 5.0 #100.0 #*3.0

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
		need_rewards = False if env != 'lin_dyn' else True
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
				a0 = torch.from_numpy(noise.get_action(a0.numpy(), 0, multiplier=1.))

		state_action = torch.cat((x0, a0),dim=2)
		if use_model:	
			#USING delta 
			x_t_1 = x0.squeeze() + self.forward(state_action.squeeze())#.unsqueeze(1)
			# x_t_1 = self.forward(state_action.squeeze())#.unsqueeze(1)
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
				a1 = torch.from_numpy(noise.get_action(a1.numpy(), 1, multiplier=1.))

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
				#USING delta
				x_next = state_action[:,:,:states_dim].squeeze() + self.forward(state_action.squeeze())#.unsqueeze(1)
				# x_next = self.forward(state_action.squeeze())#.unsqueeze(1)
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
					a = torch.from_numpy(noise.get_action(a.numpy(), s, multiplier=1.))


			# r = get_reward_fn('lin_dyn', x_next[:,:,:policy_states_dim], a).unsqueeze(2) 
			if need_rewards:
				# r = get_reward_fn(env, x_next[:,:,:policy_states_dim], a).unsqueeze(2) #this is where improvement happens when R_range = 1
				r = get_reward_fn(env, x_next[:,:,:2], a).unsqueeze(2) 
				# r = get_reward_fn('lin_dyn', state_action[:,:,:states_dim], a)
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
							lr_schedule,
							num_iters,
							losses,
							dataset,
							verbose,
							save_checkpoints,
							file_location,
							file_id,
							norms_true_pe_grads,
							norms_model_pe_grads
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
		prev_grad = torch.zeros(count_parameters(self)).double()
		lambda_mle = 0.25
		# noise = OUNoise(env.action_space)
		# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))	
		
		# model_opt = optim.SGD(self.parameters(), lr=lr)#, momentum=0.90, nesterov=True)

		# self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actionss = compute_normalization(dataset)
		# epsilon = 1
		# epsilon_decay = 1./100000
		ell = 0
		saved = False
		cos_grads = []
		for i in range(num_iters):
			batch = dataset.sample(batch_size)

			#HAVE TO MAKE THIS GRADIENT ESTIMATE COME FROM MULTIPLE next states from the same states_prev 
			#think about how to do this here
			true_states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)

			# #I could do a batch of just 1. Then unroll M times. 
			# M = 1
			# torch.autograd.set_detect_anomaly(True)
			# # true_states_next = torch.zeros((batch_size, M, states_dim)).double()
			# true_actions_np = np.zeros((batch_size, M, actions_dim))
			# # true_full_states = [samp.full_state for samp in batch]
			# true_pe_grads = torch.zeros((batch_size, count_parameters(actor))).double()
			
			# for b in range(batch_size):
			# 	true_states_next = torch.zeros((M, states_dim)).double()
			# 	env.state = true_states_prev[b]
			# 	# env.env.set_state(true_full_states[b][:9],true_full_states[b][9:])
			# 	for i in range(M):
			# 		true_actions_np[b][i] = actor.sample_action(true_states_prev[b]).detach().numpy()+np.random.normal(scale=0.3,size=1)
			# 		true_states_next[i] = torch.tensor(env.step(true_actions_np[b][i])[0]).double().to(device)
				
			# 	actor.zero_grad()
			# 	true_policy_loss = -critic(true_states_next, actor.sample_action(true_states_next)).mean()
			# 	true_pe_grads_attached = grad(true_policy_loss, actor.parameters(), create_graph=True)
			# 	true_pe_grads_ = torch.DoubleTensor()
			# 	for g in range(len(true_pe_grads_attached)):
			# 		true_pe_grads_ = torch.cat((true_pe_grads_,true_pe_grads_attached[g].detach().view(-1)))
			# 	true_pe_grads[b] = true_pe_grads_

			true_states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			# true_rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
			true_actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

			#calculate true gradients
			actor.zero_grad()

			#compute loss for actor
			# true_policy_loss = -critic(true_states_next[:,:salient_states_dim], actor.sample_action(true_states_next[:,:salient_states_dim]))
			# true_policy_loss = -critic(true_states_next, actor.sample_action(true_states_next[:,:salient_states_dim]))
			# true_policy_loss = -critic(true_states_next[:,:salient_states_dim], actor.sample_action(true_states_next))
			true_policy_loss = -critic(true_states_next, actor.sample_action(true_states_next))

			true_term = true_policy_loss.mean()
			# save_stats(None, true_rewards_tensor, None, true_states_next, value=None, prefix='true_actorcritic_')
			
			true_pe_grads = torch.DoubleTensor()
			true_pe_grads_attached = grad(true_term.mean(), actor.parameters(), create_graph=True)
			for g in range(len(true_pe_grads_attached)):
				true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

			#probably don't need these ... .grad is not accumulated with the grad() function
			# actor.zero_grad() 
			#comment the line below out after 
			# true_actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)
			step_state = torch.cat((true_states_prev, true_actions_tensor),dim=1).unsqueeze(1)
			if use_model:
				# model_x_curr, model_x_next, model_a_list, model_r_list, _ = P_hat.unroll(step_state, actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=salient_states_dim, noise=noise, epsilon=epsilon, epsilon_decay=None, env=env)
				model_x_curr, model_x_next, model_a_list, _, _ = self.unroll(step_state, actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=states_dim, noise=noise, epsilon=epsilon, epsilon_decay=None, env=env)#salient_states_dim, noise=noise, epsilon=epsilon, epsilon_decay=None, env=env)
				#can remove these after unroll is fixed
				model_x_curr = model_x_curr.squeeze(1).squeeze(1)
				model_x_next = model_x_next.squeeze(1).squeeze(1)
				# print('next states norms:', model_x_next.norm().detach().data, true_states_next.norm().detach().data)
				# print(true_states_next)
				# pdb.set_trace()
				#model_r_list = model_r_list.squeeze()[:,1:]
				model_a_list = model_a_list.squeeze(1).squeeze(1)
				# print('actions norms:', model_a_list.norm().detach().data, true_actions_tensor.norm().detach().data)
				##### DO ACTIONS ABOVE MESS WITH P_HAT GRADIENTS?!
			else:
				model_batch = dataset.sample(batch_size)
				model_x_curr = torch.tensor([samp.state for samp in model_batch]).double().to(device)
				model_x_next = torch.tensor([samp.next_state for samp in model_batch]).double().to(device)
				# model_r_list = torch.tensor([samp.reward for samp in model_batch]).double().to(device).unsqueeze(1)
				model_a_list = torch.tensor([samp.action for samp in model_batch]).double().to(device)

			# model_policy_loss = -critic(model_x_next[:,:salient_states_dim], actor.sample_action(model_x_next))
			# model_policy_loss = -critic(model_x_next[:,:salient_states_dim], actor.sample_action(model_x_next[:,:salient_states_dim]))
			# model_policy_loss = -critic(model_x_next, actor.sample_action(model_x_next[:,:salient_states_dim]))
			model_policy_loss = -critic(model_x_next, actor.sample_action(model_x_next))

			model_term = model_policy_loss.mean()

			model_pe_grads = torch.DoubleTensor()
			model_pe_grads_split = grad(model_term.mean(),actor.parameters(), create_graph=True)
			for g in range(len(model_pe_grads_split)):
				model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))
			
			# model_pe_grads = torch.zeros((batch_size, count_parameters(actor))).double()
			# for b in range(batch_size):
			# 	actor.zero_grad()
			# 	model_pe_grads_ = torch.DoubleTensor()
			# 	model_pe_grads_split = grad(model_policy_loss[b], actor.parameters(), create_graph=True)
			# 	# model_pe_grads_split = grad(model_term.mean(),actor.parameters(), create_graph=True)
			# 	for g in range(len(model_pe_grads_split)):
			# 		model_pe_grads_ = torch.cat((model_pe_grads_,model_pe_grads_split[g].view(-1)))
			# 	model_pe_grads[b] = model_pe_grads_

			cos = nn.CosineSimilarity(dim=0, eps=1e-6)
			# loss = MSE(true_pe_grads, model_pe_grads)
			# loss = (1-cos(true_pe_grads,model_pe_grads)).mean()
			# loss = MSE(model_policy_loss, true_policy_loss)
			# loss_grads_split = grad(MSE(true_policy_loss,model_policy_loss), actor.parameters(), create_graph = True) 
			# loss_grads = torch.DoubleTensor()
			# for g in range(len(loss_grads_split)):
			# 	loss_grads = torch.cat((loss_grads,loss_grads_split[g].view(-1)))
			# pdb.set_trace()
			# loss = (loss_grads / (2*(true_policy_loss - model_policy_loss))).sum(dim=1).mean()
			

			# norms_true_pe_grads.append(true_pe_grads.norm().data.cpu())
			# norms_model_pe_grads.append(model_pe_grads.norm().data.cpu())
			#mle_loss = torch.mean(torch.sum(((model_x_next - model_x_curr)-(true_states_next - true_states_prev))**2,dim=1))
			paml_loss = torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum())
			# print(cos(true_pe_grads, model_pe_grads).detach().data.cpu())
			loss = paml_loss #+ lambda_mle * mle_loss #1 - cos(true_pe_grads, model_pe_grads)#MSE(true_pe_grads,model_pe_grads) + torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum())#dim=1).mean())#/num_starting_states) 1-cos(true_pe_grads, model_pe_grads)# torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum()) #
			# if loss < 0.3:
			# 	return
			
			# if loss.detach().cpu() < best_loss and use_model:
				#Save model and losses so far
				#if save_checkpoints:
			# torch.save(self.state_dict(), os.path.join(file_location,'act_model_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)))
			# saved = True
			# if save_checkpoints:

			# np.save(os.path.join(file_location,'act_loss_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), np.asarray(losses))

			# np.save(os.path.join(file_location,'model_pe_grad_norms_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), np.asarray(norms_model_pe_grads))

			# np.save(os.path.join(file_location,'true_pe_grad_norms_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), np.asarray(norms_true_pe_grads))
			
				# best_loss = loss.detach().cpu()

			#update model
			if train: 
				opt.zero_grad()
				loss.backward()
				
				curr_grad = torch.DoubleTensor()
				for item in list(self.parameters()):
					curr_grad = torch.cat((curr_grad,item.grad.view(-1)))

				# print(curr_grad)
				# print(t)
				# print(cos(true_pe_grads, model_pe_grads))
				# print(true_pe_grads.norm())
				# print(model_pe_grads.norm())
				nn.utils.clip_grad_value_(self.parameters(), 10.0)
				# pdb.set_trace()
				opt.step()

				if torch.isnan(torch.sum(self.fc1.weight.data)):
					print('weight turned to nan, check gradients')
					pdb.set_trace()

				if (i % verbose == 0) or (i == num_iters - 1):
					print("LR: {:.5f} | batch_num: {:5d} | COS grads: {:.5f} | critic ex val: {:.3f} | paml_loss: {:.5f}".format(opt.param_groups[0]['lr'], i, nn.CosineSimilarity(dim=0)(curr_grad, prev_grad), true_policy_loss.mean().data.cpu(), loss.data.cpu()))
					# cos_grads.append(nn.CosineSimilarity(dim=0)(curr_grad, prev_grad))
				prev_grad = curr_grad
				# lr_schedule.step()

			# if train and i < 1: #and j < 1 
			# 	initial_loss = losses[0]#.data.cpu()
			# 	print('initial_loss',initial_loss)
			else: 
				print("-----------------------------------------------------------------------------------------------------")
				print("Validation loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}".format(use_model, R_range, i, loss.data.cpu()))
				print("-----------------------------------------------------------------------------------------------------")
			
			# losses.append(loss.data.cpu())

		# if train and sum(cos_grads)/len(cos_grads) < 0.1:
		lr_schedule.step()

		# if train and saved:
		# 	self.load_state_dict(torch.load(os.path.join(file_location,'act_model_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), map_location=device))
		torch.save(self.state_dict(), os.path.join(file_location,'model_paml_checkpoint_state{}_salient{}_actorcritic_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, env_name, planning_horizon, max_actions+1, file_id)))

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


	def general_train_mle(self, pe, dataset, validation_dataset, states_dim, salient_states_dim, epochs, max_actions, opt, env_name, losses, batch_size, file_location, file_id, save_checkpoints=False, verbose=20, lr_schedule=None, global_step=0):
		best_loss = 1000

		# self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actions = compute_normalization(dataset)

		# val_mean_states, val_std_states, val_mean_deltas, val_std_deltas, val_mean_actions, val_std_actions = compute_normalization(validation_dataset)		

		val_model_losses = [np.inf]
		for i in range(epochs):
			#sample from dataset 
			batch = dataset.sample(batch_size)
			states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

			# normalize states and actions
			states_norm = states_prev #(states_prev - self.mean_states) / (self.std_states + 1e-7)
			acts_norm = actions_tensor #(actions_tensor - self.mean_actions) / (self.std_actions + 1e-7)
			# normalize the state differences
			# deltas_states_norm = ((states_next - states_prev) - self.mean_deltas) / (self.std_deltas + 1e-7)
			step_state = torch.cat((states_norm, acts_norm), dim = 1).to(device)

			# step_state = torch.cat((states_prev, actions_tensor), dim=1).to(device)
			model_next_state_delta = self.forward(step_state)

			# squared_errors = ((states_next - states_prev) - model_next_state_delta)**2
			deltas_states_norm = states_next - states_prev
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
				#calculate validation loss 	
				val_batch = validation_dataset.sample(len(validation_dataset))
				val_states_prev = torch.tensor([samp.state for samp in val_batch]).double().to(device)
				val_states_next = torch.tensor([samp.next_state for samp in val_batch]).double().to(device)
				val_actions_tensor = torch.tensor([samp.action for samp in val_batch]).double().to(device)

				# normalize states and actions
				val_states_norm = val_states_prev#(val_states_prev - val_mean_states) / (val_std_states + 1e-7)
				val_acts_norm = val_actions_tensor#(val_actions_tensor - val_mean_actions) / (val_std_actions + 1e-7)
				# normalize the state differences
				val_deltas_states_norm = val_states_next - val_states_prev#((val_states_next - val_states_prev) - val_mean_deltas) / (val_std_deltas + 1e-7)
				val_step_state = torch.cat((val_states_norm, val_acts_norm), dim = 1).to(device)

				with torch.no_grad():
					val_model_next_state_delta = self.forward(val_step_state)
				
				val_squared_errors = (val_deltas_states_norm - val_model_next_state_delta)**2
				val_model_losses.append(torch.mean(torch.sum(val_squared_errors,dim=1)).data)

				if len(val_model_losses) >= 25 and val_model_losses[-1] >= (val_model_losses[1] + val_model_losses[1]*0.05):
					return model_loss	
				elif len(val_model_losses) >= 25:
					val_model_losses = val_model_losses[1:]
					# pass

				print("Iter: {} | LR: {:.7f} | salient_loss: {:.4f} | irrelevant_loss: {:.4f} | negloglik  = {:.7f} | validation loss : {:.7f}".format(i+epochs*global_step, opt.param_groups[0]['lr'], torch.mean(torch.sum(squared_errors[:,:salient_states_dim],dim=1)).detach().data.cpu(), torch.mean(torch.sum(squared_errors[:,salient_states_dim:],dim=1)).detach().data.cpu(), model_loss.data.cpu(),val_model_losses[-1]))

			opt.zero_grad()
			model_loss.backward()
			# nn.utils.clip_grad_value_(self.parameters(), 100.0)
			opt.step()
			# losses.append(model_loss.data.cpu())
		if lr_schedule is not None:
			lr_schedule.step()

		del val_squared_errors, val_model_losses

		return model_loss

	def predict(self, states, actions):
		# normalize the states and actions
		states_norm = states#(states - self.mean_states) / (self.std_states + 1e-7)
		act_norm = actions#(actions - self.mean_actions) / (self.std_actions + 1e-7)

		# concatenate normalized states and actions
		states_act_norm = torch.cat((states_norm, act_norm), dim=1)

		# predict the deltas between states and next states
		deltas = self.forward(states_act_norm)

		# calculate the next states using the predicted delta values and denormalize
		return deltas + states_norm #deltas * self.std_deltas + self.mean_deltas + states