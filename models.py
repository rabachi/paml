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

class DirectEnvModel(torch.nn.Module):
	def __init__(self, states_dim, N_ACTIONS, MAX_TORQUE):
		super(DirectEnvModel, self).__init__()
		# build network layers
		self.states_dim = states_dim
		self.n_actions = N_ACTIONS
		self.max_torque = MAX_TORQUE

		self.fc1 = nn.Linear(states_dim + N_ACTIONS, 32)
		self.fc2 = nn.Linear(32, 16)
		#self.fc3 = nn.Linear(32, 16)

		self._enc_mu = torch.nn.Linear(16, states_dim)

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
		self.state = probs.sample(torch.zeros(self.states_dim).size())
		self.steps_beyond_done = None
		return self.state.type(torch.DoubleTensor)

	def forward(self, x):
		x = self.fc1(x)	
		x = nn.ReLU()(x)
		x = self.fc2(x)
		x = nn.ReLU()(x)
		# x = self.fc3(x)
		# x = nn.ReLU()(x)

		mu = nn.Tanh()(self._enc_mu(x)) * 3.0

		return mu


	def unroll(self, state_action, policy, states_dim, steps_to_unroll=2, continuous_actionspace=True):
		if state_action is None:
			return -1	
		batch_size = state_action.shape[0]
		max_actions = state_action.shape[1]
		actions_dim = state_action[0,0,:].shape[0] - states_dim
		
		A = torch.from_numpy(
			np.tile(
				np.array([[0.9, 0.4], [-0.4, 0.9]]),
				(state_action[0,:,0].shape[0],1,1) #repeat along trajectory size, for every state
					)
				)
 		
		#x_t_1 = self.forward(state_action)
		with torch.no_grad():
			a0 = policy.sample_action(state_action[:,:,:states_dim])

		x_t_1  = torch.einsum('jik,ljk->lji',[A,state_action[:,:,:states_dim]]) + a0

		x_list = [state_action[:,:,:states_dim], x_t_1]

		with torch.no_grad():
			a1 = policy.sample_action(x_list[1])

		a_list = [a0, a1]
		r_list = [get_reward_fn('lin_dyn',x_list[0], a0).unsqueeze(2),get_reward_fn('lin_dyn',x_t_1, a1).unsqueeze(2)]

		state_action = torch.cat((x_t_1, a1),dim=2)

		for s in range(steps_to_unroll - 1):
			if torch.isnan(torch.sum(state_action)):
					print('found nan in state')
					print(state_action)
					pdb.set_trace()

			#x_next = self.forward(state_action)
			x_next = torch.einsum('jik,ljk->lji',[A,state_action[:,:,:states_dim]]) + state_action[:,:,states_dim:]

			#action_taken = state_action[:,:,states_dim:]
			with torch.no_grad():
				a = policy.sample_action(x_next)

			r = get_reward_fn('lin_dyn', x_next, a).unsqueeze(2) #this is where improvement happens when R_range = 1
			#r = get_reward_fn('lin_dyn', state_action[:,:,:states_dim], a)
			next_state_action = torch.cat((x_next, a),dim=2)
			
			############# for discrete, needs testing #######################
			#the dim of this could be wrong due to change to batch_size. NOT TESTED
			#next_state_action = torch.cat((x_next, convert_one_hot(a.double(), n_actions).unsqueeze(2)),dim=2)
			#################################################################

			a_list.append(a)
			r_list.append(r)
			x_list.append(x_next)
			state_action = next_state_action
		#x_list = torch.stack(x_list)			
		x_list = torch.cat(x_list, 2).view(batch_size, -1, steps_to_unroll+1, states_dim)
		a_list = torch.cat(a_list, 2).view(batch_size, -1, steps_to_unroll+1, actions_dim)
		r_list = torch.cat(r_list, 2).view(batch_size, -1, steps_to_unroll+1, 1)

		#NOT true: there is no gradient for P_hat from here, with R_range = 1, the first state is from true dynamics so doesn't have grad
		x_curr = x_list[:,:,:-1,:]
		x_next = x_list[:,:,1:,:]
		#print(x_next.contiguous().view(-1,2))
		a_used = a_list[:,:,:-1,:]
		r_used = r_list[:,:,:-1,:]

		return x_curr, x_next, a_used, r_used

		# #only need rewards from model for the steps we've unrolled, the rest is assumed to be equal to the environment's
		# model_rewards[:, step] = get_reward_fn('lin_dyn', shortened, a)
		# model_log_probs = get_selected_log_probabilities(policy, shortened, a)
		#don't need states, just the log probabilities 

	def paml(self, 
			pe, 
			train,
			env_name, 
			dataset,
			device,
			num_starting_states,
			num_episodes,
			batch_size, 
			max_actions, 
			states_dim, 
			actions_dim,
			num_states,
			R_range,
			discount,
			true_pe_grads_file, 
			model_pe_grads_file,
			losses,
			opt,
			opt_steps,
			num_iters,
			lr_schedule
		):

		unroll_num = num_states - R_range

		best_loss = 20
		val_loss = 0
		continuous_actionspace = pe.continuous

		num_iters = num_iters if train else num_starting_states
		opt_steps = opt_steps if train else 1
		# dataset = ReplayMemory(500000)
		# x_0 = 2*np.random.random(size=(2,)) - 0.5

		# for ep in range(num_episodes):
		# 		x_tmp, x_next_tmp, u_list, _, r_list = lin_dyn(max_actions, pe, [], x=x_0, discount=0.0)
		# 		for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_list):
		# 			dataset.push(x, x_next, u, r)

		for i in range(num_iters):
			start_at = None if train else i
			#assuming samples are disjoint
			batch = dataset.sample(batch_size, structured=True, max_actions=max_actions, num_episodes_per_start=num_episodes, num_starting_states=num_starting_states, start_at=start_at)
			states_prev = torch.zeros((batch_size, max_actions, states_dim)).double().to(device)
			states_next = torch.zeros((batch_size, max_actions, states_dim)).double().to(device)
			rewards = torch.zeros((batch_size, max_actions)).double().to(device)
			actions_tensor = torch.zeros((batch_size, max_actions, actions_dim)).double().to(device)
			discounted_rewards_tensor = torch.zeros((batch_size, max_actions, 1)).double().to(device)

			for b in range(batch_size):
				try:
					states_prev[b] = torch.tensor([samp.state for samp in batch[b]]).double().to(device)
				except:
					pdb.set_trace()
				states_next[b] = torch.tensor([samp.next_state for samp in batch[b]]).double().to(device)
				rewards[b] = torch.tensor([samp.reward for samp in batch[b]]).double().to(device)
				actions_tensor[b] = torch.tensor([samp.action for samp in batch[b]]).double().to(device)
				discounted_rewards_tensor[b] = discount_rewards(rewards[b].unsqueeze(1), discount, center=False).to(device)
			state_actions = torch.cat((states_prev,actions_tensor), dim=2)

			pe.zero_grad()
			true_log_probs_t = get_selected_log_probabilities(pe, states_prev.view(-1,states_dim), actions_tensor.view(-1,actions_dim)).view(batch_size,max_actions,actions_dim)

			true_term = torch.zeros((batch_size,unroll_num, actions_dim))
			#alt_true_term = torch.zeros((batch_size, unroll_num, R_range, actions_dim))
			#r1 = torch.zeros((unroll_num,batch_size, R_range, 1))
			for ell in range(unroll_num):
				for_true_discounted_rewards = discounted_rewards_tensor[:,ell:R_range + ell]
				#r1[ell] = for_true_discounted_rewards
				if for_true_discounted_rewards.shape[1] > 1:
					#not tested for R_range > 1
					#alt_true_term[:,ell] = true_log_probs_t[:,ell:R_range + ell]
					true_term[:,ell] = torch.einsum('ijk,ijl->ik',[true_log_probs_t[:,ell:R_range + ell], ((for_true_discounted_rewards - for_true_discounted_rewards.mean(dim=1).unsqueeze(1))/(for_true_discounted_rewards.std(dim=1).unsqueeze(1) + 1e-5))])
				else:
					true_term[:,ell] = torch.einsum('ijk,ijl->ik',[true_log_probs_t[:,ell:R_range + ell], for_true_discounted_rewards]) 

			##########################################
			true_pe_grads_attached = grad(true_term.mean(), pe.parameters(), create_graph=True)
			true_pe_grads = [true_pe_grads_attached[t].detach() for t in range(0,len(true_pe_grads_attached))]

			#print((i,true_pe_grads), file=true_pe_grads_file)
			##########################################

			rewards_np = np.asarray(rewards)

			true_rewards_after_R = torch.zeros((batch_size, unroll_num, R_range))
			for ell in range(unroll_num):
				#length of row: max_actions - ell
				rewards_ell = torch.DoubleTensor(np.hstack((np.zeros((batch_size, R_range + ell)), rewards_np[:,ell + R_range:]))).unsqueeze(2).to(device)
				discounted_rewards_after_skip = discount_rewards(rewards_ell, discount, center=False, batch_wise=True)[:,ell:ell+R_range]
				try:
					true_rewards_after_R[:, ell] = discounted_rewards_after_skip.squeeze(2).to(device)
				except RuntimeError:
					pdb.set_trace()
					print('Oops! RuntimeError')

			#opt_steps = opt_step_def if i < num_iters-2 else 1
			for j in range(opt_steps):
				if train:
					opt.zero_grad()
				pe.zero_grad() 
				model_term = torch.zeros(batch_size, unroll_num, actions_dim)
				#alt_model_term = torch.zeros(batch_size, unroll_num, R_range, actions_dim)

				step_state = state_actions.to(device)
				#all max_actions states get unrolled R_range steps
				model_x_curr, model_x_next, model_a_list, model_r_list = self.unroll(step_state[:,:unroll_num,:], pe, states_dim, steps_to_unroll=R_range, continuous_actionspace=continuous_actionspace)

				#r_norms.append(torch.norm(model_r_list.detach().data - rewards).numpy())			
				# pdb.set_trace()
				# plt.figure()
				# plt.plot(model_x_curr[0,:,0,0].detach().numpy(), model_x_curr[0,:,0,1].detach().numpy())
				# plt.plot(model_x_curr[0,:,1,0].detach().numpy(), model_x_curr[0,:,1,1].detach().numpy())
				# plt.show()

				#this doesn't happen correctly, shape of model_r_list doesn't work with this function, possibly also affecting second_returns

				second_returns = true_rewards_after_R.double().to(device)#discounted_rewards_tensor[R_range + ell + 1 + 1]
				#r2 = torch.zeros((unroll_num,batch_size, R_range, 1))

				#can I match first_returns to closest second_returns in L2 distance of states?
				
				for ell in range(unroll_num):
					#all returns match PERFECTLY, log probs look ok now 
					first_returns = discount_rewards(model_r_list[:,ell], discount, center=False,batch_wise=True).squeeze()
					total_model_returns = first_returns + second_returns[:,ell]
					#r2[ell] = total_model_returns.unsqueeze(2)	
					model_log_probs = get_selected_log_probabilities(pe, model_x_curr[:,ell,:,:].contiguous().view(-1,states_dim), model_a_list[:,ell,:,:].contiguous().view(-1,actions_dim)).view(batch_size, -1, actions_dim)

					if total_model_returns.shape[1] > 1:
						#alt_model_term[:,ell] = model_log_probs
						model_term[:,ell] = torch.einsum('ijk,ijl->ik', [model_log_probs,((total_model_returns - total_model_returns.mean(dim=1).unsqueeze(1))/(total_model_returns.std(dim=1).unsqueeze(1) + 1e-5)).unsqueeze(2)])
					else:
						model_term[:,ell] = torch.einsum('ijk,ijl->ik',[model_log_probs,total_model_returns.unsqueeze(2)])

				model_pe_grads = grad(model_term.mean(), pe.parameters(), create_graph=True)

				#print((i,model_pe_grads), file=model_pe_grads_file)
				loss = 0 
				cos = nn.CosineSimilarity(dim=0, eps=1e-6)
				for x,y in zip(true_pe_grads, model_pe_grads):
					#loss = loss + torch.sum((1-cos(x,y)))
					loss = loss + torch.norm(x-y)
				
				#r_norms.append(loss.detach().cpu())
				#loss = torch.norm(r2 - r1)
				if loss.detach().cpu() < best_loss:
					torch.save(self.state_dict(), env_name + '_paml_trained_model.pth')
					best_loss = loss.detach()  

				if train:
					loss.backward()

					nn.utils.clip_grad_value_(self.parameters(), 10.0)
					#grads.append(torch.sum(P_hat.fc1.weight.grad))
					if torch.norm(self.fc1.weight.grad) == 0:
						pdb.set_trace()
					
					if train:
						opt.step()

					if torch.isnan(torch.sum(self.fc1.weight.data)):
						print('weight turned to nan, check gradients')
						pdb.set_trace()

				losses.append(loss.data.cpu())
				lr_schedule.step()

				if train and j < 1 and i < 1:
					initial_loss = loss.data.cpu()
					#print(initial_loss)
				
				###############  FOR CURRICULUM LEARNING   #############
				if train and j == opt_steps - 1:
					if (loss.data.cpu() <= initial_loss * 0.1 or loss.data.cpu() <= 10.0) and R_range < 10:
						initial_loss = loss.data.cpu() * 10.0
						R_range += 1
						# max_actions += 1
						# num_states = max_actions + 1
						unroll_num = num_states - R_range
						lr_schedule.step()
						print("******** Horizon: {}, Traj: {} ************".format(R_range, num_states))
						# dataset = ReplayMemory(500000)
						# for ep in range(num_episodes):
						# 	x_tmp, x_next_tmp, u_list, _, r_list = lin_dyn(max_actions, pe, [], x=x_0, discount=0.0)
						# 	for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_list):
						# 		dataset.push(x, x_next, u, r)
				########################################################
				
				print("R_range:   {},  batch_num:  {},  ep:   {},  paml_loss:    {:.7f}".format(R_range, i, j, loss.data.cpu()))
				if train:
					print("R_range:   {},  batch_num:  {},  ep:   {},  paml_loss:    {:.7f}".format(R_range, i, j, loss.data.cpu()))
				else:
					val_loss += loss.data.cpu()

					#print(val_loss)
		
		if not train:
			print("---------------------------------- Validation loss -----------------------------")
			print("Validation loss: batch_num: {}, ep: {}, average validation paml_loss = {:.7f}".format(i, j, val_loss.data.cpu()/num_starting_states))
			print("---------------------------------------------------------------------------------")
		return R_range

	#next two methods should be combined
	def mle_validation_loss(self, states_next, state_actions, policy, R_range, use_model=True):
		with torch.no_grad():
			continuous_actionspace = policy.continuous
			squared_errors = torch.zeros_like(states_next)
			step_state = state_actions.to(device)

			for step in range(R_range - 1):
				
				if use_model:
					next_step_state = self.forward(step_state)
				else:
					A = torch.from_numpy(
						np.tile(
							np.array([[0.9, 0.4], [-0.4, 0.9]]),
							(step_state[0,:,0].shape[0],1,1) #repeat along trajectory size, for every state
							)
						)
					next_step_state = torch.einsum('jik,ljk->lji',[A,step_state[:,:,:2]]) + step_state[:,:,2:]

				squared_errors += F.pad(input=(states_next[:,step:,:] - next_step_state)**2, pad=(0,0,step,0,0,0), mode='constant', value=0)

				#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)

				shortened = next_step_state[:,:-1,:]
				if shortened.nelement() > 0:
					a = policy.sample_action(torch.DoubleTensor(shortened))	
					step_state = torch.cat((shortened,a),dim=2)

			#state_loss = torch.mean(squared_errors)#torch.mean((states_next - rolled_out_states_sums)**2)
			model_loss = torch.mean(squared_errors) #+ reward_loss)# + done_loss)
			#print("R_range: {}, negloglik  = {:.7f}".format(R_range, model_loss.data.cpu()))
			
		return squared_errors.mean()
	
	def train_mle(self, pe, state_actions, states_next, epochs, max_actions, R_range, opt, env_name, continuous_actionspace, losses):
		for i in range(epochs):
			opt.zero_grad()

			squared_errors = torch.zeros_like(states_next)
			step_state = state_actions.to(device)

			for step in range(R_range - 1):
				next_step_state = self.forward(step_state)

				squared_errors += F.pad(input=(states_next[:,step:,:] - next_step_state)**2, pad=(0,0,step,0,0,0), mode='constant', value=0)

				#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)

				shortened = next_step_state[:,:-1,:]
				a = pe.sample_action(torch.DoubleTensor(shortened))	
				step_state = torch.cat((shortened,a),dim=2)

			#state_loss = torch.mean(squared_errors)#torch.mean((states_next - rolled_out_states_sums)**2)

			model_loss = torch.mean(squared_errors) #+ reward_loss)# + done_loss)
			print("R_range: {}, negloglik  = {:.7f}".format(R_range, model_loss.data.cpu()))

			model_loss.backward()
			opt.step()
			losses.append(model_loss.data.cpu())
		return model_loss