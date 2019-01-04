import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
import math
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
from torch.autograd import grad

import pdb
import os
from models import *
from utils import *
from rewardfunctions import *

path = '/home/romina/CS294hw/for_viz/'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
loss_name = 'mle'

MAX_TORQUE = 2.

if __name__ == "__main__":
	#initialize pe 
	num_episodes = 1000
	max_actions = 30
	num_iters = 50
	discount = 0.9
	env = gym.make('Pendulum-v0')

	random_seeds = np.arange(1)

	n_states = env.observation_space.shape[0]
	continuous_actionspace = isinstance(env.action_space, gym.spaces.box.Box)

	if continuous_actionspace:
		n_actions = env.action_space.shape[0]
	else:
		n_actions = env.action_space.n

	starting_R_range = 1
	final_R_range = 8
	R_range_schedule_length = 1

	for rs in random_seeds: 
		#print(rs)
		#env.seed(rs)
		errors_name = env.spec.id + '_acp_errors_' + loss_name + '_' + str(final_R_range)
		#P_hat = ACPModel(n_states, n_actions) 
		P_hat = DirectEnvModel(n_states,n_actions)
		#torch.load('../mle_trained_model.pth')
		pe = Policy(n_states, n_actions, continuous=continuous_actionspace)
		#pe.load_state_dict(torch.load('policy_trained.pth'))

		P_hat.to(device)
		pe.to(device)

		if loss_name == 'paml':
			for p in pe.parameters():
				p.requires_grad = True
		else:
			for p in pe.parameters():
				p.requires_grad = False

		opt = optim.SGD(P_hat.parameters(), lr = 1e-4, momentum=0.99)
		lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[100,150,200], gamma=0.1)
		#1. Gather data, 2. train model with mle 3. reinforce 4. run policy, add new data to data

		#1. Gather data
		losses = []
		best_loss = 15

		batch_size = 5
		batch_states_prev = []
		batch_states_next = []
		batch_rewards = []
		batch_actions = []
		batch_dones = []
		batch_counter = 0


		#collect trajectories for validation of multi-step state prediction error
		val_size = 20
		val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val = collect_data(env, pe, val_size, n_actions, n_states, 0, max_actions, continuous_actionspace, device)

		state_actions_val = torch.cat((val_states_prev_tensor, actions_tensor_val),dim=2)
		squared_errors_val = torch.zeros_like(val_states_next_tensor)


		R_range_schedule = np.hstack((np.ones((20))*2,np.ones((30))*4,np.ones((30))*6,np.ones((50))*8))#np.ones((30))*4,np.ones((40))*6,np.ones((50))*8,np.ones((50))*10))

		epoch = 0
		for ep in range(num_episodes):

			s = env.reset()
			done = False
			states = [s]
			actions = []
			rewards = []
			dones = []
			while len(actions) < max_actions:
				with torch.no_grad():

					if device == 'cuda':
						action_probs = pe(torch.cuda.FloatTensor(s))
					else:
						action_probs = pe(torch.FloatTensor(s))

					if not continuous_actionspace:
						c = Categorical(action_probs[0])
						a = c.sample() 
					else:
						c = Normal(*action_probs)
						a = np.clip(c.rsample(), -MAX_TORQUE, MAX_TORQUE)

					s_prime, r, done, _ = env.step(a.cpu().numpy() - 1)
					states.append(s_prime)

					if not continuous_actionspace:
						actions.append(convert_one_hot(a, n_actions))
					else:
						actions.append(a)

					rewards.append(r)
					dones.append(done)
					s = s_prime

			if batch_counter != batch_size:
				batch_counter += 1
				batch_actions.extend(actions)
			
				batch_states_next.extend(states[1:])
				batch_states_prev.extend(states[:-1])

				batch_rewards.extend(discount_rewards(rewards, discount))
				batch_dones.extend(dones)
			else:
				#2. train model

				epoch += 1

				# states_prev = torch.FloatTensor(states)[:-1,:].to(device)
				# states_next = torch.FloatTensor(states)[1:,:].to(device)
				# actions_tensor = (torch.FloatTensor(actions[:-1])).unsqueeze(1).to(device)
				# state_actions = torch.cat((states_prev,actions_tensor),dim=1)

				rewards_tensor = torch.FloatTensor(batch_rewards).to(device).view(batch_size, -1, 1)
				states_prev = torch.FloatTensor(batch_states_prev).to(device).view(batch_size, -1, n_states)
				states_next = torch.FloatTensor(batch_states_next).to(device).view(batch_size,-1,n_states)
				
				actions_tensor = torch.stack(batch_actions).type(torch.FloatTensor).to(device).view(batch_size, -1, n_actions)

				state_actions = torch.cat((states_prev,actions_tensor), dim=2)

				#pamlize the real trajectory (states_next)

				if loss_name == 'paml':
					pe.zero_grad()
					multiplier = torch.arange(max_actions,0,-1).repeat(batch_size,1).unsqueeze(2).type(torch.FloatTensor).to(device)

					true_log_probs_t = torch.sum(
										get_selected_log_probabilities(
											pe, 
											states_prev, 
											actions_tensor, 
											batch_actions
											) * rewards_tensor #* multiplier
										, dim=1)

					true_log_probs = torch.mean(true_log_probs_t, dim=0)

					true_pe_grads = grad(true_log_probs, pe.parameters(), create_graph=True)
					#true_log_probs.backward(retain_graph=False)
					#true_pe_grads = [x.grad.data.clone() for x in pe.parameters()]

				print('Epoch:', epoch)
				if epoch < len(R_range_schedule):
					R_range = int(R_range_schedule[epoch])
				else:
					R_range = final_R_range

				print('R_range:', R_range)

				for i in range(num_iters):
					opt.zero_grad()
					
					if loss_name == 'mle':
						#rolled_out_states_sums = torch.zeros_like(states_next)
						squared_errors = torch.zeros_like(states_next)
						step_state = state_actions.to(device)

						for step in range(R_range - 1):
							next_step_state = P_hat(step_state)

							squared_errors += shift_down((states_next[:,step:,:] - next_step_state)**2, step, max_actions)

							#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)

							shortened = next_step_state[:,:-1,:]
							action_probs = pe(torch.FloatTensor(shortened))
							
							if not continuous_actionspace:
								c = Categorical(action_probs)
								a = c.sample() 
								step_state = torch.cat((shortened,convert_one_hot(a, n_actions)),dim=2)
							else:
								c = Normal(*action_probs)
								a = torch.clamp(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)
								step_state = torch.cat((shortened,a),dim=2)

						state_loss = torch.mean(squared_errors, dim=1)#torch.mean((states_next - rolled_out_states_sums)**2)

						model_loss = torch.mean(state_loss) #+ reward_loss)# + done_loss)
						print("i: {}, negloglik  = {:.7f}".format(i, model_loss.data.cpu()))
						if model_loss < best_loss:
							torch.save(P_hat.state_dict(), env.spec.id+'_mle_trained_model.pth')
							#torch.save(P_hat, 'mle_trained_model.pth')
							best_loss = model_loss  

						model_loss.backward()



					elif loss_name == 'paml':
						step_state_action = state_actions.to(device)
						k_step_log_probs = torch.zeros((R_range, batch_size, n_actions))
						pe.zero_grad()

						for step in range(R_range):
							next_step_state = P_hat(step_state_action)
							#print('states_mean:', torch.mean(next_step_state))
							shortened = next_step_state[:,:-1,:]

							with torch.no_grad():
								action_probs = pe(torch.FloatTensor(shortened))
								
								if not continuous_actionspace:
									c = Categorical(action_probs)
									actions_t_l = c.sample() 
									step_state_action = torch.cat((shortened,convert_one_hot(actions_t_l, n_actions)),dim=2)
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
							torch.save(P_hat.state_dict(), 
								env.spec.id+'_paml_trained_model.pth')
							#torch.save(P_hat, 'mle_trained_model.pth')
							best_loss = model_loss 

					#paml ize the fake trajectory
					# if loss_name == 'paml':
					# 	###have to define model_rewards_tensor
					# 	model_rewards_tensor = get_reward_fn(env, rolled_out_states_sums/R_range, rollout_actions_tensor)
					# 	pe.zero_grad()
					# 	pdb.set_trace()
					# 	model_log_probs = torch.mean(
					# 						torch.sum(
					# 							get_selected_log_probabilities(
					# 								pe, 
					# 								rolled_out_states_sums/R_range,
					# 								rollout_actions_tensor, rollout_batch_actions) * 
					# 								model_rewards_tensor
					# 								, dim=1)
					# 						, dim=0)

					# 	model_log_probs.backward(retain_graph=False)
					# 	model_pe_grads = [x.grad.data for x in pe.parameters()]


					opt.step()
					losses.append(model_loss.data.cpu())
					#lr_schedule.step()

				print("ep: {}, model_loss  = {:.3f}".format(ep, model_loss.data.cpu()))
				#print("ep: {}, PAML loss  = {:.7f}".format(ep, paml_loss.data.cpu()))

				batch_counter = 0
				batch_actions = []
				batch_rewards = []
				batch_dones = []
				batch_states_prev = []
				batch_states_next = []

				lr_schedule.step()

				if ep % 1 ==0:
					if loss_name == 'paml':
						errors_val = paml_validation_loss(env, P_hat, pe, val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val, R_range, val_size, n_actions, max_actions, device)

					elif loss_name == 'mle':
						errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, n_actions, max_actions, device)

					#if losses[-1] < best_loss:
					all_val_errs = torch.mean(errors_val, dim=0)
					print('saving multi-step errors ...')
					np.save(os.path.join(path, (errors_name+'_val')), np.asarray(all_val_errs))
					# best_loss = losses[-1]

		np.save(os.path.join(path,loss_name),np.asarray(losses))