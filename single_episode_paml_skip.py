import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
import math
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
from torch.autograd import grad, gradgradcheck

import pdb
import os
from models import *
from utils import *
from rewardfunctions import *

path = '/home/romina/CS294hw/for_viz/'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
loss_name = 'paml'

MAX_TORQUE = 2.

if __name__ == "__main__":
	#initialize pe 
	num_episodes = 1
	max_actions = 200
	num_iters = 3000
	discount = 0.9
	env = gym.make('Pendulum-v0')

	n_states = env.observation_space.shape[0]
	continuous_actionspace = isinstance(env.action_space, gym.spaces.box.Box)
	if continuous_actionspace:
		n_actions = env.action_space.shape[0]
	else:
		n_actions = env.action_space.n

	R_range = 1

 
	env.seed(0)

	errors_name = env.spec.id + '_single_episode_errors_' + loss_name + '_' + str(R_range)

	#P_hat = ACPModel(n_states, n_actions, clip_output=False)
	P_hat = DirectEnvModel(n_states,n_actions, MAX_TORQUE)
	pe = Policy(n_states, n_actions, continuous=continuous_actionspace)

	P_hat.to(device).double()
	pe.to(device).double()


	for p in P_hat.parameters():
		p.requires_grad = True

	if loss_name == 'paml':
		for p in pe.parameters():
			p.requires_grad = True
	else:
		for p in pe.parameters():
			p.requires_grad = True

	opt = optim.SGD(P_hat.parameters(), lr = 1e-5, momentum=0.99)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[300,350,400], gamma=0.1)
	#1. Gather data, 2. train model with mle 3. reinforce 4. run policy, add new data to data

	#1. Gather data
	losses = []
	best_loss = 15

	epoch = 0

	s = env.reset()
	states = [s]
	actions = []
	rewards = []

	while len(actions) < max_actions:
		with torch.no_grad():
			if device == 'cuda':
				action_probs = pe(torch.cuda.DoubleTensor(s))
			else:
				action_probs = pe(torch.DoubleTensor(s))

			if not continuous_actionspace:
				c = Categorical(action_probs[0])
				a = c.sample() 
			else:
				c = Normal(*action_probs)
				a = np.clip(c.rsample(), -MAX_TORQUE, MAX_TORQUE)

			s_prime, r, _, _ = env.step(a.cpu().numpy())#-1
			states.append(s_prime)

			if not continuous_actionspace:
				actions.append(convert_one_hot(a, n_actions))
			else:
				actions.append(a)

			rewards.append(r)
			s = s_prime

	#2. train model
	discounted_rewards_tensor = torch.DoubleTensor(discount_rewards(rewards, discount, center=False)).to(device).view(1, -1, 1)
	

	states_prev = torch.DoubleTensor(states[:-1]).to(device).view(1, -1, n_states)
	states_next = torch.DoubleTensor(states[1:]).to(device).view(1, -1,n_states)
	actions_tensor = torch.stack(actions).type(torch.DoubleTensor).to(device).view(1, -1, n_actions)
	state_actions = torch.cat((states_prev,actions_tensor), dim=2)


	#pamlize the real trajectory (states_next)
	pe.zero_grad()

	true_log_probs_t = get_selected_log_probabilities(
							pe, 
							states_prev, 
							actions_tensor, 
							range(actions_tensor.shape[0])
							)

	#1
	for_true_discounted_rewards = discounted_rewards_tensor[:, :R_range + 1]

	true_term = torch.sum(true_log_probs_t[:, :R_range + 1] * ((for_true_discounted_rewards - for_true_discounted_rewards.mean())/(for_true_discounted_rewards.std() + 1e-5)))

	true_pe_grads_attached = grad(true_term, pe.parameters(), create_graph=True)
	true_pe_grads = [true_pe_grads_attached[i].detach() for i in range(len(true_pe_grads_attached))]
	print(true_pe_grads)


	#2b
	rewards_np = np.asarray(rewards)
	unroll_num = max_actions - R_range - 1

	true_rewards_after_R = torch.zeros((unroll_num, max_actions,1))

	for ell in range(unroll_num):
		#length of row: max_actions - ell
		rewards_ell = np.hstack((np.zeros((R_range + ell + 1)), rewards_np[ell + R_range + 1:]))
		discounted_rewards_after_skip = discount_rewards(rewards_ell, discount, center=False)[ell:]

		true_rewards_after_R[ell] = torch.DoubleTensor(np.pad(discounted_rewards_after_skip, (0, ell), 'constant', constant_values=0)).to(device).unsqueeze(1)
	

	for i in range(num_iters):
		opt.zero_grad()
		pe.zero_grad()

		model_rewards = torch.zeros((unroll_num, R_range + 1, 1))

		k_step_log_probs = torch.zeros((unroll_num, R_range + 1, 1))
		
		step_state = state_actions[:,:unroll_num].to(device)


		for step in range(R_range + 1):
			next_step_state = P_hat(step_state)
			

			shortened = next_step_state#[:,:-1,:]

			###########
			with torch.no_grad():
				action_probs = pe(torch.DoubleTensor(shortened))
				
				if not continuous_actionspace:
					c = Categorical(action_probs)
					actions_t_l = c.sample() 
					step_state = torch.cat((shortened,convert_one_hot(actions_t_l, n_actions)),dim=2)
				else:
					c = Normal(*action_probs)
					actions_t_l = torch.clamp(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)
					step_state = torch.cat((shortened, actions_t_l),dim=2)
			##################


				#only need rewards from model for the steps we've unrolled, the rest is assumed to be equal to the environment's
				model_rewards[:, step] = get_reward_fn(env, shortened, actions_t_l) #need gradients of this because rewards are from the states
			
			model_log_probs = get_selected_log_probabilities(
								pe,
								shortened,
								actions_t_l,
								range(actions_t_l.shape[0])
								)
			k_step_log_probs[:,step] = model_log_probs 
			#don't need states, just the log probabilities 
		
		#unroll_num x R_range x 1
		j_step_discounted_model_rewards = discount_rewards(model_rewards, discount, center=False)

		model_returns = torch.cat(((j_step_discounted_model_rewards + true_rewards_after_R[:, :R_range + 1]), true_rewards_after_R[:, R_range + 1:]),dim=1)[:, :R_range + 1]

		centered_model_returns = (model_returns - torch.mean(model_returns, dim=1).unsqueeze(1).repeat(1,model_returns.shape[1],1)) / (torch.std(model_returns, dim=1).unsqueeze(1).repeat(1,model_returns.shape[1],1) + 1e-5) 

		model_term = torch.sum(torch.sum(k_step_log_probs * centered_model_returns, dim=0), dim=0)

		model_pe_grads = grad(model_term, pe.parameters(), create_graph=True)


		loss = 0
		for x,y in zip(true_pe_grads, model_pe_grads):
			# print(torch.sum((x-y)**2))
			loss += torch.sum((x-y)**2)


		print("i: {}, paml_loss  = {:.7f}".format(i, loss.data.cpu()))
		if loss < best_loss:
			torch.save(P_hat.state_dict(), env.spec.id + '_' + loss_name + '_trained_model.pth')
			best_loss = loss  

		loss.backward(retain_graph=True)

		#print(state_actions[:,:unroll_num].to(device))
		#print(gradgradcheck(P_hat, [state_actions[:,:unroll_num].to(device)]))

		opt.step()
		losses.append(loss.data.cpu())
		lr_schedule.step()


	# if ep % 1 ==0:
	# 	if loss_name == 'paml':
	# 		errors_val = paml_validation_loss(env, P_hat, pe, val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val, R_range, val_size, n_actions, max_actions, device)

	# 	elif loss_name == 'mle':
	# 		errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, n_actions, max_actions, device)


# all_val_errs = torch.mean(errors_val, dim=0)
# print('saving multi-step errors ...')
# np.save(os.path.join(path, (errors_name+'_val')), np.asarray(all_val_errs))

print(os.path.join(path,loss_name))
np.save(os.path.join(path,loss_name),np.asarray(losses))