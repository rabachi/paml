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

	R_range = 20

 
	env.seed(0)

	errors_name = env.spec.id + '_single_episode_errors_' + loss_name + '_' + str(R_range)

	#P_hat = ACPModel(n_states, n_actions, clip_output=False)
	P_hat = DirectEnvModel(n_states,n_actions)
	pe = Policy(n_states, n_actions, continuous=continuous_actionspace)

	P_hat.to(device)
	pe.to(device)

	if loss_name == 'paml':
		for p in pe.parameters():
			p.requires_grad = True
	else:
		for p in pe.parameters():
			p.requires_grad = False

	opt = optim.SGD(P_hat.parameters(), lr = 1e-6, momentum=0.99)
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
				action_probs = pe(torch.cuda.FloatTensor(s))
			else:
				action_probs = pe(torch.FloatTensor(s))

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
	epoch += 1

	rewards_tensor = torch.FloatTensor(discount_rewards(rewards,discount)).to(device).view(1, -1, 1)
	states_prev = torch.FloatTensor(states[:-1]).to(device).view(1, -1, n_states)
	states_next = torch.FloatTensor(states[1:]).to(device).view(1, -1,n_states)
	
	actions_tensor = torch.stack(actions).type(torch.FloatTensor).to(device).view(1, -1, n_actions)

	state_actions = torch.cat((states_prev,actions_tensor), dim=2)

	#pamlize the real trajectory (states_next)


	for i in range(num_iters):
		# errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, n_actions, max_actions, device)
		opt.zero_grad()
		
		#rolled_out_states_sums = torch.zeros_like(states_next)
		squared_errors = torch.zeros_like(states_next)
		step_state = state_actions.to(device)

		for step in range(R_range - 1):
			next_step_state = P_hat(step_state)

			#squared_errors += shift_down((states_next[:,step:,:] - next_step_state)**2, step, max_actions)
			squared_errors += (states_next - next_step_state)**2
			#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)


			###########
			# shortened = next_step_state[:,:-1,:]
			# action_probs = pe(torch.FloatTensor(shortened))
			
			# if not continuous_actionspace:
			# 	c = Categorical(action_probs)
			# 	a = c.sample() 
			# 	step_state = torch.cat((shortened,convert_one_hot(a, n_actions)),dim=2)
			# else:
			# 	c = Normal(*action_probs)
			# 	a = torch.clamp(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)
			# 	step_state = torch.cat((shortened,a),dim=2)
			##################


		state_loss = torch.sum(torch.sum(squared_errors, dim=1),dim=1)

		model_loss = torch.mean(state_loss) #+ reward_loss)# + done_loss)
		print("i: {}, negloglik  = {:.7f}".format(i, model_loss.data.cpu()))
		if model_loss < best_loss:
			torch.save(P_hat.state_dict(), env.spec.id+'_mle_trained_model.pth')
			#torch.save(P_hat, 'mle_trained_model.pth')
			best_loss = model_loss  

		model_loss.backward()

		opt.step()
		losses.append(model_loss.data.cpu())
		lr_schedule.step()


	# if ep % 1 ==0:
	# 	if loss_name == 'paml':
	# 		errors_val = paml_validation_loss(env, P_hat, pe, val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val, R_range, val_size, n_actions, max_actions, device)

	# 	elif loss_name == 'mle':
	# 		errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, n_actions, max_actions, device)


# all_val_errs = torch.mean(errors_val, dim=0)
# print('saving multi-step errors ...')
# np.save(os.path.join(path, (errors_name+'_val')), np.asarray(all_val_errs))

np.save(os.path.join(path,loss_name),np.asarray(losses))