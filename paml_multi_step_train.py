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
loss_name = 'paml'

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
	final_R_range = 30
	R_range_schedule_length = 1

	for rs in random_seeds: 
		#print(rs)
		#env.seed(rs)
		errors_name = env.spec.id + '_acp_errors_' + loss_name + '_' + str(final_R_range)
		#P_hat = ACPModel(n_states, n_actions) 
		P_hat = DirectEnvModel(n_states,n_actions, MAX_TORQUE)
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


		#R_range_schedule = np.hstack((np.ones((20))*2,np.ones((30))*4,np.ones((30))*6,np.ones((50))*8))#np.ones((30))*4,np.ones((40))*6,np.ones((50))*8,np.ones((50))*10))
		R_range_schedule = [max_actions]

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

				#Update R_range
				print('Epoch:', epoch)
				if epoch < len(R_range_schedule):
					R_range = int(R_range_schedule[epoch])
				else:
					R_range = final_R_range
				print('R_range:', R_range)


				#Train model for several epochs
				if loss_name == 'mle':
					P_hat.train_mle(env, states_next, num_iters, max_actions, R_range, opt, continuous_actionspace, losses)

				elif loss_name == 'paml':
					P_hat.train_paml_skip(env, pe, states_prev, states_next, actions_tensor, rewards_tensor, num_iters, max_actions, R_range, discount, opt, continuous_actionspace, losses, device='cpu')
					#P_hat.train_paml(env, pe, state_actions, states_next, num_iters, max_actions, R_range, discount, opt, continuous_actionspace, losses, device=device)


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