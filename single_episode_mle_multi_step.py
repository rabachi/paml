from dm_control import suite
import gym
import dm_control2gym

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

MAX_TORQUE = 1.

if __name__ == "__main__":
	#initialize pe 
	num_episodes = 100
	max_actions = 10
	num_iters = 2000
	discount = 0.9
	R_range = 3
	batch_size = 50
	##########for deepmind setup###################
	#dm_control2gym.create_render_mode('rs', show=False, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(), depth=False, scene_option=None)

	# env = dm_control2gym.make(domain_name="cartpole", task_name="balance")	
	# env.spec.id = 'dm_cartpole_balance'
	#########################################
	

	###########for openai setup####################
	# env = gym.make('CartPole-v0')
	# states_dim = env.observation_space.shape[0]
	#########################################

	###########for both
	# continuous_actionspace = isinstance(env.action_space, gym.spaces.box.Box)
	# if continuous_actionspace:
	# 	actions_dim = env.action_space.shape[0]
	# else:
	# 	actions_dim = env.action_space.n
	#env.seed(0)
	#errors_name = env.spec.id + '_single_episode_errors_' + loss_name + '_' + str(R_range)


	##########for linear system setup#############
	dataset = ReplayMemory(20000)
	x_d = np.zeros((0,2))
	x_next_d = np.zeros((0,2))
	r_d = np.zeros((0))

	states_dim = 2
	actions_dim = 2
	continuous_actionspace = True
	
	state_dim = 5
	extra_dim = state_dim - 2
	errors_name = 'lin_dyn_single_episode_errors_' + loss_name + '_' + str(R_range)
	#########################################


	#P_hat = ACPModel(states_dim, actions_dim, clip_output=False)
	P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE)
	pe = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-4.6)

	P_hat.to(device)
	pe.to(device)

	P_hat.double()
	pe.double()

	opt = optim.SGD(P_hat.parameters(), lr = 1e-6, momentum=0.99)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000,1250,1500], gamma=0.1)
	#1. Gather data, 2. train model with mle 3. reinforce 4. run policy, add new data to data

	#1. Gather data
	losses = []
	grads = []
	best_loss = 15

	epoch = 0
	x_plot = np.zeros((num_episodes * 20, 2))
	for ep in range(num_episodes):
		x_0 = 2*np.random.random( size = (2,) ) - 0.5
		x_tmp, x_next_tmp, u_list, r_tmp, _ = lin_dyn(20, pe, [], x=x_0)

		for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_tmp):
			dataset.push(x, x_next, u, r)

		x_plot[ep: ep + 20] = x_tmp


		# s = env.reset()
		# states = [s]
		# actions = []
		# rewards = []
		# done = False
		# while not done: #len(actions) < max_actions:
		# 	with torch.no_grad():
		# 		if device == 'cuda':
		# 			action_probs = pe(torch.cuda.DoubleTensor(s))
		# 		else:
		# 			action_probs = pe(torch.DoubleTensor(s))

		# 		if not continuous_actionspace:
		# 			c = Categorical(action_probs[0])
		# 			a = c.sample() 
		# 		else:
		# 			c = Normal(*action_probs)
		# 			a = torch.clamp(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)

		# 		s_prime, r, done, _ = env.step(a.cpu().numpy())#-1
		# 		states.append(s_prime)

		# 		if not continuous_actionspace:
		# 			actions.append(convert_one_hot(a, actions_dim))
		# 		else:
		# 			actions.append(a)

		# 		rewards.append(r)
		# 		s = s_prime

			# if done and len(actions) < max_actions:
			# 	filled = len(actions)
			# 	rewards += [0] * (max_actions - filled)
			# 	states += [s_prime] * (max_actions - filled)
			# 	actions += [c.sample()] * (max_actions - filled)

		#2. train model
		# rewards_tensor = torch.DoubleTensor(discount_rewards(rewards,discount)).to(device).view(1, -1, 1)
		# states_prev = torch.DoubleTensor(states[:-1]).to(device).view(1, -1, states_dim)
		# states_next = torch.DoubleTensor(states[1:]).to(device).view(1, -1,states_dim)
		
		# actions_tensor = torch.stack(actions).type(torch.DoubleTensor).to(device).view(1, -1, actions_dim - 1)

		# state_actions = torch.cat((states_prev,actions_tensor), dim=2)

		#pamlize the real trajectory (states_next)


		#unindented from here
	for i in range(num_iters):
		batch = dataset.sample(batch_size, structured=True, max_actions=max_actions, num_episodes=num_episodes)

		batch_states_prev = torch.zeros((batch_size, max_actions, states_dim)).double()
		batch_states_next = torch.zeros((batch_size, max_actions, states_dim)).double()
		batch_rewards = torch.zeros((batch_size, max_actions)).double()
		batch_actions_tensor = torch.zeros((batch_size, max_actions, actions_dim)).double()
		#discounted_rewards_tensor = torch.zeros((batch_size, max_actions, 1)).double()

		for b in range(batch_size):
			#batch = dataset.sample(max_actions, structured=True, num_episodes=num_episodes)
			batch_states_prev[b] = torch.tensor([samp.state for samp in batch[b]]).double()
			batch_states_next[b] = torch.tensor([samp.next_state for samp in batch[b]]).double()
			batch_rewards[b] = torch.tensor([samp.reward for samp in batch[b]]).double()
			batch_actions_tensor[b] = torch.tensor([samp.action for samp in batch[b]]).double()
			#discounted_rewards_tensor[b] = discount_rewards(rewards[b].unsqueeze(1), discount, center=False).to(device)
		state_actions = torch.cat((batch_states_prev,batch_actions_tensor), dim=2)

		# batch_states_prev = torch.tensor([samp.state for samp in batch]).double()
		# batch_states_next = torch.tensor([samp.next_state for samp in batch]).double()
		# batch_rewards = torch.tensor([samp.reward for samp in batch]).double()
		# batch_actions = torch.tensor([samp.action for samp in batch]).double()

		# state_actions = torch.cat((batch_states_prev,batch_actions), dim=1)
		# errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, actions_dim, max_actions, device)
		opt.zero_grad()
		
		#rolled_out_states_sums = torch.zeros_like(states_next)
		squared_errors = 0
		step_state = state_actions.to(device)

		for step in range(R_range):
			next_step_state = P_hat(step_state)

			#squared_errors += shift_down((states_next[:,step:,:] - next_step_state)**2, step, max_actions)
			if step > 0:
				squared_errors += torch.sum((batch_states_next[:,step:] - next_step_state[:,:-step,:])**2)
			else:
				squared_errors += torch.sum((batch_states_next - next_step_state)**2)

			#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)

			###########
			# shortened = next_step_state[:,:-1,:]
			action_probs = pe(torch.DoubleTensor(next_step_state))
			
			if not continuous_actionspace:
				c = Categorical(action_probs)
				a = c.sample() 
				step_state = torch.cat((shortened,convert_one_hot(a, actions_dim)),dim=2)
			else:
				c = Normal(*action_probs)
				a = torch.clamp(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)
				step_state = torch.cat((next_step_state,a),dim=2)
			##################

		#state_loss = torch.sum(torch.sum(squared_errors, dim=1),dim=1)
		#model_loss = torch.mean(state_loss) #+ reward_loss)# + done_loss)
		model_loss = squared_errors
		
		if model_loss < best_loss:
			torch.save(P_hat.state_dict(), 'lin_dyn_' + str(R_range) + '_mle_trained_model.pth')
			#torch.save(P_hat, 'mle_trained_model.pth')
			best_loss = model_loss  

		model_loss.backward()
		grads.append(torch.sum(P_hat.fc1.weight.grad))

		opt.step()
		losses.append(model_loss.data.cpu())
		lr_schedule.step()
		print("ep: {}, negloglik  = {:.7f}".format(i, model_loss.data.cpu()))

	# if ep % 1 ==0:
	# 	if loss_name == 'paml':
	# 		errors_val = paml_validation_loss(env, P_hat, pe, val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val, R_range, val_size, actions_dim, max_actions, device)

	# 	elif loss_name == 'mle':
	# 		errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, actions_dim, max_actions, device)


# all_val_errs = torch.mean(errors_val, dim=0)
# print('saving multi-step errors ...')
# np.save(os.path.join(path, (errors_name+'_val')), np.asarray(all_val_errs))
np.save(os.path.join(path,'mle_grads'),np.asarray(grads))
np.save(os.path.join(path,loss_name),np.asarray(losses))
np.save(os.path.join(path,loss_name + 'x_plot'), np.asarray(x_plot))