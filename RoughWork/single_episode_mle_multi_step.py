# from dm_control import suite
# import gym
# import dm_control2gym

import numpy as np
import matplotlib.pyplot as plt
# import gym
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
from networks import *
from utils import *
from rewardfunctions import *
import pickle

path = '/home/romina/CS294hw/for_viz/'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
loss_name = 'mle'

MAX_TORQUE = 1.

if __name__ == "__main__":
	#initialize pe 
	rs = 7
	torch.manual_seed(rs)
	np.random.seed(rs)	
	num_episodes = 1
	num_starting_states = 300
	max_actions = 3
	num_iters = 6000
	discount = 0.9
	R_range = 3 #horizon 1 --> R_range >= 2
	batch_size = num_episodes*num_starting_states
	val_num_starting_states = 75
	val_batch_size = num_episodes*val_num_starting_states
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
	dataset = ReplayMemory(200000)
	validation_dataset = ReplayMemory(200000)
	validation_num = val_batch_size
	x_d = np.zeros((0,2))
	x_next_d = np.zeros((0,2))
	r_d = np.zeros((0))

	actions_dim = 2
	continuous_actionspace = True
	
	states_dim = 2
	extra_dim = states_dim - 2
	errors_name = 'lin_dyn_single_episode_errors_' + loss_name + '_' + str(R_range)
	#########################################

	fname_training = 'training_' + str(rs) + '.pickle'
	fname_val = 'val_' + str(rs) + '.pickle'

	with open(fname_training, 'rb') as data:
		dataset = pickle.load(data)

	with open(fname_val, 'rb') as data:
		validation_dataset = pickle.load(data)

	#P_hat = ACPModel(states_dim, actions_dim, clip_output=False)
	P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE)
	pe = Policy(states_dim-extra_dim, actions_dim, continuous=continuous_actionspace, std=-2.5, max_torque=MAX_TORQUE)

	P_hat.to(device)
	pe.to(device)

	P_hat.double()
	pe.double()

	opt = optim.SGD(P_hat.parameters(), lr = 1e-5, momentum=0.99)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[2000,3000,4000], gamma=0.1)
	#1. Gather data, 2. train model with mle 3. reinforce 4. run policy, add new data to data

	#1. Gather data
	losses = []
	grads = []
	best_loss = 15

	epoch = 0
	x_plot = np.zeros((num_episodes * 20, 2))

	# for st in range(num_starting_states):
	# 	print(st)
	# 	x_0 = 2*np.random.random(size = (2,)) - 0.5
	# 	for ep in range(num_episodes):
	# 		x_tmp, x_next_tmp, u_list, r_tmp, _ = lin_dyn(max_actions, pe, [], x=x_0)

	# 		for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_tmp):
	# 			dataset.push(x, x_next, u, r)

	# 	#x_plot[ep: ep + 20] = x_tmp


	#get validation data
	# val_ststates=[]
	# for v in range(val_num_starting_states): 
	# 	x_0 = 2*np.random.random(size = (2,)) - 0.5
	# 	val_ststates.append(x_0)

	# for x_0 in val_ststates:
	# 	print(x_0)
	# 	for ep in range(validation_num):	
	# 		x_tmp, x_next_tmp, u_list, r_tmp, _ = lin_dyn(max_actions*2, pe, [], x=x_0)

	# 		for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_tmp):
	# 			validation_dataset.push(x, x_next, u, r)

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

	#validation data
	first_val_loss = 0
	for i in range(val_num_starting_states):
		val_data = validation_dataset.sample(val_batch_size, structured=True, max_actions=max_actions, num_episodes_per_start=num_episodes, num_starting_states=val_num_starting_states, start_at=None)

		val_states_prev = torch.zeros((validation_num, max_actions, states_dim)).double()
		val_states_next = torch.zeros((validation_num, max_actions, states_dim)).double()
		val_rewards = torch.zeros((validation_num, max_actions)).double()
		val_actions_tensor = torch.zeros((validation_num, max_actions, actions_dim)).double()
		#discounted_rewards_tensor = torch.zeros((batch_size, max_actions, 1)).double()

		for v in range(validation_num):
			#batch = dataset.sample(max_actions, structured=True, num_episodes=num_episodes)
			val_states_prev[v] = torch.tensor([samp.state for samp in val_data[v]]).double()
			val_states_next[v] = torch.tensor([samp.next_state for samp in val_data[v]]).double()
			val_rewards[v] = torch.tensor([samp.reward for samp in val_data[v]]).double()
			val_actions_tensor[v] = torch.tensor([samp.action for samp in val_data[v]]).double()
			#discounted_rewards_tensor[b] = discount_rewards(rewards[b].unsqueeze(1), discount, center=False).to(device)

		val_state_actions = torch.cat((val_states_prev,val_actions_tensor), dim=2)
		first_val_loss += P_hat.mle_validation_loss(val_states_next, val_state_actions, pe, R_range)

	first_val_loss = first_val_loss / val_num_starting_states
	print(first_val_loss)
	best_val_loss = first_val_loss

	for i in range(num_iters):
		batch = dataset.sample(batch_size, structured=True, max_actions=max_actions, num_episodes_per_start=num_episodes, num_starting_states=num_starting_states, start_at=None)

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
		# squared_errors = 0
		# step_state = state_actions.to(device)
		print("i: ", i)
		best = P_hat.train_mle(pe, state_actions, batch_states_next, 1, max_actions, R_range, opt, "lin_dyn", continuous_actionspace, losses)

		val_loss = 0
		#for i in range(val_num_starting_states):
		val_data = validation_dataset.sample(val_batch_size, structured=True, max_actions=max_actions, num_episodes_per_start=num_episodes, num_starting_states=val_num_starting_states, start_at=None)

		val_states_prev = torch.zeros((validation_num, max_actions, states_dim)).double()
		val_states_next = torch.zeros((validation_num, max_actions, states_dim)).double()
		val_rewards = torch.zeros((validation_num, max_actions)).double()
		val_actions_tensor = torch.zeros((validation_num, max_actions, actions_dim)).double()
		#discounted_rewards_tensor = torch.zeros((batch_size, max_actions, 1)).double()

		for v in range(validation_num):
			#batch = dataset.sample(max_actions, structured=True, num_episodes=num_episodes)
			val_states_prev[v] = torch.tensor([samp.state for samp in val_data[v]]).double()
			val_states_next[v] = torch.tensor([samp.next_state for samp in val_data[v]]).double()
			val_rewards[v] = torch.tensor([samp.reward for samp in val_data[v]]).double()
			val_actions_tensor[v] = torch.tensor([samp.action for samp in val_data[v]]).double()
			#discounted_rewards_tensor[b] = discount_rewards(rewards[b].unsqueeze(1), discount, center=False).to(device)

		val_state_actions = torch.cat((val_states_prev,val_actions_tensor), dim=2)
		if i == 0 or i == num_iters-1:
			val_loss += P_hat.mle_validation_loss(val_states_next, val_state_actions, pe, R_range, use_model=True)
		else:
			val_loss += P_hat.mle_validation_loss(val_states_next, val_state_actions, pe, R_range, use_model=True)
		
		#val_loss = val_loss #/ val_num_starting_states

		print('validation loss: ', val_loss)

		if val_loss < best_val_loss:
			#print('best_val_loss:      ', val_loss.detach().data)
			torch.save(P_hat.state_dict(), 'lin_dyn_horizon_{}_traj_{}_mle_trained_model.pth'.format(R_range,max_actions+1))
			#torch.save(P_hat, 'mle_trained_model.pth')
			best_val_loss = val_loss  

		if best < first_val_loss * 0.05 and R_range < max_actions - 1:
			R_range += 1
		
	#mle_multistep_loss(P_hat, pe, val_states_next, val_state_actions, actions_dim, max_actions, R_range, continuous_actionspace=continuous_actionspace)

		# for step in range(R_range):
		# 	next_step_state = P_hat(step_state)

		# 	#squared_errors += shift_down((states_next[:,step:,:] - next_step_state)**2, step, max_actions)
		# 	if step > 0:
		# 		squared_errors += torch.mean((batch_states_next[:,step:] - next_step_state[:,:-step,:])**2)
		# 	else:
		# 		squared_errors += torch.mean((batch_states_next - next_step_state)**2)

		# 	#rolled_out_states_sums += shift_down(next_step_state,step, max_actions)

		# 	###########
		# 	# shortened = next_step_state[:,:-1,:]
		# 	action_probs = pe(torch.DoubleTensor(next_step_state))
			
		# 	if not continuous_actionspace:
		# 		c = Categorical(action_probs)
		# 		a = c.sample() 
		# 		step_state = torch.cat((shortened,convert_one_hot(a, actions_dim)),dim=2)
		# 	else:
		# 		c = Normal(*action_probs)
		# 		a = torch.clamp(c.rsample(), min=-MAX_TORQUE, max=MAX_TORQUE)
		# 		step_state = torch.cat((next_step_state,a),dim=2)
		# 	##################

		# #state_loss = torch.sum(torch.sum(squared_errors, dim=1),dim=1)
		# #model_loss = torch.mean(state_loss) #+ reward_loss)# + done_loss)
		# model_loss = squared_errors
		
		# if model_loss < best_loss:
		# 	torch.save(P_hat.state_dict(), 'lin_dyn_' + str(R_range) + '_mle_trained_model.pth')
		# 	#torch.save(P_hat, 'mle_trained_model.pth')
		# 	best_loss = model_loss  

		# model_loss.backward()
		# grads.append(torch.sum(P_hat.fc1.weight.grad))

		# opt.step()
		# losses.append(model_loss.data.cpu())
		# lr_schedule.step()
		# if i % 20 == 0:
		# 	print("ep: {}, negloglik  = {:.7f}".format(i, model_loss.data.cpu()))

		# if i == num_iters/4:
		# 	print("R_range changed!")
		# 	R_range = 2
		# elif i == num_iters/2:
		# 	print("R_range changed!")
		# 	R_range = 3

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