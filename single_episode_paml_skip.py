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

from dm_control import suite
import gym
import dm_control2gym

path = '/home/romina/CS294hw/for_viz/'
#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
loss_name = 'paml'

MAX_TORQUE = 10.

if __name__ == "__main__":
	#initialize pe 
	num_episodes = 1
	max_actions = 5
	num_states = max_actions + 1
	num_iters = 1000
	discount = 0.9
	R_range = 1
	batch_size = 5

	true_pe_grads_file = open('true_pe_grads.txt', 'w')
	model_pe_grads_file = open('model_pe_grads.txt', 'w') 

	# dm_control2gym.create_render_mode('rs', show=False, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(), depth=False, scene_option=None)

	# # env = dm_control2gym.make(domain_name="pendulum")#, task_name="balance")
	# # env.spec.id = 'dm_pendulum'
	# env = gym.make('CartPole-v0')

	# n_states = env.observation_space.shape[0]
	# continuous_actionspace = isinstance(env.action_space, gym.spaces.box.Box)
	# if continuous_actionspace:
	# 	n_actions = env.action_space.shape[0]
	# else:
	# 	n_actions = env.action_space.n

	# R_range = 2

 
	# env.seed(0)

	# errors_name = env.spec.id + '_single_episode_errors_' + loss_name + '_' + str(R_range)
	#env_name = env.spec.id

	##########for linear system setup#############
	dataset = ReplayMemory(20000)
	x_d = np.zeros((0,2))
	x_next_d = np.zeros((0,2))
	r_d = np.zeros((0))

	n_states = 2
	n_actions = 2
	continuous_actionspace = True
	
	state_dim = 2
	extra_dim = state_dim - 2
	errors_name = 'lin_dyn_single_episode_errors_' + loss_name + '_' + str(R_range)
	env_name = 'lin_dyn'
	#########################################


	#P_hat = ACPModel(n_states, n_actions, clip_output=False)
	P_hat = DirectEnvModel(n_states,n_actions, MAX_TORQUE)	

	pe = Policy(n_states, n_actions, continuous=continuous_actionspace, std=-2.6)

	#pe.load_state_dict(torch.load('policy_reinforce_cartpole.pth', map_location=device))

	P_hat.to(device).double()
	pe.to(device).double()

	for p in P_hat.parameters():
		p.requires_grad = True


	opt = optim.SGD(P_hat.parameters(), lr = 1e-4, momentum=0.99)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[500,700,1500], gamma=0.1)
	#1. Gather data, 2. train model with mle 3. reinforce 4. run policy, add new data to data

	#1. Gather data
	losses = []
	pe_params = []
	best_loss = 20

	# batch_size = 10
	# batch_states = []
	# batch_rewards = []
	# batch_actions = []
	# batch_counter = 0

	unroll_num = num_states - R_range # T_i - j

	for ep in range(num_episodes):
		x_0 = 2*np.random.random(size=(2,)) - 0.5
		x_tmp, x_next_tmp, u_list, _, r_list = lin_dyn(max_actions, pe, [], x=x_0, discount=1.0)

		for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_list):
			dataset.push(x, x_next, u, r)

		#have to keep track of trajectories (where start and where end)
		#since max_actions is same for every episode, can just use indices in memory

	
	batch = dataset.sample(batch_size, structured=True, max_actions=max_actions, num_episodes=num_episodes)
	pdb.set_trace()
		#dataset.memory[ep:ep + max_actions]
	for i in range(num_iters):
		#batch = dataset.sample(max_actions, structured=True, num_episodes=num_episodes)
		states_prev = torch.tensor([samp.state for samp in batch]).double()
		states_next = torch.tensor([samp.next_state for samp in batch]).double()
		rewards = torch.tensor([samp.reward for samp in batch]).double()
		actions_tensor = torch.tensor([samp.action for samp in batch]).double()
		discounted_rewards_tensor = torch.DoubleTensor(discount_rewards(rewards.unsqueeze(1), discount, center=False)).to(device)

		state_actions = torch.cat((states_prev,actions_tensor), dim=1)
		#pamlize the real trajectory (states_next)
		pe.zero_grad()
		true_log_probs_t = get_selected_log_probabilities(pe, states_prev, actions_tensor)
		#1
		true_term = torch.zeros((unroll_num, R_range, n_actions))
		for ell in range(unroll_num):
			for_true_discounted_rewards = discounted_rewards_tensor[ell:R_range + ell]

			#true_term = torch.sum(true_log_probs_t[:R_range + 1] * ((for_true_discounted_rewards - for_true_discounted_rewards.mean())/(for_true_discounted_rewards.std() + 1e-5)))

			#don't sum yet
			if len(for_true_discounted_rewards) > 1:
				true_term[ell] = true_log_probs_t[ell:R_range + ell] * ((for_true_discounted_rewards - for_true_discounted_rewards.mean())/(for_true_discounted_rewards.std() + 1e-5))
			else:
				true_term[ell] = true_log_probs_t[ell:R_range + ell] *for_true_discounted_rewards

		#true_term = torch.sum(true_log_probs_t[:, :R_range + 1])

		##########################################
		true_pe_grads_attached = grad(true_term.mean(), pe.parameters(), create_graph=True)
		true_pe_grads = [true_pe_grads_attached[i].detach() for i in range(0,len(true_pe_grads_attached))]
		print(true_pe_grads, file=true_pe_grads_file)
		
		# if sum([torch.sum(i) for i in true_pe_grads]) == torch.zeros(1).double():
		# 	pdb.set_trace()
		##########################################

		#2b
		rewards_np = np.asarray(rewards)

		#true_rewards_after_R = torch.zeros((num_episodes, unroll_num, max_actions))
		true_rewards_after_R = torch.zeros((unroll_num, R_range))

		##from here double check
		for ell in range(unroll_num):
			#length of row: max_actions - ell
			#rewards_ell = np.hstack((np.zeros((R_range + ell + 1)), rewards_np[ell + R_range + 1:]))
			rewards_ell = np.hstack((np.zeros((R_range + ell)), rewards_np[ell + R_range:]))
			discounted_rewards_after_skip = discount_rewards(rewards_ell, discount, center=False)[ell:ell+R_range]
			try:
				true_rewards_after_R[ell] = torch.DoubleTensor(discounted_rewards_after_skip).to(device)#torch.DoubleTensor(np.pad(discounted_rewards_after_skip, (0, ell), 'constant', constant_values=0)).to(device)
			except RuntimeError:
				pdb.set_trace()
				print('Oops! RuntimeError')

	#for i in range(num_iters):
		opt.zero_grad()
		pe.zero_grad() 
		#simplify: unroll in one function like for true terms
		# model_rewards = torch.zeros((unroll_num, R_range + 1, 1))

		# k_step_log_probs = torch.zeros((unroll_num, R_range + 1, 2))
		
		model_term = torch.zeros(unroll_num, R_range, n_actions)
		step_state = state_actions.to(device)

		#all max_actions states get unrolled R_range steps



		#Gotta check if the below line is actually giving correct results because the shape of the output is a little weird (check unroll function)
		model_x_curr, model_x_next, model_a_list, model_r_list = P_hat.unroll(step_state, pe, n_states, steps_to_unroll=R_range, continuous_actionspace=continuous_actionspace)


		first_returns = discount_rewards(model_r_list, discount, center=False).squeeze()#transpose(-1,0)#.view(-1,1)

		for ell in range(unroll_num):
			second_returns = true_rewards_after_R.double()#discounted_rewards_tensor[R_range + ell + 1 + 1]

			total_model_returns = first_returns[ell] + second_returns[ell]#:R_range+ell].sum(dim=0) + second_returns[ell]
			model_log_probs = get_selected_log_probabilities(pe, model_x_curr[ell, :,:], model_a_list[ell,:,:])
			#the shape of model_log_probs is weird due to reason above, not sure if this multiplication would give correct results
			if len(total_model_returns) > 1:
				model_term[ell] = model_log_probs * ((total_model_returns - total_model_returns.mean())/(total_model_returns.std() + 1e-5)).unsqueeze(1)
			else:
				model_term[ell] = model_log_probs * total_model_returns.unsqueeze(1)

		model_pe_grads = grad(model_term.mean(), pe.parameters(), create_graph=True)#[1:]

		print((i,model_pe_grads), file=model_pe_grads_file)

		


			#######old stuff
			#unroll_num x R_range x 1
			# with torch.no_grad():
			# 	j_step_discounted_model_rewards = discount_rewards(model_rewards, discount, center=False)

			# 	model_returns = torch.cat(((j_step_discounted_model_rewards + true_rewards_after_R[:, :R_range + 1]), true_rewards_after_R[:, R_range + 1:]),dim=1)[:, :R_range + 1]

			# 	centered_model_returns = (model_returns - torch.mean(model_returns, dim=1).unsqueeze(1).repeat(1,model_returns.shape[1],1)) / (torch.std(model_returns, dim=1).unsqueeze(1).repeat(1,model_returns.shape[1],1) + 1e-5) 

			# model_term = torch.sum(k_step_log_probs * centered_model_returns)
			##############
		#model_term = torch.sum(torch.sum(k_step_log_probs, dim=0), dim=0)

		##########################################
		# model_pe_grads = grad(model_term, pe.parameters(), create_graph=True, only_inputs=True,allow_unused=True)[1:]

		# print(model_pe_grads, file=model_pe_grads_file)

		# print('model_term: ', model_term)
		# # print('log_probs: ', k_step_log_probs)
		# print('returns: ', centered_model_returns)

		##########################################

		loss = 0
		for x,y in zip(true_pe_grads, model_pe_grads):
			# if torch.isnan(torch.sum((x-y)**2)):
			# 	print(shortened)
			# 	pdb.set_trace()
			loss += torch.sum(torch.norm(x-y)) #do inner product instead?

		if loss < best_loss:
			torch.save(P_hat.state_dict(), env_name + '_' + loss_name + '_trained_model.pth')
			best_loss = loss.detach()  

		#grad(loss, P_hat.parameters())
		#pdb.set_trace()
		loss.backward()
		#print(P_hat.fc1.weight.grad)
		# for p in P_hat.named_parameters():

		# P_hat_grads = 
		# print((i,P_hat_grads), file=P_hat_grads_file)
		if i < 999:
			opt.step()

		if torch.isnan(torch.sum(P_hat.fc1.weight.data)):
			print('weight turned to nan, check gradients')
			pdb.set_trace()
		losses.append(loss.data.cpu())
		lr_schedule.step()

		if i < 998:
			print("ep: {}, paml_loss on PAML_trained = {:.7f}".format(i, loss.data.cpu()))
		elif i == 998:
			with torch.no_grad():
				squared_errors = torch.zeros_like(states_next)
				step_state = state_actions.to(device)

				for step in range(R_range):
					next_step_state = P_hat(step_state)
					squared_errors += (states_next - next_step_state)**2

				model_loss = torch.sum(squared_errors)

			print("ep: {}, mle_loss on PAML_trained = {:.7f}".format(i, model_loss.data.cpu()))

		else:
			print("ep: {}, paml_loss on MLE_trained = {:.7f}".format(i, loss.data.cpu()))

			with torch.no_grad():
				squared_errors = torch.zeros_like(states_next)
				step_state = state_actions.to(device)

				for step in range(R_range):
					next_step_state = P_hat(step_state)
					squared_errors += (states_next - next_step_state)**2

				model_loss = torch.sum(squared_errors)

			print("ep: {}, mle_loss on MLE_trained = {:.7f}".format(i, model_loss.data.cpu()))

		if i == 998:
			#P_hat.load_state_dict(torch.load('lin_dyn_paml_trained_model.pth', map_location=device))

			P_hat.load_state_dict(torch.load('lin_dyn_1_mle_trained_model.pth', map_location=device))
		





	# if ep % 1 ==0:
	# 	if loss_name == 'paml':
	# 		errors_val = paml_validation_loss(env, P_hat, pe, val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val, R_range, val_size, n_actions, max_actions, device)

	# 	elif loss_name == 'mle':
	# 		errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, n_actions, max_actions, device)


# all_val_errs = torch.mean(errors_val, dim=0)
# print('saving multi-step errors ...')
# np.save(os.path.join(path, (errors_name+'_val')), np.asarray(all_val_errs))
print(best_loss)
print(os.path.join(path,loss_name))
np.save(os.path.join(path,loss_name),np.asarray(losses))
# np.save(os.path.join(path,loss_name +'_pe_params'),np.asarray(pe_params))