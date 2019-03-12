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
device = 'cpu' # .... much slower with cuda ....
loss_name = 'paml'


MAX_TORQUE = 1.

if __name__ == "__main__":
	#initialize pe 
	num_episodes = 1000
	max_actions = 20
	#can't have max_actions = 1... works now
	num_states = max_actions + 1
	num_iters = 1003
	opt_step_def = 1

	discount = 0.9
	R_range = 2

	batch_size = 1000
	
	# true_log_probs_grad_file = open('true_log_probs_grad.txt', 'w')
	# model_log_probs_grad_file = open('model_log_probs_grad.txt', 'w')

	true_pe_grads_file = open('true_pe_grads.txt', 'w')
	model_pe_grads_file = open('model_pe_grads.txt', 'w') 

	true_returns_file = open('true_returns.txt', 'w')
	model_returns_file = open('model_returns.txt', 'w') 

	# dm_control2gym.create_render_mode('rs', show=False, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(), depth=False, scene_option=None)

	# # env = dm_control2gym.make(domain_name="pendulum")#, task_name="balance")
	# # env.spec.id = 'dm_pendulum'
	# env = gym.make('CartPole-v0')

	# states_dim = env.observation_space.shape[0]
	# continuous_actionspace = isinstance(env.action_space, gym.spaces.box.Box)
	# if continuous_actionspace:
	# 	actions_dim = env.action_space.shape[0]
	# else:
	# 	actions_dim = env.action_space.n

	# R_range = 2

 
	# env.seed(0)

	# errors_name = env.spec.id + '_single_episode_errors_' + loss_name + '_' + str(R_range)
	#env_name = env.spec.id

	##########for linear system setup#############
	dataset = ReplayMemory(50000)
	x_d = np.zeros((0,2))
	x_next_d = np.zeros((0,2))
	r_d = np.zeros((0))

	states_dim = 2
	actions_dim = 2
	continuous_actionspace = True
	
	state_dim = 2
	extra_dim = state_dim - 2
	errors_name = 'lin_dyn_single_episode_errors_' + loss_name + '_' + str(R_range)
	env_name = 'lin_dyn'
	#########################################

	#P_hat = ACPModel(states_dim, actions_dim, clip_output=False)
	P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE)	
	pe = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-4.6)


	#pe.load_state_dict(torch.load('policy_reinforce_cartpole.pth', map_location=device))

	P_hat.to(device).double()
	pe.to(device).double()

	for p in P_hat.parameters():
		p.requires_grad = True

	#opt = optim.SGD(P_hat.parameters(), lr=1e-5, momentum=0.90, nesterov=True)
	opt = optim.Adam(P_hat.parameters(), lr=1e-4, weight_decay=1e-5) #increase wd?
	
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000,2000,3000], gamma=0.1)
	#1. Gather data, 2. train model with mle 3. reinforce 4. run policy, add new data to data

	#1. Gather data
	losses = []
	grads = []
	pe_params = []
	r_norms = []
	best_loss = 20

	unroll_num = num_states - R_range # T_i - j

	
	x_0 = 2*np.random.random(size=(2,)) - 0.5
	for ep in range(num_episodes):
		x_tmp, x_next_tmp, u_list, _, r_list = lin_dyn(max_actions, pe, [], x=x_0, discount=0.0)
		for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_list):
			dataset.push(x, x_next, u, r)

		#have to keep track of trajectories (where start and where end)
		#since max_actions is same for every episode, can just use indices in memory

		#dataset.memory[ep:ep + max_actions]
	for i in range(num_iters):
		#assuming samples are disjoint
		batch = dataset.sample(batch_size, structured=True, max_actions=max_actions, num_episodes=num_episodes)

		states_prev = torch.zeros((batch_size, max_actions, states_dim)).double().to(device)
		states_next = torch.zeros((batch_size, max_actions, states_dim)).double().to(device)
		rewards = torch.zeros((batch_size, max_actions)).double().to(device)
		actions_tensor = torch.zeros((batch_size, max_actions, actions_dim)).double().to(device)
		discounted_rewards_tensor = torch.zeros((batch_size, max_actions, 1)).double().to(device)

		for b in range(batch_size):
			#batch = dataset.sample(max_actions, structured=True, num_episodes=num_episodes)
			states_prev[b] = torch.tensor([samp.state for samp in batch[b]]).double().to(device)
			states_next[b] = torch.tensor([samp.next_state for samp in batch[b]]).double().to(device)
			rewards[b] = torch.tensor([samp.reward for samp in batch[b]]).double().to(device)
			actions_tensor[b] = torch.tensor([samp.action for samp in batch[b]]).double().to(device)
			discounted_rewards_tensor[b] = discount_rewards(rewards[b].unsqueeze(1), discount, center=False).to(device)
		state_actions = torch.cat((states_prev,actions_tensor), dim=2)
		#pamlize the real trajectory (states_next)

		pe.zero_grad()
		true_log_probs_t = get_selected_log_probabilities(pe, states_prev.view(-1,states_dim), actions_tensor.view(-1,actions_dim)).view(batch_size,max_actions,actions_dim)

		#true_log_probs_grad = grad(true_log_probs_t.mean(), pe.parameters(), create_graph=True, retain_graph=True)

		#print((i,true_log_probs_grad), file=true_log_probs_grad_file)

		#1
		true_term = torch.zeros((batch_size,unroll_num, actions_dim))
		alt_true_term = torch.zeros((batch_size, unroll_num, R_range, actions_dim))

		r1 = torch.zeros((unroll_num,batch_size, R_range, 1))
		for ell in range(unroll_num):
			for_true_discounted_rewards = discounted_rewards_tensor[:,ell:R_range + ell]
			#r1[ell] = for_true_discounted_rewards
			#true_term = torch.sum(true_log_probs_t[:R_range + 1] * ((for_true_discounted_rewards - for_true_discounted_rewards.mean())/(for_true_discounted_rewards.std() + 1e-5)))
			#don't sum yet

			#print((i, ell, for_true_discounted_rewards.view(-1,2)), file=true_returns_file)

			if for_true_discounted_rewards.shape[1] > 1:
				#not tested for R_range > 1
				#alt_true_term[:,ell] = true_log_probs_t[:,ell:R_range + ell]
				
				true_term[:,ell] = torch.einsum('ijk,ijl->ik',[true_log_probs_t[:,ell:R_range + ell], ((for_true_discounted_rewards - for_true_discounted_rewards.mean(dim=1).unsqueeze(1))/(for_true_discounted_rewards.std(dim=1).unsqueeze(1) + 1e-5))])

				#true_term[:,ell] = torch.einsum('ijk,ijl->ik',[true_log_probs_t[:,ell:R_range + ell], true_log_probs_t[:,ell:R_range + ell]])

			else:
				#true_term[:,ell] = torch.einsum('ijk,ijl->ik', [true_log_probs_t[:,ell:R_range + ell], true_log_probs_t[:,ell:R_range + ell]])
				true_term[:,ell] = torch.einsum('ijk,ijl->ik',[true_log_probs_t[:,ell:R_range + ell], for_true_discounted_rewards]) 


		#true_term = torch.sum(true_log_probs_t[:, :R_range + 1])
		##########################################
		#is it ok to take mean across whole batch like this?
		true_pe_grads_attached = grad(true_term.mean(), pe.parameters(), create_graph=True)
		true_pe_grads = [true_pe_grads_attached[t].detach() for t in range(0,len(true_pe_grads_attached))]

		print((i,true_pe_grads), file=true_pe_grads_file)
		
		# if sum([torch.sum(i) for i in true_pe_grads]) == torch.zeros(1).double():
		# 	pdb.set_trace()
		##########################################

		#2b
		rewards_np = np.asarray(rewards)

		#true_rewards_after_R = torch.zeros((num_episodes, unroll_num, max_actions))
		true_rewards_after_R = torch.zeros((batch_size, unroll_num, R_range))
		for ell in range(unroll_num):
			#length of row: max_actions - ell
			#rewards_ell = np.hstack((np.zeros((R_range + ell + 1)), rewards_np[ell + R_range + 1:]))
			rewards_ell = torch.DoubleTensor(np.hstack((np.zeros((batch_size, R_range + ell)), rewards_np[:,ell + R_range:]))).unsqueeze(2).to(device)

			discounted_rewards_after_skip = discount_rewards(rewards_ell, discount, center=False, batch_wise=True)[:,ell:ell+R_range]
			
			try:
				true_rewards_after_R[:, ell] = discounted_rewards_after_skip.squeeze(2).to(device)#torch.DoubleTensor(np.pad(discounted_rewards_after_skip, (0, ell), 'constant', constant_values=0)).to(device)
			except RuntimeError:
				pdb.set_trace()
				print('Oops! RuntimeError')

		opt_steps = opt_step_def if i < num_iters-2 else 1
		for j in range(opt_steps):
			opt.zero_grad()
			pe.zero_grad() 
			#simplify: unroll in one function like for true terms
			# model_rewards = torch.zeros((unroll_num, R_range + 1, 1))

			# k_step_log_probs = torch.zeros((unroll_num, R_range + 1, 2))
			model_term = torch.zeros(batch_size, unroll_num, actions_dim)
			alt_model_term = torch.zeros(batch_size, unroll_num, R_range, actions_dim)

			step_state = state_actions.to(device)

			#all max_actions states get unrolled R_range steps
			model_x_curr, model_x_next, model_a_list, model_r_list = P_hat.unroll(step_state[:,:unroll_num,:], pe, states_dim, steps_to_unroll=R_range, continuous_actionspace=continuous_actionspace)

			#r_norms.append(torch.norm(model_r_list.detach().data - rewards).numpy())			
			# pdb.set_trace()
			# plt.figure()
			# plt.plot(model_x_curr[0,:,0,0].detach().numpy(), model_x_curr[0,:,0,1].detach().numpy())
			# plt.plot(model_x_curr[0,:,1,0].detach().numpy(), model_x_curr[0,:,1,1].detach().numpy())
			# plt.show()

			first_returns = discount_rewards(model_r_list.squeeze(3), discount, center=False, batch_wise=True)#.squeeze(3)#transpose(-1,0)#.view(-1,1)

			second_returns = true_rewards_after_R.double().to(device)#discounted_rewards_tensor[R_range + ell + 1 + 1]
			#r2 = torch.zeros((unroll_num,batch_size, R_range, 1))

			for ell in range(unroll_num):
				total_model_returns = first_returns[:,ell] + second_returns[:,ell]#:R_range+ell].sum(dim=0) + second_returns[ell]
				#r2[ell] = total_model_returns.unsqueeze(2)
				model_log_probs = get_selected_log_probabilities(pe, model_x_curr[:, ell, :,:].contiguous().view(-1,states_dim), model_a_list[:,ell,:,:].contiguous().view(-1,actions_dim)).view(batch_size, -1, actions_dim)

				#model_log_probs_grad = grad(model_log_probs.mean(), pe.parameters(), create_graph=True, retain_graph=True)

				#print((i,model_log_probs_grad), file=model_log_probs_grad_file)
				#the shape of model_log_probs is weird due to reason above, not sure if this multiplication would give correct results
				#might need einsum for the below multiplications
				#print((i, ell, total_model_returns), file=model_returns_file)

				if total_model_returns.shape[1] > 1:
					#alt_model_term[:,ell] = model_log_probs

					model_term[:,ell] = torch.einsum('ijk,ijl->ik', [model_log_probs, ((total_model_returns - total_model_returns.mean(dim=1).unsqueeze(1))/(total_model_returns.std(dim=1).unsqueeze(1) + 1e-5)).unsqueeze(2)])
					
					#model_term[:,ell] = torch.einsum('ijk,ijl->ik', [model_log_probs, model_log_probs])
				else:
					#model_term[:,ell] = torch.einsum('ijk,ijl->ik', [model_log_probs, total_model_returns.unsqueeze(2)])
					model_term[:,ell] = torch.einsum('ijk,ijl->ik',[model_log_probs, total_model_returns.unsqueeze(2)])
					#model_term[:,ell] = torch.einsum('ijk,ijl->ik', [model_log_probs, model_log_probs])

			model_pe_grads = grad(model_term.mean(), pe.parameters(), create_graph=True)

			#pdb.set_trace()
			print((i,model_pe_grads), file=model_pe_grads_file)
			loss = 0 
			cos = nn.CosineSimilarity(dim=0, eps=1e-6)
			for x,y in zip(true_pe_grads, model_pe_grads):
					# if torch.isnan(torch.sum((x-y)**2)):
					# 	print(shortened)
					# 	pdb.set_trace()
				# if len(x.shape) > 1:
				# 	loss = loss + torch.sum(torch.norm(torch.einsum('ij,ij->j',[x,y]))) #do inner product instead?
				# else: 
				# 	loss = loss + torch.sum(torch.norm(x.dot(y)))
				#loss = loss - torch.sum(cos(x,y))
				loss = loss + torch.norm(x-y)
			#r_norms.append(loss.detach().cpu())
			#loss = torch.norm(r2 - r1)
			if loss < best_loss:
				torch.save(P_hat.state_dict(), env_name + '_' + loss_name + '_trained_model.pth')
				best_loss = loss.detach()  

			loss.backward()

			###############  FOR CURRICULUM LEARNING   #############
			# if i > 1:
			# 	if loss.data.cpu() <= initial_loss * 0.1 and R_range < max_actions:
			# 		R_range += 1
			# 		lr_schedule.step()
			# 		print("******** Horizon: {} ************".format(R_range))
			# else:
			# 	initial_loss = loss.data.cpu()
			########################################################


			if i < num_iters-1:
				nn.utils.clip_grad_value_(P_hat.parameters(), 4.0)
				#grads.append(torch.sum(P_hat.fc1.weight.grad))
				if torch.norm(P_hat.fc1.weight.grad) == 0:
					pdb.set_trace()
				opt.step()

			if torch.isnan(torch.sum(P_hat.fc1.weight.data)):
				print('weight turned to nan, check gradients')
				pdb.set_trace()

			losses.append(loss.data.cpu())
			lr_schedule.step()

			if i < num_iters-2:
				print("ep: {}, paml_loss on PAML_trained = {:.7f}".format(i, loss.data.cpu()))
			
			elif i == num_iters-2:
				model_loss = P_hat.mle_validation_loss(states_next, state_actions.to(device), pe, R_range)
				print("ep: {}, mle_loss on PAML_trained = {:.7f}".format(i, model_loss.data.cpu()))

			else:
				print("ep: {}, paml_loss on MLE_trained = {:.7f}".format(i, loss.data.cpu()))

				model_loss = P_hat.mle_validation_loss(states_next, state_actions.to(device), pe, R_range)
				print("ep: {}, mle_loss on MLE_trained = {:.7f}".format(i, model_loss.data.cpu()))

			if i == num_iters-2:
				#P_hat.load_state_dict(torch.load('lin_dyn_paml_trained_model.pth', map_location=device))
				P_hat.load_state_dict(torch.load('lin_dyn_mle_hor_2_traj_20_batch_1000_sameStartState.pth', map_location=device))
				batch_size = num_episodes
		

	# if ep % 1 ==0:
	# 	if loss_name == 'paml':
	# 		errors_val = paml_validation_loss(env, P_hat, pe, val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val, R_range, val_size, actions_dim, max_actions, device)

	# 	elif loss_name == 'mle':
	# 		errors_val = mle_validation_loss(P_hat, pe, val_states_next_tensor, state_actions_val, actions_dim, max_actions, device)


# all_val_errs = torch.mean(errors_val, dim=0)
# print('saving multi-step errors ...')
# np.save(os.path.join(path, (errors_name+'_val')), np.asarray(all_val_errs))
print(best_loss)
print(os.path.join(path,loss_name))
# np.save(os.path.join(path,'grads'),np.asarray(grads))
np.save(os.path.join(path,'paml_rewards_norm'),np.asarray(r_norms))
np.save(os.path.join(path,loss_name),np.asarray(losses))
# np.save(os.path.join(path,loss_name +'_pe_params'),np.asarray(pe_params))