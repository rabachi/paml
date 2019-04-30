import numpy as np
# import gym
import sys

import torch
import math
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
# from torch.autograd import grad, gradgradcheck

import pdb
import os
from models import *
from networks import *
from utils import *
from get_data import save_stats

def actor_critic_paml_train(P_hat, 
							pe, 
							value_estimator,
							value_opt,
							opt, 
							num_episodes, 
							num_starting_states, 
							states_dim, 
							actions_dim, 
							use_model, 
							discount, 
							max_actions, 
							train, 
							A_numpy, 
							lr_schedule,
							num_iters,
							losses,
							true_r_list,
							true_x_curr,
							true_x_next,
							true_a_list,
							true_a_prime_list,
							step_state,
							value_loss_coeff):
	best_loss = 15000
	env_name = 'lin_dyn'
	batch_size = num_episodes*num_starting_states
	R_range = max_actions
	unroll_num = max_actions
	end_of_trajectory = 1
	num_iters = num_iters if train else 1

	ell = 0
	# true_returns = torch.zeros_like(true_r_list)
	true_returns = discount_rewards(true_r_list[:,0], discount, center=False, batch_wise=True)
	true_log_probs = get_selected_log_probabilities(pe, true_x_next, true_a_prime_list).squeeze()

	for i in range(num_iters):
		#calculate true gradients
		pe.zero_grad()

		# true_returns = discount_rewards(true_r_list[:,ell], discount, center=False, batch_wise=True) 
		true_value = value_estimator(torch.cat((true_x_next.squeeze(), true_a_prime_list.squeeze()),dim=2)) #check dims  #in a2c they use this as the baseline, here I'm just going to use it as an estimate of the advantage
		true_value_loss = (true_returns - true_value).pow(2).mean() 

		true_term = true_log_probs * true_value #torch.einsum('ijk,ijl->ik', [true_log_probs, true_value])# + true_value_loss * value_loss_coeff #should this be added here?

		save_stats(true_returns, true_r_list, true_a_prime_list, true_x_next, value=true_value, prefix='true_actorcritic_')

		true_pe_grads = []
		for st in range(num_starting_states):
			true_pe_grads_attached = grad(true_term[st*num_episodes:num_episodes*(st+1)].mean(),list(pe.parameters()) + list(value_estimator.parameters()), create_graph=True)
			true_pe_grads.append([true_pe_grads_attached[t].detach() for t in range(0,len(true_pe_grads_attached))])

		#probably don't need these ... .grad is not accumulated with the grad() function
		pe.zero_grad()
		#calculate model gradients
		step_state = torch.cat((true_x_curr, true_a_list),dim=3) 
	
		model_x_curr, model_x_next, model_a_list, model_r_list, model_a_prime_list = P_hat.unroll(step_state[:,0,:], pe, states_dim, A_numpy, steps_to_unroll=1, continuous_actionspace=True, use_model=use_model, policy_states_dim=states_dim)
		model_value = value_estimator(torch.cat((model_x_next[:,:,0,:], model_a_prime_list[:,:,0,:]),dim=2))#check dims

		model_log_probs = get_selected_log_probabilities(pe, model_x_next, model_a_prime_list).squeeze()
		model_returns = discount_rewards(model_r_list[:,0], discount, center=False, batch_wise=True) #should i take out the ell bit?
		model_value_loss = (model_returns - model_value).pow(2).mean()

		model_term = model_log_probs * model_value #torch.einsum('ijk,ijl->ik', [model_log_probs,  model_value])# + model_value_loss * value_loss_coeff

		#if i == num_iters - 1:
		save_stats(None, model_r_list, model_a_prime_list, model_x_next, value=model_value, prefix='model_actorcritic_')

		model_pe_grads = []
		for st in range(num_starting_states):
			model_pe_grads.append(list(grad(model_term[num_episodes*st:num_episodes*(st+1)].mean(),list(pe.parameters()) + list(value_estimator.parameters()), create_graph=True)))

		#model - true
		loss = 0
		for stt,stm in zip(true_pe_grads, model_pe_grads):
			for x,y in zip(stt, stm):
				loss = loss + torch.norm(x-y)**2

		loss = torch.sqrt(loss/num_starting_states)

		if loss.detach().cpu() < best_loss and use_model:
			#Save model and losses so far
			torch.save(P_hat.state_dict(), 'act_model_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states.pth'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory))
			np.save('act_loss_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory), np.asarray(losses))
			np.save('act_valueloss_paml_checkpoint_train_{}_{}_horizon_traj{}_using{}states'.format(train, env_name, R_range, max_actions+1, end_of_trajectory), np.asarray(true_value_loss.mean().detach()))
			best_loss = loss.detach().cpu()

		#update model
		if train: 
			opt.zero_grad()

			loss.backward(retain_graph=True)
			nn.utils.clip_grad_value_(P_hat.parameters(), 5.0)
			opt.step()

			# value_opt.zero_grad()
			# (model_value_loss + true_value_loss).backward()
			# value_opt.step()

			if torch.isnan(torch.sum(P_hat.fc1.weight.data)):
				print('weight turned to nan, check gradients')
				pdb.set_trace()

		losses.append(loss.data.cpu())
		lr_schedule.step()

		if train and i < 1: #and j < 1 
			initial_loss = losses[0]#.data.cpu()
			print('initial_loss',initial_loss)

		if train:
			print("R_range: {:3d} | batch_num: {:5d} | paml_loss: {:.5f} | value_loss: {:.5f}".format(R_range, i, loss.data.cpu(),(model_value_loss + true_value_loss).mean().detach()))
		else: 
			print("Validation loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}".format(use_model, R_range, i, loss.data.cpu()))
			print("---------------------------------------------------------------------------------")


if __name__=="__main__":
	
	rs = 7
	torch.manual_seed(rs)
	np.random.seed(rs)	

	num_starting_states = 200
	num_episodes = 1
	batch_size = num_starting_states * num_episodes

	val_num_episodes = 10
	val_num_starting_states = 125
	val_batch_size = val_num_starting_states*val_num_episodes
	
	num_iters = 6000
	losses = []
	unroll_num = 1

	value_loss_coeff = 1.0
	MAX_TORQUE = 1.0
	discount = 0.9

	max_actions = 10

	states_dim = 2
	actions_dim = 2
	salient_states_dim = 2
	continuous_actionspace = True
	R_range = max_actions


	use_model = True
	train_value_estimate = False
	train = True
	

	action_multiplier = 0.1
	value_estimator = Value(states_dim, actions_dim)
	value_estimator.double()
	policy_estimator = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-2.0, max_torque=MAX_TORQUE, action_multiplier=0.1)
	policy_estimator.double()

	P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE)
	P_hat.double()
	#P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon6_traj7.pth', map_location=device))
	P_hat.load_state_dict(torch.load('act_model_paml_checkpoint_train_True_lin_dyn_horizon10_traj11_using1states.pth', map_location=device))

	value_opt = optim.SGD(value_estimator.parameters(), lr=1e-3, momentum=0.90, nesterov=True) 
	value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_opt, milestones=[1500,3000], gamma=0.1)
	opt = optim.SGD(P_hat.parameters(), lr=1e-6, momentum=0.90, nesterov=True)
	#opt = optim.Adam(P_hat.parameters(), lr=1e-5, weight_decay=1e-8)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000,2000,3000], gamma=0.1)
		
	A_all = {}
	A_all[5] = np.array([[-0.2,  0.1,  0.1,  0.1,  0.1],
			       		[ 0.1,  0.1,  0.1,  0.1,  0.1],
			       		[ 0.1,  0.1,  0.5,  0.1,  0.1],
			       		[ 0.1,  0.1,  0.1,  0.8,  0.1],
			       		[ 0.1,  0.1,  0.1,  0.1, -0.9]])

	A_all[4] = np.array([[-0.2,  0.3,  0.3,  0.3],
			       		[ 0.3, -0.4,  0.3,  0.3],
			       		[ 0.3,  0.3,  0.3,  0.3],
			      		[ 0.3,  0.3,  0.3, -0.1]])

	A_all[3] = np.array([[-0.5, -0.5, -0.5],
							[ 0.3, -0.2,  0.3],
							[ 0.3,  0.3,  0.4]])

	A_all[2] = np.array([[0.9, 0.4], [-0.4, 0.9]])


	A_numpy = A_all[2]

	print('Generating sample trajectories ...')
	with torch.no_grad():
		#generate validation data
		val_step_state = torch.zeros((val_batch_size, unroll_num, states_dim+actions_dim)).double()
		for b in range(val_num_starting_states):
			x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
			val_step_state[b*val_num_episodes:val_num_episodes*(b+1),:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

		val_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(val_step_state[:,:unroll_num,:states_dim]) #I think all this does is make the visualizations look better, shouldn't affect performance (or visualizations ... )
		val_true_x_curr, val_true_x_next, val_true_a_list, val_true_r_list, val_true_a_prime_list = P_hat.unroll(val_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=False, policy_states_dim=states_dim)

		#generate training data
		train_step_state = torch.zeros((batch_size, unroll_num, states_dim+actions_dim)).double()
		for b in range(num_starting_states):
			x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
			train_step_state[b*num_episodes:num_episodes*(b+1),:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

		train_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(train_step_state[:,:unroll_num,:states_dim])#I think all this does is make the visualizations look better, shouldn't affect performance (or visualizations ... )
		train_true_x_curr, train_true_x_next, train_true_a_list, train_true_r_list, train_true_a_prime_list = P_hat.unroll(train_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=False, policy_states_dim=states_dim)

	print('Done!')
	#get accuracy of true dynamics on validation data
	true_r_list = val_true_r_list
	true_x_curr = val_true_x_curr
	true_a_list = val_true_a_list
	step_state = val_step_state

	kwargs = {
			'P_hat'              : P_hat, 
			'pe'                 : policy_estimator, 
			'value_estimator'	 : value_estimator,
			'value_opt'			 : value_opt,
			'opt' 				 : opt, 
			'num_episodes'		 : num_episodes, 
			'num_starting_states': num_starting_states, 
			'states_dim'		 : states_dim, 
			'actions_dim'		 : actions_dim, 
			'use_model'			 : False, 
			'discount'			 : discount, 
			'max_actions'		 : max_actions, 
			'train'              : False, 
			'A_numpy'			 : A_numpy, 
			'lr_schedule'		 : lr_schedule,
			'num_iters'          : int(num_iters/120),
			'losses'             : [],
			'true_r_list'        : true_r_list,
			'true_x_curr' 		 : true_x_curr,
			'true_a_list'        : true_a_list,
			'step_state'         : step_state,
			'value_loss_coeff'   : value_loss_coeff
			}

	#pretrain value function
	ell = 0
	if train_value_estimate:
		prefix='true_actorcritic_'
		epochs_value = 300
		best_loss = 1000
		true_r_list = train_true_r_list
		true_x_curr = train_true_x_curr
		true_a_list = train_true_a_list
		# true_returns = torch.zeros_like(true_r_list)
		true_returns = discount_rewards(true_r_list[:,ell], discount, center=False, batch_wise=True)
		for i in range(epochs_value):
			true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
			# true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
			
			save_stats(true_returns, true_r_list, true_a_list, true_x_curr, value=true_value, prefix='true_actorcritic_')
			np.save(prefix+'value_training', true_value.squeeze().detach().cpu().numpy())
			true_value_loss = (true_returns - true_value).pow(2).mean()
			print('Epoch: {:4d} | Value estimator loss: {:.5f}'.format(i,true_value_loss.detach().cpu()))

			if true_value_loss < best_loss:
				torch.save(value_estimator.state_dict(), 'value_estimator_horizon{}_traj{}.pth'.format(R_range, max_actions+1))
				best_loss = true_value_loss
			value_opt.zero_grad()
			true_value_loss.backward()
			value_opt.step()
			value_lr_schedule.step()
			#check validation
			if (i % 10) == 0:
				with torch.no_grad():
					true_x_curr = val_true_x_curr
					true_a_list = val_true_a_list
					true_r_list = val_true_r_list
					# true_returns = torch.zeros_like(true_r_list)
					true_returns = discount_rewards(true_r_list[:,ell], discount, center=False, batch_wise=True)
					true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
					true_value_loss = (true_returns - true_value).pow(2).mean()
					print('Validation value estimator loss: {:.5f}'.format(true_value_loss.detach().cpu()))
				true_r_list = train_true_r_list
				true_x_curr = train_true_x_curr
				true_a_list = train_true_a_list

				# true_returns = torch.zeros_like(train_true_r_list)
				
				true_returns = discount_rewards(true_r_list[:,ell], discount, center=False, batch_wise=True)
	else:
		value_estimator.load_state_dict(torch.load('value_estimator_horizon{}_traj{}.pth'.format(R_range, max_actions+1), map_location=device))


	kwargs['train'] = False
	kwargs['num_episodes'] = val_num_episodes
	kwargs['num_starting_states'] = val_num_starting_states
	kwargs['true_r_list'] = val_true_r_list
	kwargs['true_x_curr'] = val_true_x_curr
	kwargs['true_a_list'] = val_true_a_list
	kwargs['true_x_next'] = val_true_x_next
	kwargs['true_a_prime_list'] = val_true_a_prime_list
	kwargs['step_state'] = val_step_state
	kwargs['use_model'] = False
	actor_critic_paml_train(**kwargs)

	val_losses = []
	kwargs['use_model'] = use_model
	for i in range(120):
		kwargs['train'] = False
		kwargs['num_episodes'] = val_num_episodes
		kwargs['num_starting_states'] = val_num_starting_states
		kwargs['true_r_list'] = val_true_r_list
		kwargs['true_x_curr'] = val_true_x_curr
		kwargs['true_a_list'] = val_true_a_list
		kwargs['true_x_next'] = val_true_x_next
		kwargs['true_a_prime_list'] = val_true_a_prime_list
		kwargs['step_state'] = val_step_state
		kwargs['losses'] = val_losses
		actor_critic_paml_train(**kwargs)

		kwargs['train'] = train
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['true_r_list'] = train_true_r_list
		kwargs['true_x_next'] = train_true_x_next
		kwargs['true_a_prime_list'] = train_true_a_prime_list
		kwargs['step_state'] = train_step_state
		kwargs['losses'] = losses
		actor_critic_paml_train(**kwargs)





