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

def save_stats(returns, r_list, a_list, x_list, value=None, prefix=''):
	if returns is not None:
		np.save(prefix+'returns', returns.detach().cpu().numpy())
	if r_list is not None:
		np.save(prefix+'rewards', r_list.squeeze().detach().cpu().numpy())
	if a_list is not None:
		np.save(prefix+'actions', a_list.squeeze().detach().cpu().numpy())
	if x_list is not None:
		np.save(prefix+'states', x_list.squeeze().detach().cpu().numpy())
		#np.save(prefix+'statesTstates', torch.einsum('ijk,ijk->ij', [x_list.squeeze(), x_list.squeeze()]).detach().cpu().numpy())
	if value is not None:
		np.save(prefix+'value', value.squeeze().detach().cpu().numpy())


def paml_train(P_hat, 
				pe, 
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
				step_state):
	
	best_loss = 15000
	env_name = 'lin_dyn'
	batch_size = num_episodes*num_starting_states
	R_range = max_actions
	unroll_num = 1
	end_of_trajectory = 1
	ell = 0
	num_iters = num_iters if train else 1
	
	#calculate true gradients
	pe.zero_grad()
	true_log_probs = get_selected_log_probabilities(pe, true_x_next, true_a_prime_list).squeeze()
	true_returns = discount_rewards(true_r_list[:,ell, 1:], discount, center=False, batch_wise=True)
	true_term = true_log_probs * true_returns

	save_stats(true_returns, true_r_list, true_a_list, true_x_curr, prefix='true_reinforce_')

	
	true_pe_grads = torch.DoubleTensor()
	for st in range(num_starting_states):
		true_pe_grads_attached = grad(true_term[st*num_episodes:num_episodes*(st+1)].mean(),pe.parameters(), create_graph=True)
		for g in range(len(true_pe_grads_attached)):
			true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))
		# true_pe_grads.append(torch.cat([true_pe_grads_attached[t].detach().view(-1) for t in range(0,len(true_pe_grads_attached))]))

	# true_pe_grads = torch.DoubleTensor()
	# true_pe_grads_attached = grad(true_term.mean(),pe.parameters(), create_graph=True)
	# for g in range(len(true_pe_grads_attached)):
	# 	true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

	# for t in range(0,len(true_pe_grads_attached)):
	# 	true_pe_grads.append(true_pe_grads_attached[t].detach())

	for i in range(num_iters):
		opt.zero_grad()
		pe.zero_grad()
		#calculate model gradients
		#Do i need this line? seems to make a big difference .... 
		step_state = torch.cat((true_x_curr[:,:,0],true_a_list[:,:,0]),dim=2)
		model_x_curr, model_x_next, model_a_list, model_r_list, model_a_prime_list = P_hat.unroll(step_state[:,:unroll_num,:], pe, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=use_model, policy_states_dim=states_dim)

		model_returns = discount_rewards(model_r_list[:,ell, 1:], discount, center=False, batch_wise=True)
		model_log_probs = get_selected_log_probabilities(pe, model_x_next, model_a_prime_list).squeeze()
		#model_term = torch.einsum('ijk,ijl->ik', [model_log_probs,  model_returns])
		model_term = model_log_probs * model_returns

		save_stats(model_returns, model_r_list, model_a_list, model_x_curr, prefix='model_reinforce_')

		# model_pe_grads = torch.DoubleTensor()
		# model_pe_grads_split = grad(model_term.mean(),pe.parameters(), create_graph=True)
		# for g in range(len(model_pe_grads_split)):
		# 	model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))

		model_pe_grads = torch.DoubleTensor()
		for st in range(num_starting_states):
			model_pe_grads_split = list(grad(model_term[num_episodes*st:num_episodes*(st+1)].mean(),pe.parameters(), create_graph=True))
			for g in range(len(model_pe_grads_split)):
				model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))

		#model - true
		# loss = 0
		# cos = nn.CosineSimilarity(dim=0, eps=1e-6)
		# loss = cos(true_pe_grads, model_pe_grads)
		# for stt,stm in zip(true_pe_grads, model_pe_grads):
		# 	for x,y in zip(stt, stm):
		# 		loss = loss + torch.norm(x-y)**2
		
		loss = torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum())#/num_starting_states)

		if loss.detach().cpu() < best_loss and use_model:
			#Save model and losses so far
			torch.save(P_hat.state_dict(), '1model_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states.pth'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory))
			np.save('1loss_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory), np.asarray(losses))
			best_loss = loss.detach().cpu()

		#update model
		if train:
			loss.backward()
			nn.utils.clip_grad_value_(P_hat.parameters(), 5.0)
			opt.step()
			if torch.isnan(torch.sum(P_hat.fc1.weight.data)):
				print('weight turned to nan, check gradients')
				pdb.set_trace()

		losses.append(loss.data.cpu())
		lr_schedule.step()

		if train and i < 1: #and j < 1 
			initial_loss = losses[0]#.data.cpu()
			print('initial_loss',initial_loss)

		if train:
			print("R_range: {:3d} | batch_num: {:5d} | paml_loss: {:.7f}".format(R_range, i, loss.data.cpu()))
		else: 
			print("Validation loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}".format(use_model, R_range, i, loss.data.cpu()))
			print("---------------------------------------------------------------------------------")


def reinforce(pe, 
			A_numpy,
			P_hat,
			num_episodes, 
			states_dim,
			actions_dim, 
			salient_states_dim, 
			R_range,
			use_model,
			optimizer):

	all_rewards = []
	unroll_num = 1
	ell = 0
	batch_states = torch.zeros((batch_size, max_actions, states_dim)).double()
	batch_actions = torch.zeros((batch_size, max_actions, actions_dim)).double()
	batch_returns = torch.zeros((batch_size, max_actions, 1)).double()

	for ep in range(int(num_episodes/batch_size)):
		
		with torch.no_grad():
			step_state = torch.zeros((batch_size, max_actions, states_dim+actions_dim)).double()
			for b in range(batch_size):
				x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
				step_state[b,:unroll_num,:states_dim] = torch.from_numpy(x_0).double()
			
			step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(step_state[:,:unroll_num,:states_dim])
			
			model_x_curr, model_x_next, model_a_list, model_r_list, model_a_prime_list = P_hat.unroll(step_state[:,:unroll_num,:], pe, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=use_model, policy_states_dim=states_dim)
			all_rewards.extend(model_r_list[:,ell,:-1].contiguous().view(-1,max_actions).sum(dim=1).tolist())

		batch_states = model_x_curr
		batch_actions = model_a_list
		batch_returns = discount_rewards(model_r_list[:,ell,:-1], discount, center=True, batch_wise=True)
		
		if (ep == 0):
			save_stats(batch_returns, None, batch_actions, batch_states, value=None, prefix='true_policy_reinforce_')
		if (ep == int(num_episodes/batch_size) - 1):
			save_stats(batch_returns, None, batch_actions, batch_states, value=None, prefix='model_policy_reinforce_')

		model_log_probs = get_selected_log_probabilities(pe, batch_states, batch_actions/0.5).squeeze()
		model_term = model_log_probs * batch_returns

		loss = -model_term.mean()
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		print("Ep: {:4d} | Average of last 10 rewards: {:.3f}".format(ep,sum(all_rewards[-10:])/10.))

	return all_rewards



if __name__=="__main__":

	rs = 7
	torch.manual_seed(rs)
	np.random.seed(rs)	

	MAX_TORQUE = 1.0
	num_episodes = 1

	discount = 0.9
	plan = True
	num_starting_states = 200 if not plan else 4000

	max_actions = 8
	states_dim = 2
	actions_dim = 2
	salient_states_dim = 2
	R_range = max_actions
	
	use_model = True
	continuous_actionspace = True


	batch_size = num_starting_states * num_episodes

	action_multiplier = 0.0
	policy_estimator = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-2.0, max_torque=MAX_TORQUE)
	policy_estimator.double()

	P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE, mult=action_multiplier)#, action_multiplier=0.1)
	P_hat.double()
	#P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon5_traj6.pth', map_location=device))
	P_hat.load_state_dict(torch.load('1model_paml_checkpoint_train_True_lin_dyn_horizon8_traj9_using1states.pth', map_location=device))

	opt = optim.SGD(P_hat.parameters(), lr=1e-5, momentum=0.90, nesterov=True)
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
	train = True
	num_iters = 6000
	losses = []


	########## REINFORCE ############
	if plan:
		optimizer = optim.Adam(policy_estimator.parameters(), lr=0.01)
		batch_size = 5
		use_model = False
		all_rewards = reinforce(policy_estimator, 
								A_numpy,
								P_hat,
								num_episodes*num_starting_states, 
								states_dim,
								actions_dim, 
								salient_states_dim, 
								R_range,
								use_model,
								optimizer
								)

		np.save('reinforce_rewards',np.asarray(all_rewards)) 

		pdb.set_trace()
	#################################



	val_num_episodes = 10
	val_num_starting_states = 125
	val_batch_size = val_num_starting_states*val_num_episodes

	unroll_num = 1

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
	true_a_prime_list = val_true_a_prime_list
	true_x_next = val_true_x_next
	step_state = val_step_state

	kwargs = {
			'P_hat'              : P_hat, 
			'pe'                 : policy_estimator, 
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
			'true_x_next'		 : true_x_next,
			'true_a_list'        : true_a_list,
			'true_a_prime_list'  : true_a_prime_list,
			'step_state'         : step_state
			}

	kwargs['train'] = False
	kwargs['num_episodes'] = val_num_episodes
	kwargs['num_starting_states'] = val_num_starting_states
	kwargs['true_r_list'] = val_true_r_list
	kwargs['true_x_curr'] = val_true_x_curr
	kwargs['true_x_next'] = val_true_x_next
	kwargs['true_a_list'] = val_true_a_list
	kwargs['true_a_prime_list'] = val_true_a_prime_list
	kwargs['step_state'] = val_step_state
	kwargs['use_model'] = False
	paml_train(**kwargs)
	
	val_losses = []
	kwargs['use_model'] = use_model
	for i in range(120):
		kwargs['train'] = False
		kwargs['num_episodes'] = val_num_episodes
		kwargs['num_starting_states'] = val_num_starting_states
		kwargs['true_r_list'] = val_true_r_list
		kwargs['true_x_curr'] = val_true_x_curr
		kwargs['true_x_next'] = val_true_x_next
		kwargs['true_a_list'] = val_true_a_list
		kwargs['true_a_prime_list'] = val_true_a_prime_list
		kwargs['step_state'] = val_step_state
		kwargs['losses'] = val_losses
		paml_train(**kwargs)
	
		kwargs['train'] = train
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['true_r_list'] = train_true_r_list
		kwargs['true_x_curr'] = train_true_x_curr
		kwargs['true_x_next'] = train_true_x_next
		kwargs['true_a_list'] = train_true_a_list
		kwargs['true_a_prime_list'] = train_true_a_prime_list
		kwargs['step_state'] = train_step_state
		kwargs['losses'] = losses
		paml_train(**kwargs)




