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
from rewardfunctions import *

import pdb


#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def discount_rewards(list_of_rewards, discount):
	# r_tensor = torch.FloatTensor(list_of_rewards)
	# all_discounts = [discount**i for i in range(r_tensor.shape[0])]
	# all_discounts.reverse()
	# G_list = r_tensor * torch.FloatTensor(all_discounts)
	#return G_list - G_list.mean()
	if isinstance(list_of_rewards, list):
		r = np.array([discount**i * list_of_rewards[i] 
			for i in range(len(list_of_rewards))])
		# Reverse the array direction for cumsum and then
		# revert back to the original order
		r = r[::-1].cumsum()[::-1]
		return torch.FloatTensor((r - r.mean())/(r.std()+ 1e-5))

	elif isinstance(list_of_rewards, torch.Tensor):
		list_of_rewards = list_of_rewards.squeeze()
		discounted_batch = torch.zeros((list_of_rewards.shape[0],list_of_rewards.shape[1]))

		for batch in range(list_of_rewards.shape[0]):
			#r = torch.tensor([discount**i * list_of_rewards[batch, i] for i in range(list_of_rewards.shape[1])]) 
			#print([torch.sum(r[i:]) for i in range(r.shape[0])])
			discounted_batch[batch] = torch.tensor([torch.sum(discount**i * list_of_rewards[batch,i:]) for i in range(list_of_rewards.shape[1])]).requires_grad_()
		
		return discounted_batch



def convert_one_hot(a, dim):
	if dim == 2: #binary value, no need to do one-hot encoding
		return a

	if a.shape:
		retval = torch.zeros(list(a.shape)+[dim])
		retval[list(np.indices(retval.shape[:-1]))+[a]] = 1
	else: #single value tensor
		retval = torch.zeros(dim)
		retval[int(a)] = 1
	return retval


def roll_1(x, n):  
    return torch.cat((x[:, -n:], x[:, :-n]), dim=1)


def shift_down(x, step, full_size):
	if step == 0:
		return x

	if len(x.shape) == 3: #batch_wise
		#return torch.cat((torch.zeros((x.shape[0],full_size-x.shape[1]+1,x.shape[2])),x[:,:-1]),dim=1)
		return torch.cat((torch.zeros((x.shape[0], step, x.shape[2])),x),dim=1)
	elif len(x.shape) == 2: 
		return torch.cat((torch.zeros((step, x.shape[1])),x),dim=0)
	else:
		raise NotImplementedError('shape {shape_x} of input not corrent or implemented'.format(shape_x=x.shape))


def roll_left(x, n):  
    #return torch.cat((x[-n:], x[:-n]))
    return torch.cat((x[n:], x[:n]))


def get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor, batch_actions):
	action_probs = policy_estimator(states_tensor)

	if not policy_estimator.continuous:
		log_probs = torch.log(action_probs[0])
		selected_log_probs = torch.index_select(log_probs, 1, actions_tensor)[range(len(batch_actions)), range(len(batch_actions))]
	else:
		n = Normal(*action_probs)
		selected_log_probs = n.log_prob(actions_tensor)

	return selected_log_probs


def mle_validation_loss(P_hat, policy, val_states_next_tensor, state_actions_val, n_actions, max_actions, continuous_actionspace=False, device='cpu'):

	squared_errors_val = torch.zeros_like(val_states_next_tensor)
	step_state_val = state_actions_val.to(device)

	with torch.no_grad():
		for step in range(max_actions-1):
			next_step_state_val = P_hat(step_state_val)
			squared_errors_val += shift_down((val_states_next_tensor[:,step:,:] - next_step_state_val)**2, step, max_actions)

			shortened_val = next_step_state_val[:,:-1,:]
			action_probs_val = policy(torch.FloatTensor(shortened_val))
			
			if not continuous_actionspace:
				c_val = Categorical(action_probs_val[0])
				a_val = c_val.sample().type(torch.FloatTensor).to(device)
				step_state_val = torch.cat((shortened_val, convert_one_hot(a_val, n_actions).unsqueeze(2)),dim=2)
			else:
				c_val = Normal(*action_probs_val)
				a_val = torch.clamp(c_val.rsample(), min=-2.0, max=2.0)
				step_state_val = torch.cat((shortened_val, a_val),dim=2)

	# if squared_errors_val.shape < 2:
	# 	squared_errors_val.unsqueeze(0)
	
	err_1step = torch.mean(squared_errors_val[:,1])
	err_5step = torch.mean(squared_errors_val[:,5])

	try:
		err_10step = torch.mean(squared_errors_val[:,10])
		err_50step = torch.mean(squared_errors_val[:,50])
		err_100step = torch.mean(squared_errors_val[:,100])
	except:
		err_10step = 0
		err_50step = 0
		err_100step = torch.mean(squared_errors_val[:,-1])

	print("Multistep error values:", err_1step, err_5step, err_10step, err_50step, err_100step, "\n")	

	return squared_errors_val



def paml_validation_loss(env, P_hat, policy, val_states_prev_tensor, val_states_next_tensor, actions_tensor_val, rewards_tensor_val, R_range, val_size, n_actions, max_actions, continuous_actionspace=False, device='cpu'):

	state_actions_val = torch.cat((val_states_prev_tensor, actions_tensor_val),dim=2)

	policy.zero_grad()
	multiplier = torch.arange(max_actions,0,-1).repeat(val_size,1).unsqueeze(2).type(torch.FloatTensor).to(device)
	true_log_probs_t = torch.sum(
						get_selected_log_probabilities(
							policy, 
							val_states_prev_tensor, 
							actions_tensor_val, 
							range(actions_tensor_val.shape[0])
							) * rewards_tensor_val #* multiplier
						, dim=1)

	true_log_probs = torch.mean(true_log_probs_t, dim=0)
	true_pe_grads = grad(true_log_probs, policy.parameters(), create_graph=True)


	step_state_action_val = state_actions_val.to(device)
	k_step_log_probs = torch.zeros((R_range, val_size, n_actions))

	policy.zero_grad()
	for step in range(R_range):
		with torch.no_grad():
			next_step_state = P_hat(step_state_action_val)
		#print('states_mean:', torch.mean(next_step_state))
		shortened = next_step_state[:,:-1,:]

		with torch.no_grad():
			action_probs = policy(torch.FloatTensor(shortened))
			
			if not continuous_actionspace:
				c = Categorical(action_probs)
				actions_t_l = c.sample() 
				step_state_action_val = torch.cat((shortened,convert_one_hot(actions_t_l, n_actions)),dim=2)
			else:
				c = Normal(*action_probs)
				actions_t_l = torch.clamp(c.rsample(), min=-2.,max=2.)
				step_state_action_val = torch.cat((shortened, actions_t_l),dim=2)


		model_rewards_t = get_reward_fn(env, shortened, actions_t_l)
		model_log_probs = torch.sum(
						get_selected_log_probabilities(
								policy, 
								shortened,
								actions_t_l, range(actions_t_l.shape[0])) * 
								model_rewards_t
							, dim=1)

		#pdb.set_trace()
		k_step_log_probs[step] = model_log_probs#.squeeze()

	model_log_probs = torch.mean(torch.sum(k_step_log_probs, dim=0))

	model_pe_grads = grad(model_log_probs, policy.parameters(), create_graph=True)
	#total_log_probs.backward(retain_graph=True)
	#model_pe_grads = [x.grad for x in pe.parameters()]

	model_loss_val = 0
	for i in range(len(true_pe_grads)):
		model_loss_val += torch.sqrt(torch.sum((model_pe_grads[i] - true_pe_grads[i])**2))

	print('PAML validation loss:', model_loss_val.detach().data.cpu())

	return model_loss_val.detach().data



def collect_data(env, policy, episodes, n_actions, n_states, R_range, max_actions, continuous_actionspace=False, device='cpu'):
	states_prev_list = []
	states_next_list = []
	all_actions_list = []
	all_rewards_list = []

	for ep in range(episodes):
		s = env.reset()
		done = False
		states_list = [s]
		actions_list = []
		rewards_list = []
		while len(actions_list) < max_actions:
			with torch.no_grad():
				if device == 'cuda':
					action_probs = policy(torch.cuda.FloatTensor(s))
				else:
					action_probs = policy(torch.FloatTensor(s))

				if not continuous_actionspace:
					c = Categorical(action_probs[0])
					a = c.sample() 
				else:
					c = Normal(*action_probs)
					a = c.rsample()

				s_prime, r, done, _ = env.step(a.cpu().numpy() - 1)
				states_list.append(s_prime)
				rewards_list.append(r)

				if not continuous_actionspace:
					actions_list.append(convert_one_hot(a, n_actions))
				else:
					actions_list.append(a)

				s = s_prime

		states_prev_list.extend(states_list[:-R_range-1])
		states_next_list.extend(states_list[R_range+1:])
		all_actions_list.extend(actions_list)
		all_rewards_list.extend(rewards_list)

	states_prev_tensor = torch.FloatTensor(states_prev_list).to(device).view(episodes, -1, n_states)
	states_next_tensor = torch.FloatTensor(states_next_list).to(device).view(episodes, -1, n_states)
	actions_tensor = torch.stack(all_actions_list).type(torch.FloatTensor).to(device).view(episodes, -1, n_actions)
	rewards_tensor = torch.FloatTensor(all_rewards_list).to(device).view(episodes, -1, n_actions)

	return states_prev_tensor, states_next_tensor, actions_tensor, rewards_tensor



def reinforce_categorical(env, policy_estimator, episodes, use_model=False, device=torch.device('cpu')):
	#episodes = 2000
	n_actions = env.action_space.n
	discount = 0.99
	x_threshold = 2.4
	theta_threshold_radians = 12 * math.pi / 180 #.209
	optimizer = optim.Adam(policy_estimator.parameters(), 
                           lr=0.01)
	
	if use_model: 
		env.to(device)
		policy_estimator.to(device)

	all_rewards = []

	batch_size = 10
	batch_states = []
	batch_rewards = []
	batch_actions = []
	batch_counter = 0
	best_loss = 10

	theta_grad_weight = []
	theta_grad_bias = []

	for ep in range(episodes):
		#generate an episode
		if use_model:
			s = env.reset().to(device)
		else:
			s = env.reset()
		states = [s]
		actions = []
		#actions_to_use = np.load('actions_grad.npy')
		rewards = []
		done = False
		while not done:
		#for a in actions_to_use:
			with torch.no_grad():
				if not use_model:
					a_probs = policy_estimator(torch.FloatTensor(s))[0].detach().numpy()
					a = np.random.choice(range(n_actions), p=a_probs)
					s_prime, r, done, _ = env.step(a)

				else:
					a_probs = policy_estimator(s).detach()
					c = Categorical(a_probs)
					a = c.sample() 
					s_prime, _, _ = env(torch.cat((s,a.unsqueeze(0).type(torch.cuda.FloatTensor)),dim=0))
					
					x = s_prime[0]
					theta = s_prime[2]
					done = torch.abs(x) > x_threshold or torch.abs(theta) > theta_threshold_radians or len(actions) > 200
					r = torch.ones(1)
			actions.append(a)
			states.append(s_prime)
			rewards.append(r)
			#print("num steps:{}, sum rewards:{}".format(len(rewards), torch.sum(torch.stack(rewards))))
			s = s_prime

		#np.save('actions_grad',np.asarray(actions))
		if use_model:
			all_rewards.append(torch.sum(torch.stack(rewards)))
		else:
			all_rewards.append(sum(rewards))

		if batch_counter < batch_size:
			batch_counter += 1
			batch_actions.extend(actions)
			batch_states.extend(states[:-1])
			batch_rewards.extend(discount_rewards(rewards, discount))
		else:
			#update pe
			if not use_model:
				return_G = torch.FloatTensor(batch_rewards)
				states_tensor = torch.FloatTensor(batch_states)
				actions_tensor = torch.LongTensor(batch_actions)
			else:
				return_G = torch.FloatTensor(batch_rewards).to(device)
				states_tensor = torch.stack(batch_states).to(device)
				actions_tensor = torch.stack(batch_actions).type(torch.LongTensor).to(device)

			selected_log_probs = return_G * get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor, batch_actions)

			#update policy
			optimizer.zero_grad()
			loss = -selected_log_probs.mean()
			loss.backward()

			# check these two values for model vs environment
			# theta_grad_weight.append(poli.theta.weight.grad.cpu().numpy())
			# theta_grad_bias.append(pe.theta.bias.grad.cpu().numpy())

			optimizer.step()
			if loss.detach() < best_loss:
				torch.save(policy_estimator.state_dict(), 'policy_with_mle.pth')
				best_loss = loss

			batch_counter = 0
			batch_actions = []
			batch_rewards = []
			batch_states = []

		print("Average of last 10 rewards:", sum(all_rewards[-10:])/10.)
		#print("episode:{},loss:{:.3f}".format(ep,loss.detach()))
	
	# if use_model:
	# 	np.save('paml_grad_weight.npy',np.asarray(theta_grad_weight))
	# 	np.save('paml_grad_bias.npy',np.asarray(theta_grad_bias))

	# else:
	# 	np.save('env_grad_weight.npy',np.asarray(theta_grad_weight))
	# 	np.save('env_grad_bias.npy',np.asarray(theta_grad_bias))

	return all_rewards