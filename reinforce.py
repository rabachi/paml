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



def ppo_continuous(env, policy_estimator, use_model=False):
	episodes = 2000
	discount = 0.9
	eps = 0.2
	optimizer = optim.Adam(policy_estimator.parameters(), 
                           lr=1e-4)
	
	if use_model: 
		env.to(device)
		policy_estimator.to(device)

	all_rewards = []

	batch_size = 5
	batch_states = []
	batch_rewards = []
	batch_actions = []
	batch_counter = 0
	best_loss = 10

	for ep in range(episodes):
		#generate an episode
		if use_model:
			s = env.reset().to(device)
		else:
			s = env.reset()
		states = [s]
		actions = []
		rewards = []
		done = False
		while len(actions) <= 30:
			if not use_model:
				with torch.no_grad():
					params = policy_estimator(torch.DoubleTensor(s))
					params = (params[0].detach(),params[1].detach())
					m = Normal(*params)
					a = m.rsample().numpy()
					a = np.clip(a, -2, 2)
					s_prime, r, _, _ = env.step(a)

			else:
				params = policy_estimator(torch.DoubleTensor(s)).detach()
				m = Normal(*params)
				a = m.rsample()				
				s_prime, r, d = env(torch.cat((s,a.unsqueeze(0).type(torch.cuda.DoubleTensor)),dim=0))

				
				# x = s_prime[0]
				# theta = s_prime[2]
			
			actions.append(a)
			states.append(s_prime)
			rewards.append(r)
			#print("num steps:{}, sum rewards:{}".format(len(rewards), torch.sum(torch.stack(rewards))))
			s = s_prime

		if use_model:
			all_rewards.append(torch.sum(torch.stack(rewards)))
		else:
			all_rewards.append(sum(rewards))

		if batch_counter != batch_size:
			batch_counter += 1
			batch_actions.extend(actions)
			batch_states.extend(states[:-1])
			batch_rewards.extend(discount_rewards(rewards, discount))
			#print("episode: {ep}, reward: {r}".format(ep=ep, r=sum(rewards)/batch_size))
		else:
			#update pe
			if not use_model:
				return_G = torch.DoubleTensor(batch_rewards)
				states_tensor = torch.DoubleTensor(batch_states)
				actions_tensor = torch.DoubleTensor(batch_actions)
			else:
				return_G = torch.DoubleTensor(batch_rewards).to(device)
				states_tensor = torch.stack(batch_states).to(device)
				actions_tensor = torch.stack(batch_actions).type(torch.DoubleTensor).to(device)

			with torch.no_grad():
				old_selected_log_probs = get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor, range(len(actions_tensor)))

			return_G = return_G.reshape_as(old_selected_log_probs)
			for epoch in range(4):

				selected_log_probs = get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor, range(len(actions_tensor)))

				#ratio = torch.exp(selected_log_probs - old_selected_log_probs.detach())
				#clip_obj = torch.min(ratio*return_G, torch.clamp(ratio, min=1-eps, max=1+eps) * return_G)

				clip_obj = return_G*selected_log_probs
				#update policy
				optimizer.zero_grad()
				loss = -clip_obj.mean()
				# if torch.isnan(loss) or len(all_rewards) >= 1000:
				# 	pdb.set_trace()
				#pdb.set_trace()
				loss.backward()
				optimizer.step()
				if loss.detach() < best_loss:
					torch.save(policy_estimator.state_dict(), 'policy_with_mle.pth')
					best_loss = loss

			batch_counter = 0
			batch_actions = []
			batch_rewards = []
			batch_states = []

			#pdb.set_trace()
			print("Average of last 10 rewards:", sum(all_rewards[-10:])/10.)
			#print("episode:{},loss:{:.3f}".format(ep,loss.detach()))
			
	return all_rewards



#works mainly just for cartpole... 
def reinforce_categorical(env, policy_estimator, episodes, n_actions, use_model=False, device=torch.device('cpu')):
	#episodes = 2000
	max_actions = 200
	discount = 0.99
	x_threshold = 2.4
	theta_threshold_radians = 12 * math.pi / 180 #.209
	optimizer = optim.Adam(policy_estimator.parameters(), 
                           lr=0.001)
	
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

	for ep in range(episodes):
		#generate an episode
		if use_model:
			s = P_hat.reset().to(device)
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
					a_probs = policy_estimator(torch.DoubleTensor(s))
					c = Categorical(a_probs[0])
					a = c.sample() 
					s_prime, r, done, _ = env.step(a.cpu().numpy())

				else:
					a_probs = policy_estimator(s)
					c = Categorical(a_probs[0])
					a = c.sample() 
					s_prime  = env(torch.cat((s,a.unsqueeze(0).type(torch.DoubleTensor)),dim=0)).to(device)
					
					x = s_prime[0]
					theta = s_prime[2]
					done = torch.abs(x) > x_threshold or torch.abs(theta) > theta_threshold_radians or len(actions) > 200
					r = torch.ones(1)


			actions.append(a)
			states.append(s_prime)
			rewards.append(r)

			# if done and len(actions) < max_actions:
			# 	filled = len(actions)	
			# 	rewards += ([0] if not use_model else [torch.zeros(1)]) * (max_actions - filled)
			# 	states += [s_prime] * (max_actions - filled)
			# 	actions += [c.sample()] * (max_actions - filled)

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
				return_G = torch.DoubleTensor(batch_rewards)
				states_tensor = torch.DoubleTensor(batch_states)
				actions_tensor = torch.LongTensor(batch_actions)
			else:
				return_G = torch.DoubleTensor(batch_rewards).to(device)
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


if __name__ == "__main__":
	#initialize pe 
	device = 'cpu'
	num_episodes = 2500

	env = gym.make('CartPole-v0')
	n_states = env.observation_space.shape[0]
	continuous_actionspace = isinstance(env.action_space, gym.spaces.box.Box)
	if continuous_actionspace:
		n_actions = env.action_space.shape[0]
	else:
		n_actions = env.action_space.n

	R_range = 1

	errors_name = env.spec.id + '_single_episode_errors_' + 'mle' + '_' + str(R_range)
	#P_hat = DirectEnvModel(n_states,n_actions) #torch.load('../mle_trained_model.pth')
	pe = Policy(n_states, n_actions, continuous=continuous_actionspace)

	pe.double()

	P_hat = DirectEnvModel(n_states,n_actions-1,MAX_TORQUE)
	P_hat.load_state_dict(torch.load('CartPole-v0_5_mle_trained_model.pth', map_location=device))
	P_hat.double()

	# P_hat.to(device)
	# for p in P_hat.parameters():
	# 	p.requires_grad = False

	#rewards = ppo_continuous(env, pe, use_model=False)
	rewards = reinforce_categorical(env, pe, num_episodes, n_actions-1, use_model=False)
	window = 10
	smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
	                    else np.mean(rewards[:i+1]) for i in range(len(rewards))]

	plt.figure(figsize=(12,8))
	plt.plot(rewards)
	plt.plot(smoothed_rewards)
	plt.ylabel('Total Costs')
	plt.xlabel('Episodes')
	plt.show()