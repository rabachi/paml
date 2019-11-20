import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import pdb
from models import *
from utils import *

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'


def ppo(env, policy_estimator, use_model=False):
	episodes = 1000
	discount = 0.99
	eps = 0.2
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
			with torch.no_grad():
				if not use_model:
					a_probs = policy_estimator(torch.FloatTensor(s)).detach().numpy()
					a = np.random.choice([-1,0,1], p=a_probs)
					s_prime, r, done, _ = env.step(a)

				else:
					a_probs = policy_estimator(s).detach()
					c = Categorical(a_probs)
					a = c.sample() 
					s_prime, r, done = env(torch.cat((s,convert_one_hot(a,3).type(torch.FloatTensor)),dim=0).to(device))
					
					done = bool(-torch.cos(s_prime[0]) - torch.cos(s_prime[1] + s_prime[0]) > 1.)
					print(done)
					#x = s_prime[0]
					#theta = s_prime[2]
					#done = torch.abs(x) > x_threshold or torch.abs(theta) > theta_threshold_radians or len(actions) > 200
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
			if not use_model:
				return_G = torch.FloatTensor(batch_rewards)
				states_tensor = torch.FloatTensor(batch_states)
				actions_tensor = torch.LongTensor(batch_actions) + 1
			else:
				return_G = torch.FloatTensor(batch_rewards).to(device)
				states_tensor = torch.stack(batch_states).to(device)
				actions_tensor = torch.stack(batch_actions).type(torch.LongTensor).to(device)

			old_selected_log_probs = get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor, batch_actions).detach()
			for epoch in range(7):
				#update pe
				selected_log_probs = get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor, batch_actions)
				ratio = torch.exp(selected_log_probs - old_selected_log_probs)
				clip_obj = torch.min(ratio*return_G, torch.clamp(ratio, min=1-eps, max=1+eps) * return_G)

				#update policy
				optimizer.zero_grad()
				loss = -clip_obj.mean()
				pdb.set_trace()
				loss.backward()
				# check these two values for model vs environment
				#theta_grad_weight.append(pe.theta.weight.grad.cpu().numpy())
				#theta_grad_bias.append(pe.theta.bias.grad.cpu().numpy())

				#old_selected_log_probs = selected_log_probs.detach()
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
	env = gym.make('Acrobot-v1')
	env.seed(1)
	n_states = env.observation_space.shape[0]
	#pe = PolicyTorque(n_states, env.action_space.shape)
	n_actions = env.action_space.n 
	pe = Policy(n_states, n_actions)
	#pe.load_state_dict(torch.load('policy_trained.pth'))

	P_hat = DirectEnvModel(n_states,n_actions)
	P_hat.load_state_dict(torch.load('acrobot_mle_trained_model.pth', map_location=device))
	# P_hat.to(device)
	# for p in P_hat.parameters():
	# 	p.requires_grad = False

	#rewards = reinforce(env, pe, use_model=False)
	#rewards = reinforce(P_hat, pe, use_model=True)
	#rewards = ppo(env, pe)
	rewards = ppo(P_hat, pe, use_model=True)
	window = 10
	smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
	                    else np.mean(rewards[:i+1]) for i in range(len(rewards))]

	plt.figure(figsize=(12,8))
	plt.plot(rewards)
	plt.plot(smoothed_rewards)
	plt.ylabel('Total Rewards')
	plt.xlabel('Episodes')
	plt.show()


