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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
#device = 'cpu'


def ppo_continuous(env, policy_estimator, use_model=False):
	episodes = 5000
	discount = 0.9
	eps = 0.02
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
			s = env.reset().to(device)
		else:
			s = env.reset()
		states = [s]
		actions = []
		rewards = []
		done = False
		while len(actions) <= 200:
			if not use_model:
				with torch.no_grad():
					params = policy_estimator(torch.FloatTensor(s))
					params = (params[0].detach(),params[1].detach())
					m = Normal(*params)
					a = m.rsample().numpy()
					a = np.clip(a, -2, 2)
					s_prime, r, _, _ = env.step(a)

			else:
				params = policy_estimator(torch.FloatTensor(s)).detach()
				m = Normal(*params)
				a = m.rsample()				
				s_prime, r = env(torch.cat((s,a.unsqueeze(0).type(torch.cuda.FloatTensor)),dim=0))
				
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
		else:
			#update pe
			if not use_model:
				return_G = torch.FloatTensor(batch_rewards)
				states_tensor = torch.FloatTensor(batch_states)
				actions_tensor = torch.FloatTensor(batch_actions)
			else:
				return_G = torch.FloatTensor(batch_rewards).to(device)
				states_tensor = torch.stack(batch_states).to(device)
				actions_tensor = torch.stack(batch_actions).type(torch.FloatTensor).to(device)

			with torch.no_grad():
				params = policy_estimator(states_tensor)
				m = Normal(*params)
				old_selected_log_probs = m.log_prob(actions_tensor)

			return_G = return_G.reshape_as(old_selected_log_probs)
			for epoch in range(100):
				params = policy_estimator(states_tensor)
				m = Normal(*params)
				selected_log_probs = m.log_prob(actions_tensor)
				ratio = torch.exp(selected_log_probs - old_selected_log_probs.detach())
				clip_obj = torch.min(ratio*return_G, torch.clamp(ratio, min=1-eps, max=1+eps) * return_G)

				#update policy
				optimizer.zero_grad()
				loss = -clip_obj.mean()
				if torch.isnan(loss) or len(all_rewards) >= 1000:
					pdb.set_trace()
				loss.backward()
				optimizer.step()
				print(loss)
				if loss.detach() < best_loss:
					torch.save(policy_estimator.state_dict(), 'policy_with_mle.pth')
					best_loss = loss

			batch_counter = 0
			batch_actions = []
			batch_rewards = []
			batch_states = []

			print("Average of last 10 rewards:", sum(all_rewards[-10:])/10.)
			#print("episode:{},loss:{:.3f}".format(ep,loss.detach()))
			
	return all_rewards

if __name__ == "__main__":
	#initialize pe 
	env = gym.make('Pendulum-v0')
	env.seed(1)
	n_states = env.observation_space.shape[0]
	n_actions = env.action_space.shape[0]
	#P_hat = DirectEnvModel(n_states,n_actions) #torch.load('../mle_trained_model.pth')
	pe = PolicyTorque(n_states, n_actions)

	# P_hat = DirectEnvModel(n_states,n_actions)
	# P_hat.load_state_dict(torch.load('mle_trained_model.pth', map_location=device))
	# P_hat.to(device)
	# for p in P_hat.parameters():
	# 	p.requires_grad = False


	rewards = ppo_continuous(env, pe, use_model=False)
	#rewards = reinforce(P_hat, pe, use_model=True)
	window = 10
	smoothed_rewards = [np.mean(rewards[i-window:i+1]) if i > window 
	                    else np.mean(rewards[:i+1]) for i in range(len(rewards))]

	plt.figure(figsize=(12,8))
	plt.plot(rewards)
	plt.plot(smoothed_rewards)
	plt.ylabel('Total Costs')
	plt.xlabel('Episodes')
	plt.show()
