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
from torch.autograd import grad, gradgradcheck

import pdb
import os
from models import *
from utils import *
from rewardfunctions import *
import matplotlib.animation as animation



n_states = 5
n_actions = 1

policy_estimator = Policy(n_states, n_actions, continuous=True, std=-0.5)
policy_estimator.double()

optimizer = optimizer = optim.Adam(policy_estimator.parameters(), 
                           lr=1e-4)

# Load one task :
#env = suite.load(domain_name="cartpole", task_name="swingup")


dm_control2gym.create_render_mode('rs', show=False, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(), depth=False, scene_option=None)

env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
# print(env)
# action_spec = env.action_spec()
# obs_spec = env.observation_spec()
# print(action_spec)
# print(obs_spec)

episodes = 200
max_actions = 200

batch_size = 3
batch_states = []
batch_rewards = []
batch_actions = []
batch_counter = 0
best_loss = 10

all_rewards = []
ims = []

for ep in range(episodes):
	#time_step = env.reset()
	#s = np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
	s = env.reset()
	states = [s]
	actions = []		
	rewards = []
	for _ in range(max_actions):
		with torch.no_grad():
			params = policy_estimator(torch.DoubleTensor(s))
			params = (params[0].detach(),params[1].detach())
			m = Normal(*params)
			a = m.rsample()	

			observation, reward, done, _ = env.step(a)
			#time_step = env.step(a)

			s_prime = observation# np.concatenate((time_step.observation['position'], time_step.observation['velocity']))
			if done:
				print('here')
		actions.append(a)
		states.append(s_prime)
		rewards.append(reward)
		s = s_prime

		if ep > 195:
			pixels = env.render('rs')
			im = plt.imshow(pixels)
			ims.append([im])
		

		#action = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
		#time_step = env.step(action)

		#print(time_step.reward, time_step.discount, time_step.observation)


		if batch_counter < batch_size:
			batch_counter += 1
			batch_actions.extend(actions)
			batch_states.extend(states[:-1])
			batch_rewards.extend(rewards)#discount_rewards(rewards, 1.0))
		else:
			#update pe
			return_G = torch.DoubleTensor(batch_rewards)
			states_tensor = torch.DoubleTensor(batch_states)
			actions_tensor = torch.DoubleTensor(batch_actions)

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
	
	all_rewards.append(sum(rewards))
	print("Ep: {ep}, Average of last 10 rewards:{sumr}".format(ep=ep,sumr=sum(all_rewards[-10:])/10.))

env.close()

fig = plt.figure()
ani = animation.ArtistAnimation(fig, ims, interval=10, blit=True,
                                repeat_delay=60)

window = 10
smoothed_rewards = [np.mean(all_rewards[i-window:i+1]) if i > window 
                    else np.mean(all_rewards[:i+1]) for i in range(len(all_rewards))]

plt.figure(figsize=(12,8))
plt.plot(all_rewards)
plt.plot(smoothed_rewards)
plt.ylabel('Total Costs')
plt.xlabel('Episodes')
plt.show()

print('done')