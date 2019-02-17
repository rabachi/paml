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

MAX_TORQUE = 1.
device = 'cpu'

num_episodes = 750
dataset = ReplayMemory(1000)

n_states = 2
n_actions = 2
continuous_actionspace = True

batch_size = 10
policy_estimator = Policy(n_states, n_actions, continuous=continuous_actionspace, std=-2.5)
policy_estimator.double()

optimizer = optimizer = optim.Adam(policy_estimator.parameters(), lr=0.0001)

x_d = np.zeros((10,2))
x_next_d = np.zeros((10,2))
r_d = np.zeros((10))
all_rewards = []

num_iters = 1
state_dim = 2
extra_dim = state_dim - 2 # This dynamics is 2-d
batch_counter = 0
plt.figure(1)
plt.figure(2)
plt.figure(3)
#plt.figure(4)
plotted = False

best_rewards = -2000
worst_rewards = 0

best_x = np.ones((2,2)) * 0.5
worst_x = np.ones((2,2)) * 0.5

for episode in range(num_episodes):    # 2000, 10000

	#    x_0 = 0.5*np.random.randn(2)     # XXX Original

	#    x_0 = np.array([2*(np.random.random() - 0.5), 0])
	x_0 = 2*(np.random.random(size=(2,)) - 0.5) #range: [-1,1]

	#2*np.random.random((2,1000)) - 1.0    
	x_tmp, x_next_tmp, u_list, r_tmp_new, r_list = lin_dyn(40, policy_estimator, all_rewards, x=x_0)#, reward_type='Quadratic')
	#    r_tmp = np.ones_like(r_tmp) # XXX
	#    r_tmp *= 0.

	#    r_tmp = 0.0*np.ones_like(r_tmp) # XXX

	###########################
	# Add irrelevant dimensions to the state and next-state
	#x_tmp, x_next_tmp = add_irrelevant_features(x_tmp, x_next_tmp, extra_dim)    
	###########################
	
	for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_tmp_new):
	#        dataset.push(torch.from_numpy(x).float(), torch.from_numpy(x_next).float(), r )
		dataset.push(x, x_next, u, r)        

	if batch_counter < batch_size:
		batch_counter += 1
	else:
		# print(episode-batch_counter)
		# print(episode)
		batch = dataset.memory[0:episode]

		batch_states_prev = torch.tensor([samp.state for samp in batch]).double()
		# batch_states_next = torch.tensor([samp.next_state for samp in batch]).double()
		batch_rewards = torch.tensor([samp.reward for samp in batch]).double()
		batch_actions = torch.tensor([samp.action for samp in batch]).double()

		for i in range(num_iters + math.ceil(episode *0.2)):
			selected_log_probs = batch_rewards.unsqueeze(1) * get_selected_log_probabilities(policy_estimator, batch_states_prev, batch_actions)

			optimizer.zero_grad()
			loss = -selected_log_probs.mean()
			loss.backward()

			optimizer.step()

			#if ep % 10 == 0:
				# if continuous_actionspace:
				# 	print(policy_estimator.log_std.detach().data)
		print("Ep: {ep}, Average of last 10 rewards: {sumr}".format(ep=episode,sumr=sum(all_rewards[-10:])/10.))

		# dataset.clear()
		batch_counter = 0

	# print(all_rewards[-1])
	# print(np.linalg.norm(x_tmp[0] - worst_x))

	if all_rewards[-1] > best_rewards and np.linalg.norm(x_tmp[0] - worst_x[0]) <= 0.4:
		best_r = r_list
		best_x = x_tmp
		best_u = u_list
		best_rewards = all_rewards[-1]

	if all_rewards[-1] < worst_rewards and np.linalg.norm(x_tmp[0] - best_x[0]) <= 0.4:
		worst_r = r_list
		worst_x = x_tmp
		worst_u = u_list
		worst_rewards = all_rewards[-1]


print(best_x[0])
print(worst_x[0])

plt.figure(1)
plt.plot(range(best_r.shape[0]), best_r)#.numpy())

# plt.figure(4)
# plt.plot(range(r_tmp_old.shape[0]), r_tmp_old.numpy())

plt.figure(2)
plt.plot(best_x[:,0], best_x[:,1])

plt.figure(3)
plt.plot(best_u[:,0], best_u[:,1])
plotted = True


plt.figure(1)
plt.plot(range(worst_r.shape[0]), worst_r)#.numpy())

# plt.figure(4)
# plt.plot(range(r_tmp_old.shape[0]), r_tmp_old.numpy())

plt.figure(2)
plt.plot(worst_x[:,0], worst_x[:,1])

plt.figure(3)
plt.plot(worst_u[:,0], worst_u[:,1])		

plt.show()


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





