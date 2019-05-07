# from dm_control import suite
# import gym
# import dm_control2gym

import numpy as np
import matplotlib.pyplot as plt
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
from rewardfunctions import *

MAX_TORQUE = 1.
device = 'cpu'

# rs = 7
# torch.manual_seed(rs)
# np.random.seed(rs)

num_episodes = 5000
dataset = ReplayMemory(10000)

discount = 0.9

max_actions = 10
states_dim = 2
actions_dim = 2
continuous_actionspace = True

batch_size = 1
policy_estimator = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-1.0, max_torque=MAX_TORQUE)
policy_estimator.double()


P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE)
		pe = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-2.5, max_torque=MAX_TORQUE)
		
P_hat.load_state_dict(torch.load('paml_trained_lin_dyn_horizon10_traj11_using1states_statesdim2_500starts_1eps.pth', map_location=device))

optimizer = optim.Adam(policy_estimator.parameters(), lr=0.001)

x_d = np.zeros((10,2))
x_next_d = np.zeros((10,2))
r_d = np.zeros((10))
all_rewards = []

num_iters = 1
batch_counter = 0

# plt.figure(1)
# plt.figure(2)
# plt.figure(3)
#plt.figure(4)
plotted = False

best_rewards = -10
worst_rewards = 0

best_x = np.ones((2,2)) * 0.5
worst_x = np.ones((2,2)) * 0.5

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

# def discount_rewards(rewards, gamma=0.99):
# 	r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
# 	# Reverse the array direction for cumsum and then
# 	# revert back to the original order
# 	r = r[::-1].cumsum()[::-1]
# 	return (r - r.mean())/r.std()

#x_0 = 2*(np.random.random(size=(2,)) - 0.5)
################## Before training check ############################
batch_actions = np.zeros((20,10,2))
batch_states = np.zeros((20,10,2))
for episode in range(20):    # 2000, 10000
	x_0 = 2*(np.random.random(size=(2,)) - 0.5)
	states = []
	rewards = []
	actions = []
	x_tmp, x_next_tmp, u_list, returns, r_list = lin_dyn(A_numpy, max_actions, policy_estimator, all_rewards, x=x_0, discount=discount)

	for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_list): 
		states.append(x)
		actions.append(u)

	batch_states[episode]=np.asarray(states)
	batch_actions[episode]=np.asarray(actions)

np.save('policy_random_states',batch_states)
np.save('policy_random_actions',batch_actions)
#####################################################################


batch_rewards = []
batch_actions = []
batch_states = []
best_loss = 10


for episode in range(num_episodes):    # 2000, 10000
	x_0 = 2*(np.random.random(size=(2,)) - 0.5)	
	states = []
	rewards = []
	actions = []
	x_tmp, x_next_tmp, u_list, returns, r_list = lin_dyn(A_numpy, max_actions, policy_estimator, all_rewards, x=x_0, discount=discount)

	for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_list):
		dataset.push(x, x_next, u, r)  
		states.append(x)
		rewards.append(r)
		actions.append(u)      

	if batch_counter < batch_size:
		batch_rewards.extend(discount_rewards(rewards, discount))
		batch_states.extend(states)
		batch_actions.extend(actions)
		batch_counter += 1
	else:
		for t in range(num_iters):
			optimizer.zero_grad()
			# state_tensor = torch.FloatTensor(batch_states).double()
			# reward_tensor = torch.FloatTensor(batch_rewards).double()
			# # Actions are used as indices, must be LongTensor
			# action_tensor = torch.LongTensor(batch_actions).double()
			
			batch = dataset.sample(batch_size, structured=True, max_actions=max_actions, num_episodes_per_start=1, num_starting_states=batch_size, start_at=None)
			#batch = dataset.sample(batch_size*max_actions, structured=False)
			states_prev = torch.zeros((batch_size, max_actions, states_dim)).double().to(device)
			states_next = torch.zeros((batch_size, max_actions, states_dim)).double().to(device)
			rewards = torch.zeros((batch_size, max_actions)).double().to(device)
			actions_tensor = torch.zeros((batch_size, max_actions, actions_dim)).double().to(device)
			discounted_rewards_tensor = torch.zeros((batch_size, max_actions, 1)).double().to(device)

			for b in range(batch_size):
				states_prev[b] = torch.tensor([samp.state for samp in batch[b]]).double().to(device)
				states_next[b] = torch.tensor([samp.next_state for samp in batch[b]]).double().to(device)
				rewards[b] = torch.tensor([samp.reward for samp in batch[b]]).double().to(device)
				actions_tensor[b] = torch.tensor([samp.action for samp in batch[b]]).double().to(device)
				discounted_rewards_tensor[b] = discount_rewards(rewards[b].unsqueeze(1), discount, center=True, batch_wise=False) # torch.from_numpy(discount_rewards(rewards[b], gamma=discount)).unsqueeze(1)

			selected_log_probs = discounted_rewards_tensor * get_selected_log_probabilities(policy_estimator, states_prev, actions_tensor)

			loss = -selected_log_probs.mean()
			loss.backward()
			#optimizer.step()

			if loss.detach() < best_loss:
				torch.save(policy_estimator.state_dict(), 'policy_with_true.pth')
				best_loss = loss
			#print(x_0)
			print(discounted_rewards_tensor[0,0])
		batch_counter = 0
		batch_rewards = []
		batch_actions = []
		batch_states = []
		dataset = ReplayMemory(10000)

		print("Ep: {}, Average of last 10 rewards: {:.3f}".format(episode,sum(all_rewards[-10:])/10.))



################# After training check ######################

policy_estimator.load_state_dict(torch.load('policy_with_true.pth', map_location=device))

batch_actions = np.zeros((20,10,2))
batch_states = np.zeros((20,10,2))
for episode in range(20):    # 2000, 10000
	x_0 = 2*(np.random.random(size=(2,)) - 0.5)
	states = []
	rewards = []
	actions = []
	x_tmp, x_next_tmp, u_list, returns, r_list = lin_dyn(A_numpy, max_actions, policy_estimator, all_rewards, x=x_0, discount=discount)

	for x, x_next, u, r in zip(x_tmp, x_next_tmp, u_list, r_list): 
		states.append(x)
		actions.append(u)

	batch_states[episode]=np.asarray(states)
	batch_actions[episode]=np.asarray(actions)

np.save('policy_reinforced_states',batch_states)
np.save('policy_reinforced_actions',batch_actions)
################################################################

	# print(all_rewards[-1])
	# print(np.linalg.norm(x_tmp[0] - worst_x))

	# if all_rewards[-1] > best_rewards and np.linalg.norm(x_tmp[0] - worst_x[0]) <= 0.4:
	# 	best_r = r_list
	# 	best_x = x_tmp
	# 	best_u = u_list
	# 	best_rewards = all_rewards[-1]

	# if all_rewards[-1] < worst_rewards and np.linalg.norm(x_tmp[0] - best_x[0]) <= 0.4:
	# 	worst_r = r_list
	# 	worst_x = x_tmp
	# 	worst_u = u_list
	# 	worst_rewards = all_rewards[-1]
#print(all_rewards)

print(best_x[0])
print(worst_x[0])

# plt.figure(1)
# plt.plot(range(best_r.shape[0]), best_r)#.numpy())

# # plt.figure(4)
# # plt.plot(range(r_tmp_old.shape[0]), r_tmp_old.numpy())

# plt.figure(2)
# plt.plot(best_x[:,0], best_x[:,1])

# plt.figure(3)
# plt.plot(best_u[:,0], best_u[:,1])
# plotted = True


# plt.figure(1)
# plt.plot(range(worst_r.shape[0]), worst_r)#.numpy())

# # plt.figure(4)
# # plt.plot(range(r_tmp_old.shape[0]), r_tmp_old.numpy())

# plt.figure(2)
# plt.plot(worst_x[:,0], worst_x[:,1])

# plt.figure(3)
# plt.plot(worst_u[:,0], worst_u[:,1])		

# plt.show()


# window = 10
# smoothed_rewards = [np.mean(all_rewards[i-window:i+1]) if i > window 
                    # else np.mean(all_rewards[:i+1]) for i in range(len(all_rewards))]

np.save('reinforce_rewards',np.asarray(all_rewards)) 
# plt.figure(figsize=(12,8))
# plt.plot(all_rewards)
# plt.plot(smoothed_rewards)
# plt.ylabel('Total Costs')
# plt.xlabel('Episodes')
# plt.show()

print('done')





