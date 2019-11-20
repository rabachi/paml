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



device = 'cpu'

num_episodes = 10
dataset = ReplayMemory(1000)

n_states = 2
n_actions = 2
continuous_actionspace = True

batch_size = 8
policy_estimator = Policy(n_states, n_actions, continuous=continuous_actionspace, std=0)
policy_estimator.double()

optimizer = optimizer = optim.Adam(policy_estimator.parameters(), lr=0.001)

x_d = np.zeros((0,2))
x_next_d = np.zeros((0,2))
r_d = np.zeros((0))
all_rewards = []

plt.figure(1)
plt.figure(2)
plt.figure(3)
plt.figure(4)
num_iters = 10
state_dim = 2
extra_dim = state_dim - 2 # This dynamics is 2-d
batch_counter = 0
for episode in range(num_episodes):    # 2000, 10000

	#    x_0 = 0.5*np.random.randn(2)     # XXX Original

	#    x_0 = np.array([2*(np.random.random() - 0.5), 0])
	x_0 = 2*(np.random.random(size=(2,)) - 0.5)

	#2*np.random.random((2,1000)) - 1.0    
	x_tmp, x_next_tmp, u_list, r_tmp_new, r_tmp_old = lin_dyn(20, policy_estimator, all_rewards, x=x_0)

	plt.figure(1)
	plt.plot(range(r_tmp_new.shape[0]), r_tmp_new.numpy())

	plt.figure(4)
	plt.plot(range(r_tmp_old.shape[0]), r_tmp_old.numpy())

	plt.figure(2)
	plt.plot(x_tmp[:,0], x_tmp[:,1])

plt.figure(3)
plt.plot(u_list[:,0], u_list[:,1])

plt.show()
	