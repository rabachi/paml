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
from utils import *

import pdb

def collect_episode_terminates(env, policy, n_actions, n_states, 
	continuous_actionspace=False, device='cpu'):

	s = env.reset()
	done = False
	states_list = [s]
	actions_list = []
	rewards_list = []
	num_steps = 0

	while not done:
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

			s_prime, r, done, _ = env.step(a.cpu().numpy()) 
			states_list.append(s_prime)
			rewards_list.append(r)

			if not continuous_actionspace:
				actions_list.append(convert_one_hot(a, n_actions))
			else:
				actions_list.append(a)

			s = s_prime
			num_steps += 1

	#states_prev_list = states_list[:-1]
	#states_next_list = states_list[1:]

	states_prev_tensor = torch.FloatTensor(states_list[:-1]).to(device).view(-1, n_states)
	states_next_tensor = torch.FloatTensor(states_list[1:]).to(device).view(-1, n_states)
	if n_actions == 2: #binary actions not one-hotted 
		actions_tensor = torch.FloatTensor(actions_list).to(device).view(-1, 1)
		rewards_tensor = torch.FloatTensor(rewards_list).to(device).view(-1, 1)
	else:
		actions_tensor = torch.FloatTensor(actions_list).to(device).view(-1, n_actions)
		rewards_tensor = torch.FloatTensor(rewards_list).to(device).view(-1, n_actions)

	return states_prev_tensor, states_next_tensor, actions_tensor, rewards_tensor, num_steps


