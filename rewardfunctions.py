import numpy as np
import matplotlib.pyplot as plt
import gym
import sys

import torch
import math
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import pdb

MAX_TORQUE = 2.

def get_reward_fn(env, states_tensor, actions_tensor):
	if env.spec.id == 'Pendulum-v0':
		thcos = states_tensor[:,:,0]
		thsin = states_tensor[:,:,1]
		thdot = states_tensor[:,:,2]

		#pdb.set_trace()
		#tanth = thsin/thcos
		#tanth[torch.isnan(tanth)] = 0
		th = torch.atan2(thsin, thcos)

		if torch.isnan(th).any():
			pdb.set_trace()

		u = torch.clamp(actions_tensor, min=-MAX_TORQUE, max=MAX_TORQUE).squeeze()

		costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

		return -costs.unsqueeze(2)


	elif env.spec.id == 'CartPole-v0':
		theta_threshold_radians = 12 * 2 * np.pi / 360
		x_threshold = 2.4

		x = states_tensor[:,:,0]
		#x_dot = states_tensor[:,:,1]
		theta = states_tensor[:,:,2]
		#theta_dot = states_tensor[:,:,3]

		#this is a problem because ITS A BIG MATRIX WITH BATCHES AND DIFFERENT TIME STEPS!!!!!!  
		# done =  x < -x_threshold \
		# 		or x > x_threshold \
		# 		or theta < -theta_threshold_radians \
		# 		or theta > theta_threshold_radians
		# done = bool(done)

		# if not done:
  #           reward = 1.0
  #       elif self.steps_beyond_done is None:
  #           # Pole just fell!
  #           self.steps_beyond_done = 0
  #           reward = 1.0
  #       else:
  #           if self.steps_beyond_done == 0:
  #               logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
  #           self.steps_beyond_done += 1
  #           reward = 0.0



	return 0

def angle_normalize(x):
	return (((x+np.pi) % (2*np.pi)) - np.pi)