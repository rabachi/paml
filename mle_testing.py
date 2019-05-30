import numpy as np
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

device = 'cpu'
file_location = '/h/abachiro/paml/results'

states_dim = 2
salient_states_dim = 2
actions_dim = states_dim
continuous_actionspace = True
MAX_TORQUE = 2.0

policy_estimator = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-3.0, max_torque=MAX_TORQUE, small=False)
#try std=-4.3 with u = 1 and B = 0.1 --> this is 100 times smaller sigma^2, should give same results as when B = 1 and u = 0.1 with std=-2.0 --> NOPE!

policy_estimator.double()
policy_optimizer = optim.Adam(policy_estimator.parameters(), lr=0.0001)

P_hat = DirectEnvModel(states_dim, actions_dim, MAX_TORQUE, small=False)
P_hat.double()

model_opt = optim.SGD(P_hat.parameters(), lr=1e-5, momentum=0.90, nesterov=True)

max_actions = 50

R_range = max_actions

unroll_num=1
ell=0

num_starting_states = 10
num_episodes = 1

discount = 0.99
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

# A_all[3] = np.array([[ 0.9, 0.8,  0.1  ],
# 					[-0.1  ,  0.8,  0.4 ],
#    					[ 0  ,  -0.4  ,  0.96]])

A_all[3] = np.array([[0.98, 0.  , 0.  ],
					[0.  , 0.95, 0.  ],
					[0.  , 0.  , 0.99]])
   #np.array([[0.99, -0.5, -0.5],
						# [ 0.3, 0.99,  0.3],
						# [ 0.3,  0.3,  0.99]])

A_all[2] = np.array([[0.9, 0.4], [-0.4, 0.9]])
# A_all[2] = np.array([[0.9, 0.4], [-0.4, 0.9]])


A_numpy = A_all[salient_states_dim]

extra_dim =states_dim - salient_states_dim 
block = np.eye(extra_dim) * 0.98

#Either zero out B (multiplier of action) in unroll for dims of action that are added to irrelevant --> signifying that we can't control OR make policy itself only depend on relevant dimensions. Easiest option is to change B, so try that first 

# A_numpy = np.block([
# 		 [A_all[salient_states_dim],               	 np.zeros((salient_states_dim, extra_dim))],
# 		 [np.zeros((extra_dim, salient_states_dim)), block                 					  ]
# 		])

batch_size = num_starting_states * num_episodes
with torch.no_grad():
	#generate training data
	train_step_state = torch.zeros((batch_size, unroll_num, states_dim+actions_dim)).double()
	for b in range(num_starting_states):
		x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
		train_step_state[b*num_episodes:num_episodes*(b+1),:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

	train_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(train_step_state[:,:unroll_num,:states_dim])

	train_true_x_curr, train_true_x_next, train_true_a_list, train_true_r_list, train_true_a_prime_list = P_hat.unroll(train_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=False, policy_states_dim=salient_states_dim, salient_states_dim=salient_states_dim)
	train_true_returns = discount_rewards(train_true_r_list[:,0,1:], discount=discount, batch_wise=True, center=False)

losses = []
state_actions = torch.cat((train_true_x_curr.squeeze(), train_true_a_list.squeeze()), dim=2)


P_hat.train_mle(policy_estimator, state_actions, train_true_x_next.squeeze(), 500, max_actions, R_range, model_opt, "lin_dyn", losses, states_dim, salient_states_dim)

np.save(os.path.join(file_location,'mle_losses_state{}_salient{}_{}_horizon{}_traj{}'.format(states_dim, salient_states_dim, 'lin_dyn', R_range, max_actions + 1)), np.asarray(losses))





