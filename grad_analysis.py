import numpy as np
import gym
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
from actor_critic import actor_critic_DDPG
# from get_data import save_stats

import dm_control2gym

if __name__=="__main__":
	file_location = '/scratch/gobi1/abachiro/paml_results'
	# env_name = 'dm-Walker-v0'
	env_name = 'Pendulum-v0'#'dm-Cartpole-balance-v0'

	num_action_repeats = 1
	real_episodes = 100
	num_eps_per_start = 1
	max_actions = 200
	discount = 0.99
	file_id = '0'
	save_checkpoints_training = True
	verbose = 1

	# rs = 0#range(10)
	get_true_grads = True

	if env_name == 'lin_dyn':
		gym.envs.register(id='lin-dyn-v0', entry_point='gym_linear_dynamics.envs:LinDynEnv',)
		# env = gym.make('gym_linear_dynamics:lin-dyn-v0')
		env = gym.make('lin-dyn-v0')

	elif env_name == 'Pendulum-v0':
		env = gym.make('Pendulum-v0') #NormalizedEnv(gym.make('Pendulum-v0'))

	elif env_name == 'dm-Walker-v0':
		env = dm_control2gym.make(domain_name="walker", task_name="walk")
		env.spec.id = 'dm-Walker-v0'

	elif env_name == 'dm-Cartpole-balance-v0':
		env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
		env.spec.id = 'dm-Cartpole-balance-v0'

	elif env_name == 'dm-Pendulum-v0':
		env = dm_control2gym.make(domain_name="pendulum", task_name="swingup") #NormalizedEnv(dm_control2gym.make(domain_name="pendulum", task_name="swingup"))
		env.spec.id = 'dm-Pendulum-v0'
	else:
		raise NotImplementedError

	iter_count = 80000
	for rs in range(1,4):
		torch.manual_seed(rs)
		np.random.seed(rs)	
		env.seed(rs)

		#50
		num_starting_states = real_episodes
		num_episodes = num_eps_per_start 

		losses = []
		unroll_num = 1

		value_loss_coeff = 1.0

		max_torque = float(env.action_space.high[0])
		actions_dim = env.action_space.shape[0]
		states_dim = 30#env.observation_space.shape[0]
		salient_states_dim = env.observation_space.shape[0]

		continuous_actionspace = True

		use_model = True
		train_value_estimate = False
		train = True

		all_rewards = []

		noise = OUNoise(env.action_space)
		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim).double()
		actor.load_state_dict(torch.load(os.path.join(file_location, 'act_policy_paml_state30_salient3_checkpoint_Pendulum-v0_horizon3_traj201_constrainedModel_4.pth'), map_location=device))
		# actor.load_state_dict(torch.load(os.path.join(file_location, 'act_40000_policy_model_free_state24_salient24_checkpoint_dm-Walker-v0_horizon1_traj1001_nnModel_1.pth'), map_location=device))
		# critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_160000_policy_model_free_state24_salient24_checkpoint_dm-Walker-v0_horizon1_traj1001_nnModel_1.pth'), map_location=device))
		# critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_policy_paml_state5_salient5_checkpoint_dm-Cartpole-balance-v0_traj201_0.pth'), map_location=device))
		critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_policy_mle_state30_salient3_checkpoint_Pendulum-v0_traj201_constrainedModel_4.pth'), map_location=device))
		
		stablenoise = StableNoise(states_dim, salient_states_dim, 0.98)
		P_hat = DirectEnvModel(states_dim, actions_dim, max_torque, model_size='nn', hidden_size=1).double()

		dataset = ReplayMemory(1000000)
		batch_true_pe_grads = np.zeros((num_starting_states,count_parameters(actor)))
		estimate_return = torch.zeros((num_starting_states, max_actions)).double()
		returns = torch.zeros((num_starting_states, 1)).double()
		states_all = torch.zeros((num_starting_states, max_actions, states_dim)).double()

		for ep in range(num_starting_states):
			# noise.reset()
			state = env.reset()
			state = stablenoise.get_obs(state, 0)
			states = [state]
			actions = []
			rewards = []
			get_new_action = True
			
			for timestep in range(max_actions):
				with torch.no_grad():
					# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()
					action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
					# action = noise.get_action(action, timestep+1, multiplier=1.0)

				state_prime, reward, done, _ = env.step(action)
				state_prime = stablenoise.get_obs(state_prime, timestep+1)
				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)

				estimate_return[ep, timestep] = critic(torch.tensor(state).unsqueeze(0), torch.tensor(action).unsqueeze(0)).squeeze().detach().data.double()

				dataset.push(state, state_prime, action, reward)
				state = state_prime

			returns[ep] = discount_rewards(rewards, discount, center=False, batch_wise=False).detach()[0]
			states_all[ep] = torch.tensor(states[1:]).double()
			# pdb.set_trace()
			# (actor.sample_action(torch.tensor(states[1:]).double()) * returns.unsqueeze(1)).sum(dim=0)
			# true_pe_grads_attached = grad((actor.sample_action(torch.tensor(states[1:]).double()) * returns.unsqueeze(1)).sum(dim=0).norm(), actor.parameters(), create_graph=True)q
			# true_pe_grads = torch.DoubleTensor()
			# for g in range(len(true_pe_grads_attached)):
			# 	true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

			# batch_true_pe_grads[ep] = true_pe_grads.numpy()
			# pdb.set_trace()
			# for x, x_next, u, r, ret in zip(states[:-1], states[1:], actions, rewards, returns):
			# 	dataset.push(x, x_next, u, r, ret)
			# all_rewards.append(sum(rewards))
		# pdb.set_trace()
		# print(np.linalg.norm(((batch_true_pe_grads - batch_true_pe_grads.mean(axis=0))**2).mean(axis=0),2))
		# print(np.linalg.norm(batch_true_pe_grads.mean(axis=0),2))
		# true_pe_grads_attached = grad((actor.sample_action(states_all) * returns.unsqueeze(2)).sum(dim=0).mean(dim=0).norm(), actor.parameters(), create_graph=True)
		# true_pe_grads = torch.DoubleTensor()
		# for g in range(len(true_pe_grads_attached)):
		# 	true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))
		
		np.save(os.path.join(file_location, 'true_returns'+str(rs)+'_80k_'+env_name), np.asarray(returns))
		np.save(os.path.join(file_location, 'Q_estimated_returns'+str(rs)+'_80k_'+env_name), np.asarray(estimate_return))
		# pdb.set_trace()
		# print(abs(estimate_return - returns.mean(dim=0)).mean(dim=0).norm())
			# print(abs(estimate_return.double() - returns).mean())
			# print(abs(estimate_return.double() - returns).min())
			# print(abs(estimate_return.double() - returns).max())

		if get_true_grads:
			batch_true_pe_grads = np.zeros((100,count_parameters(actor)))
			batch_size = 500 #len(dataset)

			for b in range(100):
				batch = dataset.sample(batch_size)
				true_states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
				true_states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
				true_actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

				#calculate true gradients
				actor.zero_grad()

				#compute loss for actor
				true_policy_loss = -critic(true_states_next, actor.sample_action(true_states_next[:,:salient_states_dim]))

				true_term = true_policy_loss.mean()
				
				true_pe_grads = torch.DoubleTensor()
				true_pe_grads_attached = grad(true_term.mean(), actor.parameters(), create_graph=True)
				for g in range(len(true_pe_grads_attached)):
					true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

				batch_true_pe_grads[b] = true_pe_grads.numpy()

			print(np.linalg.norm(((batch_true_pe_grads - batch_true_pe_grads.mean(axis=0))**2).mean(axis=0),2))
			print(np.linalg.norm(batch_true_pe_grads.mean(axis=0),2))
			np.save(os.path.join(file_location, 'true_grads_1mil_rs_'+str(rs)+'_'+str(iter_count)+'_'+env_name), true_pe_grads.numpy())
			print(true_pe_grads) 


		model_grads = []
		for bs in [2]:#,1e3,1e4,1e5,1e6]:
			print(bs)
			batch_size = int(bs)
			batch = dataset.sample(batch_size)
			true_states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			true_states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			true_actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)
			step_state = torch.cat((true_states_prev, true_actions_tensor),dim=1).unsqueeze(1)
			
			model_x_curr, model_x_next, model_a_list, _, _ = P_hat.unroll(step_state, actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=salient_states_dim, noise=noise, env=env)

			#can remove these after unroll is fixed
			model_x_curr = model_x_curr.squeeze(1).squeeze(1)
			model_x_next = model_x_next.squeeze(1).squeeze(1)
			# print('next states norms:', model_x_next.norm().detach().data, true_states_next.norm().detach().data)
			model_a_list = model_a_list.squeeze(1).squeeze(1)
			# print('actions norms:', model_a_list.norm().detach().data, true_actions_tensor.norm().detach().data)

			a_prime = actor.sample_action(model_x_next[:,:salient_states_dim])
			model_policy_loss = -critic(model_x_next, a_prime)
			model_term = model_policy_loss[0]#.mean()

			model_pe_grads = torch.DoubleTensor()
			model_pe_grads_split = grad(model_term, actor.parameters(), create_graph=True)
			for g in range(len(model_pe_grads_split)):
				model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))
			
			pdb.set_trace()
			# dQ_daprime = grad(-critic(model_x_next, a_prime)[0], a_prime, create_graph=True)[0][0]
			# dQ_daprime_dxprime = grad(dQ_daprime.mean(), model_x_next, create_graph=True)
			
			# dpi_dtheta = torch.zeros((a_prime.shape[0], a_prime.shape[1], count_parameters(actor))).double()
			# for i in range(a_prime.shape[0]):
			# 	for j in range(a_prime.shape[1]):
			# 		grad_split = grad(a_prime[i,j], actor.parameters(), create_graph=True)
			# 		grads = torch.DoubleTensor()
			# 		for g in range(len(grad_split)):
			# 			grads = torch.cat((grads, grad_split[g].view(-1)))
			# 		dpi_dtheta[i,j] = grads.clone()
			
			# pdb.set_trace()
			# print(torch.einsum('ij,jk->ik', [dpi_dtheta[0].view(-1,6), dQ_daprime.unsqueeze(1)]).squeeze() == model_pe_grads)

			loss = (model_pe_grads - true_pe_grads).pow(2).sum()#.backward()
			# phat_grads_split = [i.grad for i in P_hat.parameters()]
			# p_hat_grads = torch.DoubleTensor()
			# for g in range(len(phat_grads_split)):
			# 	p_hat_grads = torch.cat((p_hat_grads, phat_grads_split[g].view(-1)))
			
			# model_grads.append(model_pe_grads.detach().numpy())

		# np.save(os.path.join(file_location, 'model_grads_rs_'+str(rs)+'_'+str(iter_count)+'_'+env_name), np.asarray(model_grads))
