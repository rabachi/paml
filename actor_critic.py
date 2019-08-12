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
# from get_data import save_stats

import dm_control2gym
device = 'cpu'

def pre_train_critic(actor, critic, dataset, epochs_value, discount, batch_size, q_optimizer, value_lr_schedule, states_dim, actions_dim, salient_states_dim, verbose=10):
	MSE = nn.MSELoss()
	# TAU=0.001
	TAU=0.001
	target_critic = Value(states_dim, actions_dim).double()

	for target_param, param in zip(target_critic.parameters(), critic.parameters()):
		target_param.data.copy_(param.data)

	for i in range(epochs_value):
		batch = dataset.sample(batch_size)
		states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
		states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
		rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
		actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

		actions_next = actor.sample_action(states_next[:,:salient_states_dim])
		target_q = target_critic(states_next, actions_next)
		y = rewards_tensor + discount * target_q.detach() #detach to avoid backprop target
		q = critic(states_prev, actions_tensor)

		q_optimizer.zero_grad()
		loss = MSE(y, q)
		loss.backward()
		nn.utils.clip_grad_value_(critic.parameters(), 20.0)
		q_optimizer.step()
		if value_lr_schedule is not None:
			value_lr_schedule.step()

		#soft update the target critic
		for target_param, param in zip(target_critic.parameters(), critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

		if i % verbose == 0:
			print('Epoch: {:4d} | Value estimator loss: {:.5f}'.format(i,loss.detach().cpu()))

	return critic #finish when target has converged


def actor_critic_DDPG(env, actor, noise, critic, real_dataset, batch_size, num_starting_states, max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, epsilon_decay, value_lr_schedule, file_location, file_id, num_action_repeats, planning_horizon=1, P_hat=None, model_type='paml', save_checkpoints=False, rho_ER=0.5):
 
	starting_states = num_starting_states
	env_name = env.spec.id
	max_torque = float(env.action_space.high[0])

	#rho = fraction of data used from experience replay
	# rho_ER = 0.5

	random_start_frac = 0.2 #will also use search control that searches near previously seen states
	# random_around_ER_frac = 0.8
	# from_ER_frac = 1.0 - random_start_frac - random_around_ER_frac
	radius = 0.49#np.pi/2.#0.52

	actions_dim = env.action_space.shape[0]

	R_range = planning_horizon

	# #For Pendulum
	# TAU=0.001      #Target Network HyperParameters
	# LRA=0.0001      #LEARNING RATE ACTOR
	# LRC=0.001       #LEARNING RATE CRITIC

	#For LQR
	TAU=0.001      #Target Network HyperParameters
	LRA=0.0001      #LEARNING RATE ACTOR
	LRC=0.001       #LEARNING RATE CRITIC

	# LRA=1e-6      #LEARNING RATE ACTOR
	# LRC=1e-5       #LEARNING RATE CRITIC

	buffer_start = 100
	# epsilon = 1
	epsilon_original = epsilon
	# epsilon_decay = 1./1000000
	noise_decay = 1.0 - epsilon_decay #0.99999

	# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	max_torque = float(env.action_space.high[0])
	stablenoise = StableNoise(states_dim, salient_states_dim, 0.98)

	# batch_size = 64
	best_loss = 10

	MSE = nn.MSELoss()
	# actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
	# critic = Value(states_dim, actions_dim).double()
	critic_optimizer  = optim.Adam(critic.parameters(), lr=LRC, weight_decay=1e-2)
	# if value_lr_schedule is None:
		# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(critic_optimizer, milestones=[7000,12000,20000], gamma=0.1)
		# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(critic_optimizer, milestones=[7000,9000,12000], gamma=0.1)
		# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(critic_optimizer, milestones=[100,000], gamma=0.1)
	actor_optimizer = optim.Adam(actor.parameters(), lr=LRA)

	#initialize target_critic and target_actor
	# target_actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
	target_actor = DeterministicPolicy(salient_states_dim, actions_dim, max_torque).double()
	target_critic = Value(states_dim, actions_dim).double()
	for target_param, param in zip(target_critic.parameters(), critic.parameters()):
		target_param.data.copy_(param.data)

	for target_param, param in zip(target_actor.parameters(), actor.parameters()):
		target_param.data.copy_(param.data)

	dataset = ReplayMemory(1000000)

	render = True if use_model and P_hat.model_size == 'cnn' else False 
	critic_loss = torch.tensor(2)
	policy_loss = torch.tensor(2)
	best_rewards = -np.inf
	for ep in range(starting_states):
		noise.reset()
		# epsilon *= noise_decay**ep
		# epsilon -= epsilon_decay
		# epsilon = epsilon_original
		if not use_model:
			state = env.reset()
			if env.spec.id != 'lin-dyn-v0':
				state = stablenoise.get_obs(state, 0)

			states = [state]
			actions = []
			rewards = []
			get_new_action = True
			for timestep in range(max_actions):
				if get_new_action:
					# epsilon -= epsilon_decay
					action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()
					#action += noise()*max(0, epsilon) #try without noise
					#action = np.clip(action, -1., 1.)
					action = noise.get_action(action, timestep)
					get_new_action = False
					action_counter = 1
				
				state_prime, reward, done, _ = env.step(action)

				if env.spec.id != 'lin-dyn-v0':
					state_prime = stablenoise.get_obs(state_prime, timestep + 1)

				if render:
					env.render(mode='pixels')

				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)
				
				dataset.push(state, state_prime, action, reward)
				state = state_prime

				get_new_action = True if action_counter == num_action_repeats else False
				action_counter += 1

 				#former indentation level
				if len(dataset) > batch_size:#for iteration in range(int(np.floor(max_actions/batch_size))):#
					batch = dataset.sample(batch_size)

					states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
					states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
					rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
					actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)
					actions_next = target_actor.sample_action(states_next[:,:salient_states_dim])
					#Compute target Q value
					target_Q = target_critic(states_next, actions_next)
					target_Q = rewards_tensor + discount * target_Q.detach()
					
					#Compute current Q estimates
					current_Q = critic(states_prev, actions_tensor)
					critic_loss = MSE(current_Q, target_Q)

					critic_optimizer.zero_grad()
					critic_loss.backward()
					critic_optimizer.step()
					if value_lr_schedule is not None:
						value_lr_schedule.step()

					#compute actor loss
					policy_loss = -critic(states_prev, actor.sample_action(states_prev[:,:salient_states_dim])).mean()

					#Optimize the actor
					actor_optimizer.zero_grad()
					policy_loss.backward()
					actor_optimizer.step()

					#soft update of the frozen target networks
					for target_param, param in zip(target_critic.parameters(), critic.parameters()):
						target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

					for target_param, param in zip(target_actor.parameters(), actor.parameters()):
						target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)
			# all_rewards.append(sum(rewards))
		else:
			if P_hat is None:
				raise NotImplementedError

			true_batch_size = int(np.floor(rho_ER * batch_size))
			if true_batch_size > 0:
				#use real_dataset here
				unroll_num = 1
				ell = 0
				R_range = planning_horizon

				true_batch = real_dataset.sample(true_batch_size)
				true_x_curr = torch.tensor([samp.state for samp in true_batch]).double().to(device)
				true_x_next = torch.tensor([samp.next_state for samp in true_batch]).double().to(device)
				true_a_list = torch.tensor([samp.action for samp in true_batch]).double().to(device)
				true_r_list = torch.tensor([samp.reward for samp in true_batch]).double().to(device).unsqueeze(1)

				actions_next = target_actor.sample_action(true_x_next[:,:salient_states_dim])
				target_q = target_critic(true_x_next, actions_next)
				target_q = true_r_list + discount * target_q.detach() #detach to avoid backprop target
				current_q = critic(true_x_curr, true_a_list)

				#update critic only with true data
				critic_optimizer.zero_grad()
				critic_loss = MSE(target_q, current_q)
				critic_loss.backward()
				critic_optimizer.step()
				if value_lr_schedule is not None:
					value_lr_schedule.step()


			model_batch_size = batch_size - true_batch_size
			random_start_model_batch_size = int(np.floor(random_start_frac * model_batch_size * 1.0/planning_horizon)) #randomly sample from state space

			#This is kind of slow, but it would work with any environment, the other way it to do the reset batch_wise in a custom function made for every environment separately ... but that seems like it shouldn't be done
			random_model_x_curr = torch.zeros((random_start_model_batch_size, states_dim)).double()
			for samp in range(random_start_model_batch_size):
				s0 = env.reset()
				if env.spec.id != 'lin-dyn-v0':
					s0 = stablenoise.get_obs(env.reset(), 0)
				random_model_x_curr[samp] = torch.from_numpy(s0).double()

			with torch.no_grad():

				#some starting points chosen completely randomly
				# random_model_actions = actor.sample_action(random_model_x_curr).numpy()#torch.clamp(actor.sample_action(random_model_x_curr) + torch.from_numpy(noise()*max(0, epsilon)), min=-max_torque, max=max_torque)
				# random_model_actions = torch.from_numpy(noise.get_action(random_model_actions, 1))
				# random_model_x_next = P_hat(torch.cat((random_model_x_curr, random_model_actions),1))
				# random_model_r_list = get_reward_fn(env, random_model_x_curr.unsqueeze(1), random_model_actions.unsqueeze(1))


				# #some chosen directly from Experience Replay
				# replay_start_model_batch_size = int(np.floor(from_ER_frac * model_batch_size))#model_batch_size - random_start_model_batch_size #randomly sample from replay buffer 
				# replay_model_batch = real_dataset.sample(replay_start_model_batch_size)
				# replay_model_x_curr = torch.tensor([samp.state for samp in replay_model_batch]).double().to(device)
				# replay_model_actions = torch.clamp(actor.sample_action(replay_model_x_curr) + torch.from_numpy(noise()*max(0, epsilon)), min=-max_torque, max=max_torque)
				# replay_model_x_next = P_hat(torch.cat((replay_model_x_curr, replay_model_actions),1))
				# replay_model_r_list = get_reward_fn(env, replay_model_x_curr.unsqueeze(1), replay_model_actions.unsqueeze(1))

				#some chosen from randomly sampled around area around states in ER
				replay_start_random_model_batch_size = model_batch_size - random_start_model_batch_size 
				replay_model_batch = real_dataset.sample(replay_start_random_model_batch_size)
				# have to move the states to a random position within a radius, here it's 1

				# if env_name == 'dm-Pendulum-v0' and states_dim == salient_states_dim:
				# 	random_pos_delta = np.random.uniform(size=(replay_start_random_model_batch_size, 1), low=-radius, high=radius)#
				# 	x1byx0 = torch.from_numpy(np.tan(random_pos_delta)).double()
				# 	x0 = torch.tensor([replay_model_batch[idx].state[0] for idx in range(len(replay_model_batch))]).unsqueeze(1).double()
				# 	x1 = x0*x1byx0
				# 	x2 = torch.tensor([replay_model_batch[idx].state[2] for idx in range(len(replay_model_batch))]).unsqueeze(1).double()
				# 	replay_model_x_curr = torch.cat((x0,x1,x2),1).double().to(device)
				# else:
				random_pos_delta = 2*(np.random.random(size = (replay_start_random_model_batch_size, states_dim)) - 0.5) * radius 
				replay_model_x_curr = torch.tensor([replay_model_batch[idx].state + random_pos_delta[idx] for idx in range(replay_start_random_model_batch_size)]).double()
				# replay_model_actions = actor.sample_action(replay_model_x_curr).numpy() #torch.clamp(actor.sample_action(replay_model_x_curr) + torch.from_numpy(noise()*max(0, epsilon)), min=-max_torque, max=max_torque) 
				# replay_model_actions = torch.from_numpy(noise.get_action(replay_model_actions, 1))
				# replay_model_x_next = P_hat(torch.cat((replay_model_x_curr, replay_model_actions),1))
				# replay_model_r_list = get_reward_fn(env, replay_model_x_curr.unsqueeze(1), replay_model_actions.unsqueeze(1))
			if true_batch_size > 0:
				states_prev = torch.cat((true_x_curr, random_model_x_curr, replay_model_x_curr), 0)
			else:
				states_prev = torch.cat((random_model_x_curr, replay_model_x_curr), 0)
			# actions_list = torch.cat((true_a_list, random_model_actions, replay_model_actions), 0)
			states_prime = torch.zeros((batch_size, planning_horizon, states_dim)).double().to(device)

			states_ = states_prev.clone()
			noise.reset()
			for p in range(planning_horizon):
				actions_noiseless = actor.sample_action(states_[:,:salient_states_dim]).detach().numpy() 
				actions_ = torch.from_numpy(noise.get_action(actions_noiseless, p, multiplier=0.1))

				states_prime[:, p, :] = P_hat.predict(states_, actions_).detach()
				# states_prime[:, p, :] = P_hat(torch.cat((states_, actions_),1)).detach() #+ states_
				
				# rewards_model.append(get_reward_fn(env, states_.unsqueeze(1), actions_.unsqueeze(1)))
				states_ = states_prime[:, p, :]

			states_current = torch.cat((states_prev, states_prime[:, :-1, :].contiguous().view(-1, states_dim)), 0)
			actor_optimizer.zero_grad()
			policy_loss = -critic(states_current, actor.sample_action(states_current[:,:salient_states_dim]))
			policy_loss = policy_loss.mean()
			policy_loss.backward()
			actor_optimizer.step()

			# states_next = torch.cat((true_x_next, random_model_x_next, replay_model_x_next), 0)
			for target_param, param in zip(target_critic.parameters(), critic.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

			for target_param, param in zip(target_actor.parameters(), actor.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)

			# rewards_list = torch.cat((true_r_list, random_model_r_list, replay_model_r_list), 0)

			# all_rewards.append(rewards_list.sum())
			#compute loss for actor
			# actor_optimizer.zero_grad()
			# policy_loss = -critic(states_prev, actor.sample_action(states_prev))
			# policy_loss = policy_loss.mean()
			# policy_loss.backward()
			# actor_optimizer.step()

			#soft update of the frozen target networks
			# for target_param, param in zip(target_critic.parameters(), critic.parameters()):
			# 	target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

			# for target_param, param in zip(target_actor.parameters(), actor.parameters()):
			# 	target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)

		if (ep % verbose  == 0) or (ep == starting_states - 1): #evaluate the policy using no exploration noise
			if not use_model:
				eval_rewards = []
				for ep in range(10):
					state = env.reset()
					if env.spec.id != 'lin-dyn-v0':
						state = stablenoise.get_obs(state, 0)
					episode_rewards = []
					for timestep in range(max_actions):
						action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()#[:salient_states_dim])).detach().numpy()
						state_prime, reward, done, _ = env.step(action)
						if env.spec.id != 'lin-dyn-v0':
							state_prime = stablenoise.get_obs(state_prime, timestep+1)

						episode_rewards.append(reward)
						state = state_prime
					
					eval_rewards.append(sum(episode_rewards))
				all_rewards.append(sum(eval_rewards)/10.)

				if (all_rewards[-1] > best_rewards) and save_checkpoints:
					torch.save(actor.state_dict(), os.path.join(file_location,'act_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))
					best_rewards = all_rewards[-1]
				
				print("Ep: {:5d} | Epsilon: {:.5f} | Q_Loss: {:.3f} | Pi_Loss: {:.3f} | Average rewards of 10 independent episodes:{:.4f}".format(ep, epsilon, critic_loss.detach(), policy_loss.detach(), all_rewards[-1]))
			
				#save all_rewards
				np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, use_model, env_name, R_range, max_actions + 1, file_id)), np.asarray(all_rewards))

		# if sum(all_rewards[-10:])/len(all_rewards[-10:]) > -700:
		# 	render = False
	return actor, critic


def plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, salient_states_dim, actions_dim, discount, max_actions, env, lr, num_iters, file_location, file_id, save_checkpoints_training, verbose, batch_size, num_virtual_episodes, model_type, num_action_repeats, epsilon, epsilon_decay, planning_horizon, input_rho):
	# verbose = 20
	# batch_size = 64
	R_range = planning_horizon
	losses = []

	kwargs = {
				# 'P_hat'              : P_hat,
				'actor'	             : actor, 
				'critic'	 		 : critic,
				#'target_critic'	 	 : target_critic,
				#'q_optimizer'		 : value_optimizer,
				'opt' 				 : model_opt, 
				'num_episodes'		 : num_episodes, 
				'num_starting_states': num_starting_states, 
				'states_dim'		 : states_dim, 
				'salient_states_dim' : salient_states_dim,
				'actions_dim'		 : actions_dim, 
				'batch_size'		 : batch_size,
				# 'use_model'			 : False, 
				'discount'			 : discount, 
				'max_actions'		 : max_actions, 
				'planning_horizon'	 : planning_horizon,
				# 'train'              : train, 
				'lr'		         : lr,
				'num_iters'          : num_iters,
				'losses'             : [],
				'env'				 : env,
				# 'value_loss_coeff'   : value_loss_coeff,
				'verbose'			 : verbose,
				'file_location'		 : file_location,
				'file_id'			 : file_id,
				'save_checkpoints'	 : save_checkpoints_training
			}

	unroll_num = 1
	# model_type = 'paml'
	log = 1
	ell = 0
	psi = 1.1
	# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	noise = OUNoise(env.action_space)
	kwargs['noise'] = noise

	all_rewards = []
	epochs_value = 1000
	#value_optimizer = optim.SGD(critic.parameters(), lr=1e-3, momentum=0.90, nesterov=True) 
	# value_optimizer = optim.SGD(critic.parameters(), lr=1e-5, momentum=0.90, nesterov=True) 
	value_optimizer = optim.SGD(critic.parameters(), lr=5e-4)#optim.Adam(critic.parameters(), lr=1e-4, weight_decay=1e-8)#optim.SGD(critic.parameters(), lr=1e-4)#, momentum=0.90, nesterov=True) 
	value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_optimizer, milestones=[7000,12000,20000], gamma=0.1)
	# lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[100000,200000,400000], gamma=0.1)
	
	# lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[400000,800000,1000000], gamma=0.1)

	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[40,100,500], gamma=0.1)

	dataset = ReplayMemory(1000000)
	max_torque = float(env.action_space.high[0])
	paml_losses = []
	global_step = 0
	total_eps = 10000
	env_name = env.spec.id
	true_rewards = []
	use_pixels = False
	fractions_real_data_schedule = np.linspace(1.0,0.5,num=50)

	stablenoise = StableNoise(states_dim, salient_states_dim, 0.98)
	original_batch_size = batch_size
	while(global_step <= total_eps):#/num_starting_states):
		# Generating sample trajectories 
		print("Generating sample trajectories ... epislon is {:.3f}".format(epsilon))
		# dataset, _, new_epsilon = generate_data(env, dataset, actor, num_starting_states, None, max_actions, noise, epsilon, epsilon_decay, num_action_repeats, discount=discount, all_rewards=[])#true_rewards)
		dataset, _, new_epsilon = generate_data(env, states_dim, dataset, actor, num_starting_states, None, min(200,max_actions), noise, epsilon, epsilon_decay, num_action_repeats, discount=discount, all_rewards=[], use_pixels=(P_hat.model_size=='cnn'))#true_rewards)
		batch_size = min(original_batch_size, len(dataset))
		#Evaluate policy without noise
		eval_rewards = []
		for ep in range(10):
			state = env.reset()
			if env.spec.id != 'lin-dyn-v0':
				state = stablenoise.get_obs(state, 0)
			fakes = []
			episode_rewards = []
			for timestep in range(max_actions):
				action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()
				state_prime, reward, done, _ = env.step(action)
				if env.spec.id != 'lin-dyn-v0':
					state_prime = stablenoise.get_obs(state_prime, timestep+1)
				# frewards = get_reward_fn(env, torch.DoubleTensor(state).unsqueeze(0).unsqueeze(1), torch.DoubleTensor(action).unsqueeze(0).unsqueeze(1))
				# fakes.append(frewards)
				episode_rewards.append(reward)
				state = state_prime
			# print(sum(fakes), sum(episode_rewards))
			
			eval_rewards.append(sum(episode_rewards))
		true_rewards.append(sum(eval_rewards)/10.)

		print('Average rewards on true dynamics: {:.3f}'.format(true_rewards[-1]))
		torch.save(actor.state_dict(), os.path.join(file_location,'act_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))
					
		np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)), np.asarray(true_rewards))
		# epsilon = new_epsilon
		kwargs['epsilon'] = epsilon
		print(len(dataset))
		# print("Done")

		kwargs['dataset'] = dataset
		# epochs_value = int(np.ceil(epochs_value * psi))

		# if model_type != 'mle':
		# if timestep % 1000 == 0 and ep % 10 == 0:
			# critic.reset_weights()
		critic = pre_train_critic(actor, critic, dataset, epochs_value, discount, batch_size, value_optimizer, value_lr_schedule, states_dim, actions_dim, salient_states_dim, verbose=100)
		kwargs['critic'] = critic



		if env_name == 'lin-dyn-v0':
			kwargs['train'] = False
			kwargs['num_episodes'] = num_episodes
			kwargs['num_starting_states'] = num_starting_states
			kwargs['losses'] = []
			kwargs['use_model'] = False
			P_hat.actor_critic_paml_train(**kwargs) 

		# num_iters += 10# int(np.ceil(num_iters * psi))
		kwargs['train'] = True
		# kwargs['train'] = False
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['losses'] = losses
		kwargs['num_iters'] = num_iters
		kwargs['use_model'] = True
		# kwargs['P_hat'] = P_hat

		if model_type =='mle':
			# P_hat.general_train_mle(actor, dataset, states_dim, salient_states_dim, num_iters, max_actions, model_opt, env_name, losses, batch_size, file_location, file_id, save_checkpoints=save_checkpoints_training, verbose=20, lr_schedule=lr_schedule)
			P_hat.general_train_mle(actor, dataset, states_dim, salient_states_dim, num_iters, max_actions, model_opt, env_name, losses, batch_size, file_location, file_id, save_checkpoints=True, verbose=20, lr_schedule=lr_schedule, global_step=global_step)

		elif (model_type == 'paml') or (model_type == 'pamlmean'):
			# P_hat = DirectEnvModel(states_dim, actions_dim, max_torque).double()
			if (global_step > 0) and (global_step % 15 == 0): #for lqr: 8 good
				lr = lr / 1.01
			kwargs['lr'] = lr
			# kwargs['P_hat'] = P_hat
			# P_hat, _ = actor_critic_paml_train(**kwargs)
			P_hat.actor_critic_paml_train(**kwargs)
		elif model_type == 'random':
			P_hat = DirectEnvModel(states_dim, actions_dim, max_torque).double()
			# kwargs['P_hat'] = P_hat
		else:
			raise NotImplementedError

		# kwargs['train'] = False
		# kwargs['use_model'] = True
		# kwargs['P_hat'] = P_hat
		# kwargs['losses'] = []
		# _, loss_paml = actor_critic_paml_train(**kwargs)
		# paml_losses.append(loss_paml)

		use_model = True
		# num_virtual_episodes = 600
		train = True
		#use the epsilon arrived at from generation of real data

		# Uncomment to see effect of virtual episodes (and comment out the lines below it)
		# for v_ep in range(num_virtual_episodes):
		# 	actor, critic = actor_critic_DDPG(env, actor, noise,critic, dataset, batch_size, 1, max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, epsilon_decay, value_lr_schedule, file_location, file_id, num_action_repeats, planning_horizon=planning_horizon, P_hat=P_hat, model_type=model_type)
		
		# 	#check paml_loss with the new policy
		# 	kwargs['train'] = False
		# 	kwargs['use_model'] = True
		# 	kwargs['losses'] = []
		# 	kwargs['P_hat'] = P_hat
		# 	kwargs['actor'] = actor
		# 	kwargs['critic'] = critic
		# 	_, loss_paml = actor_critic_paml_train(**kwargs)
		# 	paml_losses.append(loss_paml)
		# actor, critic = actor_critic_DDPG(env, actor, noise,critic, dataset, 8, 2*int(np.ceil(num_virtual_episodes/5.0)), max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, epsilon_decay, value_lr_schedule, file_location, file_id, num_action_repeats, planning_horizon=planning_horizon, P_hat=P_hat, model_type=model_type, save_checkpoints=save_checkpoints_training)

		# rho_ER = fractions_real_data_schedule[global_step] if global_step < 50 else 0.5



		actor, critic = actor_critic_DDPG(env, actor, noise,critic, dataset, batch_size, num_virtual_episodes, max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, epsilon_decay, value_lr_schedule, file_location, file_id, num_action_repeats, planning_horizon=planning_horizon, P_hat=P_hat, model_type=model_type, save_checkpoints=save_checkpoints_training, rho_ER=input_rho)
		




		#check paml_loss with the new policy
		# kwargs['train'] = False
		# kwargs['use_model'] = True
		# kwargs['losses'] = []
		# kwargs['P_hat'] = P_hat
		# kwargs['actor'] = actor
		# kwargs['critic'] = critic
		# _, loss_paml = actor_critic_paml_train(**kwargs)
		# paml_losses.append(loss_paml)

		if global_step % log == 0:
			np.save(os.path.join(file_location,'ac_pamlloss_model_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(model_type, env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(paml_losses))
		global_step += 1

		# all_rewards.append(sum(rewards))


	
def main(
			env_name, 
			real_episodes,
			virtual_episodes,
			num_eps_per_start,
			num_iters,
			max_actions,
			discount,
			states_dim,
			salient_states_dim,
			initial_model_lr,
			model_type,
			file_id,
			save_checkpoints_training,
			batch_size,
			verbose,
			model_size,
			num_action_repeats,
			rs,
			planning_horizon,
			hidden_size,
			input_rho,
			ensemble_size
		):

	# file_location = '/h/abachiro/paml/results'
	file_location = '/scratch/gobi1/abachiro/paml_results'
	# rs = 0
	# torch.manual_seed(rs)
	# np.random.seed(rs)

	# env = gym.make('Pendulum-v0')
	#env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
	# env.seed(rs)

	dm_control2gym.create_render_mode('pixels', show=False, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(), depth=False, scene_option=None)

	if env_name == 'lin_dyn':
		gym.envs.register(id='lin-dyn-v0', entry_point='gym_linear_dynamics.envs:LinDynEnv',)
		# env = gym.make('gym_linear_dynamics:lin-dyn-v0')
		env = gym.make('lin-dyn-v0')

	elif env_name == 'Pendulum-v0':
		if states_dim > salient_states_dim:
			# env = AddExtraDims(NormalizedEnv(gym.make('Pendulum-v0')), states_dim - salient_states_dim)
			env = gym.make('Pendulum-v0') #NormalizedEnv(gym.make('Pendulum-v0'))
		else:
			env = gym.make('Pendulum-v0')#NormalizedEnv(gym.make('Pendulum-v0')) #idk if this is good to do ... normalized env thing

	elif env_name == 'Reacher-v2':
		env = gym.make('Reacher-v2') #NormalizedEnv(gym.make('Reacher-v2'))

	elif env_name == 'Swimmer-v2':
		env = gym.make('Swimmer-v2') #NormalizedEnv(gym.make('Swimmer-v2'))

	elif env_name == 'HalfCheetah-v2':	
		env = gym.make('HalfCheetah-v2') #NormalizedEnv(gym.make('HalfCheetah-v2'))

	elif env_name == 'dm-Cartpole-swingup-v0':
		env = dm_control2gym.make(domain_name="cartpole", task_name="swingup")
		env.spec.id = 'dm-Cartpole-swingup-v0'

	elif env_name == 'dm-Pendulum-v0':
		env = dm_control2gym.make(domain_name="pendulum", task_name="swingup") #NormalizedEnv(dm_control2gym.make(domain_name="pendulum", task_name="swingup"))
		env.spec.id = 'dm-Pendulum-v0'
	else:
		raise NotImplementedError

	if model_type == 'model_free':
		plan = True
	else:
		plan = False

	torch.manual_seed(rs)
	np.random.seed(rs)	
	env.seed(rs)

	#50
	num_starting_states = real_episodes if not plan else 10000
	num_episodes = num_eps_per_start #1
	#batch_size = 64
	val_num_episodes = 1
	val_num_starting_states = 10
	val_batch_size = val_num_starting_states*val_num_episodes
	
	#num_iters = 400
	losses = []
	unroll_num = 1

	value_loss_coeff = 1.0
	max_torque = float(env.action_space.high[0])
	#discount = 0.9

	#max_actions = 200

	#states_dim = 5
	actions_dim = env.action_space.shape[0] #states_dim
	#salient_states_dim = 3 #states_dim
	continuous_actionspace = True
	R_range = planning_horizon

	use_model = True
	train_value_estimate = False
	train = True

	action_multiplier = 0.1
	epsilon = 1.
	epsilon_decay = 1./100000#1./100000
	# critic = Value(states_dim, actions_dim)
	# critic.double()
	# actor = DeterministicPolicy(states_dim, actions_dim)
	# actor.double()
	# target_actor = DeterministicPolicy(states_dim, actions_dim)
	# target_actor.double()
	# target_critic = Value(states_dim, actions_dim)
	# target_critic.double()
	if model_size == 'cnn':
		states_dim = (6, states_dim + actions_dim, states_dim)
		pdb.set_trace()

	if plan:
		all_rewards = []
		noise = OUNoise(env.action_space)
		# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
		# actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		actor = DeterministicPolicy(salient_states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim).double()
		actor_critic_DDPG(env, actor, noise, critic, None, batch_size, num_starting_states, max_actions, states_dim, salient_states_dim, discount, False, True, verbose, all_rewards, epsilon, epsilon_decay, None, file_location, file_id, num_action_repeats, P_hat=None, model_type='model_free', save_checkpoints=save_checkpoints_training)

		np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, use_model, env_name, R_range, max_actions + 1, file_id)), np.asarray(all_rewards))
		# np.save('actorcritic_pendulum_rewards',np.asarray(all_rewards)) 
		# pdb.set_trace()
	else:
		# P_hat = []
		# for ens in range(ensemble_size):
		# 	P_hat
		P_hat = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size, hidden_size=hidden_size)
		P_hat.double()
		#P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon6_traj7.pth', map_location=device))
		#P_hat.load_state_dict(torch.load('act_model_paml_checkpoint_train_True_lin_dyn_horizon5_traj6_using1states.pth', map_location=device))

		# value_optimizer = optim.SGD(critic.parameters(), lr=1e-5, momentum=0.90, nesterov=True) 
		# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_optimizer, milestones=[1500,3000], gamma=0.1)

		#1e-4
		if model_type == 'paml':
			model_opt = optim.SGD(P_hat.parameters(), lr=initial_model_lr)#, momentum=0.90, nesterov=True)
		elif model_type == 'mle':
			model_opt = optim.SGD(P_hat.parameters(), lr=initial_model_lr)
			# model_opt = optim.Adam(P_hat.parameters(), lr=initial_model_lr, weight_decay=1e-2)
		# actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		actor = DeterministicPolicy(salient_states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim).double()
		# kwargs = {
		# 		'P_hat'              : P_hat, 
		# 		'actor'	             : actor, 
		# 		'critic'	 		 : critic,
		# 		# 'target_critic'	 	 : target_critic,
		# 		'q_optimizer'		 : None,
		# 		'opt' 				 : model_opt, 
		# 		'num_episodes'		 : num_episodes, 
		# 		'num_starting_states': num_starting_states, 
		# 		'states_dim'		 : states_dim, 
		# 		'actions_dim'		 : actions_dim, 
		# 		'use_model'			 : False, 
		# 		'discount'			 : discount, 
		# 		'max_actions'		 : max_actions, 
		# 		'train'              : False,  
		# 		'lr_schedule'		 : lr_schedule,
		# 		'num_iters'          : int(num_iters/120),
		# 		'losses'             : [],
		# 		'value_loss_coeff'   : value_loss_coeff
		# 		}

		#pretrain value function
		ell = 0
		# lr = 1e-5
		plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, salient_states_dim,actions_dim, discount, max_actions, env, initial_model_lr, num_iters, file_location, file_id, save_checkpoints_training, verbose, batch_size, virtual_episodes, model_type, num_action_repeats, epsilon, epsilon_decay, planning_horizon, input_rho)
		# if train_value_estimate:
		# 	epochs_value = 300
		# 	verbose = 100
		# 	critic = pre_train_critic(actor, critic, dataset, validation_dataset, epochs_value, discount, batch_size, value_optimizer, value_lr_schedule, max_actions, verbose)
		# else:
		# 	critic.load_state_dict(torch.load('critic_horizon{}_traj{}.pth'.format(R_range, max_actions+1), map_location=device))
		# 	target_critic.load_state_dict(torch.load('critic_horizon{}_traj{}.pth'.format(R_range, max_actions+1), map_location=device))
	# print('Generating sample trajectories ...')
	# dataset, validation_dataset = generate_data(env, actor, num_starting_states, val_num_starting_states, max_actions)
	# print('Done!')

	# with torch.no_grad():
	# 	#generate validation data
	# 	val_step_state = torch.zeros((val_batch_size, unroll_num, states_dim+actions_dim)).double()
	# 	for b in range(val_num_starting_states):
	# 		x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
	# 		val_step_state[b*val_num_episodes:val_num_episodes*(b+1),:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

	# 	val_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(val_step_state[:,:unroll_num,:states_dim]) #I think all this does is make the visualizations look better, shouldn't affect performance (or visualizations ... )
	# 	val_true_x_curr, val_true_x_next, val_true_a_list, val_true_r_list, val_true_a_prime_list = P_hat.unroll(val_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=False, policy_states_dim=states_dim)

	# 	#generate training data
	# 	train_step_state = torch.zeros((batch_size, unroll_num, states_dim+actions_dim)).double()
	# 	for b in range(num_starting_states):
	# 		x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
	# 		train_step_state[b*num_episodes:num_episodes*(b+1),:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

	# 	train_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(train_step_state[:,:unroll_num,:states_dim])#I think all this does is make the visualizations look better, shouldn't affect performance (or visualizations ... )
	# 	train_true_x_curr, train_true_x_next, train_true_a_list, train_true_r_list, train_true_a_prime_list = P_hat.unroll(train_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=False, policy_states_dim=states_dim)

	
	#get accuracy of true dynamics on validation data
	# true_r_list = val_true_r_list
	# true_x_curr = val_true_x_curr
	# true_a_list = val_true_a_list
	# step_state = val_step_state
		# prefix='true_actorcritic_'
		# epochs_value = 300
		# best_loss = 1000
		# true_r_list = train_true_r_list
		# true_x_curr = train_true_x_curr
		# true_a_list = train_true_a_list
		# # true_returns = torch.zeros_like(true_r_list)
		# true_returns = discount_rewards(true_r_list[:,ell,:-1], discount, center=False, batch_wise=True)
		# for i in range(epochs_value):
		# 	true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
		# 	# true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
			
		# 	save_stats(true_returns, true_r_list, true_a_list, true_x_curr, value=true_value, prefix='true_actorcritic_')
		# 	np.save(prefix+'value_training', true_value.squeeze().detach().cpu().numpy())
		# 	true_value_loss = (true_returns - true_value).pow(2).mean()
		# 	print('Epoch: {:4d} | Value estimator loss: {:.5f}'.format(i,true_value_loss.detach().cpu()))

		# 	if true_value_loss < best_loss:
		# 		torch.save(value_estimator.state_dict(), 'value_estimator_horizon{}_traj{}.pth'.format(R_range, max_actions+1))
		# 		best_loss = true_value_loss

		# 	value_opt.zero_grad()
		# 	true_value_loss.backward()
		# 	value_opt.step()
		# 	value_lr_schedule.step()
		# 	#check validation
		# 	if (i % 10) == 0:
		# 		with torch.no_grad():
		# 			true_x_curr = val_true_x_curr
		# 			true_a_list = val_true_a_list
		# 			true_r_list = val_true_r_list
		# 			# true_returns = torch.zeros_like(true_r_list)
		# 			true_returns = discount_rewards(true_r_list[:,ell,:-1], discount, center=False, batch_wise=True)
		# 			true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
		# 			true_value_loss = (true_returns - true_value).pow(2).mean()
		# 			print('Validation value estimator loss: {:.5f}'.format(true_value_loss.detach().cpu()))
		# 		true_r_list = train_true_r_list
		# 		true_x_curr = train_true_x_curr
		# 		true_a_list = train_true_a_list
		# 		# true_returns = torch.zeros_like(train_true_r_list)
		# 		true_returns = discount_rewards(true_r_list[:,ell,:-1], discount, center=False, batch_wise=True)

# if __name__=="__main__":
# 	main()





