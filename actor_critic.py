import numpy as np
import gym
import sys

import torch
import math
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal

import pdb

import os
from models import *
from networks import *
from utils import *
import copy

import dm_control2gym
import collections
device = 'cpu'

from pympler import summary, muppy


def actor_critic_DDPG(env, actor, noise, critic, real_dataset, validation_dataset, batch_size, num_starting_states, max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, epsilon_decay, value_lr_schedule, file_location, file_id, num_action_repeats, planning_horizon=1, P_hat=None, model_type='paml', save_checkpoints=False, rho_ER=0.5):

	all_rewards = []
	starting_states = num_starting_states
	env_name = env.spec.id
	max_torque = float(env.action_space.high[0])

	random_start_frac = 0.3 #will also use search control that searches near previously seen states
	# random_around_ER_frac = 0.8
	# from_ER_frac = 1.0 - random_start_frac - random_around_ER_frac
	radius = .8#np.pi/2.#0.52

	actions_dim = env.action_space.shape[0]

	R_range = planning_horizon

	# #For Pendulum
	# TAU=0.001      #Target Network HyperParameters
	LRA=0.0001      #LEARNING RATE ACTOR
	LRC=0.001       #LEARNING RATE CRITIC

	#For LQR
	TAU=0.001      #Target Network HyperParameters
	# LRA=0.0001      #LEARNING RATE ACTOR
	# LRC=0.001       #LEARNING RATE CRITIC

	# LRA=1e-4      #LEARNING RATE ACTOR
	# LRC=1e-4       #LEARNING RATE CRITIC

	buffer_start = 100
	# epsilon = 1
	epsilon_original = epsilon
	# epsilon_decay = 1./1000000
	noise_decay = 1.0 - epsilon_decay #0.99999

	# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	max_torque = float(env.action_space.high[0])
	stablenoise = StableNoise(states_dim, salient_states_dim, 0.992)#, init=np.mean(env.reset()))

	# batch_size = 64
	best_loss = 10

	MSE = nn.MSELoss()
	critic_optimizer  = optim.Adam(critic.parameters(), lr=LRC)#, weight_decay=1e-2)
	actor_optimizer = optim.Adam(actor.parameters(), lr=LRA)

	#initialize target_critic and target_actor
	# target_actor = DeterministicPolicy(salient_states_dim, actions_dim, max_torque).double()
	target_actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
	# target_critic = Value(salient_states_dim, actions_dim).double()
	target_critic = Value(states_dim, actions_dim).double()
	for target_param, param in zip(target_critic.parameters(), critic.parameters()):
		target_param.data.copy_(param.data)

	for target_param, param in zip(target_actor.parameters(), actor.parameters()):
		target_param.data.copy_(param.data)

	if not use_model:
		dataset = ReplayMemory(200000)   

	if model_type != 'random':
		render = True if use_model and P_hat[0].model_size == 'cnn' else False 
	critic_loss = torch.tensor(2)
	policy_loss = torch.tensor(2)
	best_rewards = -np.inf
	val_losses = [100]

	iter_count = 0
	for ep in range(starting_states):
		noise.reset()
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
					action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
					# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()#[:salient_states_dim])).detach().numpy()

					action = noise.get_action(action, timestep)
					get_new_action = False
					action_counter = 1
				
				state_prime, reward, done, _ = env.step(action)
				# env.render()

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
					# actions_next = target_actor.sample_action(states_next[:,:salient_states_dim])
					actions_next = target_actor.sample_action(states_next)
					#Compute target Q value
					# target_Q = target_critic(states_next[:,:salient_states_dim], actions_next)
					target_Q = target_critic(states_next, actions_next)
					target_Q = rewards_tensor + discount * target_Q.detach()
					
					#Compute current Q estimates
					# current_Q = critic(states_prev[:,:salient_states_dim], actions_tensor)
					current_Q = critic(states_prev, actions_tensor)
					critic_loss = MSE(current_Q, target_Q)

					critic_optimizer.zero_grad()
					critic_loss.backward()
					critic_optimizer.step()
					if value_lr_schedule is not None:
						value_lr_schedule.step()

					#compute actor loss
					# policy_loss = -critic(states_prev[:,:salient_states_dim], actor.sample_action(states_prev[:,:salient_states_dim])).mean()
					# policy_loss = -critic(states_prev, actor.sample_action(states_prev[:,:salient_states_dim])).mean()
					# policy_loss = -critic(states_prev[:,:salient_states_dim], actor.sample_action(states_prev)).mean()
					policy_loss = -critic(states_prev, actor.sample_action(states_prev)).mean()#[:,:salient_states_dim])).mean()

					#Optimize the actor
					actor_optimizer.zero_grad()
					policy_loss.backward()
					actor_optimizer.step()

					iter_count += 1
					if (iter_count == 40000) or (iter_count == 80000) or (iter_count == 120000) or (iter_count == 160000):
						torch.save(actor.state_dict(), os.path.join(file_location,'act_{}_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(iter_count, model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))
						torch.save(critic.state_dict(), os.path.join(file_location,'critic_{}_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(iter_count, model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))

					#soft update of the frozen target networks
					for target_param, param in zip(target_critic.parameters(), critic.parameters()):
						target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

					for target_param, param in zip(target_actor.parameters(), actor.parameters()):
						target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)

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

				actions_next = target_actor.sample_action(true_x_next)
				# actions_next = target_actor.sample_action(true_x_next[:,:salient_states_dim])
				# target_q = target_critic(true_x_next[:,:salient_states_dim], actions_next) 
				target_q = target_critic(true_x_next, actions_next)
				target_q = true_r_list + discount * target_q.detach() #detach to avoid backprop target
				# current_q = critic(true_x_curr[:,:salient_states_dim], true_a_list)
				current_q = critic(true_x_curr, true_a_list)

				#update critic only with true data
				critic_optimizer.zero_grad()
				critic_loss = MSE(target_q, current_q)

				#calculate validation loss 
				# val_batch = validation_dataset.sample(len(validation_dataset))
				# val_states_prev = torch.tensor([samp.state for samp in val_batch]).double().to(device)
				# val_states_next = torch.tensor([samp.next_state for samp in val_batch]).double().to(device)
				# val_rewards_tensor = torch.tensor([samp.reward for samp in val_batch]).double().to(device).unsqueeze(1)
				# val_actions_tensor = torch.tensor([samp.action for samp in val_batch]).double().to(device)

				# with torch.no_grad():
				# 	val_actions_next = actor.sample_action(val_states_next[:,:salient_states_dim])
				# 	val_target_q = target_critic(val_states_next, val_actions_next)
				# 	val_y = val_rewards_tensor + discount * val_target_q #detach to avoid backprop target
				# 	val_q = critic(val_states_prev, val_actions_tensor)
				# 	val_losses.append(MSE(val_y, val_q))
				
				# if val_losses[-1] < val_losses[-2]:
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
					s0 = stablenoise.get_obs(s0, 0)
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
				replay_model_x_curr = torch.tensor([replay_model_batch[idx].state * (1 + random_pos_delta[idx]) for idx in range(replay_start_random_model_batch_size)]).double()
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
				actions_noiseless = actor.sample_action(states_).detach().numpy()
				# actions_noiseless = actor.sample_action(states_[:,:salient_states_dim]).detach().numpy() 
				#actions_ = torch.from_numpy(noise.get_action(actions_noiseless, p, multiplier=0.5))
				actions_ = torch.from_numpy(noise.get_action(actions_noiseless, p, multiplier=.5))
				#randomly sample from ensemble to choose next state
				if model_type == 'mle':
					states_prime[:, p, :] = random.sample(P_hat, 1)[0].predict(states_, actions_).detach() #P_hat.predict(states_, actions_).detach()
				else: #ADD STATE
					# states_prime[:, p, :] = random.sample(P_hat, 1)[0].forward(torch.cat((states_, actions_),1)).detach()
					states_prime[:, p, :] = random.sample(P_hat, 1)[0].predict(states_, actions_).detach()

				# states_prime[:, p, :] = P_hat(torch.cat((states_, actions_),1)).detach() #+ states_
				# rewards_model.append(get_reward_fn(env, states_.unsqueeze(1), actions_.unsqueeze(1)))
				states_ = states_prime[:, p, :]

			states_current = torch.cat((states_prev, states_prime[:, :-1, :].contiguous().view(-1, states_dim)), 0)
			actor_optimizer.zero_grad()
			# policy_loss = -critic(states_current[:,:salient_states_dim], actor.sample_action(states_current))
			# policy_loss = -critic(states_current[:,:salient_states_dim], actor.sample_action(states_current[:,:salient_states_dim]))
			# policy_loss = -critic(states_current, actor.sample_action(states_current[:,:salient_states_dim]))
			policy_loss = -critic(states_current, actor.sample_action(states_current))#[:,:salient_states_dim]))
			policy_loss = policy_loss.mean()
			policy_loss.backward()
			actor_optimizer.step()
		
			for target_param, param in zip(target_critic.parameters(), critic.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

			for target_param, param in zip(target_actor.parameters(), actor.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)


		if (ep % verbose  == 0) or (ep == starting_states - 1): #evaluate the policy using no exploration noise
			if not use_model:
				eval_rewards = []
				for ep in range(10):
					state = env.reset()
					if env.spec.id != 'lin-dyn-v0':
						state = stablenoise.get_obs(state, 0)
					episode_rewards = []
					for timestep in range(max_actions):
						action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
						# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()#[:salient_states_dim])).detach().numpy()
						state_prime, reward, done, _ = env.step(action)

						if env.spec.id != 'lin-dyn-v0':
							state_prime = stablenoise.get_obs(state_prime, timestep+1)

						episode_rewards.append(reward)
						state = state_prime
					
					eval_rewards.append(sum(episode_rewards))
				all_rewards.append(sum(eval_rewards)/10.)

				torch.save(actor.state_dict(), os.path.join(file_location,'act_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))
				torch.save(critic.state_dict(), os.path.join(file_location,'critic_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))
				
				print("Ep: {:5d} | Epsilon: {:.5f} | Q_Loss: {:.3f} | Pi_Loss: {:.3f} | Average rewards of 10 independent episodes:{:.4f}".format(ep, epsilon, critic_loss.detach(), policy_loss.detach(), all_rewards[-1]))
			
				#save all_rewards
				np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, use_model, env_name, R_range, max_actions + 1, file_id)), np.asarray(all_rewards))

	return actor, critic

def plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, salient_states_dim, actions_dim, discount, max_actions, env, lr, num_iters, file_location, file_id, save_checkpoints_training, verbose, batch_size, num_virtual_episodes, model_type, num_action_repeats, epsilon, epsilon_decay, planning_horizon, input_rho, trained_Phat_mle,rs):
	# verbose = 20
	# batch_size = 64
	R_range = planning_horizon
	losses = []
	norms_model_pe_grads = []
	norms_true_pe_grads = []

	kwargs = {
				# 'P_hat'              : P_hat,
				'actor'	             : actor, 
				'critic'	 		 : critic,
				#'target_critic'	 	 : target_critic,
				#'q_optimizer'		 : value_optimizer,
				# 'opt' 		 : model_opt, 
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
				'losses'             : None,
				'env'				 : env,
				# 'value_loss_coeff'   : value_loss_coeff,
				'verbose'			 : verbose,
				'file_location'		 : file_location,
				'file_id'			 : file_id,
				'save_checkpoints'	 : save_checkpoints_training,
				'norms_true_pe_grads' : norms_true_pe_grads,
				'norms_model_pe_grads': norms_model_pe_grads
			}

	unroll_num = 1
	# model_type = 'paml'
	log = 1
	ell = 0
	psi = 1.1
	# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	noise = OUNoise(env.action_space)
	kwargs['noise'] = noise
	skipped = -1

	all_rewards = []
	epochs_value = 5000#1000
	#value_optimizer = optim.SGD(critic.parameters(), lr=1e-3, momentum=0.90, nesterov=True) 
	# value_optimizer = optim.SGD(critic.parameters(), lr=1e-5, momentum=0.90, nesterov=True) 
	# value_optimizer = optim.SGD(critic.parameters(), lr=5e-4)#optim.Adam(critic.parameters(), lr=1e-4, weight_decay=1e-8)#optim.SGD(critic.parameters(), lr=1e-4)#, momentum=0.90, nesterov=True) 
	
	# lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[100000,200000,400000], gamma=0.1)
	
	# lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[400000,800000,1000000], gamma=0.1)
	lr_schedule = []
	for m_o in model_opt:
		if model_type == 'paml':
			# lr_schedule.append(torch.optim.lr_scheduler.MultiStepLR(m_o, milestones=[10,1000,20000000], gamma=0.1))
			# lr_schedule.append(torch.optim.lr_scheduler.MultiStepLR(m_o, milestones=[100,500,1000], gamma=0.1))
			lr_schedule.append(torch.optim.lr_scheduler.MultiStepLR(m_o, milestones=[10,500,1000], gamma=0.1))
		elif model_type == 'mle':
			lr_schedule.append(torch.optim.lr_scheduler.MultiStepLR(m_o, milestones=[10,1000,20000000], gamma=0.1)) #ADJUST AS NEEDED

	# dataset = ReplayMemory(max_actions*batch_size*10)
	# validation_dataset = ReplayMemory(max_actions*batch_size*5)
	# dataset = ReplayMemory(200000)
	dataset = ReplayMemory(200000)
	max_torque = float(env.action_space.high[0])
	paml_losses = None
	global_step = 0
	total_eps = 1000 #5000#10000
	env_name = env.spec.id
	true_rewards = []
	use_pixels = False
	fractions_real_data_schedule = np.linspace(1.0,0.5,num=50)

	stablenoise = StableNoise(states_dim, salient_states_dim, 0.992)#, init=np.mean(env.reset()))
	original_batch_size = batch_size
	start_state, start_noise = None, None

	if env_name == 'dm-Walker-v0':
		env2 = dm_control2gym.make(domain_name="walker", task_name="walk")
		env2.spec.id = 'dm-Walker-v0'

	elif env_name == 'Walker2d-v2':
		env2 = gym.make('Walker2d-v2')

	elif env_name == 'HalfCheetah-v2':
		env2 = gym.make('HalfCheetah-v2')
		env2.spec.id = 'HalfCheetah-v2'
	elif env_name == 'Reacher-v2':
		env2 = gym.make('Reacher-v2')
		env2.spec.id = 'Reacher-v2'

	elif env_name == 'dm-Reacher-v0':
		env2 = dm_control2gym.make(domain_name="reacher", task_name="easy")
		env2.spec.id = 'dm-Reacher-v0'
	elif env_name == 'dm-Cartpole-balance-v0':
		env2 = dm_control2gym.make(domain_name="cartpole", task_name="balance")
		env2.spec.id = 'dm-Reacher-Cartpole-balance-v0'

	elif env_name == 'Pendulum-v0':
		env2 = gym.make('Pendulum-v0')
		env2.spec.id = 'Pendulum-v0'
	elif env_name == 'lin-dyn-v0':
		# gym.envs.register(id='lin-dyn-v0', entry_point='gym_linear_dynamics.envs:LinDynEnv',)
		# env = gym.make('gym_linear_dynamics:lin-dyn-v0')
		env2 = gym.make('lin-dyn-v0')
		env2.seed(rs)

	# if model_type == 'paml':
	# L_paml_val = []
	# L_mle_val_relevant = []
	# L_mle_val_irrelevant = []
		# mle_L_paml_val = []
		# paml_L_paml_val = []
		# paml_L_mle_val_relevant = []
		# mle_L_mle_val_relevant = []
		# paml_L_mle_val_irrelevant = []
		# mle_L_mle_val_irrelevant = []
	# validation_dataset = ReplayMemory(2*max_actions)
	num_val_starts = 2
	validation_dataset = ReplayMemory(num_val_starts*max_actions)
	while(global_step <= total_eps):#/num_starting_states):
		# Generating sample trajectories 
		print("Global step: {:4d} | Generating sample trajectories ... epislon is {:.3f}".format(global_step, epsilon))
		# dataset, _, new_epsilon = generate_data(env, dataset, actor, num_starting_states, None, max_actions, noise, epsilon, epsilon_decay, num_action_repeats, discount=discount, all_rewards=[])#true_rewards)
		# dataset, validation_dataset, 
		# if global_step == 0:

		# validation_dataset = ReplayMemory(10*max_actions) #reset each time because ... why not? it makes more sense for the policy, and allows the whole thing to be used in the batch 

		num_steps = min(200,max_actions)
		to_reset = (max_actions <= 200) or (global_step % (max_actions/200) == 0)

		if env_name == 'dm-Walker-v0':
			num_steps = min(250, max_actions)
			to_reset = (max_actions <= 250) or (global_step % (max_actions/250) == 0)

		# num_steps = 1000
		# num_starting_states = 20
		# # if global_step == 0:
		#possible memory issue: passing dataset and validation_dataset back and forth
		start_state, start_noise, new_epsilon = generate_data(env2, states_dim, dataset, validation_dataset, actor, num_starting_states, num_val_starts, num_steps, noise, epsilon, epsilon_decay, num_action_repeats, temp_data=False, discount=discount, all_rewards=[], use_pixels=(P_hat[0].model_size=='cnn'), reset=to_reset, start_state=None if to_reset else start_state, start_noise=None if to_reset else start_noise)#true_rewards)

		batch_size = min(original_batch_size, len(dataset))
		#Evaluate policy without noise
		eval_rewards = []
		# estimate_returns = torch.zeros((10, max_actions)).double()
		# returns = torch.zeros((10, 1)).double()
		# if to_reset:
		for ep in range(10):
			state = env.reset()
			if env.spec.id != 'lin-dyn-v0':
				state = stablenoise.get_obs(state, 0)

			episode_rewards = []
			for timestep in range(max_actions):
				action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
				# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()
				# if ep % 9 == 0:
				# 	env.render()
				state_prime, reward, done, _ = env.step(action)
				# estimate_returns[ep, timestep] = critic(torch.tensor(state[:salient_states_dim]).unsqueeze(0), torch.tensor(action).unsqueeze(0)).squeeze().detach().data.double()
				# estimate_returns[ep, timestep] = critic(torch.tensor(state).unsqueeze(0), torch.tensor(action).unsqueeze(0)).squeeze().detach().data.double()
				del action
				if env.spec.id != 'lin-dyn-v0':
					state_prime = stablenoise.get_obs(state_prime, timestep+1)
				episode_rewards.append(reward)
				state = state_prime

			# returns[ep] = discount_rewards(episode_rewards, discount, center=False, batch_wise=False).detach()[0]
			eval_rewards.append(sum(episode_rewards))

		#possible memory issue: true_rewards array is getting bigger and bigger (but ultimately it should only be 5000 items large at the end)
		true_rewards.append(sum(eval_rewards)/10.)

		print('Global step: {:4d} | Average rewards on true dynamics: {:.3f}'.format(global_step, true_rewards[-1]))
		#already being saved in DDPG function
		# torch.save(actor.state_dict(), os.path.join(file_location,'act_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))

		# np.save(os.path.join(file_location,'{}_state{}_salient{}_true_returns_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}_{}'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, global_step, file_id)), np.asarray(returns))
		# np.save(os.path.join(file_location,'{}_state{}_salient{}_estimate_returns_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}_{}'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, global_step, file_id)), np.asarray(estimate_returns[:,0]))
		np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)), np.asarray(true_rewards))
		# epsilon = new_epsilon
		kwargs['epsilon'] = epsilon
		print(len(dataset))
		# print("Done")

		# epochs_value = int(np.ceil(epochs_value * psi))

		# if model_type != 'mle':
		# if timestep % 1000 == 0 and ep % 10 == 0:
			# critic.reset_weights()


		# if to_reset and env != 'lin-dyn-v0':# and model_type == 'paml':
		# 	schedule_data = 10
		# 	generate_data(env, states_dim, dataset, None, actor, schedule_data, None, max_actions, noise, epsilon, epsilon_decay, num_action_repeats, temp_data=True, discount=discount, all_rewards=[], use_pixels=(P_hat[0].model_size=='cnn'), reset=True, start_state=None, start_noise=None)#true_rewards)
		


		# critic = pre_train_critic(actor, critic, critic_dataset, None, epochs_value, discount, batch_size, value_optimizer, value_lr_schedule, states_dim, actions_dim, salient_states_dim, file_location, file_id, model_type, env_name, max_actions, verbose=100)
		# if model_type != 'mle':
			# sum1 = summary.summarize(muppy.get_objects())
		critic.pre_train(actor, dataset, epochs_value, discount, batch_size, salient_states_dim, file_location, file_id, model_type, env_name, max_actions, verbose=100)
			# sum2 = summary.summarize(muppy.get_objects())
			# summary.print_(summary.get_diff(sum1,sum2))

		if to_reset:
			dataset.clear_temp()

		kwargs['critic'] = critic
		# if env_name == 'lin-dyn-v0': #only do ensemble size 1 for lin-dyn
		# 	kwargs['opt'] = model_opt[0]
		# 	kwargs['train'] = False
		# 	kwargs['num_episodes'] = num_episodes
		# 	kwargs['num_starting_states'] = num_starting_states
		# 	kwargs['losses'] = []
		# 	kwargs['use_model'] = False
		# 	kwargs['dataset'] = dataset
		# 	kwargs['lr_schedule'] = lr_schedule[0]
		# 	P_hat[0].actor_critic_paml_train(**kwargs)

		kwargs['train'] = True
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['num_iters'] = verbose
		kwargs['use_model'] = True

		if to_reset or env_name == 'dm-Walker-v0':
			for P_hat_idx in range(len(P_hat)):
				kwargs['losses'] = None#paml_losses #losses not being recorded, may want to change this at some point
				if model_type != 'random':
					kwargs['opt'] = model_opt[P_hat_idx]
				# kwargs['P_hat'] = P_hat

				if model_type =='mle':
					# P_hat.general_train_mle(actor, dataset, states_dim, salient_states_dim, num_iters, max_actions, model_opt, env_name, losses, batch_size, file_location, file_id, save_checkpoints=save_checkpoints_training, verbose=20, lr_schedule=lr_schedule)
					P_hat[P_hat_idx].general_train_mle(actor, dataset, validation_dataset, states_dim, salient_states_dim, num_iters, max_actions, model_opt[P_hat_idx], env_name, losses, batch_size, file_location, file_id, save_checkpoints=True, verbose=verbose, lr_schedule=lr_schedule[P_hat_idx], global_step=global_step)
				elif (model_type == 'paml') or (model_type == 'pamlmean'):
					# if (global_step > 0) and (global_step % 15 == 0): #for lqr: 8 good
					# 	lr = lr / 1.01
					# kwargs['lr'] = lr
					val_losses = [100,99]
					def non_increasing(L):
						return all(x>=(y + y*0.05) for x, y in zip(L, L[1:]))
					# while False:
					while val_losses[-1] < (val_losses[-2] + val_losses[-2] * 0.0):#(val_losses[1] + val_losses[1]*0.001)): #non_increasing(val_losses)):#
					# while (val_losses[-1] <= val_losses[-2]):
						kwargs['train'] = True
						kwargs['dataset'] = dataset
						kwargs['num_iters'] = num_iters#verbose
						kwargs['batch_size'] = batch_size
						kwargs['lr_schedule'] = lr_schedule[P_hat_idx]
						P_hat[P_hat_idx].actor_critic_paml_train(**kwargs)

						kwargs['dataset'] = validation_dataset
						kwargs['train'] = False
						kwargs['num_iters'] = 1
						kwargs['batch_size'] = len(validation_dataset)
						val_losses.append(P_hat[P_hat_idx].actor_critic_paml_train(**kwargs))
						val_losses = val_losses[1:]
					del val_losses
				elif model_type == 'random': #not fixed for the ensemble case
					# P_hat = [DirectEnvModel(states_dim, actions_dim, max_torque).double()]
					pass				# kwargs['P_hat'] = P_hat
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

		# check L_paml(mle) and L_paml(paml) on validation data before training policy?
		# if model_type == 'paml':
		kwargs['dataset'] = validation_dataset
		kwargs['train'] = False
		kwargs['num_iters'] = 1
		kwargs['batch_size'] = len(validation_dataset)
		kwargs['losses'] = None
		if model_type != 'random':
			kwargs['lr_schedule'] = lr_schedule[P_hat_idx]
		# mle_L_paml_val.append(trained_Phat_mle.actor_critic_paml_train(**kwargs))
		# paml_L_paml_val.append(P_hat[0].actor_critic_paml_train(**kwargs))
		
		# L_paml_val.append(P_hat[0].actor_critic_paml_train(**kwargs))

		#Check L_mle(mle) and L_mle(paml) on validation data
		# if model_type == 'paml':
		with torch.no_grad():
			val_batch = validation_dataset.sample(len(validation_dataset))
			val_x_curr = torch.tensor([samp.state for samp in val_batch]).double().to(device)
			val_x_next = torch.tensor([samp.next_state for samp in val_batch]).double().to(device)
			val_a_list = torch.tensor([samp.action for samp in val_batch]).double().to(device)
			# paml_states_prime = random.sample(P_hat, 1)[0].predict(val_x_curr, val_a_list)
			# mle_states_prime = trained_Phat_mle.predict(val_x_curr, val_a_list)
			val_model_mle_states_prime = random.sample(P_hat, 1)[0].predict(val_x_curr, val_a_list)

		# paml_L_mle_val_relevant.append(torch.mean(torch.sum((paml_states_prime[:,:salient_states_dim] - val_x_next[:,:salient_states_dim])**2,dim=1)).data)
		# mle_L_mle_val_relevant.append(torch.mean(torch.sum((mle_states_prime[:,:salient_states_dim] - val_x_next[:,:salient_states_dim])**2,dim=1)).data)

		# paml_L_mle_val_irrelevant.append(torch.mean(torch.sum((paml_states_prime[:,salient_states_dim:] - val_x_next[:,salient_states_dim:])**2,dim=1)).data)
		# mle_L_mle_val_irrelevant.append(torch.mean(torch.sum((mle_states_prime[:,salient_states_dim:] - val_x_next[:,salient_states_dim:])**2,dim=1)).data)
		
		# L_mle_val_relevant.append(torch.mean(torch.sum((val_model_mle_states_prime[:,:salient_states_dim] - val_x_next[:,:salient_states_dim])**2,dim=1)).data)
		# L_mle_val_irrelevant.append(torch.mean(torch.sum((val_model_mle_states_prime[:,salient_states_dim:] - val_x_next[:,salient_states_dim:])**2,dim=1)).data)

		#num_virtual_episodes * (2*(skipped==global_step) + 1*(skipped!=global_step))
		all_rewards = None
		value_lr_schedule = None
		#possible memory issue: passing datasets 
		actor, critic = actor_critic_DDPG(env, actor, noise,critic, dataset, validation_dataset, batch_size, num_virtual_episodes * (2*(skipped==global_step) + 1*(skipped!=global_step)), max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, epsilon_decay, value_lr_schedule, file_location, file_id, num_action_repeats, planning_horizon=planning_horizon, P_hat=P_hat, model_type=model_type, save_checkpoints=save_checkpoints_training, rho_ER=input_rho)
		#check paml_loss with the new policy
		# kwargs['train'] = False
		# kwargs['use_model'] = True
		# kwargs['losses'] = []
		# kwargs['P_hat'] = P_hat
		# kwargs['actor'] = actor
		# kwargs['critic'] = critic
		# _, loss_paml = actor_critic_paml_train(**kwargs)
		# paml_losses.append(loss_paml)

		# if global_step % log == 0:
		# if not paml_losses == []:
			# np.save(os.path.join(file_location,'ac_pamlloss_model_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(model_type, env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(paml_losses))
		

		# if not losses == []:
		# 	np.save(os.path.join(file_location,'ac_loss_model_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(model_type, env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(losses))

		# if model_type == 'paml':
		# np.save(os.path.join(file_location,'L_paml_val_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(model_type, env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(L_paml_val))
		# np.save(os.path.join(file_location,'L_mle_val_rel_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(model_type, env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(L_mle_val_relevant))
		# np.save(os.path.join(file_location,'L_mle_val_irrel_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(model_type, env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(L_mle_val_irrelevant))

			# np.save(os.path.join(file_location,'mle_L_paml_val_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(mle_L_paml_val))
			# np.save(os.path.join(file_location,'paml_L_paml_val_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(paml_L_paml_val))
			# np.save(os.path.join(file_location,'mle_L_mle_val_rel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(mle_L_mle_val_relevant))
			# np.save(os.path.join(file_location,'paml_L_mle_val_rel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(paml_L_mle_val_relevant))
			# np.save(os.path.join(file_location,'mle_L_mle_val_irrel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(mle_L_mle_val_irrelevant))
			# np.save(os.path.join(file_location,'paml_L_mle_val_irrel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'.format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)), np.asarray(paml_L_mle_val_irrelevant))

		global_step += 1
		# for opt in model_opt:
		# 	opt.state = collections.defaultdict(dict)
		# all_rewards.append(sum(rewards))

		# del validation_dataset
		del eval_rewards, episode_rewards
	
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
		env = gym.make('Reacher-v2')

	elif env_name == 'Thrower-v2':
		env = gym.make('Thrower-v2')
		pdb.set_trace()
	elif env_name == 'Hopper-v2':
		env = gym.make('Hopper-v2')

	elif env_name == 'Acrobot-v1':
		env = gym.make('Acrobot-v1')

	elif env_name == 'MountainCarContinuous-v0':
		env = gym.make('MountainCarContinuous-v0') #NormalizedEnv(gym.make('Swimmer-v2'))

	elif env_name == 'HalfCheetah-v2':	
		env = gym.make('HalfCheetah-v2') #NormalizedEnv(gym.make('HalfCheetah-v2'))

	elif env_name == 'Walker2d-v2':
		env = gym.make('Walker2d-v2')
	
	elif env_name == 'dm-Cartpole-swingup-v0':
		env = dm_control2gym.make(domain_name="cartpole", task_name="swingup")
		env.spec.id = 'dm-Cartpole-swingup-v0'

	elif env_name == 'dm-Cartpole-balance-v0':
		env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
		env.spec.id = 'dm-Cartpole-balance-v0'

	elif env_name == 'dm-Acrobot-v0':
		env = dm_control2gym.make(domain_name="acrobot", task_name="swingup")
		env.spec.id = 'dm-Acrobot-v0'

	elif env_name == 'dm-Reacher-v0':
		env = dm_control2gym.make(domain_name="reacher", task_name="easy")
		env.spec.id = 'dm-Reacher-v0'

	elif env_name == 'dm-Cheetah-v0':
		env = dm_control2gym.make(domain_name="cheetah", task_name="run")
		env.spec.id = 'dm-Cheetah-v0'

	elif env_name == 'dm-Walker-v0':
		env = dm_control2gym.make(domain_name="walker", task_name="walk")
		env.spec.id = 'dm-Walker-v0'

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
	num_starting_states = real_episodes if not plan else 5000#10000
	num_episodes = num_eps_per_start #1
	#batch_size = 64
	val_num_episodes = 1
	val_num_starting_states = 10
	val_batch_size = val_num_starting_states*val_num_episodes
	
	#num_iters = 400
	losses = None
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
		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		# actor = DeterministicPolicy(salient_states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim).double()
		actor_critic_DDPG(env, actor, noise, critic, None,None, batch_size, num_starting_states, max_actions, states_dim, salient_states_dim, discount, False, True, verbose, all_rewards, epsilon, epsilon_decay, None, file_location, file_id, num_action_repeats, P_hat=None, model_type='model_free', save_checkpoints=save_checkpoints_training)

		np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, use_model, env_name, R_range, max_actions + 1, file_id)), np.asarray(all_rewards))
		# np.save('actorcritic_pendulum_rewards',np.asarray(all_rewards)) 
		# pdb.set_trace()
	else:
		P_hat = []
		model_opt = []
		for ens in range(ensemble_size):
			model_ens = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size, hidden_size=hidden_size).double()
			P_hat.append(model_ens)			
			if model_type == 'paml':
				model_opt.append(optim.SGD(model_ens.parameters(), lr=initial_model_lr))#, momentum=0.9))#, weight_decay=1e-2))
			elif model_type == 'mle':
				model_opt.append(optim.Adam(model_ens.parameters(), lr=initial_model_lr))

		# if model_type == 'paml' and env_name == 'Pendulum-v0':
		# 	trained_Phat_mle = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size, hidden_size=hidden_size).double()
		# 	# trained_Phat_mle.load_state_dict(torch.load(os.path.join(file_location, 'model_mle_checkpoint_state30_salient3_actorcritic_Pendulum-v0_horizon1_traj201_constrainedModel_{}.pth'.format(rs)), map_location=device))
		# elif model_type == 'paml' and env_name == 'lin_dyn':
		# 	trained_Phat_mle = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size, hidden_size=hidden_size).double()
		# 	# trained_Phat_mle.load_state_dict(torch.load(os.path.join(file_location, 'model_mle_checkpoint_state30_salient2_actorcritic_lin-dyn-v0_horizon1_traj201_hidden4_criticrestrict_actorfull_constrainedModel_{}.pth'.format(rs)), map_location=device))
		# 	# trained_Phat_mle.load_state_dict(torch.load(os.path.join(file_location, 'model_mle_checkpoint_state30_salient2_actorcritic_lin-dyn-v0_horizon1_traj201_rho0.25_hidden1_criticfull_actorfull_constrainedModel_{}.pth'.format(2)), map_location=device))
		# else:
		trained_Phat_mle = None



		# P_hat = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size, hidden_size=hidden_size)
		# P_hat.double()

		# P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon6_traj7.pth', map_location=device))
		#P_hat.load_state_dict(torch.load('act_model_paml_checkpoint_train_True_lin_dyn_horizon5_traj6_using1states.pth', map_location=device))

		# value_optimizer = optim.SGD(critic.parameters(), lr=1e-5, momentum=0.90, nesterov=True) 
		# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_optimizer, milestones=[1500,3000], gamma=0.1)

		# #1e-4
		# if model_type == 'paml':
		# 	model_opt = optim.SGD(P_hat.parameters(), lr=initial_model_lr)#, momentum=0.90, nesterov=True)
		# elif model_type == 'mle':
		# 	model_opt = optim.SGD(P_hat.parameters(), lr=initial_model_lr)

			# model_opt = optim.Adam(P_hat.parameters(), lr=initial_model_lr, weight_decay=1e-2)
		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		# actor = DeterministicPolicy(salient_states_dim, actions_dim, max_torque).double()
		# actor.load_state_dict(torch.load(os.path.join(file_location, 'act_policy_paml_state20_salient5_checkpoint_dm-Cartpole-balance-v0_horizon1_traj201_nnModel_{}.pth'.format(rs)),map_location=device))
		# actor.load_state_dict(torch.load(os.path.join(file_location, 'act_policy_model_free_state24_salient24_checkpoint_dm-Walker-v0_horizon1_traj1001_0.pth'), map_location=device))
		# actor.load_state_dict(torch.load(os.path.join(file_location, 'act_policy_model_free_state17_salient17_checkpoint_HalfCheetah-v2_horizon1_traj1001_0.pth'), map_location=device))
		
		# actor.load_state_dict(torch.load(os.path.join(file_location, 'act_policy_model_free_state3_salient3_checkpoint_Pendulum-v0_horizon1_traj201_0.pth')))
		
		# critic = Value(salient_states_dim, actions_dim).double()
		critic = Value(states_dim, actions_dim, pretrain_val_lr=1e-4, pretrain_value_lr_schedule=None).double()
		
		def load_checkpoints():
			for ens in P_hat:
				ens_cp_filename = os.path.join(file_location,'model_{}_checkpoint_state{}_salient{}_actorcritic_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, planning_horizon, max_actions+1, file_id))
				if os.path.exists(ens_cp_filename):
					ens.load_state_dict(torch.load(ens_cp_filename, map_location=device))
			
			if os.path.exists(ens_cp_filename): #only load actor and critic if model already existed, if more than 1 file in ensemble will have to make this more robust
				actor_cp_filename = os.path.join(file_location,'act_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, planning_horizon, max_actions + 1, file_id))
				if os.path.exists(actor_cp_filename):
					actor.load_state_dict(torch.load(actor_cp_filename, map_location=device))

				critic_cp_filename = os.path.join(file_location,'critic_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(model_type, states_dim, salient_states_dim, env_name, planning_horizon, max_actions + 1, file_id))
				if os.path.exists(critic_cp_filename):
					critic.load_state_dict(torch.load(critic_cp_filename, map_location=device))

		# load_checkpoints()

		# critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_policy_paml_state20_salient5_checkpoint_dm-Cartpole-balance-v0_traj201_critic0eps1000steps_nnModel_{}.pth'.format(rs)), map_location=device))

		# critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_40000_policy_model_free_state24_salient24_checkpoint_dm-Walker-v0_horizon1_traj1001_nnModel_1.pth')))
		# critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_policy_model_free_state3_salient3_checkpoint_Pendulum-v0_horizon1_traj201_0.pth')))
		
		

		# critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_policy_mle_state30_salient3_checkpoint_Pendulum-v0_traj201_constrainedModel_{}.pth'.format(rs)), map_location=device))
		



		# critic.load_state_dict(torch.load(os.path.join(file_location, 'critic_policy_model_free_state17_salient17_checkpoint_HalfCheetah-v2_horizon1_traj1001_0.pth'), map_location=device))

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
		plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, salient_states_dim,actions_dim, discount, max_actions, env, initial_model_lr, num_iters, file_location, file_id, save_checkpoints_training, verbose, batch_size, virtual_episodes, model_type, num_action_repeats, epsilon, epsilon_decay, planning_horizon, input_rho, trained_Phat_mle,rs)





