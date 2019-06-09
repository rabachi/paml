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

# import dm_control2gym
device = 'cpu'

def pre_train_critic(actor, critic, dataset, epochs_value, discount, batch_size, q_optimizer, value_lr_schedule, max_actions, states_dim, actions_dim, verbose=10):
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

		actions_next = actor.sample_action(states_next)
		target_q = target_critic(states_next, actions_next)
		y = rewards_tensor + discount * target_q.detach() #detach to avoid backprop target
		q = critic(states_prev, actions_tensor)

		q_optimizer.zero_grad()
		loss = MSE(y, q)
		loss.backward()
		nn.utils.clip_grad_value_(critic.parameters(), 20.0)
		q_optimizer.step()
		value_lr_schedule.step()

		#soft update the target critic
		for target_param, param in zip(target_critic.parameters(), critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

		if i % verbose == 0:
			print('Epoch: {:4d} | Value estimator loss: {:.5f}'.format(i,loss.detach().cpu()))

	return critic #finish when target has converged

def pre_train_critic_2(actor, critic, dataset, validation_dataset, epochs_value, discount, batch_size, q_optimizer, value_lr_schedule, max_actions, verbose):
	best_loss = 10000
	MSE = nn.MSELoss()
	TAU=0.0001
	R_range = 1

	states_dim = 3
	actions_dim = 1
	target_critic = Value(states_dim, actions_dim)
	target_critic.double()

	for target_param, param in zip(target_critic.parameters(), critic.parameters()):
		target_param.data.copy_(param.data)

	for i in range(epochs_value):
		batch = dataset.sample(batch_size)

		states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
		states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
		#target_q = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
		rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
		actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

		#compute loss for critic
		# actions_next = target_actor.sample_action(states_next)
		# target_q = target_critic(torch.cat((states_next, actions_next), dim=1))
		actions_next = actor.sample_action(states_next)
		target_q = target_critic(torch.cat((states_next, actions_next), dim=1))
		y = rewards_tensor + discount * target_q.detach() #detach to avoid backprop target
		q = critic(torch.cat((states_prev, actions_tensor),dim=1))

		q_optimizer.zero_grad()
		q_loss = MSE(q, y)
		q_loss.backward()
		q_optimizer.step()
		value_lr_schedule.step()

		# save_stats(true_returns, true_r_list, true_a_list, true_x_curr, value=true_value, prefix='true_actorcritic_')
		if i % verbose == 0:
			print('Epoch: {:4d} | Value estimator loss: {:.5f}'.format(i,q_loss.detach().cpu()))

		for target_param, param in zip(target_critic.parameters(), critic.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

		if q_loss < best_loss:
			torch.save(critic.state_dict(), 'critic_horizon{}_traj{}.pth'.format(R_range, max_actions+1))
			best_loss = q_loss.detach()

		if (i % verbose*3) == 0 and validation_dataset is not None: #check performance on validation data
			with torch.no_grad():
				val_batch = validation_dataset.sample(batch_size)

				states_prev = torch.tensor([samp.state for samp in val_batch]).double().to(device)
				states_next = torch.tensor([samp.next_state for samp in val_batch]).double().to(device)
				#target_q = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
				rewards_tensor = torch.tensor([samp.reward for samp in val_batch]).double().to(device).unsqueeze(1)
				actions_tensor = torch.tensor([samp.action for samp in val_batch]).double().to(device)

				#compute loss for critic
				# actions_next = target_actor.sample_action(states_next)
				# target_q = target_critic(torch.cat((states_next, actions_next), dim=1))
				actions_next = actor.sample_action(states_next)
				target_q = target_critic(torch.cat((states_next, actions_next), dim=1))
				y = rewards_tensor + discount * target_q.detach() #detach to avoid backprop target
				q = critic(torch.cat((states_prev, actions_tensor),dim=1))

				val_q_loss = MSE(q, y)
				print('Validation value estimator loss: {:.5f}'.format(val_q_loss.cpu()))

	with torch.no_grad():
		batch = dataset.sample(batch_size)

		states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
		states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
		#target_q = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
		rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
		actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

		#compute loss for critic
		# actions_next = target_actor.sample_action(states_next)
		# target_q = target_critic(torch.cat((states_next, actions_next), dim=1))
		actions_next = actor.sample_action(states_next)
		target_q = target_critic(torch.cat((states_next, actions_next), dim=1))
		y = rewards_tensor + discount * target_q.detach() #detach to avoid backprop target
		q = critic(torch.cat((states_prev, actions_tensor),dim=1))

		q_loss = nn.MSELoss()(q, y)
		print('inside function: ', q_loss.numpy())

	return target_critic


def actor_critic_paml_train(
							P_hat, 
							actor, 
							critic,
							env,
							noise,
							# target_critic,
							# q_optimizer,
							opt, 
							num_episodes, 
							num_starting_states, 
							batch_size,
							states_dim, 
							salient_states_dim,
							actions_dim, 
							use_model, 
							discount, 
							max_actions, 
							train, 
							lr,
							num_iters,
							losses,
							dataset,
							verbose,
							save_checkpoints,
							file_location,
							file_id
							):
	best_loss = 15000
	env_name = env.spec.id
	# batch_size = 64
	R_range = 1
	unroll_num = max_actions
	end_of_trajectory = 1
	num_iters = num_iters if train else 1
	MSE = nn.MSELoss()
	noise.reset()
	# noise = OUNoise(env.action_space)
	# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))	
	model_opt = optim.SGD(P_hat.parameters(), lr=lr)#, momentum=0.90, nesterov=True)

	epsilon = 1
	epsilon_decay = 1./100000
	ell = 0

	for i in range(num_iters):
		batch = dataset.sample(batch_size)

		true_states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
		true_states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
		true_rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
		true_actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

		#calculate true gradients
		actor.zero_grad()

		#compute loss for actor
		true_policy_loss = -critic(true_states_next, actor.sample_action(true_states_next))
		true_term = true_policy_loss.mean()

		# save_stats(None, true_rewards_tensor, None, true_states_next, value=None, prefix='true_actorcritic_')
		true_pe_grads = torch.DoubleTensor()
		true_pe_grads_attached = grad(true_term.mean(), actor.parameters(), create_graph=True)
		for g in range(len(true_pe_grads_attached)):
			true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

		#probably don't need these ... .grad is not accumulated with the grad() function
		actor.zero_grad()
		step_state = torch.cat((true_states_prev, true_actions_tensor),dim=1).unsqueeze(1)
	
		if use_model:
			model_x_curr, model_x_next, model_a_list, model_r_list, _ = P_hat.unroll(step_state, actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=salient_states_dim, noise=noise, epsilon=epsilon, epsilon_decay=epsilon_decay, env=env)

			#can remove these after unroll is fixed
			model_x_curr = model_x_curr.squeeze(1).squeeze(1)
			model_x_next = model_x_next.squeeze(1).squeeze(1)
			model_r_list = model_r_list.squeeze()[:,1:]
			model_a_list = model_a_list.squeeze(1).squeeze(1)
			##### DO ACTIONS ABOVE MESS WITH P_HAT GRADIENTS?!
		else:
			model_batch = dataset.sample(batch_size)
			model_x_curr = torch.tensor([samp.state for samp in model_batch]).double().to(device)
			model_x_next = torch.tensor([samp.next_state for samp in model_batch]).double().to(device)
			model_r_list = torch.tensor([samp.reward for samp in model_batch]).double().to(device).unsqueeze(1)
			model_a_list = torch.tensor([samp.action for samp in model_batch]).double().to(device)

		model_policy_loss = -critic(model_x_next, actor.sample_action(model_x_next))
		model_term = model_policy_loss.mean()
		# save_stats(None, model_r_list, model_a_list, model_x_next, value=None, prefix='model_actorcritic_')
		model_pe_grads = torch.DoubleTensor()
		model_pe_grads_split = grad(model_term.mean(),actor.parameters(), create_graph=True)
		for g in range(len(model_pe_grads_split)):
			model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))
		# cos = nn.CosineSimilarity(dim=0, eps=1e-6)
		# loss = 1-cos(true_pe_grads,model_pe_grads)
		loss = torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum())#/num_starting_states)


		if loss.detach().cpu() < best_loss and use_model:
			#Save model and losses so far
			#if save_checkpoints:
			torch.save(P_hat.state_dict(), os.path.join(file_location,'act_model_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)))
			if save_checkpoints:
				np.save(os.path.join(file_location,'act_loss_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), np.asarray(losses))

			best_loss = loss.detach().cpu()

		#update model
		if train: 
			opt.zero_grad()

			loss.backward()
			nn.utils.clip_grad_value_(P_hat.parameters(), 5.0)
			opt.step()

			if torch.isnan(torch.sum(P_hat.fc1.weight.data)):
				print('weight turned to nan, check gradients')
				pdb.set_trace()

		losses.append(loss.data.cpu())
		# lr_schedule.step()

		if train and i < 1: #and j < 1 
			initial_loss = losses[0]#.data.cpu()
			print('initial_loss',initial_loss)

		if train:
			if (i % verbose == 0) or (i == num_iters - 1):
				print("LR: {:.5f} | batch_num: {:5d} | critic ex val: {:.3f} | paml_loss: {:.5f}".format(lr, i, true_policy_loss.mean().data.cpu(), loss.data.cpu()))
		else: 
			print("-----------------------------------------------------------------------------------------------------")
			print("Validation loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}".format(use_model, R_range, i, loss.data.cpu()))
			print("-----------------------------------------------------------------------------------------------------")
	
	if train:
		P_hat.load_state_dict(torch.load(os.path.join(file_location,'act_model_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'.format(states_dim, salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), map_location=device))

	return P_hat, loss.data.cpu()


def actor_critic_DDPG(env, actor, noise, critic, real_dataset, batch_size, num_starting_states, max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, value_lr_schedule, file_location, file_id, num_action_repeats, P_hat=None, model_type='paml'):
 
	starting_states = num_starting_states
	env_name = env.spec.id
	max_torque = float(env.action_space.high[0])

	#rho = fraction of data used from experience replay
	rho_ER = 0.5

	random_start_frac = 0.5 #will also use search control that searches near previously seen states
	# random_around_ER_frac = 0.8
	# from_ER_frac = 1.0 - random_start_frac - random_around_ER_frac
	radius = 0.8

	actions_dim = env.action_space.shape[0]

	R_range = 1

	TAU=0.001      #Target Network HyperParameters
	LRA=0.0001      #LEARNING RATE ACTOR
	LRC=0.001       #LEARNING RATE CRITIC

	# LRA=1e-6      #LEARNING RATE ACTOR
	# LRC=1e-5       #LEARNING RATE CRITIC

	buffer_start = 100
	# epsilon = 1
	epsilon_original = epsilon
	epsilon_decay = 1./100000

	# noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	max_torque = float(env.action_space.high[0])

	# batch_size = 64
	best_loss = 10

	MSE = nn.MSELoss()
	# actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
	# critic = Value(states_dim, actions_dim).double()
	critic_optimizer  = optim.Adam(critic.parameters(), lr=LRC)
	if value_lr_schedule is None:
		value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(critic_optimizer, milestones=[1500,3000], gamma=0.1)
	actor_optimizer = optim.Adam(actor.parameters(), lr=LRA)

	#initialize target_critic and target_actor
	target_actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
	target_critic = Value(states_dim, actions_dim).double()

	for target_param, param in zip(target_critic.parameters(), critic.parameters()):
		target_param.data.copy_(param.data)

	for target_param, param in zip(target_actor.parameters(), actor.parameters()):
		target_param.data.copy_(param.data)

	dataset = ReplayMemory(1000000)
	render = False
	critic_loss = torch.tensor(2)
	policy_loss = torch.tensor(2)
	for ep in range(starting_states):
		noise.reset()
		# epsilon -= epsilon_decay
		# epsilon = epsilon_original
		if not use_model:
			state = env.reset()
			states = [state]
			actions = []
			rewards = []
			get_new_action = True
			for timestep in range(max_actions):
				if get_new_action:
					#epsilon -= epsilon_decay
					action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
					#action += noise()*max(0, epsilon) #try without noise
					#action = np.clip(action, -1., 1.)
					action = noise.get_action(action, timestep)
					get_new_action = False
					action_counter = 1

				state_prime, reward, done, _ = env.step(action)
				if render:
					env.render()

				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)
				
				dataset.push(state, state_prime, action, reward)
				state = state_prime

				get_new_action = True if action_counter == num_action_repeats else False
				action_counter += 1

				if len(dataset) > batch_size:
					batch = dataset.sample(batch_size)

					states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
					states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
					rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
					actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

					actions_next = target_actor.sample_action(states_next)
					
					#Compute target Q value
					target_Q = target_critic(states_next, actions_next)
					target_Q = rewards_tensor + discount * target_Q.detach()
					
					#Compute current Q estimates
					current_Q = critic(states_prev, actions_tensor)
					critic_loss = MSE(current_Q, target_Q)

					critic_optimizer.zero_grad()
					critic_loss.backward()
					critic_optimizer.step()
					# value_lr_schedule.step()

					#compute actor loss
					policy_loss = -critic(states_prev, actor.sample_action(states_prev)).mean()

					#Optimize the actor
					actor_optimizer.zero_grad()
					policy_loss.backward()
					actor_optimizer.step()

					#soft update of the frozen target networks
					for target_param, param in zip(target_critic.parameters(), critic.parameters()):
						target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

					for target_param, param in zip(target_actor.parameters(), actor.parameters()):
						target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)
			all_rewards.append(sum(rewards))

		else:
			if P_hat is None:
				raise NotImplementedError

			true_batch_size = int(np.floor(rho_ER * batch_size))
			#use real_dataset here
			unroll_num = 1
			ell = 0
			R_range = 1

			true_batch = real_dataset.sample(true_batch_size)
			true_x_curr = torch.tensor([samp.state for samp in true_batch]).double().to(device)
			true_x_next = torch.tensor([samp.next_state for samp in true_batch]).double().to(device)
			true_a_list = torch.tensor([samp.action for samp in true_batch]).double().to(device)
			true_r_list = torch.tensor([samp.reward for samp in true_batch]).double().to(device).unsqueeze(1)

			actions_next = target_actor.sample_action(true_x_next)
			target_q = target_critic(true_x_next, actions_next)
			target_q = true_r_list + discount * target_q.detach() #detach to avoid backprop target
			current_q = critic(true_x_curr, true_a_list)

			#update critic only with true data
			critic_optimizer.zero_grad()
			critic_loss = MSE(target_q, current_q)
			critic_loss.backward()
			critic_optimizer.step()
			value_lr_schedule.step()

			model_batch_size = batch_size - true_batch_size
			random_start_model_batch_size = int(np.floor(random_start_frac * model_batch_size)) #randomly sample from state space

			#This is kind of slow, but it would work with any environment, the other way it to do the reset batch_wise in a custom function made for every environment separately ... but that seems like it shouldn't be done
			random_model_x_curr = torch.zeros((random_start_model_batch_size, states_dim)).double()
			for samp in range(random_start_model_batch_size):
				s0 = env.reset()
				random_model_x_curr[samp] = torch.from_numpy(s0).double()

			with torch.no_grad():

				#some starting points chosen completely randomly
				# random_model_actions = torch.clamp(actor.sample_action(random_model_x_curr) + torch.from_numpy(noise()*max(0, epsilon)), min=-max_torque, max=max_torque)
				random_model_actions = actor.sample_action(random_model_x_curr)
				# random_model_actions = noise.get_action(random_model_actions, timestep)

				random_model_x_next = P_hat(torch.cat((random_model_x_curr, random_model_actions),1))
				random_model_r_list = get_reward_fn(env, random_model_x_curr.unsqueeze(1), random_model_actions.unsqueeze(1))

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
				random_pos_delta = 2*(np.random.random(size = (replay_start_random_model_batch_size, true_x_curr.shape[1])) - 0.5) * radius 

				replay_model_x_curr = torch.tensor([replay_model_batch[idx].state + random_pos_delta[idx, :] for idx in range(len(replay_model_batch))]).double().to(device)
				replay_model_actions = actor.sample_action(replay_model_x_curr) #torch.clamp(actor.sample_action(replay_model_x_curr) + torch.from_numpy(noise()*max(0, epsilon)), min=-max_torque, max=max_torque)
				replay_model_x_next = P_hat(torch.cat((replay_model_x_curr, replay_model_actions),1))
				replay_model_r_list = get_reward_fn(env, replay_model_x_curr.unsqueeze(1), replay_model_actions.unsqueeze(1))

			#check dims
			states_prev = torch.cat((true_x_curr, random_model_x_curr, replay_model_x_curr), 0)
			states_next = torch.cat((true_x_next, random_model_x_next, replay_model_x_next), 0)
			actions_list = torch.cat((true_a_list, random_model_actions, replay_model_actions), 0)
			rewards_list = torch.cat((true_r_list, random_model_r_list, replay_model_r_list), 0)

			all_rewards.append(rewards_list.sum())

			#save_stats(None, model_r_list_, model_a_list_, model_x_curr_, prefix='model_planning_')

			#save_stats(None, true_r_list, true_a_list, true_x_curr, prefix='true_planning_')

			#compute loss for actor
			actor_optimizer.zero_grad()
			policy_loss = -critic(states_prev, actor.sample_action(states_prev))
			policy_loss = policy_loss.mean()
			policy_loss.backward()
			actor_optimizer.step()

			#soft update of the frozen target networks
			for target_param, param in zip(target_critic.parameters(), critic.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

			for target_param, param in zip(target_actor.parameters(), actor.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)

		if (ep % verbose  == 0) or (ep == starting_states - 1):
			print("Ep: {:5d} | Q_Loss: {:.3f} | Pi_Loss: {:.3f} | Average of last 10 rewards:{:.4f}".format(ep, critic_loss.detach(), policy_loss.detach(), sum(all_rewards[-10:])/len(all_rewards[-10:])))
			#save all_rewards
			if not use_model:
				np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, use_model, env_name, R_range, max_actions + 1, file_id)), np.asarray(all_rewards))

		if sum(all_rewards[-10:])/len(all_rewards[-10:]) > -700:
			render = False

	return actor, critic


def plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, salient_states_dim, actions_dim, discount, max_actions, env, lr, num_iters, file_location, file_id, save_checkpoints_training, verbose, batch_size, num_virtual_episodes, model_type, num_action_repeats):
	# verbose = 20
	# batch_size = 64
	R_range = 1
	losses = []

	kwargs = {
				'P_hat'              : P_hat, 
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
	epsilon = 1.
	epsilon_decay = 1./100000
	#noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	noise = OUNoise(env.action_space)
	kwargs['noise'] = noise

	all_rewards = []
	epochs_value = 100
	#value_optimizer = optim.SGD(critic.parameters(), lr=1e-3, momentum=0.90, nesterov=True) 
	# value_optimizer = optim.SGD(critic.parameters(), lr=1e-5, momentum=0.90, nesterov=True) 
	value_optimizer = optim.SGD(critic.parameters(), lr=1e-5)#, momentum=0.90, nesterov=True) 
	value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_optimizer, milestones=[5000,8000], gamma=0.1)
	dataset = ReplayMemory(1000000)
	max_torque = float(env.action_space.high[0])
	paml_losses = []
	global_step = 0
	total_eps = 10000
	env_name = env.spec.id
	true_rewards = []

	while(global_step <= total_eps/num_starting_states):
		# Generating sample trajectories 
		print("Generating sample trajectories ... epislon is {:.3f}".format(epsilon))
		dataset, _ = generate_data(env, dataset, actor, num_starting_states, None, max_actions, noise, epsilon, epsilon_decay, num_action_repeats, discount=discount, all_rewards=true_rewards)
		# epsilon = new_epsilon
		print(len(dataset))
		print("Done")

		np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)), np.asarray(true_rewards))

		kwargs['dataset'] = dataset
		# epochs_value = int(np.ceil(epochs_value * psi))
		if model_type != 'mle':
			critic = pre_train_critic(actor, critic, dataset, epochs_value, discount, batch_size, value_optimizer, value_lr_schedule, max_actions, states_dim, actions_dim, verbose=50)
		kwargs['critic'] = critic

		kwargs['train'] = False
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['losses'] = []
		kwargs['use_model'] = False
		actor_critic_paml_train(**kwargs)

		# num_iters += 10# int(np.ceil(num_iters * psi))
		kwargs['train'] = True
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['losses'] = losses
		kwargs['num_iters'] = num_iters
		kwargs['use_model'] = True
		kwargs['P_hat'] = P_hat

		if model_type =='mle':
			P_hat.general_train_mle(actor, dataset, num_iters, max_actions, model_opt, env_name, losses, batch_size, noise, epsilon, file_location, file_id, save_checkpoints=save_checkpoints_training, verbose=20)
		elif (model_type == 'paml') or (model_type == 'pamlmean'):
			# P_hat = DirectEnvModel(states_dim, actions_dim, max_torque).double()
			if (global_step > 0) and (global_step % 3 == 0):
				lr = lr / 10.
			kwargs['lr'] = lr
			kwargs['P_hat'] = P_hat
			P_hat, _ = actor_critic_paml_train(**kwargs)
		elif model_type == 'random':
			P_hat = DirectEnvModel(states_dim, actions_dim, max_torque).double()
			kwargs['P_hat'] = P_hat
		else:
			raise NotImplementedError

		kwargs['train'] = False
		kwargs['use_model'] = True
		kwargs['P_hat'] = P_hat
		kwargs['losses'] = []
		_, loss_paml = actor_critic_paml_train(**kwargs)
		paml_losses.append(loss_paml)

		use_model = True
		# num_virtual_episodes = 600
		train = True
		#use the epsilon arrived at from generation of real data
		actor, critic = actor_critic_DDPG(env, actor, noise, critic, dataset, batch_size, num_virtual_episodes, max_actions, states_dim, salient_states_dim, discount, use_model, train, verbose, all_rewards, epsilon, value_lr_schedule, file_location, file_id, num_action_repeats, P_hat=P_hat, model_type=model_type)
		
		#check paml_loss with the new policy
		kwargs['train'] = False
		kwargs['use_model'] = True
		kwargs['losses'] = []
		kwargs['P_hat'] = P_hat
		kwargs['actor'] = actor
		kwargs['critic'] = critic
		_, loss_paml = actor_critic_paml_train(**kwargs)
		paml_losses.append(loss_paml)

		if global_step % log == 0:
			np.save(os.path.join(file_location,'paml_loss_model_{}_env_{}_horizon{}_traj{}_{}'.format(model_type, env_name, R_range, max_actions + 1, file_id)), np.asarray(paml_losses))
		global_step += 1

	
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
			small_model,
			num_action_repeats
		):

	# file_location = '/h/abachiro/paml/results'
	file_location = '/scratch/gobi1/abachiro/paml_results'
	# rs = 0
	# torch.manual_seed(rs)
	# np.random.seed(rs)	

	# env = gym.make('Pendulum-v0')
	#env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
	# env.seed(rs)
	# pdb.set_trace()

	if env_name == 'lin_dyn':
		gym.envs.register(id='lin-dyn-v0', entry_point='gym_linear_dynamics.envs:LinDynEnv',)
		# env = gym.make('gym_linear_dynamics:lin-dyn-v0')
		env = gym.make('lin-dyn-v0')
	elif env_name == 'Pendulum-v0':
		env = gym.make('Pendulum-v0')
	else:
		raise NotImplementedError

	if model_type == 'model_free':
		plan = True
	else:
		plan = False

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
	actions_dim = states_dim
	#salient_states_dim = 3 #states_dim
	continuous_actionspace = True
	R_range = 1

	use_model = True
	train_value_estimate = False
	train = True

	action_multiplier = 0.1
	epsilon = 1.
	epsilon_decay = 1./100000
	# critic = Value(states_dim, actions_dim)
	# critic.double()
	# actor = DeterministicPolicy(states_dim, actions_dim)
	# actor.double()
	# target_actor = DeterministicPolicy(states_dim, actions_dim)
	# target_actor.double()
	# target_critic = Value(states_dim, actions_dim)
	# target_critic.double()

	P_hat = DirectEnvModel(states_dim,actions_dim, max_torque, small=small_model)
	P_hat.double()
	#P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon6_traj7.pth', map_location=device))
	#P_hat.load_state_dict(torch.load('act_model_paml_checkpoint_train_True_lin_dyn_horizon5_traj6_using1states.pth', map_location=device))

	# value_optimizer = optim.SGD(critic.parameters(), lr=1e-5, momentum=0.90, nesterov=True) 
	# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_optimizer, milestones=[1500,3000], gamma=0.1)

	#1e-4
	model_opt = optim.SGD(P_hat.parameters(), lr=initial_model_lr, momentum=0.90, nesterov=True)
	#opt = optim.Adam(P_hat.parameters(), lr=1e-5, weight_decay=1e-8)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[5000,7000,8000], gamma=0.1)

	if plan:
		all_rewards = []
		noise = OUNoise(env.action_space)
		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim).double()
		actor_critic_DDPG(env, actor, noise, critic, None, batch_size, num_starting_states, max_actions, states_dim, salient_states_dim, discount, False, True, verbose, all_rewards, epsilon, None, file_location, file_id, num_action_repeats, P_hat=None, model_type='model_free')

		np.save(os.path.join(file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'.format(model_type, states_dim, salient_states_dim, use_model, env_name, R_range, max_actions + 1, file_id)), np.asarray(all_rewards))
		# np.save('actorcritic_pendulum_rewards',np.asarray(all_rewards)) 
		# pdb.set_trace()
	else:
		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
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
		plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, salient_states_dim,actions_dim, discount, max_actions, env, initial_model_lr, num_iters, file_location, file_id, save_checkpoints_training, verbose, batch_size, virtual_episodes, model_type, num_action_repeats)
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





