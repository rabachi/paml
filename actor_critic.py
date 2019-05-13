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
from get_data import save_stats

# import dm_control2gym
device = 'cpu'

def pre_train_critic(actor, critic, dataset, epochs_value, discount, batch_size, q_optimizer, value_lr_schedule, max_actions, states_dim, actions_dim, verbose):
	MSE = nn.MSELoss()
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
		q_optimizer.step()

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
							# target_critic,
							# q_optimizer,
							opt, 
							num_episodes, 
							num_starting_states, 
							states_dim, 
							actions_dim, 
							use_model, 
							discount, 
							max_actions, 
							train, 
							lr_schedule,
							num_iters,
							losses,
							dataset,
							verbose
							):
	best_loss = 15000
	env_name = env.spec.id
	batch_size = num_episodes*num_starting_states*64
	R_range = 1
	unroll_num = max_actions
	end_of_trajectory = 1
	num_iters = num_iters if train else 1
	MSE = nn.MSELoss()
	noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))	
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

		save_stats(None, true_rewards_tensor, None, true_states_next, value=None, prefix='true_actorcritic_')

		true_pe_grads = torch.DoubleTensor()
		true_pe_grads_attached = grad(true_term.mean(), actor.parameters(), create_graph=True)
		for g in range(len(true_pe_grads_attached)):
			true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

		#probably don't need these ... .grad is not accumulated with the grad() function
		actor.zero_grad()
		
		step_state = torch.cat((true_states_prev, true_actions_tensor),dim=1).unsqueeze(1)
	
		if use_model:
			model_x_curr, model_x_next, model_a_list, model_r_list, _ = P_hat.unroll(step_state, actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=states_dim, noise=noise, epsilon=epsilon, epsilon_decay=epsilon_decay,env=env)

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

		save_stats(None, model_r_list, model_a_list, model_x_next, value=None, prefix='model_actorcritic_')

		model_pe_grads = torch.DoubleTensor()
		model_pe_grads_split = grad(model_term.mean(),actor.parameters(), create_graph=True)
		for g in range(len(model_pe_grads_split)):
			model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))

		cos = nn.CosineSimilarity(dim=0, eps=1e-6)
		loss = 1-cos(true_pe_grads,model_pe_grads)
		
		# loss = MSE(true_pe_grads, model_pe_grads)

		if loss.detach().cpu() < best_loss and use_model:
			#Save model and losses so far
			torch.save(P_hat.state_dict(), 'act_model_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states.pth'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory))
			np.save('act_loss_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory), np.asarray(losses))

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
		lr_schedule.step()

		if train and i < 1: #and j < 1 
			initial_loss = losses[0]#.data.cpu()
			print('initial_loss',initial_loss)

		if train:
			if (i % verbose == 0) or (i == num_iters - 1):
				print("R_range: {:3d} | batch_num: {:5d} | paml_loss: {:.5f}".format(R_range, i, loss.data.cpu()))
		else: 
			print("-----------------------------------------------------------------------------------------------------")
			print("Validation loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}".format(use_model, R_range, i, loss.data.cpu()))
			print("-----------------------------------------------------------------------------------------------------")
	return P_hat


def actor_critic_DDPG(env, actor, critic, real_dataset, num_starting_states, max_actions, states_dim, discount, use_model, train, verbose, all_rewards, epsilon, P_hat=None):
 
	starting_states = num_starting_states
	env_name = env.spec.id


	#rho = fraction of data used from experience replay
	rho_ER = 0.5
	random_start_frac = 0.9

	actions_dim = env.action_space.shape[0]

	R_range = 1

	TAU=0.001      #Target Network HyperParameters
	LRA=0.0001      #LEARNING RATE ACTOR
	LRC=0.001       #LEARNING RATE CRITIC

	buffer_start = 100
	# epsilon = 1
	epsilon_decay = 1./100000

	noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	max_torque = float(env.action_space.high[0])

	batch_size = 128
	best_loss = 10

	MSE = nn.MSELoss()

	# actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
	# critic = Value(states_dim, actions_dim).double()

	critic_optimizer  = optim.Adam(critic.parameters(),  lr=LRC)
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
	for ep in range(starting_states):

		if not use_model:
			state = env.reset()

			states = [state]
			actions = []
			rewards = []
			for timestep in range(max_actions):
				epsilon -= epsilon_decay
				action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
				action += noise()*max(0, epsilon) #try without noise
				action = np.clip(action, -1., 1.)

				state_prime, reward, done, _ = env.step(action)
				if render:
					env.render()

				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)
				
				dataset.push(state, state_prime, action, reward)
				state = state_prime

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

					#compute actor loss
					actor_loss = -critic(states_prev, actor.sample_action(states_prev)).mean()

					#Optimize the actor
					actor_optimizer.zero_grad()
					actor_loss.backward()
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
			
			model_batch_size = batch_size - true_batch_size
			random_start_model_batch_size = int(np.floor(random_start_frac * model_batch_size)) #randomly sample from state space

			#This is kind of slow, but it would work with any environment, the other way it to do the reset batch_wise in a custom function made for every environment separately ... but that seems like it shouldn't be done
			random_model_x_curr = torch.zeros((random_start_model_batch_size, states_dim)).double()
			for samp in range(random_start_model_batch_size):
				s0 = env.reset()
				random_model_x_curr[samp] = torch.from_numpy(s0).double()

			with torch.no_grad():
				random_model_actions = torch.clamp(actor.sample_action(random_model_x_curr) + torch.from_numpy(noise()*max(0, epsilon)), min=-1.0, max=1.0)
				random_model_x_next = P_hat(torch.cat((random_model_x_curr, random_model_actions),1))
				random_model_r_list = get_reward_fn(env, random_model_x_curr.unsqueeze(1), random_model_actions.unsqueeze(1))

				replay_start_model_batch_size = model_batch_size - random_start_model_batch_size #randomly sample from replay buffer 
				replay_model_batch = real_dataset.sample(replay_start_model_batch_size)
				replay_model_x_curr = torch.tensor([samp.state for samp in replay_model_batch]).double().to(device)
				replay_model_actions = torch.clamp(actor.sample_action(replay_model_x_curr) + torch.from_numpy(noise()*max(0, epsilon)), min=-1.0, max=1.0)
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

		if ep % verbose  == 0:
			print("Ep: {:5d} | Q_Loss: {:.3f} | Pi_Loss: {:.3f} | Average of last 10 rewards:{:.4f}".format(ep, critic_loss.detach(), policy_loss.detach(), sum(all_rewards[-10:])/len(all_rewards[-10:])))
			#save all_rewards
			np.save('rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}'.format(use_model, env_name, R_range, max_actions + 1), np.asarray(all_rewards))

		if sum(all_rewards[-10:])/len(all_rewards[-10:]) > -700:
			render = False

	return target_actor, target_critic



def actor_critic_TD3(env, actor, critic, real_dataset, num_starting_states, max_actions, states_dim, actions_dim, discount, use_model, train, verbose, all_rewards, P_hat=None):#, target_actor, target_critic):

	starting_states = num_starting_states
	clip_param = 0.5
	env_name = env.spec.id

	actions_dim = env.action_space.shape[0]

	R_range = 1

	TAU=0.005      #Target Network HyperParameters
	LRA=0.0001      #LEARNING RATE ACTOR
	LRC=0.001       #LEARNING RATE CRITIC

	buffer_start = 100
	epsilon = 1
	epsilon_decay = 1./100000

	#noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	expl_noise = 0.1

	policy_noise = 0.2
	noise_clip = 0.5
	# max_torque = 5.0
	max_torque = float(env.action_space.high[0])

	policy_freq = 2

	all_rewards = []
	batch_size = 128
	# batch_states = []
	# batch_rewards = []
	# batch_actions = []
	# batch_returns = []
	# batch_counter = 0
	best_loss = 10
	# old_action_log_probs_batch = torch.zeros(batch_size).double()

	MSE = nn.MSELoss()

	# actor = DeterministicPolicy(states_dim, actions_dim)
	# actor.double()
	# critic = Value(states_dim, actions_dim)
	# critic.double()

	# q_optimizer  = optim.Adam(critic.parameters(),  lr=LRC)
	# policy_optimizer = optim.Adam(actor.parameters(), lr=LRA)

	#initialize target_critic and target_actor
	actor = Actor(states_dim, actions_dim, max_torque).double().to(device)
	target_actor = Actor(states_dim, actions_dim, max_torque).double().to(device)
	target_actor.load_state_dict(actor.state_dict())
	actor_optimizer = torch.optim.Adam(actor.parameters(), lr=LRA)

	critic = Critic(states_dim, actions_dim).double().to(device)
	target_critic = Critic(states_dim, actions_dim).double().to(device)
	target_critic.load_state_dict(critic.state_dict())
	critic_optimizer = torch.optim.Adam(critic.parameters(), lr=LRC)

	# target_actor = DeterministicPolicy(states_dim, actions_dim)
	# target_actor.double()
	# target_critic = Value(states_dim, actions_dim)
	# target_critic.double()

	for target_param, param in zip(target_critic.parameters(), critic.parameters()):
		target_param.data.copy_(param.data)

	for target_param, param in zip(target_actor.parameters(), actor.parameters()):
		target_param.data.copy_(param.data)

	dataset = ReplayMemory(1000000)
	render = False
	for ep in range(starting_states):

		if not use_model:
			state = env.reset()

			states = [state]
			actions = []
			rewards = []
			for timestep in range(max_actions):
				# epsilon -= epsilon_decay
				action = actor(torch.DoubleTensor(state)).detach().numpy()
				if expl_noise != 0:
					action = (action + np.random.normal(0, expl_noise, size=actions_dim)).clip(env.action_space.low, env.action_space.high)
				# action += noise()*max(0, epsilon) #try without noise
				# action = np.clip(action, -1., 1.)
				state_prime, reward, done, _ = env.step(action)
				if render:
					env.render()

				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)
				
				dataset.push(state, state_prime, action, reward)
				state = state_prime

				if len(dataset) > batch_size:	
					batch = dataset.sample(batch_size)

					states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
					states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
					rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
					actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

					noise = actions_tensor.data.normal_(0, policy_noise).to(device)
					noise = noise.clamp(-noise_clip, noise_clip)

					actions_next = (target_actor(states_next) + noise).clamp(-max_torque, max_torque)
					
					#Compute target Q value
					target_Q1, target_Q2 = target_critic(states_next, actions_next)#torch.cat((states_next, actions_next), dim=1))
					target_Q = torch.min(target_Q1, target_Q2)
					target_Q = rewards_tensor + discount * target_Q.detach()
					
					#Compute current Q estimates
					current_Q1, current_Q2 = critic(states_prev, actions_tensor)

					critic_loss = MSE(current_Q1, target_Q) + MSE(current_Q2, target_Q)

					critic_optimizer.zero_grad()
					critic_loss.backward()
					critic_optimizer.step()

					#Delayed policy updates
					if ep*timestep % policy_freq == 0:
						#compute actor loss
						actor_loss = -critic.Q1(states_prev, actor(states_prev)).mean()

						#Optimize the actor
						actor_optimizer.zero_grad()
						actor_loss.backward()
						actor_optimizer.step()

						#soft update of the frozen target networks
						for target_param, param in zip(target_critic.parameters(), critic.parameters()):
							target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

						for target_param, param in zip(target_actor.parameters(), actor.parameters()):
							target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)
					#compute loss for actor
					# policy_optimizer.zero_grad()
					# policy_loss = -critic(torch.cat((states_prev, actor.sample_action(states_prev)),dim=1))
					# policy_loss = policy_loss.mean()
					# policy_loss.backward()
					# policy_optimizer.step()
				all_rewards.append(sum(rewards))
		else:
			if P_hat is None:
				raise NotImplementedError

			#use real_dataset here
			unroll_num = 1
			ell = 0
			R_range = 1

			batch = real_dataset.sample(batch_size)
			true_x_curr = torch.tensor([samp.state for samp in batch]).double().to(device)
			true_x_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			true_a_list = torch.tensor([samp.action for samp in batch]).double().to(device)
			true_r_list = torch.tensor([samp.reward for samp in batch]).double().to(device)

			with torch.no_grad(): #this is not properly off-policy ... need to store the model outputs as well
				# step_state = torch.zeros((batch_size, 1, states_dim+actions_dim)).double()
				# step_state[b,:unroll_num,:states_dim] = true_x_curr
				
				# actions = actor.sample_action(step_state[:,:unroll_num,:states_dim])
				# actions += torch.from_numpy(noise()*max(0, epsilon)) #try without noise
				# actions = torch.clamp(actions, min=-1.0, max=1.0)
				# step_state[:,:unroll_num,states_dim:] = actions #actor.sample_action(step_state[:,:unroll_num,:states_dim])
				model_a_list_ = torch.clamp(actor.sample_action(true_x_curr) + torch.from_numpy(noise()*max(0,epsilon)),min=-1.,max=1.)
				step_state = torch.cat((true_x_curr, model_a_list_),dim=1)
				# model_x_curr_, model_x_next_, model_a_list_, model_r_list_, _ = P_hat.unroll(step_state[:,:unroll_num,:], actor, states_dim, None, steps_to_unroll=1, continuous_actionspace=True, use_model=True, policy_states_dim=states_dim, noise=noise, epsilon=epsilon, epsilon_decay=epsilon_decay, env=env)

				model_x_curr_ = true_x_curr
				model_x_next_ = P_hat(step_state)
				# model_a_next_ = torch.clamp(actor.sample_action(model_x_next_) + torch.from_numpy(noise()*max(0,epsilon)),min=-1.,max=1.)
				model_r_list_ = get_reward_fn(env, model_x_curr_.unsqueeze(1), model_a_list_.unsqueeze(1)) #should it be the next one?
				model_a_list_ = true_a_list

				all_rewards.append(model_r_list_.sum())

				save_stats(None, model_r_list_, model_a_list_, model_x_curr_, prefix='model_planning_')

			#compute loss for critic
			model_x_curr = torch.cat((model_x_curr_.squeeze(), true_x_curr), dim=0)
			model_a_list = torch.cat((model_a_list_.squeeze().unsqueeze(1), true_a_list), dim=0)
			model_x_next = torch.cat((model_x_next_.squeeze(), true_x_next), dim=0)
			model_r_list = torch.cat((model_r_list_.squeeze().unsqueeze(1), true_r_list.unsqueeze(1)), dim=0)

			save_stats(None, true_r_list, true_a_list, true_x_curr, prefix='true_planning_')

			actions_next = target_actor.sample_action(model_x_next)
			target_q = target_critic(torch.cat((model_x_next, actions_next), dim=1))
			y = model_r_list + GAMMA * target_q.detach() #detach to avoid backprop target
			q = critic(torch.cat((model_x_curr, model_a_list),dim=1))

			q_optimizer.zero_grad()
			q_loss = MSE(q, y)
			q_loss.backward()
			q_optimizer.step()

			#compute loss for actor
			policy_optimizer.zero_grad()
			policy_loss = -critic(torch.cat((model_x_curr, actor.sample_action(model_x_curr)),dim=1))
			policy_loss = policy_loss.mean()
			policy_loss.backward()
			policy_optimizer.step()

			#soft update of the frozen target networks
			for target_param, param in zip(target_critic.parameters(), critic.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

			for target_param, param in zip(target_actor.parameters(), actor.parameters()):
				target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)

		if ep % verbose  == 0:
			print("Ep: {:5d} | Q_Loss: {:.3f} | Pi_Loss: {:.3f} | Average of last 10 rewards:{:.4f}".format(ep, critic_loss.detach(), actor_loss.detach(), sum(all_rewards[-10:])/len(all_rewards[-10:])))
			#save all_rewards
			np.save('rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}'.format(use_model, env_name, R_range, max_actions + 1), np.asarray(all_rewards))

		if sum(all_rewards[-10:])/len(all_rewards[-10:]) > -500:
			render = False

	return target_actor, target_critic



def plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, actions_dim, discount, max_actions, env, lr_schedule, num_iters):
	
	verbose = 200
	batch_size = 64
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
				'actions_dim'		 : actions_dim, 
				# 'use_model'			 : False, 
				'discount'			 : discount, 
				'max_actions'		 : max_actions, 
				# 'train'              : train, 
				'lr_schedule'		 : lr_schedule,
				'num_iters'          : num_iters,
				'losses'             : [],
				'env'				 : env,
				# 'value_loss_coeff'   : value_loss_coeff,
				'verbose'			 : verbose
			}

	unroll_num = 1
	ell = 0
	psi = 1.1
	epsilon = 1.
	epsilon_decay = 1./100000
	noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))
	all_rewards = []
	epochs_value = 150
	value_optimizer = optim.SGD(critic.parameters(), lr=1e-3, momentum=0.90, nesterov=True) 
	value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_optimizer, milestones=[1500,3000], gamma=0.1)
	dataset = ReplayMemory(1000000)
	max_torque = float(env.action_space.high[0])

	while(True):
		# Generating sample trajectories 
		print("Generating sample trajectories ... epislon is {:.3f}".format(epsilon))
		dataset, _, new_epsilon = generate_data(env, dataset, actor, num_starting_states, None, max_actions, noise, epsilon, epsilon_decay, discount=discount)
		print(len(dataset))
		print("Done")
		kwargs['dataset'] = dataset
		epochs_value = int(np.ceil(epochs_value * psi))
		critic = pre_train_critic(actor, critic, dataset, epochs_value, discount, batch_size, value_optimizer, value_lr_schedule, max_actions, states_dim, actions_dim, 100)
		kwargs['critic'] = critic

		kwargs['train'] = False
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['losses'] = []
		kwargs['use_model'] = False
		actor_critic_paml_train(**kwargs)

		num_iters = int(np.ceil(num_iters * psi))
		P_hat = DirectEnvModel(states_dim, actions_dim, max_torque)
		P_hat.double()
		kwargs['train'] = True
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['losses'] = losses
		kwargs['num_iters'] = num_iters
		kwargs['use_model'] = True
		kwargs['P_hat'] = P_hat
		env_name = env.spec.id
		# P_hat.general_train_mle(actor, dataset, num_iters, max_actions, model_opt, env_name, losses, batch_size, noise, epsilon, verbose=20)
		P_hat = actor_critic_paml_train(**kwargs)

		kwargs['train'] = False
		kwargs['use_model'] = True
		kwargs['P_hat'] = P_hat
		kwargs['losses'] = []
		actor_critic_paml_train(**kwargs)

		use_model = True
		num_virtual_episodes = 2000
		train = True
		actor, critic = actor_critic_DDPG(env, actor, critic, dataset, num_virtual_episodes, max_actions, states_dim, discount, use_model, train, verbose, all_rewards, epsilon, P_hat=P_hat)
		
		#check paml_loss with the new policy
		kwargs['train'] = False
		kwargs['use_model'] = True
		kwargs['losses'] = []
		kwargs['P_hat'] = P_hat
		kwargs['actor'] = actor
		kwargs['critic'] = critic
		kwargs['critic'] = critic
		actor_critic_paml_train(**kwargs)

		# epsilon = new_epsilon


if __name__=="__main__":
	
	def main():
		rs = 0
		# torch.manual_seed(rs)
		# np.random.seed(rs)	

		env = gym.make('Pendulum-v0')
		#env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
		# env.seed(rs)
		# torch.manual_seed(rs)
		plan = False

		num_starting_states = 100
		num_episodes = 1
		batch_size = 64
		val_num_episodes = 1
		val_num_starting_states = 10
		val_batch_size = val_num_starting_states*val_num_episodes
		
		num_iters = 100
		losses = []
		unroll_num = 1

		value_loss_coeff = 1.0
		max_torque = float(env.action_space.high[0])
		discount = 0.9

		max_actions = 200

		states_dim = 3
		actions_dim = 1
		salient_states_dim = 3
		continuous_actionspace = True
		R_range = 1

		use_model = True
		train_value_estimate = False
		train = True

		action_multiplier = 0.1

		# critic = Value(states_dim, actions_dim)
		# critic.double()
		# actor = DeterministicPolicy(states_dim, actions_dim)
		# actor.double()
		# target_actor = DeterministicPolicy(states_dim, actions_dim)
		# target_actor.double()
		# target_critic = Value(states_dim, actions_dim)
		# target_critic.double()

		P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE)
		P_hat.double()
		#P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon6_traj7.pth', map_location=device))
		#P_hat.load_state_dict(torch.load('act_model_paml_checkpoint_train_True_lin_dyn_horizon5_traj6_using1states.pth', map_location=device))

		# value_optimizer = optim.SGD(critic.parameters(), lr=1e-5, momentum=0.90, nesterov=True) 
		# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_optimizer, milestones=[1500,3000], gamma=0.1)

		model_opt = optim.SGD(P_hat.parameters(), lr=1e-3, momentum=0.90, nesterov=True)
		#opt = optim.Adam(P_hat.parameters(), lr=1e-5, weight_decay=1e-8)
		lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[5000,7000,8000], gamma=0.1)

		if plan:
			all_rewards = []
			actor_critic_DDPG(env, None, None, None, num_starting_states, max_actions, states_dim, actions_dim, discount, False, True, 1, all_rewards, P_hat=None)#, target_policy_estimator, target_value_estimator)
			np.save('actorcritic_pendulum_rewards',np.asarray(all_rewards)) 
			pdb.set_trace()


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

		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim).double()
		kwargs = {
				'P_hat'              : P_hat, 
				'actor'	             : actor, 
				'critic'	 		 : critic,
				# 'target_critic'	 	 : target_critic,
				'q_optimizer'		 : None,
				'opt' 				 : model_opt, 
				'num_episodes'		 : num_episodes, 
				'num_starting_states': num_starting_states, 
				'states_dim'		 : states_dim, 
				'actions_dim'		 : actions_dim, 
				'use_model'			 : False, 
				'discount'			 : discount, 
				'max_actions'		 : max_actions, 
				'train'              : False,  
				'lr_schedule'		 : lr_schedule,
				'num_iters'          : int(num_iters/120),
				'losses'             : [],
				'value_loss_coeff'   : value_loss_coeff
				}

		#pretrain value function
		ell = 0
		# if train_value_estimate:
		# 	epochs_value = 300
		# 	verbose = 100
		# 	critic = pre_train_critic(actor, critic, dataset, validation_dataset, epochs_value, discount, batch_size, value_optimizer, value_lr_schedule, max_actions, verbose)
		# else:
		# 	critic.load_state_dict(torch.load('critic_horizon{}_traj{}.pth'.format(R_range, max_actions+1), map_location=device))
		# 	target_critic.load_state_dict(torch.load('critic_horizon{}_traj{}.pth'.format(R_range, max_actions+1), map_location=device))
		
		plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim, actions_dim, discount, max_actions, env, lr_schedule, num_iters)

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


		kwargs['train'] = False
		kwargs['num_episodes'] = val_num_episodes
		kwargs['num_starting_states'] = val_num_starting_states
		# kwargs['true_r_list'] = val_true_r_list
		# kwargs['true_x_curr'] = val_true_x_curr
		# kwargs['true_a_list'] = val_true_a_list
		# kwargs['true_x_next'] = val_true_x_next
		# kwargs['true_a_prime_list'] = val_true_a_prime_list
		# kwargs['step_state'] = val_step_state
		kwargs['dataset'] = validation_dataset
		kwargs['use_model'] = False
		actor_critic_paml_train(**kwargs)

		val_losses = []
		kwargs['use_model'] = use_model
		for i in range(120):
			kwargs['train'] = False
			kwargs['num_episodes'] = val_num_episodes
			kwargs['num_starting_states'] = val_num_starting_states
			# kwargs['true_r_list'] = val_true_r_list
			# kwargs['true_x_curr'] = val_true_x_curr
			# kwargs['true_a_list'] = val_true_a_list
			# kwargs['true_x_next'] = val_true_x_next
			# kwargs['true_a_prime_list'] = val_true_a_prime_list
			# kwargs['step_state'] = val_step_state
			kwargs['dataset'] = validation_dataset
			kwargs['losses'] = val_losses
			actor_critic_paml_train(**kwargs)

			kwargs['train'] = train
			kwargs['num_episodes'] = num_episodes
			kwargs['num_starting_states'] = num_starting_states
			# kwargs['true_r_list'] = train_true_r_list
			# kwargs['true_x_next'] = train_true_x_next
			# kwargs['true_a_prime_list'] = train_true_a_prime_list
			# kwargs['step_state'] = train_step_state
			kwargs['dataset'] = dataset
			kwargs['losses'] = losses
			actor_critic_paml_train(**kwargs)

	main()





