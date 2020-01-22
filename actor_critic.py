import torch
from torch import nn
from torch import optim

import pdb

from models import *
from networks import *
from utils import *

import dm_control2gym

device = 'cpu'

#from pympler import summary, muppy

def actor_critic_DDPG(env, actor, noise, critic, real_dataset, batch_size, num_starting_states,
					  max_actions, states_dim, salient_states_dim, discount, use_model, verbose,
					  value_lr_schedule, file_location, file_id, num_action_repeats,
					  planning_horizon=1, P_hat=None, model_type='paml', noise_type='random',
					  save_checkpoints=False, rho_ER=0.5):
	'''
	This function implements the "Planner" in Algorithm 1 of PAML. The algorithm used is DDPG (Lillicrap et al. 2017).
	If use_model = True, it uses the model and true env transitions collected in real_dataset to improve the actor.
	It also uses only true env data to update the critic.
	If use_model = False, it only uses true env to update both actor and critic.

	Args:
		env (gym environment)
		actor (DeterministicPolicy): policy estimator
		noise (OUNoise): exploration noise
		critic (Value): value function estimator
		real_dataset (ReplayMemory): the collection of transitions sampled from the environment
		batch_size (int)
		num_starting_states: number of episodes to use from true environment if model_free (use_model = False) or
							 from model and true env for model-based methods (use_model = True)
		max_actions (int): Number of actions in a trajectory (= number of states - 1)
		states_dim (int): full state/observation dimension
		salient_states_dim (int): dimension of relevant state dimensions
		discount (float)
		use_model (bool): if True, use the model for planning, if False, use the planner in a model-free manner
		verbose (int)
		value_lr_schedule: scheduler for learning rate decay of critic optimizer. Not used here but placeholder in case
							added in future.
		file_location (str): directory for saving model checkpoints and training stats
		file_id (str): string to append to end of file names for saving
		num_action_repeats (int): number of times to repeat each action
		planning_horizon (int): how many steps to unroll model while planning
		P_hat (DirectEnvModel): Model to be trained
		model_type (str): either 'mle', 'paml', or 'random' for no training at all
		noise_type (str): if 'redundant' use redundant type of noise if states_dim correct, otherwise
							use random type of noise (see StableNoise for definitions)
		save_checkpoints (bool): if True, save actor/critic checkpoints
		rho_ER (float \in [0,1]): fraction of real env data to use during planning (if 0, uses 100% virtual samples

	Returns:
		actor: trained policy
		critic: trained critic
	'''
	all_rewards = []
	starting_states = num_starting_states
	env_name = env.spec.id
	max_torque = float(env.action_space.high[0])

	random_start_frac = 0.3 #will also use search control that searches near previously seen states
	radius = .8#np.pi/2.#0.52

	actions_dim = env.action_space.shape[0]
	R_range = planning_horizon

	LRA=0.0001      #LEARNING RATE ACTOR
	LRC=0.001      #LEARNING RATE CRITIC
	TAU=0.001      #Target Network HyperParameters

	max_torque = float(env.action_space.high[0])
	stablenoise = StableNoise(states_dim, salient_states_dim, 0.992, type=noise_type)#, init=np.mean(env.reset()))

	MSE = nn.MSELoss()
	critic_optimizer  = optim.Adam(critic.parameters(), lr=LRC)
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
					# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()
					# #[:salient_states_dim])).detach().numpy()
					action = noise.get_action(action, timestep)
					get_new_action = False
					action_counter = 1

				state_prime, reward, done, _ = env.step(action)

				if env.spec.id != 'lin-dyn-v0':
					state_prime = stablenoise.get_obs(state_prime, timestep + 1)

				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)

				dataset.push(state, state_prime, action, reward)
				state = state_prime

				get_new_action = True if action_counter == num_action_repeats else False
				action_counter += 1

				# code below for model_free-critic-10
				# if timestep == 0:
				# 	schedule_data = 10
				# 	generate_data(env, states_dim, dataset, None, actor, schedule_data, None, max_actions, noise,
				# 					num_action_repeats, temp_data=True, discount=discount,
				# 					all_rewards=[], use_pixels=False, reset=True, start_state=None, start_noise=None)
				# 					#true_rewards)
				if timestep % 200 == 0:
					print(str(timestep) + ': pretraining crtitic ... ')
					critic.pre_train(actor, dataset, 5000, discount, batch_size, salient_states_dim, file_location,
															file_id, model_type, env_name, max_actions, verbose=100)
				# if timestep == (max_actions - 1):
				# 	dataset.clear_temp()

				if len(dataset) > batch_size:
					#FOR MF_CRITIC 10 CHECK IF THIS IS CORRECT
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
					# policy_loss = -critic(states_prev[:,:salient_states_dim],
					# 					  actor.sample_action(states_prev[:,:salient_states_dim])).mean()
					# policy_loss = -critic(states_prev, actor.sample_action(states_prev[:,:salient_states_dim])).mean()
					# policy_loss = -critic(states_prev[:,:salient_states_dim], actor.sample_action(states_prev)).mean()
					policy_loss = -critic(states_prev, actor.sample_action(states_prev)).mean()#[:,:salient_states_dim])).mean()

					#Optimize the actor
					actor_optimizer.zero_grad()
					policy_loss.backward()
					actor_optimizer.step()

					iter_count += 1
					if (iter_count == 40000) or (iter_count == 80000) or (iter_count == 120000) or (iter_count == 160000):
						torch.save(actor.state_dict(), os.path.join(
							file_location,'act_{}_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'
								.format(iter_count, model_type, states_dim, salient_states_dim, env_name, R_range,
										max_actions + 1, file_id)))
						torch.save(critic.state_dict(), os.path.join(
							file_location,'critic_{}_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'
								.format(iter_count, model_type, states_dim, salient_states_dim, env_name, R_range,
										max_actions + 1, file_id)))

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
				critic_loss.backward()
				critic_optimizer.step()

				if value_lr_schedule is not None:
					value_lr_schedule.step()

			model_batch_size = batch_size - true_batch_size
			# randomly sample from state space for starting point of virtual samples
			random_start_model_batch_size = int(np.floor(random_start_frac * model_batch_size * 1.0/planning_horizon))

			#This is kind of slow, but it would work with any environment, the other way it to do the reset
			# batch_wise in a custom function made for every environment separately ... but that seems like it
			# shouldn't be done
			random_model_x_curr = torch.zeros((random_start_model_batch_size, states_dim)).double()
			for samp in range(random_start_model_batch_size):
				s0 = env.reset()
				if env.spec.id != 'lin-dyn-v0':
					s0 = stablenoise.get_obs(s0, 0)
				random_model_x_curr[samp] = torch.from_numpy(s0).double()

			with torch.no_grad():
				#some chosen from randomly sampled around area around states in ER
				replay_start_random_model_batch_size = model_batch_size - random_start_model_batch_size
				replay_model_batch = real_dataset.sample(replay_start_random_model_batch_size)
				# have to move the states to a random position within a radius, here it's 1
				random_pos_delta = 2*(np.random.random(
					size = (replay_start_random_model_batch_size, states_dim)) - 0.5) * radius
				replay_model_x_curr = torch.tensor([replay_model_batch[idx].state * (1 + random_pos_delta[idx])
													for idx in range(replay_start_random_model_batch_size)]).double()
			if true_batch_size > 0:
				states_prev = torch.cat((true_x_curr, random_model_x_curr, replay_model_x_curr), 0)
			else:
				states_prev = torch.cat((random_model_x_curr, replay_model_x_curr), 0)
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
					states_prime[:, p, :] = random.sample(P_hat, 1)[0].predict(states_, actions_).detach()
							#P_hat.predict(states_, actions_).detach()
				else: #ADD STATE
					# states_prime[:, p, :] = random.sample(P_hat, 1)[0].forward(torch.cat((states_, actions_),1)).detach()
					states_prime[:, p, :] = random.sample(P_hat, 1)[0].predict(states_, actions_).detach()

				# states_prime[:, p, :] = P_hat(torch.cat((states_, actions_),1)).detach() #+ states_
				# rewards_model.append(get_reward_fn(env, states_.unsqueeze(1), actions_.unsqueeze(1)))
				states_ = states_prime[:, p, :]

			states_current = torch.cat((states_prev, states_prime[:, :-1, :].contiguous().view(-1, states_dim)), 0)
			actor_optimizer.zero_grad()
			# policy_loss = -critic(states_current[:,:salient_states_dim], actor.sample_action(states_current))
			# policy_loss = -critic(states_current[:,:salient_states_dim], actor.sample_action(
			# 																	states_current[:,:salient_states_dim]))
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
						# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim]))
						# 									.detach().numpy()#[:salient_states_dim])).detach().numpy()
						state_prime, reward, done, _ = env.step(action)

						if env.spec.id != 'lin-dyn-v0':
							state_prime = stablenoise.get_obs(state_prime, timestep+1)

						episode_rewards.append(reward)
						state = state_prime

					eval_rewards.append(sum(episode_rewards))
				all_rewards.append(sum(eval_rewards)/10.)

				torch.save(actor.state_dict(), os.path.join(
					file_location,'act_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(
						model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))
				torch.save(critic.state_dict(), os.path.join(
					file_location,'critic_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'.format(
						model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1, file_id)))

				print("Ep: {:5d} | Q_Loss: {:.3f} | Pi_Loss: {:.3f} | Average rewards of 10 independent episodes:{:.4f}"
					  .format(ep, critic_loss.detach(), policy_loss.detach(), all_rewards[-1]))
				#save all_rewards
				np.save(os.path.join(
					file_location,
					'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'.format(
						model_type, states_dim, salient_states_dim, use_model, env_name, R_range, max_actions + 1,
						file_id)), np.asarray(all_rewards))
	return actor, critic


def plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim,
						salient_states_dim, actions_dim, discount, max_actions, env, lr, num_iters, file_location,
						file_id, save_checkpoints_training, verbose, batch_size, num_virtual_episodes, model_type,
						num_action_repeats, planning_horizon, input_rho, trained_Phat_mle, rs, noise_type):
	'''Implements the Dyna loop in a sequential manner. The loops repeats the following steps for
		total_eps number of steps:
		1. Collect state transitions from env (trajectories of maximum 200 steps are collected here. For environments
		where we'd like to run more than 200 step-trajectories, a second env is created, so that the state of the env
		is preserved for the next iteration of the algorithm. This is because we use env to check the performance of
		the policy at various points during training so we'd like to be able to continue where we left off if the traj-
		ectory is longer than 200 steps).
		2. Pretrain critic on collected data
		3. Train model on data and using critic if with PAML, without critic if with MLE
		4. Train policy using DDPG (Lillicrap et al. 2015) with data from env and data from trained model

	Args:
		P_hat (DirectEnvModel): Model to be trained
		actor (DeterministicPolicy): policy estimator
		critic (Value): value function estimator
		model_opt: optimizer for model
		num_starting_states (int): number of episodes to gather from true env
		num_episodes (int): number of episodes to generate from true environment per starting state (mostly set to 1)
		states_dim (int): full state/observation dimension
		salient_states_dim (int): dimension of relevant state dimensions
		actions_dim (int): dimension of actions
		discount (float)
		max_actions (int): Number of actions in a trajectory (= number of states - 1)
		env (gym environment)
		lr (float): initial model learning rate
		num_iters (int): number of training iterations for the model, this is used every time the model is trained
		file_location (str): directory for saving model checkpoints and training stats
		file_id (str): string to append to end of file names for saving
		save_checkpoints_training (bool): set to True to save model checkpoints during training
		verbose (int)
		batch_size (int): batch size for training model and policy
		num_virtual_episodes (int): number of episodes to generate from the model
		model_type: either 'mle' or 'paml' or 'random' for no training at all
		num_action_repeats (int): number of times to repeat each action
		planning_horizon (int): how many steps to unroll model while planning
		input_rho (float \in [0,1]): fraction of real env data to use during planning (if 0, uses 100% virtual samples
		trained_Phat_mle: optionally can pass in a trained MLE model to compare the various losses and performances bet-
						ween an MLE model and a PAML model, not implemented to be used out-of-the-box, use with caution.
		rs: random seed
		noise_type (str): if 'redundant' use redundant type of noise if states_dim correct, otherwise
							use random type of noise (see StableNoise for definitions)
	Returns:
		None
	'''
	R_range = planning_horizon
	losses = []
	norms_model_pe_grads = []
	norms_true_pe_grads = []

	#setup model training arguments
	kwargs = {
				'actor'	             : actor,
				'critic'	 		 : critic,
				'states_dim'		 : states_dim, 
				'salient_states_dim' : salient_states_dim,
				'batch_size'		 : batch_size,
				'max_actions'		 : max_actions, 
				'planning_horizon'	 : planning_horizon,
				'num_iters'          : num_iters,
				'losses'             : None,
				'env'				 : env,
				'verbose'			 : verbose,
				'file_location'		 : file_location,
				'file_id'			 : file_id,
				'save_checkpoints'	 : save_checkpoints_training
				# 'norms_true_pe_grads' : norms_true_pe_grads,
				# 'norms_model_pe_grads': norms_model_pe_grads
			}

	noise = OUNoise(env.action_space)
	kwargs['noise'] = noise
	skipped = -1
	lr_schedule = []
	for m_o in model_opt:
		if model_type == 'paml':
			lr_schedule.append(torch.optim.lr_scheduler.MultiStepLR(m_o, milestones=[10,500,1000], gamma=0.1))
		elif model_type == 'mle':
			lr_schedule.append(torch.optim.lr_scheduler.MultiStepLR(m_o, milestones=[10,1000,20000000], gamma=0.1)) #ADJUST AS NEEDED

	dataset = ReplayMemory(200000)
	max_torque = float(env.action_space.high[0])
	global_step = 0
	total_eps = 1000 #5000#10000
	env_name = env.spec.id
	true_rewards = []
	epochs_value = 5000
	fractions_real_data_schedule = np.linspace(1.0,0.5,num=50)

	stablenoise = StableNoise(states_dim, salient_states_dim, 0.992)#, init=np.mean(env.reset()))
	original_batch_size = batch_size
	start_state, start_noise = None, None

	if env_name == 'dm-Walker-v0':
		env2 = dm_control2gym.make(domain_name="walker", task_name="walk")
		env2.spec.id = 'dm-Walker-v0'
	elif env_name == 'HalfCheetah-v2':
		env2 = gym.make('HalfCheetah-v2')
		env2.spec.id = 'HalfCheetah-v2'
	elif env_name == 'dm-Cartpole-balance-v0':
		env2 = dm_control2gym.make(domain_name="cartpole", task_name="balance")
		env2.spec.id = 'dm-Cartpole-balance-v0'
	elif env_name == 'Pendulum-v0':
		env2 = gym.make('Pendulum-v0')
		env2.spec.id = 'Pendulum-v0'
	# elif env_name == 'lin-dyn-v0':
	# 	# gym.envs.register(id='lin-dyn-v0', entry_point='gym_linear_dynamics.envs:LinDynEnv',)
	# 	# env = gym.make('gym_linear_dynamics:lin-dyn-v0')
	# 	env2 = gym.make('lin-dyn-v0')
	# 	env2.seed(rs)

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
	num_val_starts = 2
	validation_dataset = ReplayMemory(num_val_starts*max_actions)
	while(global_step <= total_eps):
		# Generating sample trajectories 
		print("Global step: {:4d} | Generating sample trajectories ...".format(global_step))

		num_steps = min(200,max_actions)
		to_reset = (max_actions <= 200) or (global_step % (max_actions/200) == 0)

		if env_name == 'dm-Walker-v0':
			num_steps = min(250, max_actions)
			to_reset = (max_actions <= 250) or (global_step % (max_actions/250) == 0)

		#possible memory issue: passing dataset and validation_dataset back and forth
		start_state, start_noise = generate_data(env2, env, states_dim, dataset, validation_dataset,
															  actor, num_starting_states, num_val_starts,
															  num_steps, noise, num_action_repeats,
															  reset=to_reset,
															  start_state=None if to_reset else start_state,
															  start_noise=None if to_reset else start_noise,
												 			  noise_type=noise_type)
		batch_size = min(original_batch_size, len(dataset))
		#Evaluate policy without noise
		eval_rewards = []
		# Commented code for estimate_returns and returns are for comparing performance of critic in comparison
		# to MC rollouts during training.
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
				# estimate_returns[ep, timestep] = critic(torch.tensor(state[:salient_states_dim]).unsqueeze(0),
				# torch.tensor(action).unsqueeze(0)).squeeze().detach().data.double()
				# estimate_returns[ep, timestep] = critic(torch.tensor(state).unsqueeze(0),
				# torch.tensor(action).unsqueeze(0)).squeeze().detach().data.double()
				del action
				if env.spec.id != 'lin-dyn-v0':
					state_prime = stablenoise.get_obs(state_prime, timestep+1)
				episode_rewards.append(reward)
				state = state_prime

			# returns[ep] = discount_rewards(episode_rewards, discount, center=False, batch_wise=False).detach()[0]
			eval_rewards.append(sum(episode_rewards))

		true_rewards.append(sum(eval_rewards)/10.)

		print('Global step: {:4d} | Average rewards on true dynamics: {:.3f}'.format(global_step, true_rewards[-1]))
		# Commented: For critic accuracy experiments
		# np.save(os.path.join(file_location,
		# '{}_state{}_salient{}_true_returns_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}_{}'
		# .format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1,
		# global_step, file_id)), np.asarray(returns))
		# np.save(os.path.join(file_location,
		# '{}_state{}_salient{}_estimate_returns_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}_{}'
		# .format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1,
		# global_step, file_id)), np.asarray(estimate_returns[:,0]))
		np.save(os.path.join(
			file_location,'{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_False_{}_horizon{}_traj{}_{}Model_hidden{}_{}.npy'
				.format(model_type, states_dim, salient_states_dim, env_name, R_range, max_actions + 1,
						P_hat[0].model_size, P_hat[0].hidden_size, file_id)),
			np.asarray(true_rewards))

		# uncomment below for critic-10 baselines
		# if to_reset and env != 'lin-dyn-v0':# and model_type == 'paml':
		# 	schedule_data = 10
		# 	generate_data(env, states_dim, dataset, None, actor, schedule_data, None, max_actions, noise,
		# 					num_action_repeats, temp_data=True, discount=discount, all_rewards=[],
		# 				  	reset=True, start_state=None, start_noise=None)#true_rewards)
		#
		critic.pre_train(actor, dataset, epochs_value, discount, batch_size, salient_states_dim, file_location,
						 file_id, model_type, env_name, max_actions, verbose=100)
		#
		# if to_reset:
		# 	dataset.clear_temp()

		kwargs['critic'] = critic
		if env_name == 'lin-dyn-v0': #only do ensemble size 1 for lin-dyn
			kwargs['opt'] = model_opt[0]
			kwargs['train'] = False
			kwargs['losses'] = []
			kwargs['use_model'] = False
			kwargs['dataset'] = dataset
			kwargs['lr_schedule'] = lr_schedule[0]
			P_hat[0].actor_critic_paml_train(**kwargs)

		kwargs['train'] = True
		kwargs['num_iters'] = verbose
		kwargs['use_model'] = True

		if to_reset or env_name == 'dm-Walker-v0':
			for P_hat_idx in range(len(P_hat)):
				kwargs['losses'] = None #losses not being recorded, may want to change this at some point
				if model_type != 'random':
					kwargs['opt'] = model_opt[P_hat_idx]
				if model_type =='mle':
					P_hat[P_hat_idx].general_train_mle(actor, dataset, validation_dataset, states_dim,
													   salient_states_dim, num_iters, max_actions, model_opt[P_hat_idx],
													   env_name, losses, batch_size, file_location, file_id,
													   save_checkpoints=True, verbose=verbose,
													   lr_schedule=lr_schedule[P_hat_idx], global_step=global_step)
				elif model_type == 'paml':
					val_losses = [100,99]
					while val_losses[-1] < (val_losses[-2] + val_losses[-2] * 0.0):
						#train PAML model
						kwargs['train'] = True
						kwargs['dataset'] = dataset
						kwargs['num_iters'] = num_iters#verbose
						kwargs['batch_size'] = batch_size
						kwargs['lr_schedule'] = lr_schedule[P_hat_idx]
						P_hat[P_hat_idx].actor_critic_paml_train(**kwargs)

						#check validation loss
						kwargs['dataset'] = validation_dataset
						kwargs['train'] = False
						kwargs['num_iters'] = 1
						kwargs['batch_size'] = len(validation_dataset)
						val_losses.append(P_hat[P_hat_idx].actor_critic_paml_train(**kwargs))
						val_losses = val_losses[1:]
					del val_losses
				elif model_type == 'random':
					pass
				else:
					raise NotImplementedError
		use_model = True
		train = True

		# For testing: Uncomment to see effect of virtual episodes on model loss to see how PAML is affected
		# for v_ep in range(num_virtual_episodes):
		# 	actor, critic = actor_critic_DDPG(env, actor, noise, critic, dataset, batch_size, 1, max_actions,
		# 									  states_dim, salient_states_dim, discount, use_model, verbose,
		# 									  file_location, file_id, num_action_repeats,
		# 									  planning_horizon=planning_horizon, P_hat=P_hat, model_type=model_type)
		# 	#check paml_loss with the new policy
		# 	kwargs['train'] = False
		# 	kwargs['use_model'] = True
		# 	kwargs['losses'] = []
		# 	kwargs['P_hat'] = P_hat
		# 	kwargs['actor'] = actor
		# 	kwargs['critic'] = critic
		# 	_, loss_paml = actor_critic_paml_train(**kwargs)
		# 	paml_losses.append(loss_paml)

		# For possibly scheduling amount of data from replay buffer vs from model
		# rho_ER = fractions_real_data_schedule[global_step] if global_step < 50 else 0.5

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

		# paml_L_mle_val_relevant.append(torch.mean(torch.sum((paml_states_prime[:,:salient_states_dim] -
		# val_x_next[:,:salient_states_dim])**2,dim=1)).data)
		# mle_L_mle_val_relevant.append(torch.mean(torch.sum((mle_states_prime[:,:salient_states_dim] -
		# val_x_next[:,:salient_states_dim])**2,dim=1)).data)

		# paml_L_mle_val_irrelevant.append(torch.mean(torch.sum((paml_states_prime[:,salient_states_dim:] -
		# val_x_next[:,salient_states_dim:])**2,dim=1)).data)
		# mle_L_mle_val_irrelevant.append(torch.mean(torch.sum((mle_states_prime[:,salient_states_dim:] -
		# val_x_next[:,salient_states_dim:])**2,dim=1)).data)
		
		# L_mle_val_relevant.append(torch.mean(torch.sum((val_model_mle_states_prime[:,:salient_states_dim] -
		# val_x_next[:,:salient_states_dim])**2,dim=1)).data)
		# L_mle_val_irrelevant.append(torch.mean(torch.sum((val_model_mle_states_prime[:,salient_states_dim:] -
		# val_x_next[:,salient_states_dim:])**2,dim=1)).data)

		#num_virtual_episodes * (2*(skipped==global_step) + 1*(skipped!=global_step))
		value_lr_schedule = None
		#possible memory issue: passing datasets
		actor, critic = actor_critic_DDPG(env, actor, noise, critic, dataset, batch_size,
										  num_virtual_episodes * (2*(skipped==global_step) + 1*(skipped!=global_step)),
										  max_actions, states_dim, salient_states_dim, discount, use_model,
										  verbose, value_lr_schedule, file_location, file_id,
										  num_action_repeats, planning_horizon=planning_horizon, P_hat=P_hat,
										  model_type=model_type, noise_type=noise_type,
										  save_checkpoints=save_checkpoints_training, rho_ER=input_rho)

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
			# np.save(os.path.join(file_location,'ac_pamlloss_model_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# .format(model_type, env_name, states_dim, salient_states_dim, R_range, max_actions + 1, file_id)),
		# np.asarray(paml_losses))

		# if not losses == []:
		# 	np.save(os.path.join(file_location,'ac_loss_model_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 						 .format(model_type, env_name, states_dim, salient_states_dim, R_range,
		# 								 max_actions + 1, file_id)), np.asarray(losses))
		# if model_type == 'paml':
		# np.save(os.path.join(file_location,'L_paml_val_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(model_type, env_name, states_dim, salient_states_dim, R_range,
		# 							 max_actions + 1, file_id)), np.asarray(L_paml_val))
		# np.save(os.path.join(file_location,'L_mle_val_rel_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(model_type, env_name, states_dim, salient_states_dim, R_range,
		# 							 max_actions + 1, file_id)), np.asarray(L_mle_val_relevant))
		# np.save(os.path.join(file_location,'L_mle_val_irrel_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(model_type, env_name, states_dim, salient_states_dim, R_range,
		# 							 max_actions + 1, file_id)), np.asarray(L_mle_val_irrelevant))
		# np.save(os.path.join(file_location,'mle_L_paml_val_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1,
		# 							 file_id)), np.asarray(mle_L_paml_val))
		# np.save(os.path.join(file_location,'paml_L_paml_val_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1,
		# 							 file_id)), np.asarray(paml_L_paml_val))
		# np.save(os.path.join(file_location,'mle_L_mle_val_rel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1,
		# 							 file_id)), np.asarray(mle_L_mle_val_relevant))
		# np.save(os.path.join(file_location,'paml_L_mle_val_rel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1,
		# 							 file_id)), np.asarray(paml_L_mle_val_relevant))
		# np.save(os.path.join(file_location,'mle_L_mle_val_irrel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1,
		# 							 file_id)), np.asarray(mle_L_mle_val_irrelevant))
		# np.save(os.path.join(file_location,'paml_L_mle_val_irrel_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
		# 					 .format(env_name, states_dim, salient_states_dim, R_range, max_actions + 1,
		# 							 file_id)), np.asarray(paml_L_mle_val_irrelevant))
		global_step += 1
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
			file_location,
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
			ensemble_size,
			noise_type
		):
	'''
	This function sets up the environment, actor, critic, model/model ensemble, optimizers, and based on model_type
	calls the Planner directly (for model-free) or calls the function that implements the model-based algorithms.
	Args:
		env_name (str)
		real_episodes (int): number of episodes to collect per iteration of Dyna (see plan_and_train_ddpg) in true env
		virtual_episodes (int): number of episodes to collect per iteration of Dyna (see plan_and_train_ddpg) from model
		num_eps_per_start (int): how many episodes to collect per starting state
		num_iters (int): number of training iterations for the model, this is used every time the model is trained
		max_actions (int): Number of actions in a trajectory (= number of states - 1)
		discount (float)
		states_dim (int): full state/observation dimension
		salient_states_dim (int): dimension of relevant state dimensions
		initial_model_lr (float): Initial model learning rate to use. This learning rate is gradually decreased
								  according to a scheduler
		model_type (str): either 'model_free', 'mle', 'paml', or 'random' for no training at all
		file_location (str): directory for saving model checkpoints and training stats
		file_id (str): string to append to end of file names for saving
		save_checkpoints (bool): if True, save model checkpoints
		batch_size (int): batch_size for training of model and policy
		verbose (int)
		model_size (str): 'small', 'constrained', or 'nn' to specify the size and type of the model
		num_action_repeats (int): number of times to repeat each action
		rs: random seed
		planning_horizon (int): how many steps to unroll model while planning
		hidden_size (int): size of hidden size for 'constrained' and 'nn' models, for 'small', it is fixed to states_dim
		input_rho (float \in [0,1]): fraction of real env data to use during planning (if 0, uses 100% virtual samples
		ensemble_size (int): number of models to use in an ensemble of models (not fully tested)

	Returns:
		None
	'''
	if env_name == 'lin_dyn':
		# gym.envs.register(id='lin-dyn-v0', entry_point='gym_linear_dynamics.envs:LinDynEnv',)
		# # env = gym.make('gym_linear_dynamics:lin-dyn-v0')
		# env = gym.make('lin-dyn-v0')
		raise NotImplementedError

	elif env_name == 'Pendulum-v0':
		# if states_dim > salient_states_dim:
		# 	env = AddExtraDims(NormalizedEnv(gym.make('Pendulum-v0')), states_dim - salient_states_dim)
		# 	NormalizedEnv(gym.make('Pendulum-v0'))
		env = gym.make('Pendulum-v0')

	elif env_name == 'HalfCheetah-v2':
		env = gym.make('HalfCheetah-v2')

	elif env_name == 'dm-Cartpole-balance-v0':
		env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
		env.spec.id = 'dm-Cartpole-balance-v0'

	elif env_name == 'dm-Walker-v0':
		env = dm_control2gym.make(domain_name="walker", task_name="walk")
		env.spec.id = 'dm-Walker-v0'

	else:
		raise NotImplementedError

	if model_type == 'model_free':
		plan = True
	else:
		plan = False

	torch.manual_seed(rs)
	np.random.seed(rs)
	env.seed(rs)

	num_starting_states = real_episodes if not plan else 5000#10000
	num_episodes = num_eps_per_start #1

	max_torque = float(env.action_space.high[0])
	actions_dim = env.action_space.shape[0]
	continuous_actionspace = True
	R_range = planning_horizon

	use_model = True
	train_value_estimate = False
	train = True

	if plan: #model_free
		all_rewards = []
		noise = OUNoise(env.action_space)
		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim).double()
		actor_critic_DDPG(env, actor, noise, critic, None, batch_size, num_starting_states,
						  max_actions, states_dim, salient_states_dim, discount, False, verbose,
						  None, file_location, file_id, num_action_repeats,
						  P_hat=None, model_type='model_free', noise_type=noise_type,
						  save_checkpoints=save_checkpoints_training)

		np.save(os.path.join(file_location,
							 '{}_state{}_salient{}_rewards_actorcritic_checkpoint_use_model_{}_{}_horizon{}_traj{}_{}'
							 .format(model_type, states_dim, salient_states_dim, use_model, env_name, R_range,
									 max_actions + 1, file_id)), np.asarray(all_rewards))
	else: #model-based
		P_hat = []
		model_opt = []
		for ens in range(ensemble_size):
			model_ens = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size,
									   hidden_size=hidden_size).double()
			P_hat.append(model_ens)
			if model_type == 'paml':
				model_opt.append(optim.SGD(model_ens.parameters(), lr=initial_model_lr))
			elif model_type == 'mle':
				model_opt.append(optim.Adam(model_ens.parameters(), lr=initial_model_lr))

		need_Phat_mle = False
		if need_Phat_mle: #this section is for additional testing. Ignore.
			if model_type == 'paml' and env_name == 'Pendulum-v0':
				trained_Phat_mle = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size,
												  hidden_size=hidden_size).double()
				trained_Phat_mle.load_state_dict(torch.load(os.path.join(file_location,
				'model_mle_checkpoint_state30_salient3_actorcritic_Pendulum-v0_horizon1_traj201_constrainedModel_{}.pth'
																		 .format(rs)), map_location=device))
			elif model_type == 'paml' and env_name == 'lin_dyn':
				trained_Phat_mle = DirectEnvModel(states_dim,actions_dim, max_torque, model_size=model_size,
												  hidden_size=hidden_size).double()
		trained_Phat_mle = None

		actor = DeterministicPolicy(states_dim, actions_dim, max_torque).double()
		critic = Value(states_dim, actions_dim, pretrain_val_lr=1e-4, pretrain_value_lr_schedule=None).double()

		def load_checkpoints():
			for ens in P_hat:
				ens_cp_filename = os.path.join(file_location,
									'model_{}_checkpoint_state{}_salient{}_actorcritic_{}_horizon{}_traj{}_{}.pth'
											   .format(model_type, states_dim, salient_states_dim, env_name,
													   planning_horizon, max_actions+1, file_id))
				if os.path.exists(ens_cp_filename):
					ens.load_state_dict(torch.load(ens_cp_filename, map_location=device))

			if os.path.exists(ens_cp_filename):
				#only load actor and critic if model already existed, if more than 1 file in ensemble
				# will have to make this more robust
				actor_cp_filename = os.path.join(file_location,
												 'act_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'
												 .format(model_type, states_dim, salient_states_dim, env_name,
														 planning_horizon, max_actions + 1, file_id))
				if os.path.exists(actor_cp_filename):
					actor.load_state_dict(torch.load(actor_cp_filename, map_location=device))

				critic_cp_filename = os.path.join(file_location,
											'critic_policy_{}_state{}_salient{}_checkpoint_{}_horizon{}_traj{}_{}.pth'
												  .format(model_type, states_dim, salient_states_dim, env_name,
														  planning_horizon, max_actions + 1, file_id))
				if os.path.exists(critic_cp_filename):
					critic.load_state_dict(torch.load(critic_cp_filename, map_location=device))

		plan_and_train_ddpg(P_hat, actor, critic, model_opt, num_starting_states, num_episodes, states_dim,
							salient_states_dim,actions_dim, discount, max_actions, env, initial_model_lr,
							num_iters, file_location, file_id, save_checkpoints_training, verbose, batch_size,
							virtual_episodes, model_type, num_action_repeats, planning_horizon,
							input_rho, trained_Phat_mle, rs, noise_type)
