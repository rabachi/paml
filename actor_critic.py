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
def actor_critic_paml_train(P_hat, 
							actor, 
							value_estimator,
							value_opt,
							opt, 
							num_episodes, 
							num_starting_states, 
							states_dim, 
							actions_dim, 
							use_model, 
							discount, 
							max_actions, 
							train, 
							A_numpy, 
							lr_schedule,
							num_iters,
							losses,
							true_r_list,
							true_x_curr,
							true_x_next,
							true_a_list,
							true_a_prime_list,
							step_state,
							value_loss_coeff):
	best_loss = 15000
	env_name = 'lin_dyn'
	batch_size = num_episodes*num_starting_states
	R_range = max_actions
	unroll_num = max_actions
	end_of_trajectory = 1
	num_iters = num_iters if train else 1

	ell = 0

	batch = dataset.sample(batch_size)

	states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
	states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
	#target_q = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
	rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
	actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

	# true_returns = torch.zeros_like(true_r_list)
	true_returns = discount_rewards(true_r_list[:,0,1:], discount, center=False, batch_wise=True)
	true_log_probs = get_selected_log_probabilities(actor, true_x_next, true_a_prime_list).squeeze()

	for i in range(num_iters):
		#calculate true gradients
		actor.zero_grad()

		#compute loss for critic
		true_actions_next = actor.sample_action(true_x_next)
		true_target_q = critic(torch.cat((true_x_next, true_actions_next), dim=1))
		true_y = true_r_list + GAMMA * true_target_q.detach() #detach to avoid backprop target
		true_q = critic(torch.cat((true_x_curr, true_a_list),dim=1))

		#q_optimizer.zero_grad()
		q_loss = MSE(q, y)
		#q_loss.backward()
		#q_optimizer.step()

		#compute loss for actor
		#policy_optimizer.zero_grad()
		true_policy_loss = -critic(torch.cat((true_x_curr, actor.sample_action(true_x_curr)),dim=1))
		policy_loss = policy_loss.mean()
		#policy_loss.backward()
		#policy_optimizer.step()

		# # true_returns = discount_rewards(true_r_list[:,ell], discount, center=False, batch_wise=True) 
		# true_value = value_estimator(torch.cat((true_x_next.squeeze(), true_a_prime_list.squeeze()),dim=2)) #check dims  #in a2c they use this as the baseline, here I'm just going to use it as an estimate of the advantage
		# true_value_loss = (true_returns - true_value).pow(2).mean() 

		# true_term = true_log_probs * true_value #torch.einsum('ijk,ijl->ik', [true_log_probs, true_value])# + true_value_loss * value_loss_coeff #should this be added here?

		save_stats(true_returns, true_r_list, true_a_prime_list, true_x_next, value=true_value, prefix='true_actorcritic_')

		true_pe_grads = torch.DoubleTensor()
		true_pe_grads_attached = grad(true_term.mean(),pe.parameters(), create_graph=True)
		for g in range(len(true_pe_grads_attached)):
			true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

		# true_pe_grads = torch.DoubleTensor()
		# for st in range(num_starting_states):
		# 	true_pe_grads_attached = grad(true_term[st*num_episodes:num_episodes*(st+1)].mean(),pe.parameters(), create_graph=True)
		# 	for g in range(len(true_pe_grads_attached)):
		# 		true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

		#probably don't need these ... .grad is not accumulated with the grad() function
		pe.zero_grad()
		#calculate model gradients
		step_state = torch.cat((true_x_curr, true_a_list),dim=3) 
	
		model_x_curr, model_x_next, model_a_list, model_r_list, model_a_prime_list = P_hat.unroll(step_state[:,0,:], actor, states_dim, A_numpy, steps_to_unroll=1, continuous_actionspace=True, use_model=use_model, policy_states_dim=states_dim)
		

		model_actions_next = actor.sample_action(model_x_next)
		model_target_q = critic(torch.cat((model_x_next, model_actions_next), dim=1))
		model_y = model_r_list + GAMMA * model_target_q.detach() #detach to avoid backprop target
		model_q = critic(torch.cat((model_x_curr, model_a_list),dim=1))

		#q_optimizer.zero_grad()
		q_loss = MSE(q, y)
		#q_loss.backward()
		#q_optimizer.step()

		#compute loss for actor
		#policy_optimizer.zero_grad()
		model_policy_loss = -critic(torch.cat((model_x_curr, actor.sample_action(model_x_curr)),dim=1))
		policy_loss = policy_loss.mean()
		#policy_loss.backward()
		#policy_optimizer.step()


		# model_value = value_estimator(torch.cat((model_x_next[:,:,0,:], model_a_prime_list[:,:,0,:]),dim=2))#check dims

		# model_log_probs = get_selected_log_probabilities(actor, model_x_next, model_a_prime_list).squeeze()
		# model_returns = discount_rewards(model_r_list[:,0,1:], discount, center=False, batch_wise=True) #should i take out the ell bit?
		# model_value_loss = (model_returns - model_value).pow(2).mean()

		# model_term = model_log_probs * model_value #torch.einsum('ijk,ijl->ik', [model_log_probs,  model_value])# + model_value_loss * value_loss_coeff

		#if i == num_iters - 1:
		save_stats(None, model_r_list, model_a_prime_list, model_x_next, value=model_value, prefix='model_actorcritic_')


		# model_pe_grads = torch.DoubleTensor()
		# for st in range(num_starting_states):
		# 	model_pe_grads_split = list(grad(model_term[num_episodes*st:num_episodes*(st+1)].mean(),pe.parameters(), create_graph=True))
		# 	for g in range(len(model_pe_grads_split)):
		# 		model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))

		model_pe_grads = torch.DoubleTensor()
		model_pe_grads_split = grad(model_term.mean(),pe.parameters(), create_graph=True)
		for g in range(len(model_pe_grads_split)):
			model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))

		loss = torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum())#/num_starting_states)
		# cos = nn.CosineSimilarity(dim=0, eps=1e-6)
		# loss = 1-cos(true_pe_grads,model_pe_grads)
		#model - true
		# loss = 0
		# for stt,stm in zip(true_pe_grads, model_pe_grads):
		# 	for x,y in zip(stt, stm):
		# 		loss = loss + torch.norm(x-y)**2

		# loss = torch.sqrt(loss/num_starting_states)

		if loss.detach().cpu() < best_loss and use_model:
			#Save model and losses so far
			torch.save(P_hat.state_dict(), 'act_model_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states.pth'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory))
			np.save('act_loss_paml_checkpoint_train_{}_{}_horizon{}_traj{}_using{}states'.format(train, env_name, R_range, max_actions + 1, end_of_trajectory), np.asarray(losses))
			np.save('act_valueloss_paml_checkpoint_train_{}_{}_horizon_traj{}_using{}states'.format(train, env_name, R_range, max_actions+1, end_of_trajectory), np.asarray(true_value_loss.mean().detach()))
			best_loss = loss.detach().cpu()

		#update model
		if train: 
			opt.zero_grad()

			loss.backward(retain_graph=True)
			nn.utils.clip_grad_value_(P_hat.parameters(), 5.0)
			opt.step()

			# value_opt.zero_grad()
			# (model_value_loss + true_value_loss).backward()
			# value_opt.step()

			if torch.isnan(torch.sum(P_hat.fc1.weight.data)):
				print('weight turned to nan, check gradients')
				pdb.set_trace()

		losses.append(loss.data.cpu())
		lr_schedule.step()

		if train and i < 1: #and j < 1 
			initial_loss = losses[0]#.data.cpu()
			print('initial_loss',initial_loss)

		if train:
			print("R_range: {:3d} | batch_num: {:5d} | paml_loss: {:.5f} | value_loss: {:.5f}".format(R_range, i, loss.data.cpu(),(model_value_loss + true_value_loss).mean().detach()))
		else: 
			print("Validation loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}".format(use_model, R_range, i, loss.data.cpu()))
			print("---------------------------------------------------------------------------------")


def actor_critic(env, actor, critic, target_actor, target_critic):

	starting_states = 600
	max_actions = 200
	states_dim = 3
	actions_dim = 1
	clip_param = 0.5

	TAU=0.001       #Target Network HyperParameters
	LRA=0.0001      #LEARNING RATE ACTOR
	LRC=0.001       #LEARNING RATE CRITIC
	H1=400   #neurons of 1st layers
	H2=300
	GAMMA=0.9

	buffer_start = 100
	epsilon = 1
	epsilon_decay = 1./100000

	MSE = nn.MSELoss()

	q_optimizer  = optim.Adam(critic.parameters(),  lr=LRC)
	policy_optimizer = optim.Adam(actor.parameters(), lr=LRA)

	for target_param, param in zip(target_critic.parameters(), critic.parameters()):
		target_param.data.copy_(param.data)

	for target_param, param in zip(target_actor.parameters(), actor.parameters()):
		target_param.data.copy_(param.data)

	noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(actions_dim))

	# opt = optim.SGD(actor.parameters(), lr=LRA, momentum=0.90, nesterov=True)
	# opt = optim.SGD(critic.parameters(), lr=LRC, momentum=0.90, nesterov=True)

	all_rewards = []

	batch_size = 64
	batch_states = []
	batch_rewards = []
	batch_actions = []
	batch_returns = []
	batch_counter = 0
	best_loss = 10
	old_action_log_probs_batch = torch.zeros(batch_size).double()

	dataset = ReplayMemory(1000000)
	render = False

	for ep in range(starting_states):
		state = env.reset()

		states = [state]
		actions = []
		rewards = []

		for timestep in range(max_actions):
			epsilon -= epsilon_decay
			with torch.no_grad():
				action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
				#action += noise()*max(0, epsilon) #try without noise
				#action = np.clip(action, -1., 1.)

			state_prime, reward, done, _ = env.step(action)
			if render:
				env.render()

			actions.append(action)
			states.append(state_prime)
			rewards.append(reward)

			dataset.push(state, state_prime, action, reward)

			state = state_prime

		#deltas = calc_actual_state_values(target_critic, rewards, states[1:], discount)
		# for x, x_next, u, r in zip(states[:-1], states[1:], actions, rewards):
			


			if len(dataset) > batch_size:	

				batch = dataset.sample(batch_size)

				states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
				states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
				#target_q = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
				rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
				actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

				states_tensor = states_prev 

				#compute loss for critic
				# actions_next = target_actor.sample_action(states_next)
				# target_q = target_critic(torch.cat((states_next, actions_next), dim=1))
				actions_next = actor.sample_action(states_next)
				target_q = critic(torch.cat((states_next, actions_next), dim=1))
				y = rewards_tensor + GAMMA * target_q.detach() #detach to avoid backprop target
				q = critic(torch.cat((states_prev, actions_tensor),dim=1))

				q_optimizer.zero_grad()
				q_loss = MSE(q, y)
				q_loss.backward()
				q_optimizer.step()

				#compute loss for actor
				policy_optimizer.zero_grad()
				policy_loss = -critic(torch.cat((states_prev, actor.sample_action(states_prev)),dim=1))
				policy_loss = policy_loss.mean()
				policy_loss.backward()
				policy_optimizer.step()

				# if pe.continuous:
				# 	actions_tensor = torch.DoubleTensor(batch_actions).view(-1,max_actions,actions_dim)
				# else:
				# 	actions_tensor = torch.LongTensor(batch_actions).view(-1,max_actions)

				# log_probs = get_selected_log_probabilities(pe, states_tensor, actions_tensor).squeeze()
				# values = pe.get_value(states_tensor)
				
				# # returns_tensor_mean = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-5)
				# advantages = (true_values - values).squeeze()
				# advantages_norm = (advantages - advantages.mean())/(advantages.std() + 1e-5)

				# ratio = torch.exp(log_probs - old_action_log_probs_batch)
				# surr1 = ratio * advantages_norm
				# surr2 = torch.clamp(ratio, 1.0 - clip_param,
				#                     		1.0 + clip_param) * advantages_norm

				# actor_loss = -torch.min(surr1, surr2).mean()


				# entropy = Normal(*pe.get_action_probs(states_tensor)).entropy().mean()

				# value_loss = advantages.pow(2).mean()
				# # actor_loss = (-log_probs.view(batch_size,max_actions) * advantages.detach()).mean()

				# loss = actor_loss + 0.5*value_loss - 1e-4*entropy

				# opt.zero_grad()
				# loss.backward()
				# nn.utils.clip_grad_value_(pe.parameters(),2.0)
				# opt.step()

				#soft update of the frozen target networks
				for target_param, param in zip(target_critic.parameters(), critic.parameters()):
					target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data * TAU)

				for target_param, param in zip(target_actor.parameters(), actor.parameters()):
					target_param.data.copy_(target_param.data * (1.0 - TAU) + param.data *TAU)


				# batch_counter = 0
				# # old_action_log_probs_batch = log_probs.detach()
				# dataset = ReplayMemory(100000)

			# print(actor_loss.detach().numpy())
		all_rewards.append(sum(rewards))
		print("Ep: {:5d} | Loss: {:.3f}   | Average of last 10 rewards:{:.4f}".format(ep, q_loss.detach(), sum(all_rewards[-10:])/len(all_rewards[-10:])))
		if sum(all_rewards[-10:])/len(all_rewards[-10:]) > -500:
			render = False

		# if ep == starting_states - 1:
		# 	pdb.set_trace()
	return all_rewards


if __name__=="__main__":
	
	rs = 0
	# torch.manual_seed(rs)
	# np.random.seed(rs)	

	env = gym.make('Pendulum-v0')
	#env = dm_control2gym.make(domain_name="cartpole", task_name="balance")
	
	#env.seed(rs)
	#torch.manual_seed(rs)

	num_starting_states = 100
	num_episodes = 1
	batch_size = num_starting_states * num_episodes

	val_num_episodes = 10
	val_num_starting_states = 125
	val_batch_size = val_num_starting_states*val_num_episodes
	
	num_iters = 6000
	losses = []
	unroll_num = 1

	value_loss_coeff = 1.0
	MAX_TORQUE = 2.0
	discount = 0.95

	max_actions = 10

	states_dim = 3
	actions_dim = 1
	salient_states_dim = 2
	continuous_actionspace = True
	R_range = max_actions


	use_model = True
	train_value_estimate = False
	train = True
	

	action_multiplier = 0.1

	value_estimator = Value(states_dim, actions_dim)
	value_estimator.double()
	policy_estimator = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=1.5, max_torque=MAX_TORQUE)
	policy_estimator.double()
	target_policy_estimator = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-1.5, max_torque=MAX_TORQUE)
	target_policy_estimator.double()
	target_value_estimator = Value(states_dim, actions_dim)
	target_value_estimator.double()

	P_hat = DirectEnvModel(states_dim,actions_dim, MAX_TORQUE)
	P_hat.double()
	#P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon6_traj7.pth', map_location=device))
	#P_hat.load_state_dict(torch.load('act_model_paml_checkpoint_train_True_lin_dyn_horizon5_traj6_using1states.pth', map_location=device))

	# value_opt = optim.SGD(value_estimator.parameters(), lr=1e-3, momentum=0.90, nesterov=True) 
	# value_lr_schedule = torch.optim.lr_scheduler.MultiStepLR(value_opt, milestones=[1500,3000], gamma=0.1)
	opt = optim.SGD(P_hat.parameters(), lr=1e-6, momentum=0.90, nesterov=True)
	#opt = optim.Adam(P_hat.parameters(), lr=1e-5, weight_decay=1e-8)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[1000,2000,3000], gamma=0.1)
		
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

	A_all[3] = np.array([[-0.5, -0.5, -0.5],
							[ 0.3, -0.2,  0.3],
							[ 0.3,  0.3,  0.4]])

	A_all[2] = np.array([[0.9, 0.4], [-0.4, 0.9]])


	A_numpy = A_all[2]


	all_rewards = actor_critic(env, policy_estimator, value_estimator, target_policy_estimator, target_value_estimator)

	np.save('actorcritic_pendulum_rewards',np.asarray(all_rewards)) 

	pdb.set_trace()

	print('Generating sample trajectories ...')
	with torch.no_grad():
		#generate validation data
		val_step_state = torch.zeros((val_batch_size, unroll_num, states_dim+actions_dim)).double()
		for b in range(val_num_starting_states):
			x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
			val_step_state[b*val_num_episodes:val_num_episodes*(b+1),:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

		val_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(val_step_state[:,:unroll_num,:states_dim]) #I think all this does is make the visualizations look better, shouldn't affect performance (or visualizations ... )
		val_true_x_curr, val_true_x_next, val_true_a_list, val_true_r_list, val_true_a_prime_list = P_hat.unroll(val_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=False, policy_states_dim=states_dim)

		#generate training data
		train_step_state = torch.zeros((batch_size, unroll_num, states_dim+actions_dim)).double()
		for b in range(num_starting_states):
			x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
			train_step_state[b*num_episodes:num_episodes*(b+1),:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

		train_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(train_step_state[:,:unroll_num,:states_dim])#I think all this does is make the visualizations look better, shouldn't affect performance (or visualizations ... )
		train_true_x_curr, train_true_x_next, train_true_a_list, train_true_r_list, train_true_a_prime_list = P_hat.unroll(train_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True, use_model=False, policy_states_dim=states_dim)

	print('Done!')
	#get accuracy of true dynamics on validation data
	true_r_list = val_true_r_list
	true_x_curr = val_true_x_curr
	true_a_list = val_true_a_list
	step_state = val_step_state

	kwargs = {
			'P_hat'              : P_hat, 
			'pe'                 : policy_estimator, 
			'value_estimator'	 : value_estimator,
			'value_opt'			 : value_opt,
			'opt' 				 : opt, 
			'num_episodes'		 : num_episodes, 
			'num_starting_states': num_starting_states, 
			'states_dim'		 : states_dim, 
			'actions_dim'		 : actions_dim, 
			'use_model'			 : False, 
			'discount'			 : discount, 
			'max_actions'		 : max_actions, 
			'train'              : False, 
			'A_numpy'			 : A_numpy, 
			'lr_schedule'		 : lr_schedule,
			'num_iters'          : int(num_iters/120),
			'losses'             : [],
			'true_r_list'        : true_r_list,
			'true_x_curr' 		 : true_x_curr,
			'true_a_list'        : true_a_list,
			'step_state'         : step_state,
			'value_loss_coeff'   : value_loss_coeff
			}

	#pretrain value function
	ell = 0
	if train_value_estimate:
		prefix='true_actorcritic_'
		epochs_value = 300
		best_loss = 1000
		true_r_list = train_true_r_list
		true_x_curr = train_true_x_curr
		true_a_list = train_true_a_list
		# true_returns = torch.zeros_like(true_r_list)
		true_returns = discount_rewards(true_r_list[:,ell,:-1], discount, center=False, batch_wise=True)
		for i in range(epochs_value):
			true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
			# true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
			
			save_stats(true_returns, true_r_list, true_a_list, true_x_curr, value=true_value, prefix='true_actorcritic_')
			np.save(prefix+'value_training', true_value.squeeze().detach().cpu().numpy())
			true_value_loss = (true_returns - true_value).pow(2).mean()
			print('Epoch: {:4d} | Value estimator loss: {:.5f}'.format(i,true_value_loss.detach().cpu()))

			if true_value_loss < best_loss:
				torch.save(value_estimator.state_dict(), 'value_estimator_horizon{}_traj{}.pth'.format(R_range, max_actions+1))
				best_loss = true_value_loss
			value_opt.zero_grad()
			true_value_loss.backward()
			value_opt.step()
			value_lr_schedule.step()
			#check validation
			if (i % 10) == 0:
				with torch.no_grad():
					true_x_curr = val_true_x_curr
					true_a_list = val_true_a_list
					true_r_list = val_true_r_list
					# true_returns = torch.zeros_like(true_r_list)
					true_returns = discount_rewards(true_r_list[:,ell,:-1], discount, center=False, batch_wise=True)
					true_value = value_estimator(torch.cat((true_x_curr.squeeze(), true_a_list.squeeze()),dim=2))
					true_value_loss = (true_returns - true_value).pow(2).mean()
					print('Validation value estimator loss: {:.5f}'.format(true_value_loss.detach().cpu()))
				true_r_list = train_true_r_list
				true_x_curr = train_true_x_curr
				true_a_list = train_true_a_list

				# true_returns = torch.zeros_like(train_true_r_list)
				
				true_returns = discount_rewards(true_r_list[:,ell,:-1], discount, center=False, batch_wise=True)
	else:
		value_estimator.load_state_dict(torch.load('value_estimator_horizon{}_traj{}.pth'.format(R_range, max_actions+1), map_location=device))


	kwargs['train'] = False
	kwargs['num_episodes'] = val_num_episodes
	kwargs['num_starting_states'] = val_num_starting_states
	kwargs['true_r_list'] = val_true_r_list
	kwargs['true_x_curr'] = val_true_x_curr
	kwargs['true_a_list'] = val_true_a_list
	kwargs['true_x_next'] = val_true_x_next
	kwargs['true_a_prime_list'] = val_true_a_prime_list
	kwargs['step_state'] = val_step_state
	kwargs['use_model'] = False
	actor_critic_paml_train(**kwargs)

	val_losses = []
	kwargs['use_model'] = use_model
	for i in range(120):
		kwargs['train'] = False
		kwargs['num_episodes'] = val_num_episodes
		kwargs['num_starting_states'] = val_num_starting_states
		kwargs['true_r_list'] = val_true_r_list
		kwargs['true_x_curr'] = val_true_x_curr
		kwargs['true_a_list'] = val_true_a_list
		kwargs['true_x_next'] = val_true_x_next
		kwargs['true_a_prime_list'] = val_true_a_prime_list
		kwargs['step_state'] = val_step_state
		kwargs['losses'] = val_losses
		actor_critic_paml_train(**kwargs)

		kwargs['train'] = train
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['true_r_list'] = train_true_r_list
		kwargs['true_x_next'] = train_true_x_next
		kwargs['true_a_prime_list'] = train_true_a_prime_list
		kwargs['step_state'] = train_step_state
		kwargs['losses'] = losses
		actor_critic_paml_train(**kwargs)





