import torch
from torch import nn
import torch.nn.functional as F
from networks import *
from torch.autograd import grad

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class DirectEnvModel(torch.nn.Module):
	'''A class for all environment models'''
	def __init__(self, states_dim, N_ACTIONS, MAX_TORQUE, model_size='nn', hidden_size=2, limit_output=False):
		super(DirectEnvModel, self).__init__()
		# build network layers
		self.states_dim = states_dim
		self.n_actions = N_ACTIONS
		self.max_torque = MAX_TORQUE
		self.model_size = model_size
		self.hidden_size = hidden_size if model_size != 'small' else self.states_dim
		self.limit_output = limit_output
		self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_actions, \
		self.std_actions = (0, 1, 0, 1, 0, 1)
		if self.model_size == 'small':
			self.fc1 = nn.Linear(states_dim + N_ACTIONS, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
		elif self.model_size == 'constrained':
			self. fc1 = nn.Linear(states_dim + N_ACTIONS, hidden_size)
			self._enc_mu = torch.nn.Linear(hidden_size, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
			torch.nn.init.xavier_uniform_(self._enc_mu.weight)
		elif self.model_size == 'nn':
			self.fc1 = nn.Linear(states_dim + N_ACTIONS, hidden_size) #128 limited, 512 good for halfcheetah
			self.fc2 = nn.Linear(hidden_size, hidden_size)
			self._enc_mu = torch.nn.Linear(hidden_size, states_dim)
			torch.nn.init.xavier_uniform_(self.fc1.weight)
			torch.nn.init.xavier_uniform_(self.fc2.weight)
			torch.nn.init.xavier_uniform_(self._enc_mu.weight)
		else:
			raise NotImplementedError

	def forward(self, x):
		if self.model_size == 'small':
			x = self.fc1(x)
			if self.limit_output:
				mu = nn.Tanh()(x) * 2.0
			else:
				mu = x
		elif self.model_size == 'constrained':
			mu = self._enc_mu(self.fc1(x))
			#mu = x#nn.Tanh()(x) * 360.0

		elif self.model_size == 'nn':
			x = nn.ReLU()(self.fc1(x))
			# x = self.fc2(x)
			x = nn.ReLU()(self.fc2(x))
			# x = self.fc3(x)
			# x = nn.ReLU()(x)
			if self.limit_output:
				mu = nn.Tanh()(self._enc_mu(x)) * 8.0 #100.0 #*3.0
			else:
				mu = self._enc_mu(x) #nn.Tanh()(self._enc_mu(x)) * 5.0 #100.0 #*3.0
		else:
			raise NotImplementedError
		return mu

	def unroll(self, state_action, policy, states_dim, A_numpy, steps_to_unroll=2, continuous_actionspace=True,
			   use_model=True, salient_states_dim=2, extra_dims_stable=False, noise=None, env='lin_dyn',
			   using_delta=False):
		'''This method repeatedly applies the model in self to generate virtual next-states.
		Args:
			state_action (batch, number of steps, states_dim + action_dim): a batch of starting state and actions
			concatenated
			policy (DeterministicPolicy or Policy class): policy to unroll model with
			states_dim (int): full state/observation dimension
			salient_states_dim (int): dimension of relevant state dimensions
			A_numpy (numpy array, states_dim x states_dim): only for the LQR environment, set to None for all others
			steps_to_unroll (int): number of times to apply model (how many steps to unroll the model)
			continuous_actionspace (bool): specifies whether the action_space is continuous, the case of discrete
											actionspaces has not been tested!!
			use_model (bool): use the model to unroll or A_numpy to unroll. Only applies to lin_dyn
			extra_dims_stable (bool): whether the added irrelevant dimensions have stable dynamics
			noise (class of noise with certain methods (see OUNoise)
			env: name of environment
			using_delta (bool): whether the model predicts next_state - current_state or it directly predicts next state
		Returns:
			torch arrays:
			x_curr: previous states in unrolling process
			x_next: next states after applying model to x_curr and a_used
			a_used: actions applied by using policy on model
			r_used: rewards from model, only returns None unless env == 'lin_dyn'
			a_prime: actions resulting from applying policy to x_next
			'''

		need_rewards = False if env != 'lin_dyn' else True
		if state_action is None:
			return -1	
		batch_size = state_action.shape[0]
		max_actions = state_action.shape[1]
		actions_dim = state_action[0,0,states_dim:].shape[0]
		
		#initialize dynamics arrays if they exist (this function can be used in model mode or true mode if in
		# linear dynamic system)
		if A_numpy is not None:
			A = torch.from_numpy(
				np.tile(
					A_numpy,
					(state_action[0,:,0].shape[0],1,1) #repeat along trajectory size, for every state
					)
				).to(device)
			extra_dim = states_dim - salient_states_dim
			B = torch.from_numpy(
				np.block([
						 [0.1*np.eye(salient_states_dim),            np.zeros((salient_states_dim, extra_dim))],
						 [np.zeros((extra_dim, salient_states_dim)), np.zeros((extra_dim,extra_dim))          ]
						])
				).to(device)
		elif A_numpy is None and not use_model:
			raise NotImplementedError

		x0 = state_action[:,:,:states_dim]
		with torch.no_grad():
			a0 = policy.sample_action(x0[:,:,:states_dim])
			if isinstance(policy, DeterministicPolicy):
				a0 = torch.from_numpy(noise.get_action(a0.numpy(), 0, multiplier=1.))

		state_action = torch.cat((x0, a0),dim=2)
		if use_model:	
			#USING delta 
			if using_delta:
				x_t_1 = x0.squeeze() + self.forward(state_action.squeeze())#.unsqueeze(1)
			else:
				x_t_1 = self.forward(state_action.squeeze())#.unsqueeze(1)
			if len(x_t_1.shape) < 3:
				x_t_1 = x_t_1.unsqueeze(1)
		else:
			x_t_1  = torch.einsum('jik,ljk->lji',[A,x0[:,:,:states_dim]])
			if not extra_dims_stable:
				x_t_1 = add_irrelevant_features(x_t_1, states_dim-salient_states_dim, noise_level=0.4)
				x_t_1 = x_t_1 + 0.1*a0
			else:
				a0_appended = a0[:]
				if actions_dim < states_dim:
					a0_appended = torch.cat((a0, torch.zeros((a0.shape[0], a0.shape[1],
															  states_dim-actions_dim)).double()),2)
				x_t_1 = x_t_1 + torch.einsum('ijk,kk->ijk',[a0_appended,B])

		x_list = [x0, x_t_1]

		with torch.no_grad():
			a1 = policy.sample_action(x_list[1][:,:,:states_dim])
			if isinstance(policy, DeterministicPolicy):
				a1 = torch.from_numpy(noise.get_action(a1.numpy(), 1, multiplier=1.))

		a_list = [a0, a1]
		if need_rewards:
			r_list = [get_reward_fn(env, x_list[0][:,:,:salient_states_dim], a0).unsqueeze(2),
					  get_reward_fn(env, x_t_1[:,:,:salient_states_dim], a1).unsqueeze(2)]

		state_action = torch.cat((x_t_1, a1),dim=2)

		for s in range(steps_to_unroll - 1):
			if torch.isnan(torch.sum(state_action)):
				print('found nan in state')
				print(state_action)
				pdb.set_trace()

			if use_model:
				#USING delta
				if using_delta:
					x_next = state_action[:,:,:states_dim].squeeze() + self.forward(state_action.squeeze())
				else:
					x_next = self.forward(state_action.squeeze())

				if len(x_next.shape) < 3:		
					x_next = x_next.unsqueeze(1)
			else:
				x_next  = torch.einsum('jik,ljk->lji',[A,state_action[:,:,:states_dim]])
				if not extra_dims_stable:
					x_next = add_irrelevant_features(x_next, states_dim-salient_states_dim, noise_level=0.4)
					x_next = x_next + 0.1*state_action[:,:,states_dim:]
				else:
					curr_action = state_action[:,:,states_dim:]
					if actions_dim < states_dim:
						curr_action = torch.cat((curr_action, torch.zeros((state_action.shape[0],state_action.shape[1],
																		   states_dim-actions_dim)).double()),2)
					x_next = x_next + torch.einsum('ijk,kk->ijk',[curr_action,B])

			with torch.no_grad():
				a = policy.sample_action(x_next[:,:,:states_dim])
				if isinstance(policy, DeterministicPolicy):
					a = torch.from_numpy(noise.get_action(a.numpy(), s, multiplier=1.))

			if need_rewards:
				r = get_reward_fn(env, x_next[:,:,:salient_states_dim], a).unsqueeze(2) 
			next_state_action = torch.cat((x_next, a),dim=2)

			############# for discrete, needs testing #######################
			#the dim of this could be wrong due to change to batch_size. NOT TESTED
			#next_state_action = torch.cat((x_next, convert_one_hot(a.double(), n_actions).unsqueeze(2)),dim=2)
			#################################################################
			a_list.append(a)
			if need_rewards:
				r_list.append(r)
			x_list.append(x_next)
			state_action = next_state_action

		x_list = torch.cat(x_list, 2).view(batch_size, -1, steps_to_unroll+1, states_dim)
		a_list = torch.cat(a_list, 2).view(batch_size, -1, steps_to_unroll+1, actions_dim)
		if need_rewards:
			r_list = torch.cat(r_list, 2).view(batch_size, -1, steps_to_unroll+1, 1)

		x_curr = x_list[:,:,:-1,:]
		x_next = x_list[:,:,1:,:]
		a_used = a_list[:,:,:-1,:]
		a_prime = a_list[:,:,1:,:]
		if need_rewards:
			r_used = r_list
		else:
			r_used = None
		return x_curr, x_next, a_used, r_used, a_prime

	def actor_critic_paml_train(
							self, 
							actor,
							critic,
							env,
							noise,
							opt,
							batch_size,
							states_dim,
							salient_states_dim,
							use_model,
							max_actions,
							planning_horizon,
							train,
							losses,
							lr_schedule,
							num_iters,
							dataset,
							verbose,
							file_location,
							file_id,
							save_checkpoints
							):
		'''Trains the model in self by using the actor-critic formulation of PAML, specifically for DDPG
		(Lillicrap et al. 2015)

		Args:
			losses (list) : Not used but could add functionality of saving losses,
			actor: policy function
			critic: value function estimator
			env: environment used
			noise: OUNoise class instant
			opt: optimizer for model training
			batch_size (int): batch size for model training
			states_dim (int)
			salient_states_dim (int)
			use_model (bool): Whether to use the model to generate virtual samples or only the real data
			max_actions (int): Number of actions in a trajectory (= number of states - 1)
			planning_horizon (int): planning horizon used in planning. In this method this value is only used for book-
				keeping purposes
			train (bool): Whether to train model, if False return validation loss
			lr_schedule (scheduler): learning rate scheduler for opt
			num_iters (int): number of training iterations
			dataset (ReplayMemory): the collection of transitions sampled from the environment
			verbose (bool)
			save_checkpoints (bool): save training checkpoints if True
			file_location (str): Directory where to save checkpoints and loss data
			file_id (str): what to append to end of file name for saved info

		Returns:
			loss: value of the final loss at end of training
		'''

		best_loss = 15000
		env_name = env.spec.id
		R_range = planning_horizon
		num_iters = num_iters if train else 1
		noise.reset()
		prev_grad = torch.zeros(count_parameters(self)).double()

		for i in range(num_iters):
			batch = dataset.sample(batch_size)

			true_states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			true_states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			# true_rewards_tensor = torch.tensor([samp.reward for samp in batch]).double().to(device).unsqueeze(1)
			true_actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

			#calculate true gradients
			actor.zero_grad()

			#compute loss for actor
			true_policy_loss = -critic(true_states_next, actor.sample_action(true_states_next))

			true_term = true_policy_loss.mean()
			
			true_pe_grads = torch.DoubleTensor()
			true_pe_grads_attached = grad(true_term.mean(), actor.parameters(), create_graph=True)
			for g in range(len(true_pe_grads_attached)):
				true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

			step_state = torch.cat((true_states_prev, true_actions_tensor),dim=1).unsqueeze(1)
			if use_model:
				model_x_curr, model_x_next, model_a_list, _, _ = self.unroll(step_state, actor, states_dim, None,
																			 steps_to_unroll=1,
																			 continuous_actionspace=True,
																			 use_model=True,
																			 salient_states_dim=states_dim,
																			 noise=noise, env=env)
				#salient_states_dim, noise=noise, env=env)
				model_x_curr = model_x_curr.squeeze(1).squeeze(1)
				model_x_next = model_x_next.squeeze(1).squeeze(1)
			else:
				model_batch = dataset.sample(batch_size)
				model_x_curr = torch.tensor([samp.state for samp in model_batch]).double().to(device)
				model_x_next = torch.tensor([samp.next_state for samp in model_batch]).double().to(device)
				# model_r_list = torch.tensor([samp.reward for samp in model_batch]).double().to(device).unsqueeze(1)
				model_a_list = torch.tensor([samp.action for samp in model_batch]).double().to(device)
			# model_policy_loss = -critic(model_x_next[:,:salient_states_dim], actor.sample_action(model_x_next))
			# model_policy_loss = -critic(model_x_next[:,:salient_states_dim], actor.sample_action(
			# 																	model_x_next[:,:salient_states_dim]))
			# model_policy_loss = -critic(model_x_next, actor.sample_action(model_x_next[:,:salient_states_dim]))
			model_policy_loss = -critic(model_x_next, actor.sample_action(model_x_next))
			model_term = model_policy_loss.mean()
			model_pe_grads = torch.DoubleTensor()
			model_pe_grads_split = grad(model_term.mean(),actor.parameters(), create_graph=True)
			for g in range(len(model_pe_grads_split)):
				model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))
			paml_loss = torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum())
			loss = paml_loss
			#update model
			if train: 
				opt.zero_grad()
				loss.backward()
				
				curr_grad = torch.DoubleTensor()
				for item in list(self.parameters()):
					curr_grad = torch.cat((curr_grad,item.grad.view(-1)))

				nn.utils.clip_grad_value_(self.parameters(), 10.0)
				opt.step()

				if torch.isnan(torch.sum(self.fc1.weight.data)):
					print('weight turned to nan, check gradients')
					pdb.set_trace()

				if (i % verbose == 0) or (i == num_iters - 1):
					print("LR: {:.5f} | batch_num: {:5d} | COS grads: {:.5f} | critic ex val: {:.3f} | paml_loss: {:.5f}"
							.format(opt.param_groups[0]['lr'], i, nn.CosineSimilarity(dim=0)(curr_grad, prev_grad),
									true_policy_loss.mean().data.cpu(), loss.data.cpu()))
				prev_grad = curr_grad
			else: 
				print("-----------------------------------------------------------------------------------------------------")
				print("Validation_loss model: {} | R_range: {:3d} | batch_num: {:5d} | average validation paml_loss = {:.7f}"
					  .format(use_model, R_range, i, loss.data.cpu()))
				print("-----------------------------------------------------------------------------------------------------")
			
		lr_schedule.step()
		# if train and saved:
		# 	self.load_state_dict(torch.load(os.path.join(file_location,
		# 	'act_model_paml_state{}_salient{}_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'.format(states_dim,
		# 	salient_states_dim, train, env_name, R_range, max_actions + 1, file_id)), map_location=device))
		if save_checkpoints:
			torch.save(self.state_dict(), os.path.join(file_location,
										'model_paml_checkpoint_state{}_salient{}_actorcritic_{}_horizon{}_traj{}_{}.pth'
												   .format(states_dim, salient_states_dim, env_name, planning_horizon,
														   max_actions+1, file_id)))

		return loss.data.cpu()

	def train_mle(self, pe, state_actions, states_next, epochs, max_actions, R_range, opt, env_name,
				  losses, states_dim, salient_states_dim, file_location, file_id, save_checkpoints=False, verbose=10):
		'''Trains model with the squared error distance between states objective. This method works for multiple steps
		or whole trajectories.

		Args:
			pe: policy estimator
			state_actions (batch_size, number of steps, states_dim + actions_dim):
										concatenated states and actions to unroll model from, given for multiple steps
			states_next (batch_size, number of steps, states_dim):
			epochs (int): number of iterations to train model for
			max_actions (int): number of actions in length of whole trajectories
			R_range (int): the number of steps to unroll model for calculating loss
			opt: model optimizer
			env_name (str): name of environment
			losses (list): list of all model losses for bookkeeping
			states_dim (int): dimension of observations/states
			salient_states_dim (int): dimension of relevant dimensions
			file_location (str): directory for saving model checkpoints and training stats
			file_id (str): string to append to end of file names for saving
			save_checkpoints (bool): if True, save model checkpoints
			verbose (bool)

		Returns:
			final model loss at end of training
		'''
		best_loss = 1000
		for i in range(epochs):
			opt.zero_grad()
			squared_errors = torch.zeros_like(states_next)
			step_state = state_actions.to(device)
			for step in range(R_range - 1):
				next_step_state = self.forward(step_state)
				squared_errors += F.pad(input=(states_next[:,step:,:] - next_step_state)**2, pad=(0,0,step,0,0,0),
										mode='constant', value=0)
				shortened = next_step_state[:,:-1,:]
				# a = pe.sample_action(torch.DoubleTensor(shortened[:,:,:states_dim]))	
				a = state_actions[:, step+1:, states_dim:]
				step_state = torch.cat((shortened,a),dim=2)
			model_loss = torch.mean(squared_errors)#torch.mean(torch.sum(squared_errors,dim=2))
			if model_loss.detach().data.cpu() < best_loss and save_checkpoints:
				torch.save(self.state_dict(), os.path.join(file_location,
										'model_mle_checkpoint_state{}_salient{}_reinforce_{}_horizon{}_traj{}_{}.pth'
																.format(states_dim, salient_states_dim, env_name,
																		R_range, max_actions + 1, file_id)))
				best_loss = model_loss.detach().data.cpu()
			if i % verbose == 0:
				print("Epoch: {}, negloglik  = {:.7f}".format(i, model_loss.data.cpu()))
			model_loss.backward()
			opt.step()
			losses.append(model_loss.data.cpu())
		return model_loss


	def general_train_mle(self, pe, dataset, validation_dataset, states_dim, salient_states_dim,
						  epochs, max_actions, opt, env_name, losses, batch_size, file_location,
						  file_id, save_checkpoints=False, verbose=20, lr_schedule=None, global_step=0):
		'''This method also uses the squared distance objective, but is only used for one-step comparisons and is thus
		much simpler than train_mle(). This method either trains the model for a set number of iterations or terminates
		training early if the validation loss starts to go up instead of down and returns.

		Args:
			pe: policy estimator
			dataset (ReplayMemory): the collection of transitions sampled from the environment for training
			validation_dataset (ReplayMemory): the collection of transitions sampled from the environment for validation
			states_dim (int)
			salient_states_dim (int)
			epochs (int): number of iterations to train model, unless validation loss starts to go up, in which case
						  training is stopped
			max_actions (int): total number of actions in a trajectory
			opt: optimizer for model
			env_name (str): name of environment
			losses (list): not implemented, but in future could be used for saving model losses
			batch_size (int)
			file_location (str): directory for saving model checkpoints and training stats
			file_id (str): string to append to end of file names for saving
			save_checkpoints (bool): if True, save model checkpoints
			verbose (bool)
			lr_schedule: scheduler for learning rate of model optimizer
			global_step (int): for record-keeping. Refers to iteration from the main loop of calling function

		Returns:
			final mode loss at end of training
		'''
		best_loss = 1000
		# self.mean_states, self.std_states, self.mean_deltas, self.std_deltas, self.mean_actions, self.std_actions =
		# compute_normalization(dataset)
		# val_mean_states, val_std_states, val_mean_deltas, val_std_deltas, val_mean_actions, val_std_actions =
		# compute_normalization(validation_dataset)
		val_model_losses = [np.inf]
		for i in range(epochs):
			#sample from dataset 
			batch = dataset.sample(batch_size)
			states_prev = torch.tensor([samp.state for samp in batch]).double().to(device)
			states_next = torch.tensor([samp.next_state for samp in batch]).double().to(device)
			actions_tensor = torch.tensor([samp.action for samp in batch]).double().to(device)

			# normalize states and actions
			states_norm = states_prev #(states_prev - self.mean_states) / (self.std_states + 1e-7)
			acts_norm = actions_tensor #(actions_tensor - self.mean_actions) / (self.std_actions + 1e-7)
			# normalize the state differences
			# deltas_states_norm = ((states_next - states_prev) - self.mean_deltas) / (self.std_deltas + 1e-7)
			step_state = torch.cat((states_norm, acts_norm), dim = 1).to(device)

			# step_state = torch.cat((states_prev, actions_tensor), dim=1).to(device)
			model_next_state_delta = self.forward(step_state)

			# squared_errors = ((states_next - states_prev) - model_next_state_delta)**2
			deltas_states_norm = states_next - states_prev
			squared_errors = (deltas_states_norm - model_next_state_delta)**2

			#step_state = torch.cat((next_step_state,a),dim=1)
			model_loss = torch.mean(torch.sum(squared_errors,dim=1))

			if model_loss.detach().data.cpu() < best_loss and save_checkpoints:
				torch.save(self.state_dict(), os.path.join(file_location,
										'model_mle_checkpoint_state{}_salient{}_actorcritic_{}_horizon{}_traj{}_{}.pth'
														   .format(states_dim, salient_states_dim, env_name, 1,
																   max_actions + 1, file_id)))
				best_loss = model_loss.detach().data.cpu()

			if (i % verbose == 0) or (i == epochs - 1):
				#calculate validation loss 	
				val_batch = validation_dataset.sample(len(validation_dataset))
				val_states_prev = torch.tensor([samp.state for samp in val_batch]).double().to(device)
				val_states_next = torch.tensor([samp.next_state for samp in val_batch]).double().to(device)
				val_actions_tensor = torch.tensor([samp.action for samp in val_batch]).double().to(device)

				# normalize states and actions
				val_states_norm = val_states_prev#(val_states_prev - val_mean_states) / (val_std_states + 1e-7)
				val_acts_norm = val_actions_tensor#(val_actions_tensor - val_mean_actions) / (val_std_actions + 1e-7)
				# normalize the state differences
				val_deltas_states_norm = val_states_next - val_states_prev#((val_states_next - val_states_prev) -
																			# val_mean_deltas) / (val_std_deltas + 1e-7)
				val_step_state = torch.cat((val_states_norm, val_acts_norm), dim = 1).to(device)

				with torch.no_grad():
					val_model_next_state_delta = self.forward(val_step_state)
				
				val_squared_errors = (val_deltas_states_norm - val_model_next_state_delta)**2
				val_model_losses.append(torch.mean(torch.sum(val_squared_errors,dim=1)).data)

				if len(val_model_losses) >= 25 and val_model_losses[-1] >= (val_model_losses[1] + val_model_losses[1]*0.05):
					return model_loss	
				elif len(val_model_losses) >= 25:
					val_model_losses = val_model_losses[1:]
				print("Iter: {} | LR: {:.7f} | salient_loss: {:.4f} | irrelevant_loss: {:.4f} | negloglik  = {:.7f} | validation loss : {:.7f}"
					  .format(i+epochs*global_step, opt.param_groups[0]['lr'],
							  torch.mean(torch.sum(squared_errors[:,:salient_states_dim],dim=1)).detach().data.cpu(),
							  torch.mean(torch.sum(squared_errors[:,salient_states_dim:],dim=1)).detach().data.cpu(),
							  model_loss.data.cpu(),
							  val_model_losses[-1]))
			opt.zero_grad()
			model_loss.backward()
			# nn.utils.clip_grad_value_(self.parameters(), 100.0)
			opt.step()
			# losses.append(model_loss.data.cpu())
		if lr_schedule is not None:
			lr_schedule.step()
		del val_squared_errors, val_model_losses
		return model_loss

	def predict(self, states, actions):
		'''Predicts next states given states and actions using model in self. Only predicts for one step

		Args:
			states (batch_size, states_dim): torch Tensor of states
			actions (batch_size, actions_dim): Tensor of actions

		Returns:
			Tensor of next states as predicted by model in self
		'''
		# normalize the states and actions
		states_norm = states#(states - self.mean_states) / (self.std_states + 1e-7)
		act_norm = actions#(actions - self.mean_actions) / (self.std_actions + 1e-7)
		# concatenate normalized states and actions
		states_act_norm = torch.cat((states_norm, act_norm), dim=1)
		# predict the deltas between states and next states
		deltas = self.forward(states_act_norm)
		# calculate the next states using the predicted delta values and denormalize
		return deltas + states_norm #deltas * self.std_deltas + self.mean_deltas + states