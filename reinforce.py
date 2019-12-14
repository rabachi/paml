import torch
from torch import optim
from torch.autograd import grad
import pdb
from models import *
from networks import *
from utils import *

device = 'cpu'

def paml_train(P_hat,
				pe,
				model_opt,
				num_episodes,
				num_starting_states,
				states_dim,
				salient_states_dim,
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
				verbose,
				save_checkpoints,
				file_location,
				file_id,
				extra_dims_stable
				):
	'''This function is used for multiple purposes:
		1. To train the model P_hat using the PAML loss formulated for REINFORCE (Williams, 1992) as the planner. This
			happens if the train flag and use_model flags are set to True
		2. To check the PAML loss of the model validation data: train flag set to False and use_model set to True
		3. To check the PAML loss of the true environment (since we have access to the linear system here): train flag
			set to False and use_model set to False
		In general, this function calculate the PAML loss, and thus can either be used for training or validation

	Args:
		P_hat (DirectEnvModel): Model to be trained
		pe (Policy instant): policy estimator
		model_opt: optimizer of model
		num_episodes (int): number of episodes to roll out in true environment per starting state (mostly set to 1)
		num_starting_states (int): number of starting states for true environment trajectories
		states_dim (int): full state/observation dimension
		salient_states_dim (int): dimension of relevant state dimensions
		actions_dim (int): dimension of actions
		use_model (bool): see above for usage. Set to true if we want to calculate PAML loss on model.
		discount (float)
		max_actions (int): number of actions in length of whole trajectories
		train (bool): If true, train model, else calculate validation loss
		A_numpy (numpy array): this array specifies the true dynamics for the linear dynamic system
		lr_schedule: learning rate scheduler for model optimizer
		num_iters (int): number of training iterations for the model, this is used every time the model is trained
		losses (list): list of training losses for record keeping
		true_r_list (Tensor, (batch_size, number of steps, 1)): rewards collected from true env
		true_x_curr (Tensor, (batch_size, number of steps, states_dim)): states collected from true env
		true_x_next (Tensor, (batch_size, number of steps, states_dim)): next states collected from true env
		true_a_list (Tensor, (batch_size, number of steps, actions_dim)): actions collected from true env
		true_a_prime_list (Tensor, (batch_size, number of steps, actions_dim)): next actions collected from true env
		file_location (str): directory for saving model checkpoints and training stats
		file_id (str): string to append to end of file names for saving
		save_checkpoints (bool): if True, save model checkpoints
		verbose (int)
		extra_dims_stable (bool): whether the added irrelevant dimensions have stable dynamics

	Returns:
		P_hat: the trained model
	'''
	best_loss = 15000
	env_name = 'lin_dyn'
	R_range = max_actions
	unroll_num = 1
	ell = 0
	num_iters = num_iters if train else 1
	#calculate true gradients
	pe.zero_grad()
	true_log_probs = get_selected_log_probabilities(pe, true_x_next, true_a_prime_list).squeeze()
	true_returns = true_r_list#discount_rewards(true_r_list[:,ell, 1:], discount, center=False, batch_wise=True)
	true_term = true_log_probs * true_returns

	true_pe_grads = torch.DoubleTensor()
	for st in range(num_starting_states):
		true_pe_grads_attached = grad(true_term[st*num_episodes:num_episodes*(st+1)].mean(),pe.parameters(),
									  create_graph=True)
		for g in range(len(true_pe_grads_attached)):
			true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

	# COMMENTED OUT: alternate way of calculating model_pe_grads, however, doesn't work as well, why?
	# true_pe_grads = torch.DoubleTensor()
	# true_pe_grads_attached = grad(true_term.mean(),pe.parameters(), create_graph=True)
	# for g in range(len(true_pe_grads_attached)):
	# 	true_pe_grads = torch.cat((true_pe_grads,true_pe_grads_attached[g].detach().view(-1)))

	step_state = torch.cat((true_x_curr[:,:,0],true_a_list[:,:,0]),dim=2)
	policy_states_dim = salient_states_dim #if not extra_dims_stable else states_dim
	for i in range(num_iters):
		pe.zero_grad()
		#calculate model gradients
		model_x_curr, model_x_next, model_a_list, model_r_list, model_a_prime_list = P_hat.unroll(
			step_state[:,:unroll_num,:], pe, states_dim, A_numpy, steps_to_unroll=R_range, continuous_actionspace=True,
			use_model=use_model, salient_states_dim=policy_states_dim, extra_dims_stable=extra_dims_stable,
			using_delta=False)

		model_returns = discount_rewards(model_r_list[:,ell, 1:], discount, center=False, batch_wise=True)
		model_log_probs = get_selected_log_probabilities(pe, model_x_next, model_a_prime_list).squeeze()
		#model_term = torch.einsum('ijk,ijl->ik', [model_log_probs,  model_returns])
		model_term = model_log_probs * model_returns

		#COMMENTED OUT: alternate way of calculating model_pe_grads, however, doesn't work as well, why?
		# model_pe_grads = torch.DoubleTensor()
		# model_pe_grads_split = grad(model_term.mean(),pe.parameters(), create_graph=True)
		# for g in range(len(model_pe_grads_split)):
		# 	model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))
		model_pe_grads = torch.DoubleTensor()
		for st in range(num_starting_states):
			model_pe_grads_split = list(grad(model_term[num_episodes*st:num_episodes*(st+1)].mean(),
											 pe.parameters(), create_graph=True))
			for g in range(len(model_pe_grads_split)):
				model_pe_grads = torch.cat((model_pe_grads,model_pe_grads_split[g].view(-1)))
		loss = torch.sqrt((true_pe_grads-model_pe_grads).pow(2).sum()/num_starting_states)
		if loss.detach().cpu() < best_loss and use_model:
			#Save model and losses so far
			if save_checkpoints:
				torch.save(P_hat.state_dict(), os.path.join(file_location,
														'model_paml_checkpoint_train_{}_{}_horizon{}_traj{}_{}.pth'\
														.format(train, env_name, R_range, max_actions + 1, file_id)))
				np.save(os.path.join(file_location,
									 'loss_paml_checkpoint_train_{}_{}_horizon{}_traj{}_{}'.format(
										 train, env_name, R_range, max_actions + 1, file_id)), np.asarray(losses))
			best_loss = loss.detach().cpu()

		#update model
		if train:
			pe.zero_grad()
			model_opt.zero_grad()
			loss.backward()
			nn.utils.clip_grad_value_(P_hat.parameters(), 5.0)
			model_opt.step()

			if torch.isnan(torch.sum(P_hat.fc1.weight.data)):
				print('weight turned to nan, check gradients')
				pdb.set_trace()

		losses.append(loss.data.cpu())
		lr_schedule.step()

		if train and i < 1: #and j < 1
			initial_loss = losses[0]#.data.cpu()
			print('initial_loss',initial_loss)

		if train:
			if (i % verbose == 0):
				if save_checkpoints:
					np.save(os.path.join(file_location,'loss_paml_checkpoint_train_{}_{}_horizon{}_traj{}_{}'.format(
						train, env_name, R_range, max_actions + 1,file_id)), np.asarray(losses))

				print("LR: {:.10} | R_range: {:3d} | batch_num: {:5d} | policy sigma^2: {:.3f} | paml_loss: {:.7f}"\
					  .format(model_opt.param_groups[0]['lr'], R_range, i, pe.sigma_sq.mean(dim=1)[0][0].detach().cpu(),
							  loss.data.cpu()))
		else:
			print("-------------------------------------------------------------------------------------------")
			print("Val loss model: {} | R_range: {:3d} | batch_num: {:5d} | policy sig^2: {:.3f} | avg val paml_loss = {:.7f}"\
				  .format(use_model, R_range, i, pe.sigma_sq.mean(dim=1)[0][0].cpu(), loss.data.cpu()))
			print("-------------------------------------------------------------------------------------------")

	return P_hat


def reinforce(
			policy_estimator,
			A_numpy,
			P_hat,
			num_episodes,
			states_dim,
			actions_dim,
			salient_states_dim,
			R_range,
			use_model,
			optimizer,
			discount,
			true_x_curr,
			true_x_next,
			true_a_list,
			true_r_list,
			true_a_prime_list,
			file_location,
			file_id,
			save_checkpoints,
			train=True,
			verbose=100,
			all_rewards=[],
			model_type='paml',
			extra_dims_stable=False
			):
	'''This functions performs REINFORCE (Williams, 1992) on the policy_estimater using either the learned model or
	the true dynamics for the model-free version. To use the model-free version, set use_model = False.
	For the model-based version, set use_model = True.
	This function can also be used with the train flag set to False to calculate performance of policy_estimator on
	validation data.

	Args:
		policy_estimator (Policy instant): policy to be trained
		A_numpy (numpy array): this array specifies the true dynamics for the linear dynamic system
		P_hat (DirectEnvModel): Model to be used for generating virtual samples if necessary
																				(i.e. if use_model set to True))
		num_episodes (int): number of episodes to use in total for training the policy. These episodes may be
							generated by the model or true environment
		states_dim (int): full state/observation dimension
		salient_states_dim (int): dimension of relevant state dimensions
		actions_dim (int): dimension of actions
		R_range (int): planning horizon, since we're using REINFORCE, this is permanently set to length of trajectory
		use_model (bool): set to True if would like to use model to perform planning, False if we would like to use
							only real data (model-free)
		optimizer: optimizer for policy_estimator
		discount (float)
		true_r_list (Tensor, (batch_size, number of steps, 1)): rewards collected from true env
		true_x_curr (Tensor, (batch_size, number of steps, states_dim)): states collected from true env
		true_x_next (Tensor, (batch_size, number of steps, states_dim)): next states collected from true env
		true_a_list (Tensor, (batch_size, number of steps, actions_dim)): actions collected from true env
		true_a_prime_list (Tensor, (batch_size, number of steps, actions_dim)): next actions collected from true env
		file_location (str): directory for saving model checkpoints and training stats
		file_id (str): string to append to end of file names for saving
		save_checkpoints (bool): if True, save model checkpoints
		verbose (int)
		train (bool): Set to True if policy_estimator should be trained, False otherwise
		all_rewards (list): list to keep track of rewards collected by policy as it trains
		model_type (str): either 'mle', 'paml', or 'random' for no training at all
		extra_dims_stable (bool): True if the extra dimensions of noise should have stable dynamics

	Returns:
		best_pe: final trained policy estimator
	'''
	env_name = 'lin_dyn'
	best_loss = 1000
	# all_rewards = []
	unroll_num = 1
	# unroll_num = R_range
	ell = 0
	max_actions = R_range
	batch_size = 5 if train else num_episodes
	batch_states = torch.zeros((batch_size, max_actions, states_dim)).double()
	batch_actions = torch.zeros((batch_size, max_actions, actions_dim)).double()
	batch_returns = torch.zeros((batch_size, max_actions, 1)).double()
	print(use_model)
	policy_states_dim = salient_states_dim #if not extra_dims_stable else states_dim
	for ep in range(int(num_episodes/batch_size)):
		with torch.no_grad():
			step_state = torch.zeros((batch_size, max_actions, states_dim+actions_dim)).double()
			for b in range(batch_size):
				x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
				step_state[b,:unroll_num,:states_dim] = torch.from_numpy(x_0).double()

			step_state[:,:unroll_num,states_dim:] = \
				policy_estimator.sample_action(step_state[:,:unroll_num,:states_dim])

			model_x_curr, model_x_next, model_a_list, model_r_list, model_a_prime_list = P_hat.unroll(
				step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy, steps_to_unroll=R_range,
				continuous_actionspace=True, use_model=use_model, salient_states_dim=policy_states_dim,
				extra_dims_stable=extra_dims_stable, using_delta=False)

			all_rewards.extend(model_r_list[:,ell,:-1].contiguous().view(-1,max_actions).sum(dim=1).tolist())

		# batch_states = torch.cat((model_x_curr,true_x_curr),dim=0)
		# batch_actions = torch.cat((model_a_list,true_a_list),dim=0)
		# batch_rewards = torch.cat((model_r_list, true_r_list), dim=0)
		batch_states = model_x_curr
		batch_actions = model_a_list
		batch_rewards = model_r_list
		batch_returns = discount_rewards(batch_rewards[:,ell,:-1], discount, center=True, batch_wise=True)

		model_log_probs = get_selected_log_probabilities(policy_estimator, batch_states, batch_actions).squeeze()

		model_term = model_log_probs * batch_returns
		loss = -model_term.mean()

		if train:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if loss < best_loss:
			if save_checkpoints:
				torch.save(policy_estimator.state_dict(), os.path.join(file_location,
																'policy_reinforce_use_model_{}_horizon{}_traj{}_{}.pth'\
																.format(use_model, R_range, max_actions + 1, file_id)))
			best_loss= loss
			best_pe = policy_estimator

		if (ep % verbose == 0) or (ep == int(num_episodes/batch_size) - 1):
			print("Ep: {:4d} | policy sigma^2: {:.3f} | Average of last 10 rewards: {:.3f}".format(
				ep,policy_estimator.sigma_sq.mean(dim=1)[0][0].detach().cpu(),
				sum(all_rewards[-10:])/len(all_rewards[-10:])))
			#save all_rewards
			if not use_model:
				np.save(os.path.join(file_location,
					'model_free_state{}_salient{}_rewards_reinforce_checkpoint_use_model_{}_lin_dyn_horizon{}_traj{}_{}.npy'\
					.format(states_dim, salient_states_dim, use_model, R_range, max_actions + 1,
							file_id)), np.asarray(all_rewards))
			if len(all_rewards) >= 100000: #dump some data to save space
				all_rewards = []
	return best_pe


def plan_and_train(P_hat, policy_estimator, model_opt, policy_optimizer, num_starting_states, num_episodes, states_dim,
				   salient_states_dim, actions_dim, discount, max_actions, A_numpy, lr_schedule, num_iters, losses,
				   rewards_log, verbose, num_virtual_episodes, file_location, file_id, save_checkpoints, model_type,
				   extra_dims_stable):
	'''Implements the Dyna loop in a sequential manner. The loops repeats the following steps for
		total_eps/num_starting_states number of steps:
		1. Collect trajectories from the environment (linear dynamics)
		2. Check PAML loss on the true dynamics since in the linear dynamics case we have access to it. Also check it
			on the model
		3. Train the model either using PAML loss or MLE loss
		4. Train the policy using REINFORCE (Williams, 1992) with data from model and real collected trajectories
		5. Check how much the new policy changes the PAML losses calculated in step 2.

	Args:
		P_hat (DirectEnvModel): Model to train
		policy_estimator (Policy): Stochastic policy estimator
		model_opt: optimizer for model training
		policy_optimizer: optimizer for policy training
		num_starting_states (int): number of starting states for true environment trajectories
		num_episodes (int): number of episodes to roll out in true environment per starting state (mostly set to 1)
		states_dim (int): full state/observation dimension
		salient_states_dim (int): dimension of relevant state dimensions
		actions_dim (int): dimension of actions
		discount (float)
		max_actions (int): Number of actions in a trajectory (= number of states - 1)
		A_numpy (numpy array): this array specifies the true dynamics for the linear dynamic system
		lr_schedule (scheduler): learning rate scheduler for opt
		num_iters (int): number of training iterations for the model, this is used every time the model is trained
		losses (list): list of training losses for record keeping
		rewards_log (list): list for keeping track of rewards throughout training
		verbose (int)
		num_virtual_episodes (int): number of episodes to generate from the model
		file_location (str): directory for saving model checkpoints and training stats
		file_id (str): string to append to end of file names for saving
		save_checkpoints (bool): if True, save model checkpoints
		model_type (str): either 'mle', 'paml', or 'random' for no training at all
		extra_dims_stable (bool): True if the extra dimensions of noise should have stable dynamics

	Returns:
		None
	'''
	batch_size = num_starting_states * num_episodes
	R_range = max_actions
	kwargs = {
				'P_hat'              : P_hat,
				'pe'                 : policy_estimator,
				'model_opt' 				 : model_opt,
				'num_episodes'		 : num_episodes,
				'num_starting_states': num_starting_states,
				'states_dim'		 : states_dim,
				'discount'			 : discount,
				'max_actions'		 : max_actions,
				'A_numpy'			 : A_numpy,
				'lr_schedule'		 : lr_schedule,
				'num_iters'          : num_iters,
				'verbose'			 : verbose,
				'salient_states_dim' : salient_states_dim,
				'save_checkpoints'   : save_checkpoints,
				'file_location'		 : file_location,
				'file_id'			 : file_id,
				'extra_dims_stable'  : extra_dims_stable
			}
	unroll_num = 1
	true_rewards = []
	total_eps = 10000.
	global_step = 0
	skip_next_training = False
	policy_states_dim = salient_states_dim #if not extra_dims_stable else states_dim
	while(global_step <= total_eps/num_starting_states):
		if (global_step >= total_eps/num_starting_states - 5) or not skip_next_training:
			with torch.no_grad():
				#generate training data
				train_step_state = torch.zeros((batch_size, unroll_num, states_dim+actions_dim)).double()
				for b in range(num_starting_states):
					x_0 = 2*(np.random.random(size=(states_dim,)) - 0.5)
					train_step_state[b*num_episodes:num_episodes*(b+1),:unroll_num,:states_dim] = \
						torch.from_numpy(x_0).double()
				train_step_state[:,:unroll_num,states_dim:] = policy_estimator.sample_action(
					train_step_state[:,:unroll_num,:states_dim])
				train_true_x_curr, train_true_x_next, train_true_a_list, train_true_r_list, train_true_a_prime_list = \
					P_hat.unroll(train_step_state[:,:unroll_num,:], policy_estimator, states_dim, A_numpy,
								 steps_to_unroll=R_range, continuous_actionspace=True, use_model=False,
								 salient_states_dim=policy_states_dim, extra_dims_stable=extra_dims_stable,
								 using_delta=False)
				train_true_returns = discount_rewards(train_true_r_list[:,0,1:], discount=discount,
													  batch_wise=True, center=False)
				print("Checking policy performance on true dynamics ...", train_true_r_list.squeeze().sum(dim=1).mean())
				true_rewards.append(train_true_r_list.squeeze().sum(dim=1).mean())

			np.save(os.path.join(file_location,
		'{}_state{}_salient{}_rewards_reinforce_checkpoint_use_model_False_{}_horizon{}_traj{}_{}Model_hidden{}_{}.npy'\
								 .format(model_type, states_dim, salient_states_dim, 'lin_dyn', R_range,
										 max_actions + 1, P_hat.model_size, P_hat.hidden_size, file_id)),
																							np.asarray(true_rewards))
		#check paml loss on true dynamics
		loss_false = []
		kwargs['train'] = False
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['true_r_list'] = train_true_returns
		kwargs['true_x_curr'] = train_true_x_curr
		kwargs['true_x_next'] = train_true_x_next
		kwargs['true_a_list'] = train_true_a_list
		kwargs['true_a_prime_list'] = train_true_a_prime_list
		# kwargs['step_state'] = train_step_state
		kwargs['losses'] = loss_false
		kwargs['use_model'] = False
		paml_train(**kwargs)

		#check paml loss on model, before training model
		kwargs['train'] = True
		kwargs['num_episodes'] = num_episodes
		kwargs['num_starting_states'] = num_starting_states
		kwargs['true_r_list'] = train_true_returns
		kwargs['true_x_curr'] = train_true_x_curr
		kwargs['true_x_next'] = train_true_x_next
		kwargs['true_a_list'] = train_true_a_list
		kwargs['true_a_prime_list'] = train_true_a_prime_list
		# kwargs['step_state'] = train_step_state
		kwargs['losses'] = losses
		kwargs['num_iters'] = num_iters
		kwargs['use_model'] = True
		kwargs['P_hat'] = P_hat

		if model_type == 'mle':
			state_actions = torch.cat((train_true_x_curr.squeeze(), train_true_a_list.squeeze()), dim=2)
			P_hat.train_mle(policy_estimator, state_actions, train_true_x_next.squeeze(), num_iters, max_actions,
							R_range, model_opt, "lin_dyn", losses, states_dim, salient_states_dim, file_location,
							file_id, save_checkpoints=save_checkpoints)
		elif model_type == 'paml':
			kwargs['P_hat'] = P_hat
			kwargs['use_model'] = True
			if not skip_next_training:
				P_hat = paml_train(**kwargs)
			else:
				skip_next_training = False
		elif model_type == 'random':
			P_hat = DirectEnvModel(states_dim, actions_dim, MAX_TORQUE).double()
		else:
			raise NotImplementedError

		if global_step % verbose == 0:
			np.save(os.path.join(file_location,'reinforce_loss_model_{}_env_{}_state{}_salient{}_horizon{}_traj{}_{}'
								 .format(model_type, 'lin-dyn', states_dim, salient_states_dim, R_range,
										 max_actions + 1, file_id)), np.asarray(losses))
		#check paml loss with current policy, after model training
		kwargs['train'] = False
		kwargs['use_model'] = True
		kwargs['P_hat'] = P_hat
		kwargs['losses'] = []
		paml_train(**kwargs)

		use_model = True
		policy_estimator = reinforce(policy_estimator,
									A_numpy,
									P_hat,
									num_virtual_episodes,
									states_dim,
									actions_dim,
									salient_states_dim,
									R_range,
									True,
									policy_optimizer,
									discount,
									train_true_x_curr,
									train_true_x_next,
									train_true_a_list,
									train_true_returns,
									train_true_a_prime_list,
									file_location,
									file_id,
									save_checkpoints,
									train=True,
									verbose=10,
									all_rewards=rewards_log,
									model_type=model_type,
									extra_dims_stable=extra_dims_stable
									)
		paml_model = []
		paml_true = []
		#check paml_loss with the new policy
		kwargs['train'] = False
		kwargs['use_model'] = True
		kwargs['losses'] = paml_model
		kwargs['P_hat'] = P_hat
		kwargs['pe'] = policy_estimator
		paml_train(**kwargs)

		kwargs['train'] = False
		kwargs['use_model'] = False
		kwargs['losses'] = paml_true
		kwargs['P_hat'] = P_hat
		kwargs['pe'] = policy_estimator
		paml_train(**kwargs)

		global_step += 1


def main(
			max_torque,
			real_episodes,
			virtual_episodes,
			num_eps_per_start,
			num_iters,
			discount,
			max_actions,
			states_dim,
			salient_states_dim,
			initial_model_lr,
			model_type,
			file_location,
			file_id,
			save_checkpoints_training,
			verbose,
			extra_dims_stable,
			model_size,
			rs,
		):
	'''This function initializes the model and policy and some other variables, then calls either REINFORCE
	(Williams 1992) directly for the model_free version or calls the plan_and_train function for the model-based ones.

	Args:
		max_torque (float): maximum action value that can be applied in the environment
		real_episodes (int): number of starting states for true environment trajectories
		virtual_episodes (int): number of episodes to generate from the model
		num_eps_per_start (int): number of episodes to roll out in true environment per starting state (mostly set to 1)
		num_iters (int): number of training iterations for the model, this is used every time the model is trained
		discount (float)
		max_actions (int): Number of actions in a trajectory (= number of states - 1)
		states_dim (int): full state/observation dimension
		salient_states_dim (int): dimension of relevant state dimensions
		initial_model_lr (float): Initial model learning rate to use. This learning rate is gradually decreased
								  according to a scheduler
		model_type (str): either 'mle', 'paml', or 'random' for no training at all
		file_location (str): directory for saving model checkpoints and training stats
		file_id (str): string to append to end of file names for saving
		save_checkpoints (bool): if True, save model checkpoints
		batch_size (int): batch size for training model and policy
		verbose (int)
		extra_dims_stable (bool): True if the extra dimensions of noise should have stable dynamics
		model_size (str): 'small', 'constrained', or 'nn' to specify the size and type of the model
		rs (int): random seed

	Returns:
		None
	'''
	torch.manual_seed(rs)
	np.random.seed(rs)

	MAX_TORQUE = max_torque#2.0
	num_episodes = num_eps_per_start#1

	if model_type == 'model_free':
		plan = True
	else:
		plan = False

	num_starting_states = real_episodes if not plan else 1000000
	actions_dim = salient_states_dim#states_dim
	R_range = max_actions

	use_model = True
	continuous_actionspace = True

	action_multiplier = 0.0
	policy_estimator = Policy(states_dim, actions_dim, continuous=continuous_actionspace, std=-3.0,
							  max_torque=MAX_TORQUE, small=False)
	policy_estimator.double()
	policy_optimizer = optim.Adam(policy_estimator.parameters(), lr=0.0001)
	# policy_estimator.load_state_dict(torch.load('policy_reinforce_use_model_True_horizon20_traj21.pth',
	# map_location=device))
	if plan:
		model_size = 'small'
	P_hat = DirectEnvModel(states_dim, actions_dim, MAX_TORQUE, model_size=model_size, limit_output=True)
	P_hat.double()
	# P_hat.load_state_dict(torch.load('trained_model_paml_lindyn_horizon5_traj6.pth', map_location=device))
	# P_hat.load_state_dict(torch.load('1model_paml_checkpoint_train_False_lin_dyn_horizon20_traj21_using1states.pth',
	# map_location=device))

	model_opt = optim.SGD(P_hat.parameters(), lr=initial_model_lr)
	lr_schedule = torch.optim.lr_scheduler.MultiStepLR(model_opt, milestones=[500,1200,1800], gamma=0.1)

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

	A_all[3] = np.array([[0.98, 0.  , 0.  ],
						[0.  , 0.95, 0.  ],
						[0.  , 0.  , 0.99]])

	A_all[2] = np.array([[0.9, 0.4], [-0.4, 0.9]])

	if extra_dims_stable:
		extra_dim = states_dim - salient_states_dim
		block = np.eye(extra_dim) * 0.98
		A_numpy = np.block([
						 [A_all[salient_states_dim],               	 np.zeros((salient_states_dim, extra_dim))],
						 [np.zeros((extra_dim, salient_states_dim)), block                 					  ]
						  ])
	else:
		A_numpy = A_all[salient_states_dim]

	train = True

	losses = []
	rewards_log = []
	if not plan:
		plan_and_train(P_hat, policy_estimator, model_opt, policy_optimizer, num_starting_states, num_episodes,
					   states_dim, salient_states_dim, actions_dim, discount, max_actions, A_numpy, lr_schedule,
					   num_iters, losses, rewards_log, verbose, virtual_episodes, file_location, file_id,
					   save_checkpoints_training, model_type, extra_dims_stable)

	########## REINFORCE ############
	if plan:
		batch_size = 5
		use_model = True
		reinforce(
					policy_estimator,
					A_numpy,
					P_hat,
					num_episodes*num_starting_states,
					states_dim,
					actions_dim,
					salient_states_dim,
					R_range,
					False,
					policy_optimizer,
					discount,
					None,
					None,
					None,
					None,
					None,
					file_location,
					file_id,
					save_checkpoints_training,
					train=True,
					verbose=verbose,
					all_rewards=[],
					model_type='model_free',
					extra_dims_stable=extra_dims_stable
				)
