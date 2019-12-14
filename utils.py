import torch
from torch.distributions import Categorical, Normal, MultivariateNormal
from rewardfunctions import *
import pdb

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

def count_parameters(model):
	'''Count total number of trainable parameters in argument
	Args:
		model: the function whose parameters to count
	Returns:
		int: number of trainable parameters in model
	'''
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def non_decreasing(L):
	''' Determine whether a list of numbers is in non-decreasing order
	Args:
		L: list of floats
	Returns:
		bool: if list is in non-decreasing order
	'''
	return all(x>=y for x, y in zip(L, L[1:]))


def discount_rewards(list_of_rewards, discount, center=True, batch_wise=False):
	'''
	This is a function for calculating the returns from a given list of rewards and discount factor. There is an option
	to subtract the mean of the rewards and divide by std(rewards) as a baseline. There is also an option to calculate
	returns in a batch_wise manner if list_of_rewards is given as a Tensor.
	Args:
		list_of_rewards (type may be list, numpy array, or Tensor):
										list of rewards collected from interacting with an environment or model
		discount (float): discount factor to calculate returns
		center (bool): if True, the returns are mean-subtracted and divided by the standard deviation of the rewards
						set to True for doing REINFORCE. Otherwise, for doing PAML, it should be False.
		batch_wise (bool): if list_of_rewards is a Tensor, batch_wise = True will calculate the returns in a batchwise
							manner, by consider each row along the first axis a separate trajectory.
							Otherwise it assumes that the rewards are given for one trajectory.

	Returns:
		returns of list_of_rewards by using discount
	'''
	if isinstance(list_of_rewards, list) or isinstance(list_of_rewards, np.ndarray):
		list_of_rewards = np.asarray(list_of_rewards, dtype=np.float32)
		r = np.zeros_like(list_of_rewards)

		for i in range(len(list_of_rewards)):
			r = r + discount**i * np.pad(list_of_rewards,(0,i),'constant')[i:]

		if center:
			return torch.DoubleTensor((r - list_of_rewards.mean())/(list_of_rewards.std()+ 1e-5))
		else:
			return torch.DoubleTensor(r.copy())

	elif isinstance(list_of_rewards, torch.Tensor):
		r = torch.zeros_like(list_of_rewards)
		if batch_wise:
			lim_range = list_of_rewards.shape[1]
		else:
			lim_range = list_of_rewards.shape[0]

		for i in range(lim_range):
			r = r + discount**i * shift(list_of_rewards,i, dir='up')

		if center:
			return (r - torch.mean(list_of_rewards))/(torch.std(list_of_rewards) + 1e-5)
		else:
			return r


class StableNoise(object):
	'''A class for adding different types of noise to the agent's observations'''
	def __init__(self, states_dim, salient_states_dim, param, init=1, type='random'):
		'''
		The inputs to this class will have dimension salient_states_dim and the outputs
		should have dimension states_dim
		Args:
			states_dim: total observation dims wanted
			salient_states_dim: dimension of inputs
			param (float \in (0,1]): decay rate for the noise
			init (float): initial undecayed noise is sampled from Unif(-init,init)
		'''
		self.states_dim = states_dim
		self.salient_states_dim = salient_states_dim
		self.extra_dim = self.states_dim - self.salient_states_dim
		self.param = param
		self.random_initial = 2*init*np.random.random(size = (self.extra_dim,)) - init
		self.random_w = np.random.random(size = (self.salient_states_dim, self.salient_states_dim))
		self.type = type

	def get_obs(self, obs, t=0):
		'''
		Add extra dimensions to obs. If type is 'redundant', the noise added will be determined
		based on self.states_dim as follows:
		 	1. cos(obs), 2. cos(obs), sin(obs), 3. cos(obs), sin(obs), random_matrix*obs
		If type is NOT 'redundant', the noise added will be \eta \sim Unif(-init, init) * param ** t, where \eta
		is sampled once per random seed, param \in (0,1] and t is the time-step of the observation. Therefore, the
		noise added is time-decaying.

		Args:
			obs (batch x salient_states_dim): batch of observations to add extra dimensions to
			t (int): the time step of the observations (single value) to determine how much to decay noise
			type: 'redundant' adds the redundant type of noise to obs. Any other input adds the random, decaying noise

		Returns:
			new_obs (batch x states_dim): obs stacked with extra observations, dimension now equal to self.states_dim
		'''
		if self.extra_dim == 0: #no extra dims
			return obs

		if self.type=='redundant': #redundant extra dims
			extra_obs1 = np.cos(obs)
			extra_obs2 = np.sin(obs)
			extra_obs3 = self.random_w @ obs

			if self.states_dim == 2 * self.salient_states_dim:
				new_obs = np.hstack([obs, extra_obs1])
			elif self.states_dim == 3 * self.salient_states_dim:
				new_obs = np.hstack([obs, extra_obs1, extra_obs2])
			elif self.states_dim == 4 * self.salient_states_dim:
				new_obs = np.hstack([obs, extra_obs1, extra_obs2, extra_obs3])
			else:
				raise ValueError('states_dim for redundant extra dimensions should be a multiple of salient_states_dim')
		else: #random decaying extra dims
			extra_obs = self.random_initial * self.param**t
			new_obs = np.hstack([obs, extra_obs])

		return new_obs


def generate_data(env, env_val, states_dim, dataset, val_dataset, actor, train_starting_states, val_starting_states,
				  max_actions, noise, num_action_repeats, temp_data=False, reset=True, start_state=None,
				  start_noise=None, noise_type='random'):
	'''This function generates the training and validation datasets from the true environments. Since we would like
	some environments to be run for long trajectories but would still like to be able to train a model and improve our
	policy during an episode (and not wait until the end), we add a flag reset, which should be set to False if we'd
	like to continue sampling from a given start_state and start_noise (exploration noise also has a state since it is
	correlated in time). We also have two sets of environments. "env" should not be used at all between calls of this
	function if we'd like to continue trajectories. Thus, env_val is used for validation data collection.
	#TODO: this function can be written more robustly. In a way that makes sure env doesn't get used in unexpected places.

	Args:
		env (gym environment): for collecting training data
		env_val (gym environment): for collecting validation set. Different from above because we'd like to preserve the
									state that the above env reaches.
		states_dim (int): dimension of states
		dataset (ReplayMemory): training dataset to add data to
		val_dataset (ReplayMemory): validation dataset to add data to
		actor (DeterministicPolicy): policy to collect data with
		train_starting_states (int): number of episodes to collect for training set
		val_starting_states (int): number of episodes to collect for validation set, if set to None will not collect
									validation data
		max_actions (int): length of trajectory - 1 of the episodes to collect data from environment
		noise (OUNoise): exploration noise
		num_action_repeats (int): how many times to repeat each action for (not used for PAML yet)
		temp_data (bool): set to True if the data collected should be stored in the temporary part of the replay buffer
		reset (bool): True if should reset env. Otherwise set to False (to continue from start_state and start_noise if
					env hasn't been reset in-between invocations of this function)
		start_state (numpy array of size states_dim): the state from which we should start sampling from env from
												if reset = False .setting to None has the same effect as reset = True
		start_noise (OUNoise): the noise from which we should start when sampling from env if reset = False.
		noise_type (str): if 'redundant' use redundant type of noise if states_dim correct, otherwise
							use random type of noise (see StableNoise for definitions)
	Returns:
		state: last state reached while collecting real data. This can be used (if env is not reset) to continue
				trajectory from this state
		noise: snapshot of the noise state reached while collecting data. Can be used to continue trajectory using this.
	'''
	if env.spec.id != 'lin-dyn-v0':
		salient_states_dim = env.observation_space.shape[0]
	else:
		salient_states_dim = 2 #for lin_dyn, it is fixed at 2. Observation_space would give dimension of actual obs
								#which is equivalent to states_dim not salient_states_dim
	stablenoise = StableNoise(states_dim, salient_states_dim, 0.992, type=noise_type) #The last value is the decay rate of the noise
																	#it is assumed to be a property of the environment
	#Gather training dataset
	for ep in range(train_starting_states):
		if reset or start_state is None:
			state = env.reset()
		else:
			state = start_state #have to find a way to set env to this state ... otherwise this is doing nothing
		# full_state = env.env.state_vector().copy()
		if env.spec.id != 'lin-dyn-v0' and reset:
			state = stablenoise.get_obs(state, 0)
		if reset or start_noise is None:
			noise.reset()
		else:
			noise = start_noise
		states = [state]
		actions = []
		rewards = []
		get_new_action = True
		for timestep in range(max_actions):
			if get_new_action:
				with torch.no_grad():
					action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
					action = noise.get_action(action, timestep, multiplier=1.0)
					get_new_action = False
					action_counter = 1

			state_prime, reward, done, _ = env.step(action)
			if env.spec.id != 'lin-dyn-v0':
				state_prime = stablenoise.get_obs(state_prime, timestep+1)
				
			if reward is not None: #WHY IS REWARD NONE sometimes?
				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)
				# dataset.push(full_state, state, state_prime, action, reward)
				if temp_data:
					dataset.temp_push(state, state_prime, action, reward)
				else:
					dataset.push(state, state_prime, action, reward)
			state = state_prime
			# full_state = env.env.state_vector().copy()
			get_new_action = True if action_counter == num_action_repeats else False
			action_counter += 1

	# Collect validation dataset
	if val_starting_states is not None:
		for ep in range(val_starting_states):
			state = env_val.reset()
			# full_state = env.env.state_vector().copy()
			if env_val.spec.id != 'lin-dyn-v0':
				state = stablenoise.get_obs(state, 0)
			states = [state]
			actions = []
			rewards = []
			for timestep in range(max_actions):
				with torch.no_grad():
					action = actor.sample_action(torch.DoubleTensor(state)).detach().numpy()
					# action = actor.sample_action(torch.DoubleTensor(state[:salient_states_dim])).detach().numpy()
					action = noise.get_action(action, timestep+1, multiplier=1.0)
				state_prime, reward, done, _ = env_val.step(action)
				if env_val.spec.id != 'lin-dyn-v0':
					state_prime = stablenoise.get_obs(state_prime, timestep+1)
				actions.append(action)
				states.append(state_prime)
				rewards.append(reward)
				# val_dataset.push(full_state, state, state_prime, action, reward)
				val_dataset.push(state, state_prime, action, reward)
				state = state_prime
				# full_state = env.env.state_vector().copy()
	return state, noise


def add_irrelevant_features(x, extra_dim, noise_level = 0.4):
	'''
	Another function to add irrelevant dimensions to states, this function is only used in the function unroll for
	DirectEnvModel class. Should be merged with StableNoise in future
	Args:
		x: numpy array or batch of Tensor
		extra_dim: number of extra dimensions to add to x
		noise_level: magnitude of noise for the extra dimensions

	Returns:
		numpy array or batch of Tensors of x and the added extra dimensions
	'''
#    x_irrel= np.random.random((x.shape[0], extra_dim))
	if isinstance(x, np.ndarray):
		x_irrel= noise_level*np.random.randn(1, extra_dim).reshape(-1,)
		return np.hstack([x, x_irrel])
	elif isinstance(x, torch.Tensor):
		x_irrel= noise_level*torch.randn(x.shape[0],x.shape[1],extra_dim).double().to(device)
		return torch.cat((x, x_irrel),2)


def convert_one_hot(a, dim):
	'''
	Convert a given value a into a one-hot-coded version with dimensions dim
	Args:
		a: integer to convert to one-hot encoding
		dim: dimension of encoding

	Returns:
		return one-hot-encoded version of a
	'''
	if dim == 2: #binary value, no need to do one-hot encoding
		return a

	if a.shape:
		retval = torch.zeros(list(a.shape)+[dim])
		retval[list(np.indices(retval.shape[:-1]))+[a]] = 1
	else: #single value tensor
		retval = torch.zeros(dim)
		retval[int(a)] = 1
	return retval


def shift(x, step, dir='up'):
	'''
	Given a Tensor with either 3 or 2 dimensions, this function will move "step" number of items forward ('down')
	or backward ('up') along the 2nd or 1st dimensions respectively, and pad with zeros.
	Args:
		x (tensor, could have either of the following shapes (batch_size, trajectory_length, dimensions)
					or (trajectory_length, dimensions)): input to be shift along the second or first dimension
		step (int): number of items to shift
		dir (str: 'up' or 'down'):

	Returns:
		Shifted x
	'''

	#up works, not tested down
	if step == 0:
		return x

	if len(x.shape) == 3: #batch_wise
		if dir=='down':
			return torch.cat((torch.zeros((x.shape[0], step, x.shape[2])).double().to(device),x),dim=1)[:,:-step]
		elif dir=='up':
			return torch.cat((x,torch.zeros((x.shape[0], step, x.shape[2])).double().to(device)),dim=1)[:,step:]

	elif len(x.shape) == 2: 
		if dir=='down':
				return torch.cat((torch.zeros((step, x.shape[1])).double().to(device),x),dim=0)[:-step]
		elif dir=='up':
			return torch.cat((x,torch.zeros((step, x.shape[1])).double().to(device)),dim=0)[step:]

	else:
		raise NotImplementedError('shape {shape_x} of input not correct or implemented'.format(shape_x=x.shape))


def roll_left(x, n):  
	'''
	Very simple function for shifting each element in list x to the left by n items in a circular fashion.
	Args:
		x: list whose items are to be shifted
		n: number of items to shift to the left in list

	Returns:
		new list whose items are shifted left by n items
	'''
	return torch.cat((x[n:], x[:n]))


def get_selected_log_probabilities(policy_estimator, states_tensor, actions_tensor):
	'''Return the score function of policy_estimator evaluated at actions_tensor given states_tensor

	Args:
		policy_estimator (Policy class instant): Stochastic policy
		states_tensor (batch_size, dimension of states): Tensor of states
		actions_tensor (batch_size, dimensions of actions): Tensor of actions

	Returns:
		selected_log_probs (batch_size, dimension of actions) :
			log probability of policy_estimator at actions_tensor given states_tensor
	'''
	action_probs = policy_estimator.get_action_probs(states_tensor)
	if not policy_estimator.continuous:
		c = Categorical(action_probs[0])
		selected_log_probs = c.log_prob(actions_tensor)

	else:
		n = Normal(*action_probs)
		selected_log_probs = n.log_prob(actions_tensor)

	return selected_log_probs

