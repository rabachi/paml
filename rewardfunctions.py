import numpy as np
import matplotlib.pyplot as plt
# import gym
import sys

import torch
import math
from torch import nn
from torch import optim
from torch.distributions import Categorical, Normal, MultivariateNormal
import pdb

MAX_TORQUE = 2.
_DEFAULT_VALUE_AT_MARGIN = 0.1

def get_reward_fn(env, states_tensor, actions_tensor):
	if (env == 'lin_dyn') or (env.spec.id == 'lin-dyn-v0'):
		#set actions multiplier to 0 to try with reinforce
		rewards = -(torch.einsum('ijk,ijk->ij', [states_tensor, states_tensor]) +
					torch.einsum('ijk,ijk->ij', [actions_tensor, actions_tensor]))
		#rewards = torch.clamp(states_tensor[:,0]**2, min=0., max=1.0)
		return rewards

	if env.spec.id == 'Pendulum-v0': 
		thcos = states_tensor[:,:,0]
		thsin = states_tensor[:,:,1]
		thdot = states_tensor[:,:,2]

		#pdb.set_trace()
		#tanth = thsin/thcos
		#tanth[torch.isnan(tanth)] = 0
		th = torch.atan2(thsin, thcos)

		if torch.isnan(th).any():
			pdb.set_trace()

		#u = torch.clamp(actions_tensor, min=-MAX_TORQUE, max=MAX_TORQUE).squeeze()
		u = actions_tensor.squeeze().unsqueeze(1)

		costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

		return -costs#.unsqueeze(2)

	elif env.spec.id == 'HalfCheetah-v2': 
		dt = 0.05 #from stepping through env
		xposbefore = states_tensor[:,0]
		# xposbefore = self.sim.data.qpos[0]
		# self.do_simulation(action, self.frame_skip) #can't do this step because this is also for stepping through environment, but I actually HAVE the next states, and can compare them directly here 
		xposafter = states_tensor #self.sim.data.qpos[0]
		ob = self._get_obs()

		reward_ctrl = - 0.1 * torch.square(actions_tensor).sum()

		reward_run = (xposafter - xposbefore)/dt

		reward = reward_ctrl + reward_run


		def _get_obs(self):
			return np.concatenate([
				self.sim.data.qpos.flat[1:],
				self.sim.data.qvel.flat,
			])

	elif env.spec.id == 'dm-Pendulum-v0':
		COS_BND = np.cos(np.deg2rad(8))

		rewards = (torch.le(states_tensor[:,:,0], 1) == torch.ge(states_tensor[:,:,0], COS_BND))
		return rewards.double()

	elif env.spec.id == 'dm-Cartpole-swingup-v0': #TAKES NEXT STATE FOR REWARD NOT CURRENT STATE
		rewards_to_return = torch.zeros(states_tensor.shape[0])
		for d in range(states_tensor.shape[0]):
			pole_angle_cosine = states_tensor[d,1] 
			upright = (pole_angle_cosine + 1) / 2 
			centered = tolerance(states_tensor[d,0], margin=2) 
			centered = (1 + centered) / 2 
			small_control = tolerance(actions_tensor[d,:], margin=1,
		                                value_at_margin=0,
		                                sigmoid='quadratic')[0]
			small_control = (4 + small_control) / 5
			small_velocity = tolerance(states_tensor[d,4], margin=5).min()
			small_velocity = (1 + small_velocity) / 2 
		
			rewards_to_return[d] = upright.mean() * small_control * small_velocity * centered #torch.from_numpy(centered)
			# OrderedDict([('position', array([ 0.01871485, -0.99999419, -0.00340747])), ('velocity', array([0.04293839, 0.06518433]))])

		return rewards_to_return


	elif env.spec.id == 'CartPole-v0':
		theta_threshold_radians = 12 * 2 * np.pi / 360
		x_threshold = 2.4
		x = states_tensor[:,:,0]
		#x_dot = states_tensor[:,:,1]
		theta = states_tensor[:,:,2]
		#theta_dot = states_tensor[:,:,3]

		#this is a problem because ITS A BIG MATRIX WITH BATCHES AND DIFFERENT TIME STEPS!!!!!!  
		done =  (x < -x_threshold) \
				| (x > x_threshold) \
				| (theta < -theta_threshold_radians) \
				| (theta > theta_threshold_radians)
		#done = bool(done)
		return done.transpose(1,0)
		# if not done:
  #           reward = 1.0
  #       elif self.steps_beyond_done is None:
  #           # Pole just fell!
  #           self.steps_beyond_done = 0
  #           reward = 1.0
  #       else:
  #           if self.steps_beyond_done == 0:
  #               logger.warn("You are calling 'step()' even though this environment has already returned done = True. You should always call 'reset()' once you receive 'done = True' -- any further steps are undefined behavior.")
  #           self.steps_beyond_done += 1
  #           reward = 0.0

  	# else:
  	# 	raise NotImplementedError

	# elif env.spec.id == 'dm_cartpole_balance':

	# 	states = states_tensor.cpu().detach().numpy()
	# 	print(states)
	# 	states = np.swapaxes(np.atleast_3d(states), 1,2)
	# 	pole_angle_cosine = states[:,:,1]
	# 	cart_position = states[:,:,0]
	# 	angular_vel = states[:,:,]

	# 	control = actions_tensor.cpu().detach().numpy().squeeze()


	# 	upright = (pole_angle_cosine + 1) / 2

	# 	centered = tolerance(cart_position, margin=2)
	# 	centered = (1 + centered) / 2
	# 	small_control = tolerance(actions_tensor, margin=1,
	# 					value_at_margin=0.000000001,
	# 					sigmoid='quadratic')[0]
	# 	small_control = (4 + small_control) / 5

	# 	small_velocity = tolerance(angular_vel, margin=5).min()
	# 	small_velocity = (1 + small_velocity) / 2

	# 	return torch.FloatTensor(np.expand_dims(upright.mean(axis=0),axis=1) * small_control * small_velocity * centered.T)


	return 0


def tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid='gaussian',
	value_at_margin=_DEFAULT_VALUE_AT_MARGIN):
	"""Returns 1 when `x` falls inside the bounds, between 0 and 1 otherwise.
	Args:
	x: A scalar or numpy array.
	bounds: A tuple of floats specifying inclusive `(lower, upper)` bounds for
	the target interval. These can be infinite if the interval is unbounded
	at one or both ends, or they can be equal to one another if the target
	value is exact.
	margin: Float. Parameter that controls how steeply the output decreases as
	`x` moves out-of-bounds.
	* If `margin == 0` then the output will be 0 for all values of `x`
	outside of `bounds`.
	* If `margin > 0` then the output will decrease sigmoidally with
	increasing distance from the nearest bound.
	sigmoid: String, choice of sigmoid type. Valid values are: 'gaussian',
	'linear', 'hyperbolic', 'long_tail', 'cosine', 'tanh_squared'.
	value_at_margin: A float between 0 and 1 specifying the output value when
	the distance from `x` to the nearest bound is equal to `margin`. Ignored
	if `margin == 0`.
	Returns:
	A float or numpy array with values between 0.0 and 1.0.
	Raises:
	ValueError: If `bounds[0] > bounds[1]`.
	ValueError: If `margin` is negative.
	"""
	lower, upper = bounds
	if lower > upper:
		raise ValueError('Lower bound must be <= upper bound.')
	if margin < 0:
		raise ValueError('`margin` must be non-negative.')

	in_bounds = np.logical_and(lower <= x, x <= upper)
	if margin == 0:
		value = np.where(in_bounds, 1.0, 0.0)
	else:
		d = np.where(x < lower, lower - x, x - upper) / margin
		value = np.where(in_bounds, 1.0, _sigmoids(d, value_at_margin, sigmoid))

	return float(value) if np.isscalar(x) else value


def _sigmoids(x, value_at_1, sigmoid):
	"""Returns 1 when `x` == 0, between 0 and 1 otherwise.
	Args:
	x: A scalar or numpy array.
	value_at_1: A float between 0 and 1 specifying the output when `x` == 1.
	sigmoid: String, choice of sigmoid type.
	Returns:
	A numpy array with values between 0.0 and 1.0.
	Raises:
	ValueError: If not 0 < `value_at_1` < 1, except for `linear`, `cosine` and
	`quadratic` sigmoids which allow `value_at_1` == 0.
	ValueError: If `sigmoid` is of an unknown type.
	"""
	if sigmoid in ('cosine', 'linear', 'quadratic'):
		if not 0 <= value_at_1 < 1:
			raise ValueError('`value_at_1` must be nonnegative and smaller than 1, '
				'got {}.'.format(value_at_1))
	else:
		if not 0 < value_at_1 < 1:
			raise ValueError('`value_at_1` must be strictly between 0 and 1, '
				'got {}.'.format(value_at_1))

	if sigmoid == 'gaussian':
		scale = np.sqrt(-2 * np.log(value_at_1))
		return np.exp(-0.5 * (x*scale)**2)

	elif sigmoid == 'hyperbolic':
		scale = np.arccosh(1/value_at_1)
		return 1 / np.cosh(x*scale)

	elif sigmoid == 'long_tail':
		scale = np.sqrt(1/value_at_1 - 1)
		return 1 / ((x*scale)**2 + 1)

	elif sigmoid == 'cosine':
		scale = np.arccos(2*value_at_1 - 1) / np.pi
		scaled_x = x*scale
		return np.where(abs(scaled_x) < 1, (1 + np.cos(np.pi*scaled_x))/2, 0.0)

	elif sigmoid == 'linear':
		scale = 1-value_at_1
		scaled_x = x*scale
		return np.where(abs(scaled_x) < 1, 1 - scaled_x, 0.0)

	elif sigmoid == 'quadratic':
		scale = np.sqrt(1-value_at_1)
		scaled_x = x*scale
		return np.where(abs(scaled_x) < 1, 1 - scaled_x**2, 0.0)

	elif sigmoid == 'tanh_squared':
		scale = np.arctanh(np.sqrt(1-value_at_1))
		return 1 - np.tanh(x*scale)**2

	else:
		raise ValueError('Unknown sigmoid type {!r}.'.format(sigmoid))


def angle_normalize(x):
	return (((x+np.pi) % (2*np.pi)) - np.pi)