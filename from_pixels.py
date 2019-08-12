import gym
import dm_control2gym
import numpy as np
import pdb

# make the dm_control environment
env1 = dm_control2gym.make(domain_name="pendulum", task_name="swingup")
# env2 = gym.make('Pendulum-v0')

dm_control2gym.create_render_mode('pixels', show=False, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(),
             depth=False, scene_option=None)

env1.reset()
# env2.reset()
for t in range(1000):
	pdb.set_trace()
	a = env1.action_space.sample()
	observation, reward, done, info = env1.step(env1.action_space.sample()) # take a random action
	# tup = env2.step(a)
	# env2.render(mode='rgb_array')
	pixels = env1.render(mode='pixels')





# https://github.com/openai/gym/blob/master/gym/core.py
class AddExtraDims(gym.ObservationWrapper):
    """ Wrap action """
    def __init__(self, env, extra_dims):
        super(AddExtraDims, self).__init__(env)

        self.extra_dims = extra_dims

    def observation(self, observation):
        new_observation = self.add_irrelevant_features(observation, self.extra_dims, noise_level=0.4)
        return new_observation

    def add_irrelevant_features(self, x, extra_dim, noise_level = 0.4):
        x_irrel= noise_level*np.random.randn(1, extra_dim).reshape(-1,)
        return np.hstack([x, x_irrel])


class ContinuousReward(gym.RewardWrapper):
	# def __init__():
	# 	pass

	def reward(self, reward):
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


env = gym.make('Pendulum-v0')
env.reset()

w = AddExtraDims(env, 7)
w.reset()
print(w.step([0]))


