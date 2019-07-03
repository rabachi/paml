import gym
import dm_control2gym
import numpy as np
import pdb


import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


env = gym.make('CartPole-v0').unwrapped

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if gpu is to be used
device = torch.device("cpu")

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


def get_cart_location(screen_width):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render(mode='rgb_array').transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)
    
env.reset()
plt.figure()
plt.imshow(get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
           interpolation='none')
plt.title('Example extracted screen')
# plt.show()

# make the dm_control environment
# env1 = dm_control2gym.make(domain_name="pendulum", task_name="swingup")
# env2 = gym.make('Pendulum-v0')

# dm_control2gym.create_render_mode('pixels', show=False, return_pixel=True, height=240, width=320, camera_id=-1, overlays=(),
#              depth=False, scene_option=None)

# env1.reset()
# env2.reset()
# for t in range(1000):
# 	pdb.set_trace()
# 	a = env1.action_space.sample()
# 	observation, reward, done, info = env1.step(env1.action_space.sample()) # take a random action
# 	tup = env2.step(a)
# 	env2.render(mode='rgb_array')
# 	pixels = env1.render(mode='pixels')





# # https://github.com/openai/gym/blob/master/gym/core.py
# class AddExtraDims(gym.ObservationWrapper):
#     """ Wrap action """
#     def __init__(self, env, extra_dims):
#         super(AddExtraDims, self).__init__(env)

#         self.extra_dims = extra_dims

#     def observation(self, observation):
#         new_observation = self.add_irrelevant_features(observation, self.extra_dims, noise_level=0.4)
#         return new_observation

#     def add_irrelevant_features(self, x, extra_dim, noise_level = 0.4):
#         x_irrel= noise_level*np.random.randn(1, extra_dim).reshape(-1,)
#         return np.hstack([x, x_irrel])


# class ContinuousReward(gym.RewardWrapper):
# 	# def __init__():
# 	# 	pass

# 	def reward(self, reward):
# 		thcos = states_tensor[:,:,0]
# 		thsin = states_tensor[:,:,1]
# 		thdot = states_tensor[:,:,2]

# 		#pdb.set_trace()
# 		#tanth = thsin/thcos
# 		#tanth[torch.isnan(tanth)] = 0
# 		th = torch.atan2(thsin, thcos)

# 		if torch.isnan(th).any():
# 			pdb.set_trace()

# 		#u = torch.clamp(actions_tensor, min=-MAX_TORQUE, max=MAX_TORQUE).squeeze()
# 		u = actions_tensor.squeeze().unsqueeze(1)

# 		costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

# 		return -costs#.unsqueeze(2)


# env = gym.make('Pendulum-v0')
# env.reset()

# w = AddExtraDims(env, 7)
# w.reset()
# print(w.step([0]))


