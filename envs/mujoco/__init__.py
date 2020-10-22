from gym.envs.mujoco.mujoco_env import MujocoEnv
# ^^^^^ so that user gets the correct error
# message if mujoco is not installed correctly
from envs.mujoco.ant import MBRLAntEnv
# from gym.envs.mujoco.half_cheetah import HalfCheetahEnv
from envs.mujoco.hopper import MBRLHopperEnv
# from gym.envs.mujoco.walker2d import Walker2dEnv
# from gym.envs.mujoco.humanoid import HumanoidEnv
# from gym.envs.mujoco.inverted_pendulum import InvertedPendulumEnv
# from gym.envs.mujoco.inverted_double_pendulum import InvertedDoublePendulumEnv
# from gym.envs.mujoco.reacher import ReacherEnv
from envs.mujoco.swimmer import MBRLSwimmerEnv
from envs.mujoco.noisyswimmer import MBRLNoisySwimmerEnv

from envs.mujoco.walker import MBRLWalkerEnv
# from gym.envs.mujoco.humanoidstandup import HumanoidStandupEnv
# from gym.envs.mujoco.pusher import PusherEnv
# from gym.envs.mujoco.thrower import ThrowerEnv
# from gym.envs.mujoco.striker import StrikerEnv
