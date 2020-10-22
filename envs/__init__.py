from gym.envs.registration import register
# Mujoco
# ----------------------------------------

#

# register(
#     id='HalfCheetah-v2',
#     entry_point='gym.envs.mujoco:HalfCheetahEnv',
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )

# register(
#     id='HalfCheetah-v3',
#     entry_point='gym.envs.mujoco.half_cheetah_v3:HalfCheetahEnv',
#     max_episode_steps=1000,
#     reward_threshold=4800.0,
# )

register(
    id='MBRLHopper-v0',
    entry_point='envs.mujoco:MBRLHopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

# register(
#     id='Hopper-v3',
#     entry_point='gym.envs.mujoco.hopper_v3:HopperEnv',
#     max_episode_steps=1000,
#     reward_threshold=3800.0,
# )

register(
    id='MBRLSwimmer-v0',
    entry_point='envs.mujoco:MBRLSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='MBRLNoisySwimmer-v0',
    entry_point='envs.mujoco:MBRLNoisySwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)


register(
    id='MBRLWalker-v0',
    max_episode_steps=1000,
    entry_point='envs.mujoco:MBRLWalkerEnv',
)

# register(
#     id='Walker2d-v3',
#     max_episode_steps=1000,
#     entry_point='gym.envs.mujoco.walker2d_v3:Walker2dEnv',
# )

register(
    id='MBRLAnt-v0',
    entry_point='envs.mujoco:MBRLAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

# register(
#     id='Ant-v3',
#     entry_point='gym.envs.mujoco.ant_v3:AntEnv',
#     max_episode_steps=1000,
#     reward_threshold=6000.0,
# )

