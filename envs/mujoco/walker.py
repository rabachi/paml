import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class MBRLWalkerEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "walker2d.xml", 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        old_ob = self._get_obs()
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        if getattr(self, 'action_space', None):
            a = np.clip(a, self.action_space.low,
                             self.action_space.high)

        reward_ctrl = -0.1 * np.square(a).sum()
        reward_run = old_ob[8]
        reward_height = -3.0 * np.square(old_ob[0] - 1.3)

        reward = reward_run + reward_ctrl + reward_height + 1
        done = False

        # posbefore = self.sim.data.qpos[0]
        # self.do_simulation(a, self.frame_skip)
        # posafter, height, ang = self.sim.data.qpos[0:3]
        # alive_bonus = 1.0
        # reward = ((posafter - posbefore) / self.dt)
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()
        # done = False #not (height > 0.8 and height < 2.0 and
        #        #     ang > -1.0 and ang < 1.0)
        # ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        # return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
