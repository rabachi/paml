import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class MBRLHopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.ctrl_cost_coeff = 0.01
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0

        # lb, ub = self.action_bounds
        scaling = 1. #(ub - lb) * 0.5

        # reward = (posafter - posbefore) / self.dt
        # reward += alive_bonus
        # reward -= 1e-3 * np.square(a).sum()

        ob = self._get_obs()

        reward = (posafter - posbefore) / self.dt - \
            0.5 * self.ctrl_cost_coeff * np.sum(np.square(a / scaling)) -\
            np.sum(np.maximum(np.abs(ob[2:]) - 100, 0)) -\
            10*np.maximum(0.45 - height, 0) - \
            10*np.maximum(abs(ang) - .2, 0)

        # s = self.state_vector()
        # done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    # (height > .7) and (abs(ang) < .2))
        done = False
        # ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20
