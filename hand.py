import numpy as np
from gym import utils
from gym import spaces
from gym.envs.mujoco import mujoco_env

class HandEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hand_angle.xml', 5)
        utils.EzPickle.__init__(self)
        low = np.array([0, 0, 0, 0])
        high = np.array([0.05, 0.05, 0.04, 0.065])
        est_high = np.array([0.3, 0.3, 1.5, 1.5])
        # self.action_space = spaces.Box(low=low, high=high)
        # self.estimate_space = spaces.Box(low=low, high=est_high)

    def _step(self, action):
        # xposbefore = self.model.data.qpos[0, 0]
        self.do_simulation(action, self.frame_skip)
        # xposafter = self.model.data.qpos[0, 0]
        ob = self._get_obs()
        # reward_ctrl = - 0.1 * np.square(action).sum()
        # reward_run = (xposafter - xposbefore)/self.dt
        # reward = reward_ctrl + reward_run
        # reward = reward_run
        done = None
        reward = 0
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            # self.model.data.qpos[0:8],
            self.model.data.qpos.flat[4:8]
            # self.model.data.qvel.flat,
        ])

    def reset_model(self):
        # qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        # qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 1.8
        self.viewer.cam.elevation = -20
        # self.viewer.cam.fovy += .5
        # self.viewer.cam.lookat[0] +=
        # self.viewer.cam.lookat[1] +=
        # self.viewer.cam.lookat[2] += 10
        # self.viewer.cam.azimuth = 30