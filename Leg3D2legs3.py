import numpy as np
from gym import spaces
from gym import utils
from gym.envs.mujoco import mujoco_env2 as mujoco_env

class Leg3D2legsEnv3(mujoco_env.MujocoEnv, utils.EzPickle):
    zd2 = 0
    check = 0
    gait = np.array((0,0,0,0))
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'Leg3D2legs3.xml', 4)

        self.max_speed = 5.
        self.max_torque = 6.
        # self.viewer = None

        high = np.array([0.1,0.1,0.1, 1, 1, 1, 1, \
                         1. ,1. ,1. ,1. ,1. ,1. ,1. ,1.,
                         1., 1., 1., 1., 1., 1., 1., 1.,
                         self.max_speed, self.max_speed, self.max_speed, self.max_speed,
                         self.max_speed, self.max_speed, self.max_speed, self.max_speed,
                         self.max_speed, self.max_speed, self.max_speed, self.max_speed,
                         self.max_speed, self.max_speed,
                         0.])

        action_high = self.max_torque*np.array([1.,1.,1.,1.,1.,1.,1.,1.])
        self.action_space = spaces.Box(low=-action_high, high=action_high)
       # self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(3,))
        self.observation_space = spaces.Box(low=-high, high=high)


    def _step(self, a):

        # if self.check !=0 :
        #     a[0][8:] = np.random.uniform(-self.max_torque, self.max_torque, size=(1,4))

        self.check += 1
        a_swing = np.array(np.random.uniform(low=-15,high=15,size=(1,4)))
        #a_swing = np.array([0,0,0,0])
        a2=np.concatenate([a, a_swing[0]])
        # a[0][8:] = np.random.uniform(low=-6,high=6,size=(1,4))
        self.do_simulation(a2, self.frame_skip)
        s = np.concatenate([self.model.data.qpos, self.model.data.qvel])

        a3=np.reshape(a2[:8],(8,1))
        amag=np.dot(a2[:8],a3)

        reward = (s[0]**2+(self.zd2-s[1])**2+s[2]**2)*100+((s[3]-1)**2+s[4]**2+s[5]**2+s[6]**2)*100\
                 + 0.01*amag


        self.zd2 = 0

        return self._get_obs(s, self.gait, self.zd2), -reward, False, {}

    def reset_model(self):
        # qpos = self.init_qpos+[0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,1]
        qvel = self.init_qvel
        qpos = self.init_qpos
        # qpos[:5] = [0, 0, 0, 0, 0]

        # qpos[:5] = [0,0,0,0,0]
        # qpos[:5] = np.random.uniform(low=0, high=0.05, size=5)
        # qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.05, high=0.05)
        self.state = self.set_state(qpos, qvel)

        self.zd2 = 0
        self.gait = np.array((0, 0, 0, 0))
        return self._get_obs(self.state, self.gait, self.zd2)

    def _get_obs(self, s, gait, zd):
        # return np.concatenate([self.model.data.qpos[0], self.model.data.qpos[1], self.model.data.qpos[2], \
        #                        self.model.data.qpos[3], self.model.data.qpos[4], self.model.data.qpos[5],
        #                        self.model.data.qpos[6],
        #                        np.cos(self.model.data.qpos[7]), np.sin(self.model.data.qpos[7]),
        #                        np.cos(self.model.data.qpos[8]), np.sin(self.model.data.qpos[8]),
        #                        np.cos(self.model.data.qpos[9]), np.sin(self.model.data.qpos[9]),
        #                        np.cos(self.model.data.qpos[10]), np.sin(self.model.data.qpos[10]),
        #                        np.cos(self.model.data.qpos[11]), np.sin(self.model.data.qpos[11]),
        #                        np.cos(self.model.data.qpos[12]), np.sin(self.model.data.qpos[12]),
        #                        np.cos(self.model.data.qpos[13]), np.sin(self.model.data.qpos[13]),
        #                        np.cos(self.model.data.qpos[14]), np.sin(self.model.data.qpos[14]),
        #                        np.cos(self.model.data.qpos[15]), np.sin(self.model.data.qpos[15]),
        #                        np.cos(self.model.data.qpos[16]), np.sin(self.model.data.qpos[16]),
        #                        np.cos(self.model.data.qpos[17]), np.sin(self.model.data.qpos[17]),
        #                        np.cos(self.model.data.qpos[18]), np.sin(self.model.data.qpos[18]),
        #                        self.model.data.qvel[0], self.model.data.qvel[1], self.model.data.qvel[2],
        #                        self.model.data.qvel[3], self.model.data.qvel[4], self.model.data.qvel[5],
        #                        self.model.data.qvel[6], self.model.data.qvel[7], self.model.data.qvel[8],
        #                        self.model.data.qvel[9], self.model.data.qvel[10], self.model.data.qvel[11],
        #                        self.model.data.qvel[12], self.model.data.qvel[13], self.model.data.qvel[14],
        #                        self.model.data.qvel[15], self.model.data.qvel[16], self.model.data.qvel[17],
        #                        [zd]])

        return np.concatenate([self.model.data.qpos[0], self.model.data.qpos[1], self.model.data.qpos[2], \
                               self.model.data.qpos[3], self.model.data.qpos[4], self.model.data.qpos[5],
                               self.model.data.qpos[6],
                               np.cos(self.model.data.qpos[7]), np.sin(self.model.data.qpos[7]),
                               np.cos(self.model.data.qpos[8]), np.sin(self.model.data.qpos[8]),
                               np.cos(self.model.data.qpos[9]), np.sin(self.model.data.qpos[9]),
                               np.cos(self.model.data.qpos[10]), np.sin(self.model.data.qpos[10]),
                               # np.cos(self.model.data.qpos[11]), np.sin(self.model.data.qpos[11]),
                               # np.cos(self.model.data.qpos[12]), np.sin(self.model.data.qpos[12]),
                               np.cos(self.model.data.qpos[13]), np.sin(self.model.data.qpos[13]),
                               # np.cos(self.model.data.qpos[14]), np.sin(self.model.data.qpos[14]),
                               # np.cos(self.model.data.qpos[15]), np.sin(self.model.data.qpos[15]),
                               np.cos(self.model.data.qpos[16]), np.sin(self.model.data.qpos[16]),
                               np.cos(self.model.data.qpos[17]), np.sin(self.model.data.qpos[17]),
                               np.cos(self.model.data.qpos[18]), np.sin(self.model.data.qpos[18]),
                               self.model.data.qvel[0], self.model.data.qvel[1], self.model.data.qvel[2],
                               self.model.data.qvel[3], self.model.data.qvel[4], self.model.data.qvel[5],
                               self.model.data.qvel[6], self.model.data.qvel[7], self.model.data.qvel[8],
                               self.model.data.qvel[9], self.model.data.qvel[10], self.model.data.qvel[13],
                               self.model.data.qvel[16], self.model.data.qvel[17],[zd]])

        # return np.concatenate([s[0],s[1],s[2], s[3],s[4],s[5],s[6], \
        #                        np.cos(s[7]), np.sin(s[7]), np.cos(s[8]), np.sin(s[8]), np.cos(s[9]), np.sin(s[9]),
        #                        np.cos(s[10]), np.sin(s[10]),
        #                        # np.cos(s[11]), np.sin(s[11]), np.cos(s[12]), np.sin(s[12]),
        #                        np.cos(s[13]), np.sin(s[13]),
        #                        # np.cos(s[14]), np.sin(s[14]), np.cos(s[15]), np.sin(s[15]),
        #                        np.cos(s[16]), np.sin(s[16]), np.cos(s[17]), np.sin(s[17]), np.cos(s[18]), np.sin(s[18]),
        #                        s[19],s[20],s[21], s[22], s[23], s[24],
        #                        s[25], s[26], s[27], s[28],
        #                        # s[29], s[30],
        #                        s[31],
        #                        # s[32], s[33],
        #                        s[34], s[35], s[36], [gait[0]], [gait[1]], [gait[2]], [gait[3]], [zd]])

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = v.model.stat.extent * 1.1
        v.cam.azimuth = 90
        v.cam.elevation = 2


    def angle_normalize(x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)
