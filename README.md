# HandControl

### Development Environments
- Ubuntu 16.04
- [Mujoco 150 linux](https://mujoco.org)
- [Mujoco-py 1.50.1.35](https://github.com/openai/mujoco-py)
- [Openai gym](https://github.com/openai/gym)

---

#### 1. Make Mujoco Model
First of all, to simulate Mujoco in openai gym, we need a MuJoCo XML model file in its native **MJCF** format. You can find details about a modeling process in [here](http://www.mujoco.org/book/modeling.html).
I made 'hand_angle.xml' model to simulate.

#### 2. Build gym environment
After making the model, we can build gym environment to simulate the mujoco model in openai gym framework.
1. Put the model (`hand_angle.xml`) to `~your path to /gym/gym/envs/mujoco/assets`
2. Make a new environment python file ('yourenvname.py') in `~ your path to /gym/gym/envs/mujoco`. In my case, `hand.py`.
I refer to the pre existing files given by gym framework. They provide some examples kind of 'ant.py', 'half_cheetah.py' etc. for newbies like me.
The envirionment python file is composed as below.

**Notice**: from the mujoco-py 1.50 we need to use `step(self, action)` rather than `_step(self, action)`

```python
import numpy as np
from gym import utils
from gym import spaces
from gym.envs.mujoco import mujoco_env

class HandEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'hand_angle.xml', 5)
        utils.EzPickle.__init__(self)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()
        reward = 0
        done = None
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[4:8]
        ])

    def reset_model(self):
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 1.8
        self.viewer.cam.elevation = -20
```

#### 3. Register Environment to mujoco_env in the gym
After building the environment python file, we need to register this environment to the gym. First, to register to the gym, we need to register the environment to mujoco_env by modifying `~your path to/gym/gym/envs/mujoco/__init__.py`
Add your environment end of the `__init__.py` as below.

```python
from gym.envs.mujoco.hand import HandEnv
```

#### 4. Register Environment to the gym
Finally, we can register our environment to the gym framework. To do that, we need to modify file `~your path to/gym/gym/envs/__init__.py`. Open the file then you can find Mujoco section. At the end of the Mujoco section add some lines as below to register your environment.

```python
register(
    id='hand-v0',
    entry_point='gym.envs.mujoco:HandEnv',
    max_episode_steps=1000,
)
```

#### 5. Test the environment
You can test your environment using this simple code.

```python
import gym

env = gym.make('hand-v0')
env.reset()

for i in range(1000):
    env.render()
    action = [0, 0, 0, 0]
    state, reward, done, info = env.step(action)
    print(state)
```

If you made mujoco model and registered to gym environment properly as explained above, your environment works well and you can see simulation window like this.

<p align="center">
    <img src="https://github.com/gemst1/gemst1.github.io/blob/master/assets/images/openai-gym-mujoco-py.png?raw=true" alt="OpenAI gym with mujoco-py"/>
    <figcaption class="caption">OpenAI gym with mujoco-py</figcaption>
</p>
