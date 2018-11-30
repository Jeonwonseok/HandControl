import gym
import numpy as np

env = gym.make('finger-v1')
env.reset()

tendon1 = 0.062
tendon2 = 0.062
tendon3 = 0.062
tendon4 = 0.107
tendon5 = 0.132

for i in range(20000):
    env.render()
    # action = [tendon1, tendon2, tendon3-0.01, tendon4-0.01, tendon5-0.01, 0.3, 0.8, 0.8, 0.8]
    action = [-0., -0., -0., -0., -0., 0.3, 0.8, 0.8, 0.8]
    # action = [0, -0.12, -0.28, -0.28, -0.35]
    # action = [tendon5 - 0.021]
    # action = [-0.14, tendon2, -0.34, -0.36, -0.36]
    state, reward, done, info = env.step(action)
    print(state)

