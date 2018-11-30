import gym
import numpy as np

env = gym.make('hand-v0')
env.reset()

o0 = np.array([0, 0, 0])
is0 = np.array([0, -0.01, 0]) + o0

o1 = np.array([0, 0, 0.05]) + o0
is1 = np.array([0.015, -0.009, 0.05]) + o1
is2 = np.array([0.027, -0.009, 0.06]) + o1
is10 = np.array([0.022, 0, 0.077]) + o1
is11 = np.array([0.038, 0, 0.074]) + o1

o2 = np.array([0.03, 0, 0.075]) + o1
is3 = np.array([0, -0.007, 0.014]) + o2
is4 = np.array([0, -0.007, 0.033]) + o2
is8 = np.array([-0.007, 0, 0.014]) + o2
is9 = np.array([0.007, 0, 0.014]) + o2

o3 = np.array([0, 0, 0.045]) + o2
is5 = np.array([0, -0.0065, 0.01]) + o3
is6 = np.array([0, -0.0065, 0.02]) + o3

o4 = np.array([0, 0, 0.03]) + o3
is7 = np.array([0, -0.006, 0.01]) + o4

length01 = np.sqrt(np.square(is1-is0).sum())
length12 = np.sqrt(np.square(is2-is1).sum())
length23 = np.sqrt(np.square(is3-is2).sum())
length34 = np.sqrt(np.square(is4-is3).sum())
length45 = np.sqrt(np.square(is5-is4).sum())
length56 = np.sqrt(np.square(is6-is5).sum())
length67 = np.sqrt(np.square(is7-is6).sum())

length110 = np.sqrt(np.square(is10-is1).sum())
length111 = np.sqrt(np.square(is11-is10).sum())
length108 = np.sqrt(np.square(is8-is10).sum())
length119 = np.sqrt(np.square(is9-is11).sum())

tendon1 = length01+length111+length119
tendon2 = length01+length110+length108
tendon3 = length01+length12+length23+length34
tendon4 = tendon3+length45+length56
tendon5 = tendon4+length67

for i in range(1000):
    env.render()
    action = [tendon1, tendon2, tendon3, tendon4, tendon5]
    state, reward, done, info = env.step(action)
    print(state)

