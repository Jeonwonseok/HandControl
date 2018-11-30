import gym
from gym.wrappers import Monitor
import tensorflow as tf
from control_network import ControlNetwork
from estimate_network import EstimateNetwork
import numpy as np
import datetime
# import matplotlib.pyplot as plt
from gym import wrappers

ENV_NAME = 'hand-v0'
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

MAX_EPISODES = 10001
EPISODE_STEPS = 500

EXPLORATION_STEPS = 3000

SUMMARY_DIR = './results/{}/{}/tf'.format(ENV_NAME, DATETIME)
MONITOR_DIR = './results/{}/{}/gym'.format(ENV_NAME, DATETIME)
MODEL_DIR = './results/{}/{}/model/model.ckpt'.format(ENV_NAME, DATETIME)

def init_pose():
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

    length01 = np.sqrt(np.square(is1 - is0).sum())
    length12 = np.sqrt(np.square(is2 - is1).sum())
    length23 = np.sqrt(np.square(is3 - is2).sum())
    length34 = np.sqrt(np.square(is4 - is3).sum())
    length45 = np.sqrt(np.square(is5 - is4).sum())
    length56 = np.sqrt(np.square(is6 - is5).sum())
    length67 = np.sqrt(np.square(is7 - is6).sum())

    length110 = np.sqrt(np.square(is10 - is1).sum())
    length111 = np.sqrt(np.square(is11 - is10).sum())
    length108 = np.sqrt(np.square(is8 - is10).sum())
    length119 = np.sqrt(np.square(is9 - is11).sum())

    tendon1 = length01 + length111 + length119
    tendon2 = length01 + length110 + length108
    tendon3 = length01 + length12 + length23 + length34
    tendon4 = tendon3 + length45 + length56
    tendon5 = tendon4 + length67

    init_action = np.array([tendon1, tendon2, tendon3, tendon4, tendon5])
    return init_action

def summaries():
    estimate_error_1 = tf.Variable(0.)
    estimate_error_2 = tf.Variable(0.)
    estimate_error_3 = tf.Variable(0.)
    estimate_error_4 = tf.Variable(0.)
    tf.summary.scalar('Estimate_error_1', estimate_error_1)
    tf.summary.scalar('Estimate_error_2', estimate_error_2)
    tf.summary.scalar('Estimate_error_3', estimate_error_3)
    tf.summary.scalar('Estimate_error_4', estimate_error_4)

    control_error_1 = tf.Variable(0.)
    control_error_2 = tf.Variable(0.)
    control_error_3 = tf.Variable(0.)
    control_error_4 = tf.Variable(0.)
    tf.summary.scalar('Control_error_1', control_error_1)
    tf.summary.scalar('Control_error_2', control_error_2)
    tf.summary.scalar('Control_error_3', control_error_3)
    tf.summary.scalar('Control_error_4', control_error_4)

    estimate_rmse = tf.Variable(0.)
    control_rmse = tf.Variable(0.)

    tf.summary.scalar('Estimate_RMSE', estimate_rmse)
    tf.summary.scalar('Control_RMSE', control_rmse)

    desired_input_1 = tf.Variable(0.)
    desired_input_2 = tf.Variable(0.)
    desired_input_3 = tf.Variable(0.)
    desired_input_4 = tf.Variable(0.)


    tf.summary.scalar('Desired_1', desired_input_1)
    tf.summary.scalar('Desired_2', desired_input_2)
    tf.summary.scalar('Desired_3', desired_input_3)
    tf.summary.scalar('Desired_4', desired_input_4)

    summary_vars = [estimate_error_1, estimate_error_2, estimate_error_3, estimate_error_4,
                    control_error_1, control_error_2, control_error_3, control_error_4,
                    estimate_rmse, control_rmse, desired_input_1, desired_input_2, desired_input_3, desired_input_4]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(sess, env, control_net, estimate_net, action_bound, init_pose):
    summary_ops, summary_vars = summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    saver = tf.train.Saver()

    # joint1_bias = np.array([0.025, 0.025, 0., 0., 0.])
    joint1_bias = np.array([0.0, 0.0, 0., 0., 0.])
    desired_bound = np.array([0.7, 1.57, 1.57, 1.57])
    desired_range = np.array([0.5, 0, 0, 0])
    estimate_net.update_target_network()

    for i in range(MAX_EPISODES):
        env.reset()
        desired_state = np.multiply(desired_bound, np.random.rand(1, 4) - desired_range)

        if i < EXPLORATION_STEPS:
            action = init_pose - np.multiply(np.random.rand(1, 5), action_bound - np.array([0.4, 0.4, 0, 0, 0]))
        else:
            action = control_net.action(np.reshape(desired_state, (1, 4)))
            estimate_net.learning_rate = 0.0005

        if i > 8000:
            control_net.learning_rate = 0.0001
            estimate_net.learning_rate = 0.0001

        for j in range(EPISODE_STEPS):
            if i % 500 == 0 or i == MAX_EPISODES-1:
                env.render()

            state, reward, done, info = env.step(action)

        # env.render(close=True)

        estimate = estimate_net.estimate(np.reshape(action, (1, 5)))
        estimate_target = estimate_net.estimate_target(np.reshape(action, (1, 5)))

        if i > EXPLORATION_STEPS:
            action_gradients = estimate_net.action_gradients(np.reshape(action, (1, 5)), np.reshape(desired_state, (1, 4)))

            control_net.train(np.reshape(desired_state, (1, 4)), action_gradients[0])

        estimate_net.train(np.reshape(action, (1, 5)), np.reshape(state, (1, 4)))

        if i % 10 == 0:
            estimate_net.update_target_network()

        estimate_error = estimate - state
        control_error = desired_state - state
        error = np.append(estimate_error, control_error)
        estimate_rmse = np.sqrt((estimate_error ** 2).mean())
        control_rmse = np.sqrt((control_error ** 2).mean())
        sum_vars = np.append(np.append(error, [estimate_rmse, control_rmse]), desired_state)
        summary_str = sess.run(summary_ops, feed_dict={summary_vars[k]: sum_vars[k] for k in range(len(summary_vars))})
        writer.add_summary(summary_str, i)
        writer.flush()

        if i % 100 == 0:
            if i < EXPLORATION_STEPS:
                print(i)
                print('real state:      ', state)
                print('estimate:        ', estimate)
                print('estimate_error:  ', estimate - state)
                print('action:          ', action)
            else:
                print(i)
                print('desired:         ', desired_state)
                # print('estimate_target: ', estimate_target)
                # print('estimate:        ', estimate)
                print('real state:      ', state)
                print('target_est_error:', estimate_target - state)
                print('estimate_error:  ', estimate-state)
                print('control_error:   ', desired_state - state)
                print('action:          ', action)
    saver.save(sess, MODEL_DIR)

def main(_):
    with tf.Session() as sess:
        env = gym.make('hand-v0')
        env.reset()

        init_pos = init_pose()

        state_dim = 4
        action_dim = 5
        action_bound = np.array([0.22, 0.22, 0.3, 0.3, 0.32])
        estimation_bound = np.array([0.7, 1.9, 1.9, 1.9])

        estimate_net = EstimateNetwork(sess, state_dim, action_dim, estimation_bound, 0.001)
        control_net = ControlNetwork(sess, state_dim, action_dim, action_bound, init_pos, 0.0005)

        env = Monitor(env, MONITOR_DIR, force=True)
        # env.spec.timestep_limit = 0.6
        train(sess, env, control_net, estimate_net, action_bound, init_pos)


if __name__ == '__main__':
    tf.app.run()

# for i in range(1000):
#     env.render()
#     action = [0,0,0,-0.09,0,0,0,0,0,0,0,0,0]
#     state, reward, done, info = env.step(action)
#     print(state)
