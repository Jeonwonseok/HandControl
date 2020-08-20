import gym
from gym.wrappers import Monitor
import tensorflow as tf
from control_network_finger import ControlNetwork
from estimate_network_finger import EstimateNetwork
import numpy as np
import datetime

ENV_NAME = 'finger-v1'
DATETIME = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

MAX_EPISODES = 10
EPISODE_STEPS = 300

EXPLORATION_STEPS = 3000

SUMMARY_DIR = './results/{}/{}/tf'.format(ENV_NAME, DATETIME)
MONITOR_DIR = './results/{}/{}/gym'.format(ENV_NAME, DATETIME)
MODEL_DIR = './results/{}/{}/model/model.ckpt'.format(ENV_NAME, DATETIME)

def init_pose():
    tendon1 = 0.062
    tendon2 = 0.062
    tendon3 = 0.062
    tendon4 = 0.107
    tendon5 = 0.132

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

    action_1 = tf.Variable(0.)
    action_2 = tf.Variable(0.)
    action_3 = tf.Variable(0.)
    action_4 = tf.Variable(0.)
    action_5 = tf.Variable(0.)
    tf.summary.scalar('Action_1', action_1)
    tf.summary.scalar('Action_2', action_2)
    tf.summary.scalar('Action_3', action_3)
    tf.summary.scalar('Action_4', action_4)
    tf.summary.scalar('Action_5', action_5)

    summary_vars = [estimate_error_1, estimate_error_2, estimate_error_3, estimate_error_4,
                    control_error_1, control_error_2, control_error_3, control_error_4,
                    estimate_rmse, control_rmse, desired_input_1, desired_input_2, desired_input_3, desired_input_4,
                    action_1, action_2, action_3, action_4, action_5]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def train(sess, env, control_net, estimate_net, action_bound, init_pose):
    summary_ops, summary_vars = summaries()

    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
    saver = tf.train.Saver()
    saver.restore(sess, tf.train.latest_checkpoint('./results/finger-v0/20180214202615/model'))

    desired_bound = np.array([0.7, 1.57, 1.57, 1.57])
    desired_range = np.array([0.5, 0, 0, 0])
    estimate_net.update_target_network()

    for i in range(MAX_EPISODES):
        env.reset()
        desired_state = np.multiply(desired_bound, np.random.rand(1, 4) - desired_range)

        if i < EXPLORATION_STEPS:
            action = np.multiply(np.array([-0.1, -0.1, -0.1, -0.1, -0.1]), np.random.rand(1, 5))
            action = action.reshape([5])
        else:
            action = control_net.action(np.reshape(desired_state, (1, 4)))
            action = action.reshape([5])
            estimate_net.learning_rate = 0.0002

        if i > 12000:
            control_net.learning_rate = 0.0001
            estimate_net.learning_rate = 0.0001

        action_mod = np.zeros([9])
        action_mod[0:5] = action[0:5]
        action_mod[5:9] = desired_state.reshape([4])

        for j in range(EPISODE_STEPS):
            if i % 500 == 0 or i == MAX_EPISODES-1:
                env.render()

            state, reward, done, info = env.step(action_mod)

        # env.render(close=True)

        estimate = estimate_net.estimate(np.reshape(action, (1, 5)))
        estimate_target = estimate_net.estimate_target(np.reshape(action, (1, 5)))

        if i > EXPLORATION_STEPS:
            action_gradients = estimate_net.action_gradients(np.reshape(action, (1, 5)), np.reshape(desired_state, (1, 4)))

            control_net.train(np.reshape(desired_state, (1, 4)), action_gradients[0])

        # estimate_net.train(np.reshape(action, (1, 5))
        #                    + np.multiply(np.array([-0.04, -0.04, -0.04, -0.04, -0.04])
        #                                  , np.random.rand(1, 5) - np.array([0.5, 0.5, 0.5, 0.5, 0.5]))
        #                    , np.reshape(state, (1, 4)))

        estimate_net.train(np.reshape(action, (1, 5)), np.reshape(state, (1, 4)))

        if i % 30 == 0:
            estimate_net.update_target_network()

        estimate_error = estimate - state
        control_error = desired_state - state
        error = np.append(estimate_error, control_error)
        estimate_rmse = np.sqrt((estimate_error ** 2).mean())
        control_rmse = np.sqrt((control_error ** 2).mean())
        sum_vars = np.append(np.append(np.append(error, [estimate_rmse, control_rmse]), desired_state), action)
        summary_str = sess.run(summary_ops, feed_dict={summary_vars[k]: sum_vars[k] for k in range(len(summary_vars))})
        writer.add_summary(summary_str, i)
        writer.flush()

        if i % 100 == 0:
            if i < EXPLORATION_STEPS:
                print(i)
                print('real state:      ', state)
                print('estimate:        ', estimate)
                print('estimate_error:  ', estimate - state)
                print('action:          ', action_mod[0:5])
            else:
                print(i)
                print('desired:         ', desired_state)
                # print('estimate_target: ', estimate_target)
                # print('estimate:        ', estimate)
                print('real state:      ', state)
                print('target_est_error:', estimate_target - state)
                print('estimate_error:  ', estimate-state)
                print('control_error:   ', desired_state - state)
                print('action:          ', action_mod[0:5])
    saver.save(sess, MODEL_DIR)

def main(_):
    with tf.Session() as sess:
        env = gym.make(ENV_NAME)
        env.reset()

        init_pos = init_pose()

        state_dim = 4
        estimate_dim = 3
        action_dim = 5
        action_bound = np.array([-0.19, -0.19, -0.13, -0.1, -0.1])
        estimation_bound = np.array([0.7, 1.9, 1.9, 1.9])

        estimate_net = EstimateNetwork(sess, estimate_dim, action_dim, estimation_bound, 0.001)
        control_net = ControlNetwork(sess, state_dim, action_dim, action_bound, init_pos, 0.001)

        # env = Monitor(env, MONITOR_DIR, force=True)
        train(sess, env, control_net, estimate_net, action_bound, init_pos)


if __name__ == '__main__':
    tf.app.run()

