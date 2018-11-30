import tensorflow as tf
from control_network_finger import ControlNetwork
from estimate_network_finger import EstimateNetwork
import numpy as np

def main(_):
    with tf.Session() as sess:


        state_dim = 4
        estimate_dim = 3
        action_dim = 5
        action_bound = np.array([-0.19, -0.19, -0.13, -0.1, -0.1])
        estimation_bound = np.array([0.7, 1.9, 1.9, 1.9])

        estimate_net = EstimateNetwork(sess, estimate_dim, action_dim, estimation_bound, 0.001)
        control_net = ControlNetwork(sess, state_dim, action_dim, action_bound, init_pos, 0.001)

        # env = Monitor(env, MONITOR_DIR, force=True)
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('./results/finger-v0/20180214202615/model'))


if __name__ == '__main__':
    tf.app.run()