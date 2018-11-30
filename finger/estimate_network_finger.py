import tensorflow as tf

class EstimateNetwork(object):

    def __init__(self, sess, state_dim, action_dim, estimate_bound, learning_rate):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.estimate_bound = estimate_bound
        self.learning_rate = learning_rate
        self.layer_node_1 = 300
        self.layer_node_2 = 300
        self.layer_node_3 = 400
        self.layer_node_4 = 400
        self.weight_estimate = tf.constant([1., 1., 1., 1.])
        self.weight_target = tf.constant([16., 1., 1., 1.])
        self.output_range = tf.constant([0.5, 0., 0., 0.])

        # Estimate Network
        self.actions, self.estimates = self.create_estimate_network('est')
        self.net_params = tf.trainable_variables()

        # Target Network
        self.actions_target, self.estimates_target = self.create_estimate_network('target')
        self.net_params_target = tf.trainable_variables()[len(self.net_params):]

        self.update_net_params_target = \
            [self.net_params_target[i].assign(self.net_params[i])
             for i in range(len(self.net_params_target))]

        self.observe_state = tf.placeholder(tf.float32, [None, self.state_dim])
        self.error = tf.scalar_mul(100, self.estimates - self.observe_state)
        self.loss = tf.reduce_mean(tf.multiply(tf.multiply(self.error, self.error), self.weight_estimate))
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.desired_state = tf.placeholder(tf.float32, [None, self.state_dim])
        # self.torque_sum = tf.scalar_mul(0.001, tf.reduce_mean(tf.square(self.actions_target)))
        self.error_target = tf.scalar_mul(100, (self.estimates_target - self.desired_state))
        self.loss_target = tf.reduce_mean(tf.multiply(tf.multiply(self.error_target, self.error_target)
                                                      , self.weight_target))

        self.actions_grads = tf.gradients(self.loss_target, self.actions_target)

    def create_estimate_network(self, name):
        actions = tf.placeholder(tf.float32, [None, self.action_dim])

        with tf.name_scope(name):
            with tf.name_scope('fc1'):
                w_fc1 = self.weight_variable([self.action_dim, self.layer_node_1])
                b_fc1 = self.bias_variable([self.layer_node_1])

                h_fc1 = tf.nn.relu(tf.matmul(actions, w_fc1) + b_fc1)

            with tf.name_scope('fc2'):
                w_fc2 = self.weight_variable([self.layer_node_1, self.layer_node_2])
                b_fc2 = self.bias_variable([self.layer_node_2])

                h_fc2 = tf.nn.relu(tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2))

            with tf.name_scope('fc3'):
                w_fc3 = self.weight_variable([self.layer_node_2, self.layer_node_3])
                b_fc3 = self.bias_variable([self.layer_node_3])

                h_fc3 = tf.nn.relu(tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3))

            # with tf.name_scope('fc4'):
            #     w_fc4 = self.weight_variable([self.layer_node_3, self.layer_node_4])
            #     b_fc4 = self.bias_variable([self.layer_node_4])
            #
            #     h_fc4 = tf.nn.relu(tf.nn.relu(tf.matmul(h_fc3, w_fc4) + b_fc4))

            with tf.name_scope('fc5'):
                w_fc5 = self.weight_variable([self.layer_node_4, self.state_dim])
                b_fc5 = self.bias_variable([self.state_dim])

                outputs = tf.subtract(tf.nn.sigmoid(tf.matmul(h_fc3, w_fc5) + b_fc5), self.output_range)
                estimates = tf.multiply(outputs, self.estimate_bound)

        return actions, estimates

    def train(self, actions, observe_state):
        return self.sess.run(self.optimize, feed_dict={self.actions: actions, self.observe_state: observe_state})

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def action_gradients(self, actions, desired_state):
        return self.sess.run(self.actions_grads, feed_dict={self.actions_target: actions, self.desired_state: desired_state})

    def update_target_network(self):
        self.sess.run(self.update_net_params_target)

    def estimate(self, actions):
        return self.sess.run(self.estimates, feed_dict={self.actions: actions})

    def estimate_target(self, actions):
        return self.sess.run(self.estimates_target, feed_dict={self.actions_target: actions})