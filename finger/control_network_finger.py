import tensorflow as tf

class ControlNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, init_pose, learning_rate):
        self.sess = sess
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.init_pose = init_pose
        self.learning_rate = learning_rate
        self.layer_node_1 = 300
        self.layer_node_2 = 400
        self.layer_node_3 = 400
        self.output_range = tf.constant([0.5, 0., 0., 0.])

        # Control Network
        self.inputs, self.scaled_outputs = self.create_control_network()
        self.net_params = tf.trainable_variables()

        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])

        self.train_gradients = tf.gradients(self.scaled_outputs, self.net_params, self.action_gradients)

        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.train_gradients, self.net_params))

    def create_control_network(self):
        inputs = tf.placeholder(tf.float32, [None, self.state_dim])

        with tf.name_scope('fc1'):
            w_fc1 = self.weight_variable([self.state_dim, self.layer_node_1])
            b_fc1 = self.bias_variable([self.layer_node_1])

            h_fc1 = tf.nn.relu(tf.matmul(inputs, w_fc1) + b_fc1)

        with tf.name_scope('fc2'):
            w_fc2 = self.weight_variable([self.layer_node_1, self.layer_node_2])
            b_fc2 = self.bias_variable([self.layer_node_2])

            h_fc2 = tf.nn.relu(tf.nn.relu(tf.matmul(h_fc1, w_fc2) + b_fc2))

        with tf.name_scope('fc3'):
            w_fc3 = self.weight_variable([self.layer_node_2, self.layer_node_3])
            b_fc3 = self.bias_variable([self.layer_node_3])

            h_fc3 = tf.nn.relu(tf.nn.relu(tf.matmul(h_fc2, w_fc3) + b_fc3))

        with tf.name_scope('fc4'):
            w_fc4 = self.weight_variable([self.layer_node_3, self.action_dim])
            b_fc4 = self.bias_variable([self.action_dim])

            outputs = tf.nn.sigmoid(tf.matmul(h_fc3, w_fc4) + b_fc4)
            scaled_outputs = tf.multiply(outputs, self.action_bound)

        return inputs, scaled_outputs

    def train(self, inputs, action_gradients):
        return self.sess.run(self.optimize, feed_dict={self.inputs: inputs, self.action_gradients: action_gradients})

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    def action(self, inputs):
        # inputs = tf.reshape(inputs, [None, self.state_dim])
        return self.sess.run(self.scaled_outputs, feed_dict={self.inputs: inputs})
    # def train(self, inputs, ):