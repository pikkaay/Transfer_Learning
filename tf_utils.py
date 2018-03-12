#Placeholder
import tensorflow as tf
import time


def create_placeholders(n_H0, n_W0, n_C0, n_y): 
    X = tf.placeholder(tf.float32, [None, n_H0, n_W0, n_C0])
    Y = tf.placeholder(tf.float32, [None, n_y])
    return X, Y

def conv_layer(input, size_in, size_out, name="conv"):
    with tf.name_scope(name):
        w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
        with tf.name_scope("relu"):
            act = tf.nn.relu(conv + b)
            tf.summary.histogram("weights", w)
            tf.summary.histogram("biases", b)
            tf.summary.histogram("activations", act)
        with tf.name_scope("max_pool"):
            max_pool = tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
    return max_pool

def compute_cost(Z, Y):
    with tf.name_scope("cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z, labels = Y))
        tf.summary.scalar("cost", cost)
    return cost

#Initialize parameters
def initialize_parameters():
    W1 = tf.get_variable("W1", [4,4,3,8], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    W2 = tf.get_variable("W2", [2,2,8,16], initializer = tf.contrib.layers.xavier_initializer(seed = 0))
    parameters = {"W1": W1,
                  "W2": W2}
    return parameters

def fc_layer(input, size_in, size_out, name="fc", activation = None):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.matmul(input, w) + b
    if activation == 'relu' :
        act = tf.nn.relu(act)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act

def delete_summary_dir(summaries_dir):
    if tf.gfile.Exists(summaries_dir):
        tf.gfile.DeleteRecursively(summaries_dir)
    tf.gfile.MakeDirs(summaries_dir)

def save_tensorboard(graph):
    with tf.Session(graph=graph) as sess:
        filename = "./summary_log/VS-"+time.strftime("%H%M%S")
        writer = tf.summary.FileWriter(filename, sess.graph)
    print("Tensorboard summary saved to "+filename) 
















