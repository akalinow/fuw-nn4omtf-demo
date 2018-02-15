import tensorflow as tf
from nn4omtf import utils
from nn4omtf.dataset.const import HITS_TYPE 

def create_nn():
    arr = [5, 10, 16, 50, 100, 500]
    l = len(arr) + 1
    FL = 21
    x = tf.placeholder(tf.float32, [None, 18, 2])
    rx = tf.reshape(x, [-1, 36])
    with tf.name_scope("fc1"):
        # First layer, fully-connected
        W_fc1 = utils.weight_variable([36, FL])
        b_fc1 = utils.bias_variable([FL])
        h_fc1 = tf.nn.relu(tf.matmul(rx, W_fc1) + b_fc1)
        utils.add_summary(W_fc1, add_hist=False)

    with tf.name_scope("fc2"):
        # Second layer, fully-connected
        W_fc2 = utils.weight_variable([FL, l])
        b_fc2 = utils.bias_variable([l])
        y = tf.matmul(h_fc1, W_fc2) + b_fc2
    
    return x, y, arr, HITS_TYPE.REDUCED

