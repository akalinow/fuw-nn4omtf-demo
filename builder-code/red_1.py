import tensorflow as tf
from nn4omtf import utils
from nn4omtf.dataset.const import HITS_TYPE 

def create_nn():
    arr = [2, 5, 10, 12.5, 15, 20, 25, 30]
    l = len(arr) + 1
    l_sgn = 2
    FL = 12
    SL = 24
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
        W_fc2 = utils.weight_variable([FL, SL])
        b_fc2 = utils.bias_variable([SL])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
        utils.add_summary(W_fc2, add_hist=False)

    # Map the features on classes
    with tf.name_scope('fc3'):
        W_fc3 = utils.weight_variable([SL, l])
        b_fc3 = utils.bias_variable([l])
        y_pt = tf.matmul(h_fc2, W_fc3) + b_fc3
    
    with tf.name_scope('fc3_sgn'):
        W_fc3_sgn = utils.weight_variable([SL, l_sgn])
        b_fc3_sgn = utils.bias_variable([l_sgn])
        y_sgn = tf.matmul(h_fc2, W_fc3_sgn) + b_fc3_sgn
    
    return x, y_pt, y_sgn, arr, HITS_TYPE.REDUCED

