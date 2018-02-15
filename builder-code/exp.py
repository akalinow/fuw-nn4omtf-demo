import tensorflow as tf
from nn4omtf import utils
from nn4omtf.dataset.const import HITS_TYPE 

def create_nn():
    arr = [5, 10, 16, 50, 100, 500]
    l = len(arr) + 1
    FL = 12
    SL = 24
    TL = 24
    x = tf.placeholder(tf.float32, [None, 18, 2])
    rx = tf.reshape(x, [-1, 36])
    
    with tf.name_scope("fc1_1"):
        W_fc1_1 = utils.weight_variable([36, FL])
        b_fc1_1 = utils.bias_variable([FL])
        h_fc1_1 = tf.nn.relu(tf.matmul(rx, W_fc1_1) + b_fc1_1)
        utils.add_summary(W_fc1_1, add_hist=False)

    with tf.name_scope("fc1_2"):
        W_fc1_2 = utils.weight_variable([36, FL])
        b_fc1_2 = utils.bias_variable([FL])
        h_fc1_2 = tf.nn.sigmoid(tf.matmul(rx, W_fc1_2) + b_fc1_2)
        utils.add_summary(W_fc1_2, add_hist=False)

    with tf.name_scope("fc2"):
        # Second layer, fully-connected
        W_fc2 = utils.weight_variable([FL, SL])
        b_fc2 = utils.bias_variable([SL])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_1 + h_fc1_2, W_fc2) + b_fc2)
        utils.add_summary(W_fc2, add_hist=False)

    with tf.name_scope("fc3"):
        # Second layer, fully-connected
        W_fc3 = utils.weight_variable([SL, TL])
        b_fc3 = utils.bias_variable([TL])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)
        utils.add_summary(W_fc3, add_hist=False)

    # Map the features on classes
    with tf.name_scope('fc4'):
        W_fc4 = utils.weight_variable([TL, l])
        b_fc4 = utils.bias_variable([l])
        y = tf.matmul(h_fc3, W_fc4) + b_fc4
    
    return x, y, arr, HITS_TYPE.REDUCED

