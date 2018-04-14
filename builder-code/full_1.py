import tensorflow as tf
from nn4omtf import utils
from nn4omtf.dataset.const import HITS_TYPE
from nn4omtf.network import SIGN_OUT_SZ

def create_nn():
    arr = [0, 5, 10, 15, 20, 25, 30]
    l = len(arr) + 1
    in_sz = 18 * 14 # HITS_FULL size
    ls_pt = [in_sz, 128, 64, 32, l]
    ls_sgn = [in_sz, 128, 32, 16, SIGN_OUT_SZ]
    x = tf.placeholder(tf.float32, [None, 18, 14])
    rx = tf.reshape(x, [-1, in_sz])
    
    y = rx
    for sz in ls_pt[1:-1]:
        y = utils.mk_fc_layer(y, sz, act_fn=tf.nn.relu)
    y_pt = utils.mk_fc_layer(y, ls_pt[-1])
    
    y = rx
    for sz in ls_sgn[1:-1]:
        y = utils.mk_fc_layer(y, sz, act_fn=tf.nn.relu)
    y_sgn = utils.mk_fc_layer(y, ls_sgn[-1])

    return x, y_pt, y_sgn, arr, HITS_TYPE.FULL

