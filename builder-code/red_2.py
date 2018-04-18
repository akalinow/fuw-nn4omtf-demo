import tensorflow as tf
from nn4omtf import utils
from nn4omtf.const import HITS_TYPE
import numpy as np

def create_nn(is_training_ph, sgn_sz):
    """
    This method builds TensorFlow graph inside OMTFNN object.
    It must have following arguments which are automaticaly feed 
    by framework.
    Args:
        istraining_ph: placeholder for indicating train or valid/test phase
        sgn_sz: size of sign output layer
    """
    in_sz = np.prod(HITS_TYPE.REDUCED_SHAPE)
    hits_ph = tf.placeholder(tf.float32, shape=HITS_TYPE.REDUCED_PH_SHAPE)
    arr = [0, 10, 20, 30]
    pt_sz = len(arr) + 1
    ls_pt = [in_sz, 128, 64, 32, pt_sz]
    ls_sgn = [in_sz, 64, 32, 16, sgn_sz]
    x = tf.reshape(hits_ph, [-1, in_sz])
    y = x
    for sz in ls_pt[1:-1]:
        # Pass is_training_ph to setup batch normalization on these layers
        y = utils.mk_fc_layer(y, sz, act_fn=tf.nn.relu, 
                is_training=is_training_ph)
    y_pt = utils.mk_fc_layer(y, ls_pt[-1], is_training=is_training_ph)
    
    y = x
    for sz in ls_sgn[1:-1]:
        # Batch normalization will not be applied here
        y = utils.mk_fc_layer(y, sz, act_fn=tf.nn.relu, is_training=is_training_ph)
    y_sgn = utils.mk_fc_layer(y, ls_sgn[-1],is_training=is_training_ph)

    return hits_ph, y_pt, y_sgn, arr, HITS_TYPE.REDUCED

