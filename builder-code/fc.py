# -*- coding: utf-8 -*-
"""
    Copyright(C) 2018 Jacek Łysiak
    MIT License
    
    Muon classifier - fully connected layers
"""


def create_nn(x, x_shape, is_training):
    """
    Args:
        x: input hits array
        x_shape: input tensor shape for single event
        is_training: placeholder for indicating train or valid/test phase

    Note: Only code in `create_nn` function scope will be exctracted and saved
        in model directory. It's important to provide all necessary imports
        within.
    """
    import tensorflow as tf
    from nn4omtf import utils
    import numpy as np
    arr = [0, 5, 10, 15, 20, 25, 30]
    out_sz = 2 * len(arr) + 1
    in_sz = np.prod(x_shape)

    hidden_layers = [128, 64, 64]
    x = tf.reshape(x, [-1, in_sz])
    for sz in hidden_layers:
        # Pass is_training to setup batch normalization on these layers
        x = utils.mk_fc_layer(x, sz, act_fn=tf.nn.relu, is_training=is_training)
    logits = utils.mk_fc_layer(x, out_sz, is_training=is_training)

    return logits, arr 

