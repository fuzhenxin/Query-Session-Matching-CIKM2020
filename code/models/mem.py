import tensorflow as tf
import numpy as np
import pickle as pickle

import utils.layers as layers
import utils.operations as op

def mem_update(x5_enc_cur, x5_mask_cur, x5_enc_cur_last, x5_mask_cur_last, r_enc, r_enc_mask, initializer_opt, g_layer1=[None, None]):
    """
    x5_enc_cur
    x5_mask_cur
    x5_enc_cur_last
    x5_mask_cur_last
    x5_enc_cur_last
    x5_mask_cur_last
    x5_enc_cur_list
    x5_mask_cur_list
    """

    x5_enc_w = layers.attention(x5_enc_cur_last,x5_enc_cur,x5_enc_cur,x5_mask_cur_last,x5_mask_cur,use_len=True)
    x5_enc_w_w = tf.concat([x5_enc_cur_last, x5_enc_w], axis=2)
    w_w = tf.layers.dense(x5_enc_w_w, 1, activation=tf.nn.tanh, name="type75", kernel_initializer=initializer_opt)
    all_mem_weight = w_w
    x5_enc_cur_last_ori = x5_enc_cur_last
    x5_enc_cur_last = x5_enc_cur_last + x5_enc_w*w_w

    x5_enc_cur_last2 = x5_enc_cur_last

    if g_layer1[0]:
        with tf.variable_scope("gg1", reuse=tf.AUTO_REUSE):
            x5_enc_w = layers.attention(x5_enc_cur,g_layer1[0][0],g_layer1[0][0],x5_mask_cur,g_layer1[0][1],use_len=True)
            x5_enc_w = layers.attention(x5_enc_cur_last2,x5_enc_w,x5_enc_w,x5_mask_cur_last,x5_mask_cur,use_len=True)
            x5_enc_w_w = tf.concat([x5_enc_cur_last2, x5_enc_w], axis=2)
            w_w = tf.layers.dense(x5_enc_w_w, 1, activation=tf.nn.tanh, name="type751", kernel_initializer=initializer_opt)
            x5_enc_cur_last = x5_enc_cur_last + x5_enc_w*w_w
    return x5_enc_cur_last, x5_mask_cur_last, x5_enc_cur_last, x5_mask_cur_last, all_mem_weight
