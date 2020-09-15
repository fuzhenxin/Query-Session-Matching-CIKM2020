import tensorflow as tf
import numpy as np
import pickle as pickle

import utils.layers as layers
import utils.operations as op
from models.mem import mem_update
import copy

# turn3 is the future
def cs_model(input_x, input_x_mask, input_x_len, input_x2, input_x_mask2, input_x_len2, input_x3, input_x_mask3, input_x_len3, word_emb, conf, initializer_opt=None):

    turn_num1 = input_x.shape[1] #conf["max_turn_num"] # tf.shape(input_x)[1]
    turn_num2 = input_x2.shape[1] # conf["max_turn_num"] #tf.shape(input_x2)[1]
    turn_num3 = input_x3.shape[1] # conf["max_turn_num"] #tf.shape(input_x2)[1]
    sent_len = conf["max_turn_len"]
    emb_dim = conf["emb_size"]
    matchin_include_x = conf["matchin_include_x"]

    data_type = []
    if "history" in conf["cs_type"]: data_type.append("history")
    if "future" in conf["cs_type"]: data_type.append("future")

    merge_hf = False

    # EMB
    x_e = tf.nn.embedding_lookup(word_emb, input_x)
    x2_e = tf.nn.embedding_lookup(word_emb, input_x2)
    x3_e = tf.nn.embedding_lookup(word_emb, input_x3)

    x_e_mb = tf.reshape(x_e, [-1, sent_len, emb_dim])
    x2_e_mb = tf.reshape(x2_e, [-1, sent_len, emb_dim])
    x3_e_mb = tf.reshape(x3_e, [-1, sent_len, emb_dim])

    x_len_mb = tf.reshape(input_x_len, [-1])
    x_len2_mb = tf.reshape(input_x_len2, [-1])
    x_len3_mb = tf.reshape(input_x_len3, [-1])

    x_mask = tf.to_float(input_x_mask) # bs turn_num1 sent_len
    x2_mask = tf.to_float(input_x_mask2) # bs turn_num2 sent_len
    x3_mask = tf.to_float(input_x_mask3) # bs turn_num3 sent_len


    # ==================================== Encoder Layer =============================
    with tf.variable_scope("Encode", reuse=tf.AUTO_REUSE):
        with tf.variable_scope('enc_self_att', reuse=tf.AUTO_REUSE):
            x_enc_mb = layers.block(x_e_mb,x_e_mb,x_e_mb,Q_lengths=x_len_mb,K_lengths=x_len_mb) # bs*turn_num1 sent_len emb
            x2_enc_mb = layers.block(x2_e_mb,x2_e_mb,x2_e_mb,Q_lengths=x_len2_mb,K_lengths=x_len2_mb) # bs*turn_num2 sent_len emb
            x3_enc_mb = layers.block(x3_e_mb,x3_e_mb,x3_e_mb,Q_lengths=x_len3_mb,K_lengths=x_len3_mb) # bs*turn_num2 sent_len emb


    x_enc = tf.reshape(x_enc_mb, [-1, turn_num1, sent_len, emb_dim]) # bs turn_num1 sent_len emb
    x_mask_flat = tf.reshape(x_mask, [-1, turn_num1*sent_len]) # bs turn_num1*sent_len
    x_enc_ts = tf.reshape(x_enc, [-1, turn_num1*sent_len, emb_dim])
    x_mask_ts = tf.reshape(x_mask_flat, [-1, turn_num1*sent_len])


    iter_rep = []
    input_all_dict = {"history": [], "future": []}
    input_all_dict["history"] = [x2_e_mb, x2_enc_mb, x_len2_mb, turn_num2, x2_mask]
    input_all_dict["future"] = [x3_e_mb, x3_enc_mb, x_len3_mb, turn_num3, x3_mask]
    save_dynamic_dict = {}
    all_mem_weight_dict = {}
    sim_ori_all = []

    for match_type in data_type:
        x5_e_mb, x5_enc_mb, x_len5_mb, turn_num5, x5_mask = input_all_dict[match_type]
        scope_name = "Model"
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

            
            x5_enc = tf.reshape(x5_enc_mb, [-1, turn_num5, sent_len, emb_dim]) # bs turn_num1 sent_len emb

            x5_enc_ts = tf.reshape(x5_enc_mb, [-1, turn_num5*sent_len, emb_dim])
            x5_mask_ts = tf.reshape(x5_mask, [-1, turn_num5*sent_len])

            # ==================================== Static Memory Layer =============================
            if conf["use_static_memory"]:
                with tf.variable_scope("static_memory", reuse=tf.AUTO_REUSE):

                    if merge_hf:
                        x_enc_ts = tf.reshape(x_enc, [-1, turn_num1*sent_len, emb_dim])
                        x_mask_ts = tf.reshape(x_mask_flat, [-1, turn_num1*sent_len])
                        iter_rep_per1, iter_rep_per5 = match_layer(x5_enc_ts, x5_mask_ts, x_enc_ts, x_mask_ts, emb_dim, initializer_opt, turn_num5=None)
                        iter_rep_per_out = tf.concat([iter_rep_per1, iter_rep_per5], axis=1)
                        print(iter_rep_per_out.shape)
                        iter_rep.append(iter_rep_per_out)
                    else:
                        x5_enc_mb = x5_enc_mb # bs*turn_num2 sent_len emb_dim
                        x5_mask_mb = tf.reshape(x5_mask, [-1, sent_len])
                        x_enc_mb2_ts = tf.reshape(tf.tile(tf.expand_dims(x_enc, axis=1), [1, turn_num5, 1, 1, 1]), [-1, turn_num1*sent_len, emb_dim]) # bs*turn_num2 sent_len*turn_num1 emb_dim
                        x_mask_mb2_ts = tf.reshape(tf.tile(tf.expand_dims(x_mask_flat, axis=1), [1, turn_num5, 1]), [-1, turn_num1*sent_len])
                        x5_enc_ts = tf.reshape(x5_enc_mb, [-1, turn_num5*sent_len, emb_dim])
                        x5_e_ts = tf.reshape(x5_e_mb, [-1, turn_num5*sent_len, emb_dim])
                        x5_mask_ts = tf.reshape(x5_mask, [-1, turn_num5*sent_len])
                        x_e_ts = tf.reshape(x_e, [-1, turn_num1*sent_len, emb_dim])
                        iter_rep_per_out, sim_ori = match_layer_selfatt(x5_e_ts, x5_enc_ts, x5_mask_ts, x_e_ts, x_enc_ts, x_mask_ts, emb_dim, initializer_opt, turn_num5=None, matchin_include_x=matchin_include_x)
                        sim_ori_all.append(sim_ori)
                        iter_rep.append(iter_rep_per_out)

            # ==================================== Dynamic Memory Layer Local =============================
            if conf["use_dynamic_memory"]:
                with tf.variable_scope("dynamic_memory", reuse=tf.AUTO_REUSE):
                    x5_enc_list = tf.unstack(x5_enc, axis=1) # bs [turn_num2] sent_len emb_dim
                    x5_mask_list = tf.unstack(x5_mask, axis=1) # bs [turn_num2] sent_len

                    x_enc_list = tf.unstack(x_enc, axis=1)
                    x_mask_list = tf.unstack(x_mask, axis=1)
                    x_enc_cur_list, x_mask_cur_list, _, _, _, _, all_mem_weight = mem_all_update(x_enc_list, x_mask_list, initializer_opt, need_reverse=True)
                    all_mem_weight = tf.stack(all_mem_weight, axis=1)
                    all_mem_weight_dict[match_type+"_query"] = all_mem_weight

                    x_enc_mb2_ts = tf.reshape(x_enc_cur_list, [-1, turn_num1*sent_len, emb_dim])
                    x_mask_mb2_ts = tf.reshape(x_mask_cur_list, [-1, turn_num1*sent_len])

                    if match_type=="history": need_reverse=True
                    else: need_reverse=False

                    x5_enc_cur_list, x5_mask_cur_list, x5_enc_list, x5_mask_list, x5_enc_cur_last, x5_mask_cur_last, all_mem_weight = mem_all_update(x5_enc_list, x5_mask_list, initializer_opt, need_reverse=need_reverse)
                    all_mem_weight = tf.stack(all_mem_weight, axis=1)
                    all_mem_weight_dict[match_type] = all_mem_weight

                    turn_xishu=turn_num5
                    # else: turn_xishu=1
                    x5_enc_cur_list = tf.reshape(x5_enc_cur_list, [-1, turn_xishu*sent_len, emb_dim])
                    x5_mask_cur_list = tf.reshape(x5_mask_cur_list, [-1, turn_xishu*sent_len])

                    save_dynamic_dict[match_type] = [x5_e_mb, x5_enc_cur_list, x5_mask_cur_list, x_enc_mb2_ts, x_mask_mb2_ts, x5_enc_list, x5_mask_list, turn_num5, x5_enc_ts, x5_mask_ts, x5_enc_cur_last, x5_mask_cur_last]


    # ==================================== Dynamic Memory Layer Global =============================
    data_type_dm2 = copy.deepcopy(data_type)
    # data_type_dm2 = []
    if conf["use_dynamic_memory"] and conf["dynamic_memory_global"]:
        for match_type in data_type:
            scope_name = "Model"
            with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
                with tf.variable_scope("dynamic_memory_part2", reuse=tf.AUTO_REUSE):
                    x5_e_mb, x5_enc_cur_list, x5_mask_cur_list, x_enc_mb2_ts, x_mask_mb2_ts, x5_enc_list, x5_mask_list, turn_num5, x5_enc_ts, x5_mask_ts, x5_enc_cur_last, x5_mask_cur_last = save_dynamic_dict[match_type]
                    

                    h1 = save_dynamic_dict["history"][-4]
                    h2 = save_dynamic_dict["history"][-3]
                    hq1 = x_enc_ts
                    hq2 = x_mask_ts
                    f1 = save_dynamic_dict["future"][-4]
                    f2 = save_dynamic_dict["future"][-3]
                    g_last_his = [h1, h2 ]
                    g_last_fut = [f1, f2 ]

                    g_last_fut = None
                    g1 = tf.concat([h1, f1, hq1], axis=1)
                    g2 = tf.concat([h2, f2, hq2], axis=1)
                    g_last_his = [g1, g2]
                    x5_mask_cur_last, x5_enc_cur_last = None, None

                    x_enc_cur_list, x_mask_cur_list, _, _, _, _, all_mem_weight = mem_all_update(x_enc_list, x_mask_list, initializer_opt, need_reverse=True, g_last_his=g_last_his, g_last_fut=g_last_fut)

                    x5_enc_cur_list, x5_mask_cur_list, x5_enc_list, x5_mask_list, x5_enc_cur_last, x5_mask_cur_last, all_mem_weight = mem_all_update(x5_enc_list, x5_mask_list, initializer_opt, x_enc_cur_last=x5_enc_cur_last, x_mask_cur_last=x5_mask_cur_last, g_last_his=g_last_his, g_last_fut=g_last_fut)

                    x_enc_mb2_ts = tf.reshape(x_enc_cur_list, [-1, turn_num1*sent_len, emb_dim])
                    x_mask_mb2_ts = tf.reshape(x_mask_cur_list, [-1, turn_num1*sent_len])

                    turn_xishu=turn_num5
                    x5_enc_cur_list = tf.reshape(x5_enc_cur_list, [-1, turn_xishu*sent_len, emb_dim])
                    x5_mask_cur_list = tf.reshape(x5_mask_cur_list, [-1, turn_xishu*sent_len])

                    save_dynamic_dict[match_type+"_global"] = [x5_e_mb, x5_enc_cur_list, x5_mask_cur_list, x_enc_mb2_ts, x_mask_mb2_ts, x5_enc_list, x5_mask_list, turn_num5, x5_enc_ts, x5_mask_ts, x5_enc_cur_last, x5_mask_cur_last]
                    data_type_dm2.append(match_type+"_global")

    # ==================================== Dynamic Memory Layer AGG =============================
    for match_type in data_type_dm2:
        scope_name = "Model"
        # if conf["sepqrate_cs"]: scope_name = "Model"+match_type
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            if conf["use_dynamic_memory"]:
                scope_name1 = "dynamic_memory_part3" if "global" not in match_type else "dynamic_memory_part3_global"
                with tf.variable_scope(scope_name1, reuse=tf.AUTO_REUSE):
                    x5_e_mb, x5_enc_cur_list, x5_mask_cur_list, x_enc_mb2_ts, x_mask_mb2_ts, _, _, turn_num5, _, _, _, _ =  save_dynamic_dict[match_type]

                    x_e_ts = tf.reshape(x_e, [-1, turn_num1*sent_len, emb_dim])
                    x5_e_ts = tf.reshape(x5_e_mb, [-1, turn_num5*sent_len, emb_dim])
                    iter_rep_per_out, sim_ori = match_layer_selfatt(x5_e_ts, x5_enc_cur_list, x5_mask_cur_list, x_e_ts, x_enc_mb2_ts, x_mask_mb2_ts, emb_dim, initializer_opt, turn_num5=None, matchin_include_x=matchin_include_x)
                    iter_rep.append(iter_rep_per_out)

    iter_rep_con = tf.concat(iter_rep, axis=1)
    return iter_rep_con, iter_rep, all_mem_weight_dict, save_dynamic_dict, sim_ori_all



def agg_layer2(iter_rep_per1, initializer_opt):
    with tf.variable_scope("agg2layer"):
        iter_rep_per_out1_mean = tf.reduce_sum(iter_rep_per1, 1)
        iter_rep_per_out1_max = tf.reduce_max(iter_rep_per1, 1)
        iter_rep_per_out1 = tf.concat([iter_rep_per_out1_max, iter_rep_per_out1_mean], axis=1)
    return iter_rep_per_out1


def mem_all_update(x_enc_list, x_mask_list, initializer_opt, need_reverse=False, x_enc_cur_last=None, x_mask_cur_last=None, g_last_his=None, g_last_fut=None):

    # print(g_last_his[0].shape) # bs 120 200
    # print(g_last_his[1].shape) # bs 120 200
    # print(g_last_fut[0].shape) # bs 120 200
    # print(g_last_fut[1].shape) # bs 120 200

    if need_reverse:
        x_enc_list.reverse()
        x_mask_list.reverse() 

    if x_enc_cur_last is None:
        x_enc_cur_last = x_enc_list[0]
    if x_mask_cur_last is None:
        x_mask_cur_last = x_mask_list[0]

    x_enc_cur_list, x_mask_cur_list = [], []
    all_mem_weight = []
    for i in range(len(x_enc_list)):
        x_enc_cur = x_enc_list[i] # bs sent_len emb_dim
        x_mask_cur = x_mask_list[i] # bs sent_len

        x_enc_cur_last, x_mask_cur_last, update1, update2, all_mem_weight_per = mem_update(x_enc_cur, x_mask_cur, x_enc_cur_last, x_mask_cur_last, x_enc_list[0], x_mask_list[0], initializer_opt, g_layer1=[g_last_his,g_last_fut])

        x_enc_cur_list.append(update1)
        x_mask_cur_list.append(update2)
        all_mem_weight.append(all_mem_weight_per)

    x_enc_cur_list = tf.stack(x_enc_cur_list, axis=1) # bs turn sent_len emb_dim
    x_mask_cur_list = tf.stack(x_mask_cur_list, axis=1) # bs turn sent_len
    return x_enc_cur_list, x_mask_cur_list, x_enc_list, x_mask_list, x_enc_cur_last, x_mask_cur_last, all_mem_weight


def match_layer(x5_enc_cur_list, x5_mask_cur_list, x_enc_mb2_ts, x_mask_mb2_ts, emb_dim, initializer_opt, turn_num5):

    # att_t1_batch bs*turn_num5 sent_len*turn_num1 emb_dim
    # att_t5_batch bs*turn_num5 sent_len emb_dim
    att_t5_batch, att_t1_batch = layers.local_inference(x5_enc_cur_list, x5_mask_cur_list, x_enc_mb2_ts, x_mask_mb2_ts)

    t5_match_batch = tf.concat([x5_enc_cur_list, att_t5_batch, x5_enc_cur_list*att_t5_batch, x5_enc_cur_list-att_t5_batch], 2)
    t1_match_batch = tf.concat([x_enc_mb2_ts, att_t1_batch, x_enc_mb2_ts*att_t1_batch, x_enc_mb2_ts-att_t1_batch], 2)

    t5_match_small_batch = tf.layers.dense(t5_match_batch, emb_dim/2, activation=tf.nn.relu, name="fnn", kernel_initializer=initializer_opt)
    t1_match_small_batch = tf.layers.dense(t1_match_batch, emb_dim/2, activation=tf.nn.relu, name="fnn", kernel_initializer=initializer_opt) 

    t5_match_rnn_batch, _ = layers.bilstm_layer(t5_match_small_batch, 1, emb_dim/2, initializer_opt=initializer_opt)
    t1_match_rnn_batch, _ = layers.bilstm_layer(t1_match_small_batch, 1, emb_dim/2, initializer_opt=initializer_opt)

    t5_match_rnn_mean = tf.reduce_sum(t5_match_rnn_batch * tf.expand_dims(x5_mask_cur_list, -1), 1) / tf.expand_dims(tf.reduce_sum(x5_mask_cur_list, 1), 1)
    t5_match_rnn_max = tf.reduce_max(t5_match_rnn_batch * tf.expand_dims(x5_mask_cur_list, -1), 1)
    t1_match_rnn_mean = tf.reduce_sum(t1_match_rnn_batch * tf.expand_dims(x_mask_mb2_ts, -1), 1) / tf.expand_dims(tf.reduce_sum(x_mask_mb2_ts, 1), 1)
    t1_match_rnn_max = tf.reduce_max(t1_match_rnn_batch * tf.expand_dims(x_mask_mb2_ts, -1), 1)

    t5_match_rnn_batch = tf.concat([t5_match_rnn_max, t5_match_rnn_mean], axis=1) # bs*turn_num5 emb_dim*5
    t1_match_rnn_batch = tf.concat([t1_match_rnn_max, t1_match_rnn_mean], axis=1) # bs*turn_num5 emb_dim*5

    if turn_num5 is not None:
        iter_rep_per5 = tf.reshape(t5_match_rnn_batch, [-1, turn_num5, t5_match_rnn_batch.shape[1]], name="Final_Reshape5") # bs turn_num5 emb_dim*5
        iter_rep_per1 = tf.reshape(t1_match_rnn_batch, [-1, turn_num5, t1_match_rnn_batch.shape[1]], name="Final_Reshape1") # bs turn_num2 emb_dim*2
        return iter_rep_per1, iter_rep_per5
    else:
        return t1_match_rnn_batch, t5_match_rnn_batch
    

def match_layer_selfatt(x5_e_ts, x5_enc_cur_list, x5_mask_cur_list, x_e_ts, x_enc_mb2_ts, x_mask_mb2_ts, emb_dim, initializer_opt, turn_num5, matchin_include_x=False):
    # print(x5_enc_cur_list.shape)  # bs*turn5 sent_len emb_dim
    # print(x5_mask_cur_list.shape) # bs*turn5 sent_len
    # print(x_enc_mb2_ts.shape) # bs*turn5 turn1*sent_len emb_dim
    # print(x_mask_mb2_ts.shape) # bs*turn5 turn1*sent_len

    if matchin_include_x:
        a_list = [x5_e_ts, x5_enc_cur_list]
        b_list = [x_e_ts, x_enc_mb2_ts]
    else:
        a_list = [x5_enc_cur_list]
        b_list = [x_enc_mb2_ts]

    with tf.variable_scope("atb", reuse=tf.AUTO_REUSE):
        atb = layers.block(x5_enc_cur_list, x_enc_mb2_ts, x_enc_mb2_ts, Q_lengths=x5_mask_cur_list, K_lengths=x_mask_mb2_ts, use_len=True)
    with tf.variable_scope("bta", reuse=tf.AUTO_REUSE):
        bta = layers.block(x_enc_mb2_ts, x5_enc_cur_list, x5_enc_cur_list, Q_lengths=x_mask_mb2_ts, K_lengths=x5_mask_cur_list, use_len=True)
    a_list.append(atb)
    b_list.append(bta)

    a_list = tf.stack(a_list, axis=-1)
    b_list = tf.stack(b_list, axis=-1)
    sim_ori = tf.einsum('biks,bjks->bijs', a_list, b_list)/tf.sqrt(200.0)
    sim = layers.CNN_FZX(sim_ori)
    if turn_num5 is not None:
        sim = tf.reshape(sim, [-1, turn_num5, sim.shape[-1]])
    return sim, sim_ori
