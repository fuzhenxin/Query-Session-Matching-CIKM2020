import tensorflow as tf
import numpy as np
import pickle as pickle
import time

import utils.layers as layers
import utils.operations as op

from models.cs_net import cs_model

class Net(object):
    '''Add positional encoding(initializer lambda is 0),
       cross-attention, cnn integrated and grad clip by value.

    Attributes:
        conf: a configuration paramaters dict
        word_embedding_init: a 2-d array with shape [vocab_size+1, emb_size]
    '''
    def __init__(self, conf, is_train=False):
        self._graph = tf.Graph()
        self._conf = conf
        # con2 con3 con4 gru
        self.is_train = is_train

        if self._conf['word_emb_init'] is not None:
            print('loading word emb init')
            self._word_embedding_init = pickle.load(open(self._conf['word_emb_init'], 'rb'))
        else:
            self._word_embedding_init = None

    def build_graph(self):
        with self._graph.as_default():
            rand_seed = self._conf['rand_seed']
            tf.set_random_seed(rand_seed)

            #word embedding
            if self._word_embedding_init is not None:
                word_embedding_initializer = tf.constant_initializer(self._word_embedding_init)
            else:
                word_embedding_initializer = tf.random_normal_initializer(stddev=0.1)

            self._word_embedding = tf.get_variable(
                name='word_embedding',
                shape=[self._conf['vocab_size']+1, self._conf['emb_size']],
                dtype=tf.float32,
                initializer=word_embedding_initializer, trainable=True)

            batch_size = None
            initializer_opt = tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                             mode='FAN_AVG',
                                                             uniform=True,
                                                             dtype=tf.float32)
            initializer_opt = tf.truncated_normal_initializer(stddev=0.02)

            self.turns_sess_num = self._conf["max_turn_num_hf"]*2+1
            self.turns_q_num = self._conf["max_turn_num"]

            #define placehloders
            self.turns1 = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num"], self._conf["max_turn_len"]], name="turns1")
            self.tt_turns_len1 = tf.placeholder(tf.int32, shape=[batch_size,], name="tt_turns_len1")
            self.every_turn_len1 = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num"]], name="every_turn_len1")
            self.turns2 = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num_hf"], self._conf["max_turn_len"]], name="turns2")
            self.tt_turns_len2 = tf.placeholder(tf.int32, shape=[batch_size,], name="tt_turns_len2")
            self.every_turn_len2 = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num_hf"]], name="every_turn_len2")
            self.turnsf = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num_hf"], self._conf["max_turn_len"]], name="turnsf")
            self.tt_turns_lenf = tf.placeholder(tf.int32, shape=[batch_size,], name="tt_turns_lenf")
            self.every_turn_lenf = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num_hf"]], name="every_turn_lenf")
            self.response = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_len"]], name="response")
            self.response_len = tf.placeholder(tf.int32, shape=[batch_size,], name="response_len")
            self.turnsa = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_len"]*self.turns_sess_num], name="turnsa")
            self.turnsa_len = tf.placeholder(tf.int32, shape=[batch_size,], name="turnsa_len")
            self.turnsq = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_len"]*self.turns_q_num], name="turnsq")
            self.turnsq_len = tf.placeholder(tf.int32, shape=[batch_size,], name="turnsq_len")
            self.keep_rate = tf.placeholder(tf.float32, [], name="keep_rate") 

            self.turns_sess = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num_sess"], self._conf["max_turn_len"]], name="turns_sess")
            self.tt_turns_len_sess = tf.placeholder(tf.int32, shape=[batch_size,], name="tt_turns_len_sess")
            self.every_turn_len_sess = tf.placeholder(tf.int32, shape=[batch_size, self._conf["max_turn_num_sess"]], name="every_turn_len_sess")

            self.label = tf.placeholder(tf.float32, shape=[batch_size,])


            # ==================================== CS Model =============================
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "Starting build CS Model")

            input_x =self.turns1
            input_x_len = self.every_turn_len1
            input_x_mask = tf.sequence_mask(input_x_len, self._conf["max_turn_len"])
            
            input_xf = self.turnsf
            input_x_lenf = self.every_turn_lenf

            input_xf = tf.concat([tf.expand_dims(self.response, axis=1), input_xf], axis=1)
            input_x_lenf = tf.concat([ input_x_lenf, tf.expand_dims(self.response_len, axis=1)], axis=1)
            input_x_maskf = tf.sequence_mask(input_x_lenf, self._conf["max_turn_len"])

            input_x2 = self.turns2
            input_x_len2 = self.every_turn_len2
            input_x2 = tf.concat([ input_x2, tf.expand_dims(self.response, axis=1)], axis=1)
            input_x_len2 = tf.concat([ input_x_len2, tf.expand_dims(self.response_len, axis=1)], axis=1)
            input_x_mask2 = tf.sequence_mask(input_x_len2, self._conf["max_turn_len"])


            with tf.variable_scope('model_crdms'):
                final_info_cs, final_info_css, self.all_mem_weight_dict, self.save_dynamic_dict, self.sim_ori = cs_model(input_x, input_x_mask, input_x_len, input_x2, input_x_mask2, input_x_len2, input_xf, input_x_maskf, input_x_lenf, self._word_embedding, self._conf)
                final_info_cs = tf.layers.dense(final_info_cs, 50, kernel_initializer=tf.contrib.layers.xavier_initializer())


            # ==================================== Calculate Loss =============================
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "Starting calculate Loss")
            self.trainops = {"cs": dict()}
            all_loss_inouts = [
                    ["cs", final_info_cs],
                ]
            for loss_type, loss_input in all_loss_inouts:
                if loss_type!=self._conf["train_type"] and loss_type!="cr": continue
                with tf.variable_scope('loss_'+loss_type):

                    self.trainops[loss_type]["loss"], self.trainops[loss_type]["logits"] = layers.loss(loss_input, self.label)

                    use_loss_weight = True
                    loss_added, logits_added = [], []
                    num_loss = len(final_info_css)
                    for i,j in enumerate(final_info_css):
                        with tf.variable_scope("losscc"+str(i)): loss_per, logits_per = layers.loss(j, self.label)
                        if num_loss==6 and i>=2 and use_loss_weight:
                            loss_per = loss_per*0.5
                            logits_per = logits_per*0.5
                        loss_added.append(loss_per)
                        logits_added.append(logits_per)
                    if num_loss==6 and use_loss_weight: num_loss=num_loss-2
                    self.trainops[loss_type]["loss"] += sum(loss_added)/num_loss
                    self.trainops[loss_type]["logits"] += sum(logits_added)/num_loss

                    self.trainops[loss_type]["global_step"] = tf.Variable(0, trainable=False)
                    initial_learning_rate = self._conf['learning_rate']
                    self.trainops[loss_type]["learning_rate"] = tf.train.exponential_decay(
                        initial_learning_rate,
                        global_step=self.trainops[loss_type]["global_step"],
                        decay_steps=self._conf["decay_step"],
                        decay_rate=0.9,
                        staircase=True)

                    Optimizer = tf.train.AdamOptimizer(self.trainops[loss_type]["learning_rate"])
                    self.trainops[loss_type]["optimizer"] = Optimizer.minimize(self.trainops[loss_type]["loss"])

                    self.trainops[loss_type]["grads_and_vars"] = Optimizer.compute_gradients(self.trainops[loss_type]["loss"])

                    self.trainops[loss_type]["capped_gvs"] = [(tf.clip_by_value(grad, -1, 1), var) for grad, var in self.trainops[loss_type]["grads_and_vars"] if grad!=None]
                    self.trainops[loss_type]["g_updates"] = Optimizer.apply_gradients(
                         self.trainops[loss_type]["capped_gvs"],
                        global_step=self.trainops[loss_type]["global_step"])


            self.all_variables = tf.global_variables()
            self.init = tf.global_variables_initializer()
            self.saver = tf.train.Saver(max_to_keep = self._conf["max_to_keep"])
            
            self.all_operations = self._graph.get_operations()

        return self._graph

