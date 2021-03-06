import sys
import os
import time
import math

import pickle as pickle
import tensorflow as tf
import numpy as np

import utils.reader as reader
import utils.evaluation as eva
# import utils.evaluation_class as eva_class

def train(conf, _model):
    
    if conf['rand_seed'] is not None:
        np.random.seed(conf['rand_seed'])

    if not os.path.exists(conf['save_path']):
        os.makedirs(conf['save_path'])

    train_type = conf["train_type"]

    # load data
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'starting loading data')
    data_collections = pickle.load(open(conf["data_path"], 'rb'))
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'finish loading data')


    file_names = ["train.mix", "valid.mix", "test.mix"]

    train_data = data_collections[file_names.index("train.mix")]
    batch_num = math.ceil(float(len(train_data['y']))/conf["batch_size"])

    valid_data = data_collections[file_names.index("valid.mix")]
    val_batches = reader.build_batches(valid_data, conf, train_type=train_type)
    val_batch_num = len(val_batches["response"])

    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "finish building test batches")


    print('configurations: %s' %conf)
    _graph = _model.build_graph()
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'build graph sucess')

    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(graph=_graph, config=config) as sess:
        _model.init.run()
        if conf["init_model"]:
            _model.saver_load.restore(sess, conf["init_model"])
            print("sucess init %s" %conf["init_model"])

        average_loss = 0.0
        batch_index = 0
        step = 0
        best_result = (0, 0, 0, 0)

        g_updates = _model.trainops[train_type]["g_updates"]
        loss = _model.trainops[train_type]["loss"]
        global_step = _model.trainops[train_type]["global_step"]
        learning_rate = _model.trainops[train_type]["learning_rate"]
        logits = _model.trainops[train_type]["logits"]

        early_stop_count = 0
        for step_i in range(conf["num_scan_data"]):
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'starting shuffle train data')
            shuffle_train = reader.unison_shuffle(train_data)
            train_batches = reader.build_batches(shuffle_train, conf, train_type=train_type)
            print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'finish building train data')
            for batch_index in range(batch_num):
                feed = {
                    _model.turns1: train_batches["turns1"][batch_index],
                    _model.turnsf: train_batches["turnsf"][batch_index],
                    _model.tt_turns_len1: train_batches["tt_turns_len1"][batch_index],
                    _model.every_turn_len1: train_batches["every_turn_len1"][batch_index],
                    _model.tt_turns_lenf: train_batches["tt_turns_lenf"][batch_index],
                    _model.every_turn_lenf: train_batches["every_turn_lenf"][batch_index],
                    _model.response: train_batches["response"][batch_index], 
                    _model.response_len: train_batches["response_len"][batch_index],
                    _model.label: train_batches["label"][batch_index],
                    _model.keep_rate: 1.0,
                    _model.turnsa: train_batches["turnsa"][batch_index],
                    _model.turnsa_len: train_batches["turnsa_len"][batch_index],
                    _model.turnsq: train_batches["turnsq"][batch_index],
                    _model.turnsq_len: train_batches["turnsq_len"][batch_index],
                }

                feed[_model.turns2] = train_batches["turns2"][batch_index]
                feed[_model.tt_turns_len2] = train_batches["tt_turns_len2"][batch_index]
                feed[_model.every_turn_len2] = train_batches["every_turn_len2"][batch_index]


                _, curr_loss = sess.run([g_updates, loss], feed_dict = feed)
                average_loss += curr_loss
                step += 1

                if step < 500: print_step_time = int(conf["print_step"]/10)
                else: print_step_time = conf["print_step"]
                if step % print_step_time == 0 and step > 0:
                    g_step, lr = sess.run([global_step, learning_rate])
                    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), 'epoch: %d, step: %.5d, lr: %-.6f, loss: %s' %(step_i, g_step, lr, average_loss / print_step_time) )
                    average_loss = 0


            #--------------------------evaluation---------------------------------
            score_file_path = conf['save_path'] + '/score.' + str(step_i)
            score_file = open(score_file_path, 'w')

            for batch_index in range(val_batch_num):
                feed = {
                    _model.turns1: val_batches["turns1"][batch_index],
                    _model.turnsf: val_batches["turnsf"][batch_index],
                    _model.tt_turns_len1: val_batches["tt_turns_len1"][batch_index],
                    _model.every_turn_len1: val_batches["every_turn_len1"][batch_index],
                    _model.tt_turns_lenf: val_batches["tt_turns_lenf"][batch_index],
                    _model.every_turn_lenf: val_batches["every_turn_lenf"][batch_index],
                    _model.response: val_batches["response"][batch_index], 
                    _model.response_len: val_batches["response_len"][batch_index],
                    _model.keep_rate: 1.0,
                    _model.turnsa: val_batches["turnsa"][batch_index],
                    _model.turnsa_len: val_batches["turnsa_len"][batch_index],
                    _model.turnsq: val_batches["turnsq"][batch_index],
                    _model.turnsq_len: val_batches["turnsq_len"][batch_index],
                }
                if train_type=="class":
                    feed[_model.turns_sess] = val_batches["turns2"][batch_index]
                    feed[_model.tt_turns_len_sess] = val_batches["tt_turns_len2"][batch_index]
                    feed[_model.every_turn_len_sess] = val_batches["every_turn_len2"][batch_index]
                else:
                    feed[_model.turns2] = val_batches["turns2"][batch_index]
                    feed[_model.tt_turns_len2] = val_batches["tt_turns_len2"][batch_index]
                    feed[_model.every_turn_len2] = val_batches["every_turn_len2"][batch_index]

                scores = sess.run(logits, feed_dict = feed)

                for i in range(len(scores)):
                    score_file.write(
                        str(scores[i]) + '\t' +
                        str(val_batches["label"][batch_index][i]) + '\n')
            score_file.close()



            #write evaluation result
            if train_type=="class":
                result = eva_class.evaluate(score_file_path)
            else:
                result = eva.evaluate(score_file_path)
            
            print(time.strftime('%Y-%m-%d %H:%M:%S result: ',time.localtime(time.time())), *result)

            if result[1] + result[2] > best_result[1] + best_result[2]:
                early_stop_count = 0
                best_result = result
                _save_path = _model.saver.save(sess, conf["save_path"] + "/model", global_step=step_i)
                print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())), "succ saving model in " + _save_path)
            else:
                early_stop_count+=1
                if early_stop_count>=4: break
            print(time.strftime('%Y-%m-%d %H:%M:%S best result',time.localtime(time.time())), *best_result)
            #_model.saver.save(sess, conf["save_path"] + "/model-all-"+str(global_step))
