import pickle as pickle
import numpy as np
import math

def unison_shuffle(data, seed=None):
    if seed is not None:
        np.random.seed(seed)

    y = np.array(data['y'])
    c1 = np.array(data['q_c'])
    c2 = np.array(data['c_c'])
    cf = np.array(data['c_f'])
    r = np.array(data['c_r'])

    assert len(y) == len(c1) == len(c2) == len(r)
    p = np.random.permutation(len(y))
    shuffle_data = {'y': y[p], 'q_c': c1[p], 'c_c': c2[p], "c_f": cf[p], 'c_r': r[p]}
    return shuffle_data

def split_c(c, split_id):
    '''c is a list, example context
       split_id is a integer, conf[_EOS_]
       return nested list
    '''
    turns = [[]]
    for _id in c:
        if _id != split_id:
            turns[-1].append(_id)
        else:
            turns.append([])
    if turns[-1] == [] and len(turns) > 1:
        turns.pop()
    return turns

def normalize_length(_list, length, cut_type='tail', process_context=False):
    '''_list is a list or nested list, example turns/r/single turn c
       cut_type is head or tail, if _list len > length is used
       return a list len=length and min(read_length, length)
    '''
    real_length = len(_list)
    if real_length == 0:
        return [0]*length, 0 # hh

    if real_length <= length:
        if not isinstance(_list[0], list):
            if process_context:
                if cut_type=="tail": _list = [0]*(length - real_length) + _list
                else: _list = _list + [0]*(length - real_length)
            else:
                _list.extend([0]*(length - real_length))
        else:
            if process_context:
                if cut_type=="tail": _list = [[0] for jj in range(length - real_length)] + _list
                else: _list = _list + [[0] for jj in range(length - real_length)]
            else:
                _list.extend([[0] for jj in range(length - real_length)])
        return _list, real_length

    if cut_type == 'head':
        return _list[:length], length
    if cut_type == 'tail':
        return _list[-length:], length


def build_batches(data, conf, turn_cut_type='tail', term_cut_type='tail', train_type="cr"):
    _turns_batches1 = []
    _turns_batches2 = []
    _turns_batchesf = []
    _turnsa_batches = []
    _turnsq_batches = []
    _tt_turns_len_batches1 = []
    _every_turn_len_batches1 = []
    _tt_turns_len_batches2 = []
    _every_turn_len_batches2 = []
    _tt_turns_len_batchesf = []
    _every_turn_len_batchesf = []
    _response_batches = []
    _response_len_batches = []
    _turnsa_len_batches = []
    _turnsq_len_batches = []
    _label_batches = []

    max_turn_num_2 = conf['max_turn_num_sess'] if train_type=="class" else conf['max_turn_num_hf']
    turn_cut_type_2 = 'head' if train_type=="class" else 'tail'

    #batch_len = int(len(data['y'])/conf['batch_size'])
    batch_len = math.ceil(float(len(data['y']))/conf['batch_size'])
    for batch_index in range(batch_len):

        _turns1 = []
        _turns2 = []
        _turnsf = []
        _turnsa = []
        _turnsq = []
        _tt_turns_len1 = []
        _every_turn_len1 = []
        _tt_turns_len2 = []
        _every_turn_len2 = []
        _tt_turns_lenf = []
        _every_turn_lenf = []
        _response = []
        _response_len = []
        _turnsa_len = []
        _turnsq_len = []
        _label = []

        for i in range(conf['batch_size']):
            index = batch_index * conf['batch_size'] + i
            if index>=len(data['y']):
                break

            c1 = data['q_c'][index]
            c2 = data['c_c'][index]
            cf = data['c_f'][index]
            r = data['c_r'][index]
            y = data['y'][index]

            turns1 = split_c(c1, conf['_EOS_'])
            assert len(turns1)
            nor_turns1, turn_len1 = normalize_length(turns1, conf['max_turn_num'], turn_cut_type, process_context=True)

            nor_turns_nor_c1 = []
            term_len1 = []
            for c in nor_turns1:
                nor_c, nor_c_len = normalize_length(c, conf['max_turn_len'], term_cut_type)
                nor_turns_nor_c1.append(nor_c)
                term_len1.append(nor_c_len)

            turns2 = split_c(c2, conf['_EOS_'])
            assert len(turns2)
            nor_turns2, turn_len2 = normalize_length(turns2, max_turn_num_2, turn_cut_type_2, process_context=True)

            nor_turns_nor_c2 = []
            term_len2 = []
            for c in nor_turns2:
                nor_c, nor_c_len = normalize_length(c, conf['max_turn_len'], term_cut_type)
                nor_turns_nor_c2.append(nor_c)
                term_len2.append(nor_c_len)

            turnsf = split_c(cf, conf['_EOS_'])
            assert len(turnsf)
            nor_turnsf, turn_lenf = normalize_length(turnsf, conf['max_turn_num_hf'], cut_type="head")

            nor_turns_nor_cf = []
            term_lenf = []
            for c in nor_turnsf:

                nor_c, nor_c_len = normalize_length(c, conf['max_turn_len'], term_cut_type)

                nor_turns_nor_cf.append(nor_c)
                term_lenf.append(nor_c_len)


            r = [int(i) for i in r]
            nor_r, r_len = normalize_length(r, conf['max_turn_len'], term_cut_type)

            turns1 = split_c(c1, conf['_EOS_'])
            turns_q = turns1[-conf["max_turn_num_hf"]:]
            turns_q_tmp = [i[-conf["max_turn_len"]:] for i in turns_q]
            turns_q = []
            for iii in turns_q_tmp:
                if len(iii)==1 and iii[0]==0:
                    continue
                for jjj in iii:
                    turns_q.append(jjj)
            nor_turns_q, nor_turns_q_len = normalize_length(turns_q, conf["max_turn_num"]*conf["max_turn_len"])
            _turnsq.append(nor_turns_q)
            _turnsq_len.append(nor_turns_q_len)

            turns2 = split_c(c2, conf['_EOS_'])
            turnsf = split_c(cf, conf['_EOS_'])
            turns_all = turns2[-conf["max_turn_num_hf"]:] + [data['c_r'][index]] + turnsf[:conf["max_turn_num_hf"]]
            turns_all_tmp = [i[-conf["max_turn_len"]:] for i in turns_all]
            turns_all = []
            for iii in turns_all_tmp:
                if len(iii)==1 and iii[0]==0:
                    continue
                for jjj in iii:
                    turns_all.append(jjj)
            nor_turns_all, nor_turns_all_len = normalize_length(turns_all, (conf["max_turn_num_hf"]*2+1)*conf["max_turn_len"])
            _turnsa.append(nor_turns_all)
            _turnsa_len.append(nor_turns_all_len)
            
            _turns1.append(nor_turns_nor_c1)
            _turns2.append(nor_turns_nor_c2)
            _turnsf.append(nor_turns_nor_cf)
            _every_turn_len1.append(term_len1)
            _tt_turns_len1.append(turn_len1)
            _every_turn_len2.append(term_len2)
            _tt_turns_len2.append(turn_len2)
            _every_turn_lenf.append(term_lenf)
            _tt_turns_lenf.append(turn_lenf)
            _response.append(nor_r)
            _response_len.append(r_len)
            _label.append(y)

        _turns_batches1.append(_turns1)
        _turns_batches2.append(_turns2)
        _turns_batchesf.append(_turnsf)
        _turnsa_batches.append(_turnsa)
        _turnsq_batches.append(_turnsq)
        _tt_turns_len_batches1.append(_tt_turns_len1)
        _every_turn_len_batches1.append(_every_turn_len1)
        _tt_turns_len_batches2.append(_tt_turns_len2)
        _every_turn_len_batches2.append(_every_turn_len2)
        _tt_turns_len_batchesf.append(_tt_turns_lenf)
        _every_turn_len_batchesf.append(_every_turn_lenf)
        _response_batches.append(_response)
        _response_len_batches.append(_response_len)
        _turnsa_len_batches.append(_turnsa_len)
        _turnsq_len_batches.append(_turnsq_len)
        _label_batches.append(_label)

    ans = { 
        "turns1": _turns_batches1, "turns2": _turns_batches2, "tt_turns_len1": _tt_turns_len_batches1, "every_turn_len1":_every_turn_len_batches1,
        "tt_turns_len2": _tt_turns_len_batches2, "every_turn_len2":_every_turn_len_batches2,
        "turnsf": _turns_batchesf, "tt_turns_lenf": _tt_turns_len_batchesf, "every_turn_lenf": _every_turn_len_batchesf,
        "turnsa": _turnsa_batches, "turnsa_len": _turnsa_len_batches, "turnsq": _turnsq_batches, "turnsq_len": _turnsq_len_batches,
        "response": _response_batches, "response_len": _response_len_batches, "label": _label_batches
    }   

    return ans 

if __name__ == '__main__':
    conf = { 
        "batch_size": 5,
        "max_turn_num": 6, 
        "max_turn_len": 20, 
        "_EOS_": 1,
    }
    train, val, test, test_human = pickle.load(open('../data_ali/data.cc.cc.pkl', 'rb'))
    print('load data success')
    
    train_batches = build_batches(train, conf)
    val_batches = build_batches(val, conf)
    test_batches = build_batches(test, conf)
    test_batches = build_batches(test_human, conf)
    print('build batches success')
    
    #pickle.dump([train_batches, val_batches, test_batches], open('../data/batches.pkl', 'wb'))
    #print('dump success')
