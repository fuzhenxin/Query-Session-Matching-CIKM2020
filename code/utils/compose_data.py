#encoding="utf-8"

import sys
import codecs
import pickle as pkl
import numpy as np

def get_voc(data_dir):

    file_name_train = data_dir+"/train.txt"
    file_name_train = codecs.open(file_name_train, "r")
    lines = [i.strip() for i in file_name_train.readlines()]

    voc_cnt = dict()
    voc_dict = dict()
    voc_dict["UNK"] = 0
    voc_dict["SENT"] = 1
    words_all = ["UNK", "SENT"]
    word_global_index = 2
    for line in lines:
        if "ecommerce" in data_dir:
            line = line.split("|")[:-1]
            line = "|".join(line)
            line = line.replace("|", " ")
            line = line.split()
        elif "douban" in data_dir:
            line = line.split()
        elif "ubuntu" in data_dir:
            line = line.replace("|", " ")
            line = line.split()[1:]
        else:
            assert False
        for word in line:
            if word in voc_cnt:
                voc_cnt[word] += 1
            else:
                voc_cnt[word] = 1
    voc_item = sorted(voc_cnt.items(), key=lambda v:(v[1]), reverse=True)
    print(voc_item[:10])
    print("Words before filt: ", len(voc_item))
    #voc_item = voc_item[:60000]
    #voc_item = [i for i in voc_item if i[1]>10]
    if "douban" in data_dir:
        lines = open("../../data/douban_processed_all/vocab.txt", "r").readlines()
        lines = [ [i.strip(), 1] for i in lines]
        voc_item = lines
    for word_per in voc_item:
        word_per = word_per[0]
        voc_dict[word_per] = word_global_index
        words_all.append(word_per)
        word_global_index += 1
    print("Words after filt: ", word_global_index)

    f_w = codecs.open(data_dir+"/voc.txt", "w", "utf-8")
    for word_per in words_all:
        f_w.write(word_per+"\n")
    write_emb(words_all, data_dir)
    return voc_dict

def write_emb(words, data_dir):
    f = codecs.open(data_dir+"/vectors.txt", "r", "utf-8", errors='ignore')
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(len(lines))
    emb = dict()
    all_emb = []
    for line_index,line in enumerate(lines):
        line_w = line.split()[0]
        line_emb = np.array([float(i) for i in line.split()[1:]])
        if line_emb.shape[0]!=200:
            print("Error in: ", line_index)
            continue
        assert line_emb.shape[0]==200, (line_emb.shape[0], line_index)
        emb[line_w] = line_emb
        all_emb.append(line_emb)
    all_emb = np.array(all_emb)
    print(all_emb.shape)
    all_emb = np.mean(all_emb, axis=0)
    print("UNK emc shape: ", all_emb.shape)
    res = []
    for word in words:
        if word in emb:
            res.append(emb[word])
        else:
            res.append(all_emb)
    res = np.array(res)
    print("Glove shape: ", res.shape)
    pkl.dump(res, open(data_dir+"/emb.pkl", "wb"))


def convert_to_id(line, voc):
    #print("==============")
    #print(line)
    line = line.replace("\t", " | ")
    line = line.split()
    ids = []
    for word in line:
        if word == "|":
            word_id = voc["SENT"]
        else:
            word_id = voc[word] if word in voc else voc["UNK"]
        ids.append(word_id)
    if len(ids)==0:
        ids = [voc["UNK"]]
    return ids

def compose_data(data_dir, voc=None):
    res = []


    file_names = ["train.mix", "valid.mix", "test.mix"]
    for file_name_per in file_names:
        f = codecs.open(data_dir+"/"+file_name_per, "r", "utf-8")
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        q_c, c_c, c_r, c_f, y = [], [], [], [], []
        for line in lines:
            line = line.split("|")
            #assert len(line)==5
            ids_c1 = convert_to_id(line[0], voc)
            ids_c2 = convert_to_id(line[1], voc)
            ids_r = convert_to_id(line[2], voc)
            ids_f = convert_to_id(line[3], voc)
            q_c.append(ids_c1)
            c_c.append(ids_c2)
            c_r.append(ids_r)
            c_f.append(ids_f)
            y.append(int(line[-1]))
        tvt = {"q_c": q_c, "c_c": c_c, "c_r": c_r, "c_f": c_f, "y": y}
        res.append(tvt)

    pkl.dump(res, open(data_dir+"/data.pkl", "wb"))

if __name__=="__main__":
    data_dir = sys.argv[1]
    voc = get_voc(data_dir)

    compose_data(data_dir, voc)

