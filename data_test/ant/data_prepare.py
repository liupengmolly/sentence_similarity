#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import os
import random
import sys

import jieba
import numpy as np
import pandas as pd
import jieba.posseg as seg

reload(sys)
sys.setdefaultencoding('utf8')

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)
from pypinyin import pinyin
from common.data_helper import DataHelper
from word2vec.train_method import gensim_trainer
from word2vec.word2vec_model import Word2vecModel
from lib.feature.idf import Idf


def generate_data4_single_ensemble(train_path):

    data = pd.read_csv(train_path, sep='\t')
    data = list(np.array(data)[:, 1:4])
    data = [(x[0], x[1], str(x[2])) for x in data]
    train_data_pos = [x for x in data if x[2] == '1']
    train_data_neg = [x for x in data if x[2] == '0']
    parent_path = os.path.dirname(train_path)
    random.shuffle(train_data_pos)
    random.shuffle(train_data_neg)
    pos_index = int(len(train_data_pos) * 4 / 5.0)
    neg_index = int(len(train_data_neg) * 4 / 5.0)
    single_pos = train_data_pos[:pos_index]
    ensemble_pos = train_data_pos[pos_index:]
    single_neg = train_data_neg[:neg_index]
    ensemble_neg = train_data_neg[neg_index:]
    for i in range(5):
        single_train = single_pos + random.sample(single_neg, len(single_pos))
        pd.DataFrame(single_train).to_csv(
            os.path.join(parent_path, str(i).join(['single_train_data', '.csv'])), sep='\t'
        )

    ensemble_train = ensemble_pos * 5 + random.sample(ensemble_neg, len(ensemble_pos) * 5)
    pd.DataFrame(ensemble_train).to_csv(
        os.path.join(parent_path, 'ensemble_train.csv'), sep='\t'
    )


def generate_2models_train_data(train_path):
    data = pd.read_csv(train_path, sep='\t')
    data = list(np.array(data)[:, 1:4])
    data = [(x[0], x[1], str(x[2])) for x in data]
    train_data_pos = [x for x in data if x[2] == '1']
    train_data_neg = [x for x in data if x[2] == '0']
    parent_path = os.path.dirname(train_path)
    for i in [10, 11]:
        neg_sample = random.sample(train_data_neg, len(train_data_pos) * 3)
        data = train_data_pos * 3 + neg_sample
        pd.DataFrame(data).to_csv(
            os.path.join(parent_path, str(i).join(['train_data', '.csv'])), sep='\t')


def generate_train_data(train_path):
    data = pd.read_csv(train_path, sep='\t')
    data = list(np.array(data)[:, 1:4])
    data = [(x[0], x[1], str(x[2])) for x in data]
    train_data_pos = [x for x in data if x[2] == '1']
    train_data_neg = [x for x in data if x[2] == '0']
    parent_path = os.path.dirname(train_path)
    for i in xrange(10):
        neg_sample = random.sample(train_data_neg, len(train_data_pos))
        data = train_data_pos + neg_sample
        pd.DataFrame(data).to_csv(
            os.path.join(parent_path, str(i).join(['train_data', '.csv'])), sep='\t')


def generate_data(data_path):
    data = open(data_path, 'r').readlines()
    data = [line.strip().split('\t') for line in data]
    data = [((x[1].replace(u'\ufeff', ''), x[2]), x[3]) for x in data]
    print "样本总数： ", len(data)
    print "正样本总数： ", len([x for x in data if x[1] == '0'])

    data = zip(*data)
    x_train, y_train, x_valid, y_valid, x_test, y_test = DataHelper.data_split(
        data, valid_size=0.05, test_size=0.06
    )
    train_data = [(x_train[ind][0],
                   x_train[ind][1],
                   y_train[ind])
                  for ind in range(len(x_train))]
    train_data_pos = [x for x in train_data if x[2] == '1']
    train_data_neg = [x for x in train_data if x[2] == '0']

    train_data_sample = train_data_pos + random.sample(train_data_neg, 6000)

    train_data_pos_extend = train_data_pos * 4 + train_data_neg

    valid_data = [(x_valid[ind][0],
                  x_valid[ind][1],
                  y_valid[ind])
                  for ind in range(len(x_valid))]
    test_data = [(x_test[ind][0],
                  x_test[ind][1],
                  y_test[ind])
                 for ind in range(len(x_test))]

    parent_path = os.path.dirname(data_path)
    train_data_path = os.path.join(parent_path, 'train_data.csv')
    pd.DataFrame(train_data).to_csv(train_data_path, sep='\t')

    train_neg_sample_path = os.path.join(parent_path, 'train_neg_sample.csv')
    pd.DataFrame(train_data_sample).to_csv(train_neg_sample_path, sep='\t')
    train_data_pos_extend_path = os.path.join(parent_path, 'train_data_pos_extend.csv')
    pd.DataFrame(train_data_pos_extend).to_csv(train_data_pos_extend_path, sep='\t')
    valid_data_path = os.path.join(parent_path, 'valid_data.csv')
    pd.DataFrame(valid_data).to_csv(valid_data_path, sep='\t')
    test_data_path = os.path.join(parent_path, 'test_data.csv')
    pd.DataFrame(test_data).to_csv(test_data_path, sep='\t')


def w2v_train(data_path, save_path):
    data = open(data_path, 'r').readlines()
    data = [line.strip().split('\t') for line in data]
    sentences = []
    for x in data:
        sentences.append(list(jieba.cut(x[1].replace(u'\ufeff', ''))))
        sentences.append(list(jieba.cut(x[2])))
    w2v_dict = gensim_trainer.train(sentences)
    f = open(save_path, 'w')
    f.write(json.dumps(w2v_dict))


def w2v_train_with_pinyin(data_path,save_path):
    data = open(data_path, 'r').readlines()
    data = [line.strip().split('\t') for line in data]
    sentences = []
    for x in data:
        if len(x) != 4:
            continue
        sent1 = x[1].decode('utf8')
        sent2 = x[2].decode('utf8')
        chars1 = [c for c in sent1.replace(u'\ufeff','')]
        chars2 = [c for c in sent2.replace(u'\ufeff','')]
        sentences.append(chars1)
        sentences.append(chars2)
        sentences.append([p[0] for p in pinyin(chars1)])
        sentences.append([p[0] for p in pinyin(chars2)])
    w2v_dict = gensim_trainer.train(sentences)
    f = open(save_path, 'w')
    f.write(json.dumps(w2v_dict))


def generate_word_idf(data_path, word2vec_file):
    data = open(data_path, 'r').readlines()
    data = [line.strip().split('\t') for line in data]
    word2vec = Word2vecModel.load(word2vec_file)
    word_id_map, id_vector_map = word2vec.generate_word_id_map(np.float32)
    sentences = []
    for x in data:
        s1 = list(jieba.cut(unicode(x[1]).replace(u'\ufeff', '')))
        s1 = [word_id_map.get(y, 0) for y in s1]
        sentences.append(s1)
        s2 = list(jieba.cut(x[2]))
        s2 = [word_id_map.get(y, 0) for y in s2]
        sentences.append(s2)
    id_idf_map = [5e-5] * (max(word_id_map.values()) + 1)
    idf = Idf(sentences)

    for item in idf.words_idf.items():
        index = item[0]
        if index not in [0, 1]:
            id_idf_map[index] = item[1]
    parent_path = os.path.dirname(data_path)
    f = open(os.path.join(parent_path, 'id_idf.json'), 'w')
    f.write(json.dumps(id_idf_map))
    f.close()


def generate_pos_id_map(data_path):
    data = open(data_path, 'r').readlines()
    data = [line.strip().split('\t') for line in data]
    pos_set = set()
    for (ind, line) in enumerate(data):
        if ind % 200 == 1:
            print "%d  done " % ind
        s1 = list(seg.cut(unicode(line[1]).replace(u'\ufeff', '')))
        pos_set = pos_set | set([x.flag for x in s1])
        s2 = list(seg.cut(line[2]))
        pos_set = pos_set | set([x.flag for x in s2])
    pos_set = sorted(list(pos_set))
    pos_id_map = {}
    for (ind, ele) in enumerate(pos_set):
        pos_id_map[ele] = ind + 2
    parent_path = os.path.dirname(data_path)
    f = open(os.path.join(parent_path, 'pos_id.json'), 'w')
    f.write(json.dumps(pos_id_map))


def w2v_c2v_combine(w2v_map, c2v_map):
    unknown_char = [5e-5] * 60
    new_w2v_map = {}
    for word in w2v_map:
        vec = w2v_map[word]
        tmp = np.zeros([60])
        for char in word:
            cv = c2v_map.get(char, unknown_char)
            tmp += cv
        tmp = tmp / len(word)
        tmp = [float(x) for x in tmp]
        new_w2v_map[word] = vec + list(tmp)
    f = open('ant_data_w2v_combined_char.json', 'w')
    f.write(json.dumps(new_w2v_map))
    f.close()


def w2v_p2v_combine(w2v_map, c2v_map):
    unknown_char = [5e-5] * 60
    new_w2v_map = {}
    count = 0
    for word in w2v_map:
        vec = w2v_map[word]
        tmp = np.zeros([60])
        word_py = pinyin(word)
        for ch_py in word_py:
            if ch_py[0] in c2v_map:

                cv = c2v_map.get(ch_py[0])
            else:
                count += 1
                cv = unknown_char
            tmp += cv
        tmp = tmp / len(word)
        tmp = [float(x) for x in tmp]
        new_w2v_map[word] = vec + list(tmp)
    f = open('ant_data_w2v_combined_p2v.json', 'w')
    f.write(json.dumps(new_w2v_map))
    f.close()


def c2v_p2v_combine(c2v_map):
    unknown_char = [5e-5] * 60
    new_w2v_map = {}
    count = 0
    for ch in c2v_map:
        if len(json.dumps(ch)) == 8:
            py = pinyin(ch)
            vec = c2v_map.get(ch, unknown_char)
            if py[0][0] in c2v_map:
                tmp = c2v_map.get(py[0][0])
            else:
                count += 1
                tmp = unknown_char
            tmp = [float(x) for x in tmp]
            new_w2v_map[ch] = vec + tmp
    f = open('ant_data_c2v_combined_p2v.json', 'w')
    f.write(json.dumps(new_w2v_map))
    f.close()


def w2v_c2v_p2v_combine(w2v_map, c2v_map):
    unknown_char = [5e-5] * 60
    new_w2v_map = {}
    for word in w2v_map:
        vec = w2v_map[word]
        tmp = np.zeros([60])
        word_py = pinyin(word)
        for ch_py in word_py:
            if ch_py[0] in c2v_map:
                cv = c2v_map.get(ch_py[0], unknown_char)
                tmp += np.array(cv)
        for char in word:
            cv = c2v_map.get(char, unknown_char)
            tmp += np.array(cv)
        tmp = tmp / len(word * 2)
        tmp = [float(x) for x in tmp]
        new_w2v_map[word] = vec + list(tmp)
    f = open('ant_data_w2v_c2v_p2v_combined.json', 'w')
    f.write(json.dumps(new_w2v_map))
    f.close()


def ner_statistic(data):
    tag_label_count = {'nr': {}, 'ns': {}, 'nt': {}}
    tag_label_count['nr']['pos'] = 1
    tag_label_count['nr']['neg'] = 0
    tag_label_count['ns']['pos'] = 1
    tag_label_count['ns']['neg'] = 0
    tag_label_count['nt']['pos'] = 1
    tag_label_count['nt']['neg'] = 0
    for pair in data:
        tokens1 = seg.cut(pair[1])
        tokens2 = seg.cut(pair[2])
        label = int(pair[3].strip())
        for tag in ['nr', 'ns', 'nt']:
            tokens1_ = set([x.word for x in tokens1 if x.flag == tag])
            tokens2_ = set([x.word for x in tokens2 if x.flag == tag])
            if tokens1_ & tokens2_:
                if label == 1:
                    tag_label_count[tag]['pos'] += 1
                else:
                    tag_label_count[tag]['neg'] += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='input data path')
    parser.add_argument('--word2vec_file', type=str, default='', help='input data path')
    args = parser.parse_args()
    # generate_train_data(args.data_path)
    generate_word_idf(args.data_path, args.word2vec_file)
    # generate_pos_id_map(args.data_path)
