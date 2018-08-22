#! /usr/bin/python
# -*- coding: utf-8 -*-

import math
import os
import sys
import pickle
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
import networkx as nx
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def generate_graph(data):
    data = zip(list(data.label), list(data.q1), list(data.q2))
    graph = nx.Graph()
    for ele in data:
        if ele[0] == 1:
            graph.add_edge(ele[1], ele[2])
    pickle.dump(graph, open(os.path.join(root_path, 'data_test/pai/data/cache/graph.pickle'),'w'))

def load_graph():
    graph = pickle.load(open(os.path.join(root_path,"data_test/pai/data/cache/graph.pickle"), 'r'))
    return graph


def generate_powerful_word(data, fp):
    """
    计算数据中词语的影响力，格式如下：
        词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，
        3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
    """
    words_power = {}
    train_subset_data = data
    for index, row in train_subset_data.iterrows():
        label = int(row['label'])
        q1_words = str(row['words_chars1']).lower().split()
        q2_words = str(row['words_chars2']).lower().split()
        all_words = set(q1_words + q2_words)
        q1_words = set(q1_words)
        q2_words = set(q2_words)
        for word in all_words:
            if word not in words_power:
                words_power[word] = [0. for i in range(7)]
            # 计算出现语句对数量
            words_power[word][0] += 1.
            words_power[word][1] += 1.
            if ((word in q1_words) and (word not in q2_words)) or \
                    ((word not in q1_words) and (word in q2_words)):
                # 计算单侧语句数量
                words_power[word][3] += 1.
                if 0 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算单侧语句正确比例
                    words_power[word][4] += 1.
            if (word in q1_words) and (word in q2_words):
                # 计算双侧语句数量
                words_power[word][5] += 1.
                if 1 == label:
                    # 计算正确语句对数量
                    words_power[word][2] += 1.
                    # 计算双侧语句正确比例
                    words_power[word][6] += 1.
    for word in words_power:
        # 计算出现语句对比例
        words_power[word][1] /= len(data)
        # 计算正确语句对比例
        words_power[word][2] /= words_power[word][0]
        # 计算单侧语句对正确比例
        if words_power[word][3] > 1e-6:
            words_power[word][4] /= words_power[word][3]
        # 计算单侧语句对比例
        words_power[word][3] /= words_power[word][0]
        # 计算双侧语句对正确比例
        if words_power[word][5] > 1e-6:
            words_power[word][6] /= words_power[word][5]
        # 计算双侧语句对比例
        words_power[word][5] /= words_power[word][0]
    sorted_words_power = sorted(words_power.iteritems(), key=lambda d: d[1][0], reverse=True)
    print("INFO", "power words calculation done, len(words_power)=%d"
                % len(sorted_words_power))
    f = open(fp, 'w')
    for ele in sorted_words_power:
        f.write("%s" % ele[0])
        for num in ele[1]:
            f.write("\t%.5f" % num)
        f.write("\n")
    f.close()
    return sorted_words_power

def load_powerful_word(fp):
    powful_word = []
    f = open(fp, 'r')
    for line in f:
        subs = line.split('\t')
        word = subs[0]
        stats = [float(num) for num in subs[1:]]
        powful_word.append((word, stats))
    f.close()
    return powful_word

def init_powerful_word_dside(pword, thresh_num, thresh_rate):
    pword_dside = []
    pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
    pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
    pword_dside.extend(map(lambda x: x[0],
                           filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
    return pword_dside

def extract_row_dside(row,pword_dside):
    tags = []
    q1_words = str(row['words_chars1']).lower().split()
    q2_words = str(row['words_chars2']).lower().split()
    for word in pword_dside:
        if (word in q1_words) and (word in q2_words):
            tags.append(1.0)
        else:
            tags.append(0.0)
    return sum(tags)

def init_powerful_word_oside(pword, thresh_num, thresh_rate):
    pword_oside = []
    pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
    pword_oside.extend(
        map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
    return pword_oside

def extract_row_oside(row,pword_oside):
    tags = []
    q1_words = set(str(row['words_chars1']).lower().split())
    q2_words = set(str(row['words_chars2']).lower().split())
    for word in pword_oside:
        if (word in q1_words) and (word not in q2_words):
            tags.append(1.0)
        elif (word not in q1_words) and (word in q2_words):
            tags.append(1.0)
        else:
            tags.append(0.0)
    return sum(tags)

def extract_row_rate_dside(row,pword_dict):
    num_least = 300
    rate = [1.0]
    q1_words = set(str(row['words_chars1']).lower().split())
    q2_words = set(str(row['words_chars2']).lower().split())
    share_words = list(q1_words.intersection(q2_words))
    for word in share_words:
        if word not in pword_dict:
            continue
        if pword_dict[word][0] * pword_dict[word][5] < num_least:
            continue
        rate[0] *= (1.0 - pword_dict[word][6])
    rate = [1 - num for num in rate]
    return rate

def extract_row_rate_oside(row, pword_dict):
    num_least = 300
    rate = [1.0]
    q1_words = set(str(row['words_chars1']).lower().split())
    q2_words = set(str(row['words_chars2']).lower().split())
    q1_diff = list(set(q1_words).difference(set(q2_words)))
    q2_diff = list(set(q2_words).difference(set(q1_words)))
    all_diff = set(q1_diff + q2_diff)
    for word in all_diff:
        if word not in pword_dict:
            continue
        if pword_dict[word][0] * pword_dict[word][3] < num_least:
            continue
        rate[0] *= (1.0 - pword_dict[word][4])
    rate = [1 - num for num in rate]
    return rate

# class PowerfulWord(object):
#     @staticmethod
#     def load_powerful_word(fp):
#         powful_word = []
#         f = open(fp, 'r')
#         for line in f:
#             subs = line.split('\t')
#             word = subs[0]
#             stats = [float(num) for num in subs[1:]]
#             powful_word.append((word, stats))
#         f.close()
#         return powful_word
#
#     @staticmethod
#     def generate_powerful_word(data, subset_indexs):
#         """
#         计算数据中词语的影响力，格式如下：
#             词语 --> [0. 出现语句对数量，1. 出现语句对比例，2. 正确语句对比例，
#             3. 单侧语句对比例，4. 单侧语句对正确比例，5. 双侧语句对比例，6. 双侧语句对正确比例]
#         """
#         words_power = {}
#         train_subset_data = data.iloc[subset_indexs, :]
#         for index, row in train_subset_data.iterrows():
#             label = int(row['is_duplicate'])
#             q1_words = str(row['question1']).lower().split()
#             q2_words = str(row['question2']).lower().split()
#             all_words = set(q1_words + q2_words)
#             q1_words = set(q1_words)
#             q2_words = set(q2_words)
#             for word in all_words:
#                 if word not in words_power:
#                     words_power[word] = [0. for i in range(7)]
#                 # 计算出现语句对数量
#                 words_power[word][0] += 1.
#                 words_power[word][1] += 1.
#
#                 if ((word in q1_words) and (word not in q2_words)) or \
#                         ((word not in q1_words) and (word in q2_words)):
#                     # 计算单侧语句数量
#                     words_power[word][3] += 1.
#                     if 0 == label:
#                         # 计算正确语句对数量
#                         words_power[word][2] += 1.
#                         # 计算单侧语句正确比例
#                         words_power[word][4] += 1.
#                 if (word in q1_words) and (word in q2_words):
#                     # 计算双侧语句数量
#                     words_power[word][5] += 1.
#                     if 1 == label:
#                         # 计算正确语句对数量
#                         words_power[word][2] += 1.
#                         # 计算双侧语句正确比例
#                         words_power[word][6] += 1.
#         for word in words_power:
#             # 计算出现语句对比例
#             words_power[word][1] /= len(subset_indexs)
#             # 计算正确语句对比例
#             words_power[word][2] /= words_power[word][0]
#             # 计算单侧语句对正确比例
#             if words_power[word][3] > 1e-6:
#                 words_power[word][4] /= words_power[word][3]
#             # 计算单侧语句对比例
#             words_power[word][3] /= words_power[word][0]
#             # 计算双侧语句对正确比例
#             if words_power[word][5] > 1e-6:
#                 words_power[word][6] /= words_power[word][5]
#             # 计算双侧语句对比例
#             words_power[word][5] /= words_power[word][0]
#         sorted_words_power = sorted(words_power.iteritems(), key=lambda d: d[1][0], reverse=True)
#         LogUtil.log("INFO", "power words calculation done, len(words_power)=%d"
#                     % len(sorted_words_power))
#         return sorted_words_power
#
#     @staticmethod
#     def save_powerful_word(words_power, fp):
#         f = open(fp, 'w')
#         for ele in words_power:
#             f.write("%s" % ele[0])
#             for num in ele[1]:
#                 f.write("\t%.5f" % num)
#             f.write("\n")
#         f.close()
#
#
# class PowerfulWordDoubleSide(Extractor):
#
#     def __init__(self, config_fp, thresh_num=500, thresh_rate=0.9):
#         Extractor.__init__(self, config_fp)
#
#         powerful_word_fp = '%s/words_power.%s.txt' % (
#             self.config.get('DIRECTORY', 'devel_fp'),
#             self.config.get('MODEL', 'train_subset_name'))
#         self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
#         self.pword_dside = PowerfulWordDoubleSide\
#             .init_powerful_word_dside(self.pword, thresh_num, thresh_rate)
#
#     @staticmethod
#     def init_powerful_word_dside(pword, thresh_num, thresh_rate):
#         pword_dside = []
#         pword = filter(lambda x: x[1][0] * x[1][5] >= thresh_num, pword)
#         pword_sort = sorted(pword, key=lambda d: d[1][6], reverse=True)
#         pword_dside.extend(map(lambda x: x[0],
#                                filter(lambda x: x[1][6] >= thresh_rate, pword_sort)))
#         LogUtil.log('INFO', 'Double side power words(%d): %s' %
#                     (len(pword_dside), str(pword_dside)))
#         return pword_dside
#
#     def extract_row(self, row):
#         tags = []
#         q1_words = str(row['words_chars1']).lower().split()
#         q2_words = str(row['words_chars2']).lower().split()
#         for word in self.pword_dside:
#             if (word in q1_words) and (word in q2_words):
#                 tags.append(1.0)
#             else:
#                 tags.append(0.0)
#         return tags
#
#     def get_feature_num(self):
#         return len(self.pword_dside)
#
#
# class PowerfulWordOneSide(Extractor):
#
#     def __init__(self, config_fp, thresh_num=500, thresh_rate=0.9):
#         Extractor.__init__(self, config_fp)
#
#         powerful_word_fp = '%s/words_power.%s.txt' % (
#             self.config.get('DIRECTORY', 'devel_fp'),
#             self.config.get('MODEL', 'train_subset_name'))
#         self.pword = PowerfulWord.load_powerful_word(powerful_word_fp)
#         self.pword_oside = PowerfulWordOneSide\
#             .init_powerful_word_oside(self.pword, thresh_num, thresh_rate)
#
#     @staticmethod
#     def init_powerful_word_oside(pword, thresh_num, thresh_rate):
#         pword_oside = []
#         pword = filter(lambda x: x[1][0] * x[1][3] >= thresh_num, pword)
#         pword_oside.extend(
#             map(lambda x: x[0], filter(lambda x: x[1][4] >= thresh_rate, pword)))
#         LogUtil.log('INFO', 'One side power words(%d): %s' % (
#             len(pword_oside), str(pword_oside)))
#         return pword_oside
#
#     def extract_row(self, row):
#         tags = []
#         q1_words = set(str(row['question1']).lower().split())
#         q2_words = set(str(row['question2']).lower().split())
#         for word in self.pword_oside:
#             if (word in q1_words) and (word not in q2_words):
#                 tags.append(1.0)
#             elif (word not in q1_words) and (word in q2_words):
#                 tags.append(1.0)
#             else:
#                 tags.append(0.0)
#         return tags
#
#     def get_feature_num(self):
#         return len(self.pword_oside)
#
#
# class PowerfulWordDoubleSideRate(Extractor):
#     def __init__(self, config_fp):
#         Extractor.__init__(self, config_fp)
#
#         powerful_word_fp = '%s/words_power.%s.txt' % (
#             self.config.get('DIRECTORY', 'devel_fp'),
#             self.config.get('MODEL', 'train_subset_name'))
#         self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))
#
#     def extract_row(self, row):
#         num_least = 300
#         rate = [1.0]
#         q1_words = set(str(row['question1']).lower().split())
#         q2_words = set(str(row['question2']).lower().split())
#         share_words = list(q1_words.intersection(q2_words))
#         for word in share_words:
#             if word not in self.pword_dict:
#                 continue
#             if self.pword_dict[word][0] * self.pword_dict[word][5] < num_least:
#                 continue
#             rate[0] *= (1.0 - self.pword_dict[word][6])
#         rate = [1 - num for num in rate]
#         return rate
#
#     def get_feature_num(self):
#         return 1
#
#
# class PowerfulWordOneSideRate(Extractor):
#     def __init__(self, config_fp):
#         Extractor.__init__(self, config_fp)
#
#         powerful_word_fp = '%s/words_power.%s.txt' % (
#             self.config.get('DIRECTORY', 'devel_fp'),
#             self.config.get('MODEL', 'train_subset_name'))
#         self.pword_dict = dict(PowerfulWord.load_powerful_word(powerful_word_fp))
#
#     def extract_row(self, row):
#         num_least = 300
#         rate = [1.0]
#         q1_words = set(str(row['question1']).lower().split())
#         q2_words = set(str(row['question2']).lower().split())
#         q1_diff = list(set(q1_words).difference(set(q2_words)))
#         q2_diff = list(set(q2_words).difference(set(q1_words)))
#         all_diff = set(q1_diff + q2_diff)
#         for word in all_diff:
#             if word not in self.pword_dict:
#                 continue
#             if self.pword_dict[word][0] * self.pword_dict[word][3] < num_least:
#                 continue
#             rate[0] *= (1.0 - self.pword_dict[word][4])
#         rate = [1 - num for num in rate]
#         return rate
#
#     def get_feature_num(self):
#         return 1