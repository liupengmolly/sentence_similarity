#! /usr/bin/env python
# -*- coding: utf-8 -*-
from pypinyin import pinyin
import os
from random import random
import jieba

import numpy as np
from sklearn.model_selection import train_test_split
from common.tf_vocabulary_processor import TfVocabularyProcessor
UNKNOWN = 1  # 表示找不到的词
PAD = 0  # 表示向量的扩展位

def f1(pos,pred_pos,pred_correct):
    r=pred_correct/pred_pos
    f=pred_correct/pos
    return f*r*2/(f+r)


class DataHelper(object):

    def __init__(self,
                 data=None,
                 max_document_length=50,
                 word_index_dic=None,
                 by_word_index_dic=False):

        self.data = data
        self.max_document_length = max_document_length
        self.by_word_index_dic = by_word_index_dic
        if self.by_word_index_dic:
            if not word_index_dic:
                raise Exception("请传入有效的 word_index_dic 或者 "
                                "初始化 by_trained_word2vec 为False")
            self.word_index_dic = word_index_dic
        else:
            x_1 = [" ".join(x[0]) for x in self.data[0]]
            x_2 = [" ".join(x[1]) for x in self.data[0]]
            documents = np.concatenate((x_1, x_2), axis=0)
            vocab_processor = TfVocabularyProcessor(max_document_length)
            vocab_processor.fit_transform(documents)
            print("Length of loaded vocabulary ={}".format(
                len(vocab_processor.vocabulary_)))
            self.vocab_processor = vocab_processor

    def set_by_trained_word2vec(self, value):
        self.by_word_index_dic = value

    def get_vocab_processor(self):
        return self.vocab_processor

    def documents_transform_and_padding(self, cfg):
        """
        将句子转换成index, unknow 0, 不足最大长度的位用 -1 填充
        """

        x, y = self.data
        x_new = []
        if self.by_word_index_dic and self.word_index_dic:
            for pair in x:
                pair_1, pair_2 = pair
                pair_1 = self.sentence_transform_and_padding(
                    pair_1, self.max_document_length, cfg)
                pair_2 = self.sentence_transform_and_padding(
                    pair_2, self.max_document_length, cfg)

                if random() > 0.5:
                    x_new.append((pair_1, pair_2))
                else:
                    x_new.append((pair_2, pair_1))
            return x_new, y
        else:
            x_1, x_2 = zip(*x)
            x_1 = [" ".join(ele) for ele in x_1]
            x_2 = [" ".join(ele) for ele in x_2]
            x_1 = np.asanyarray(list(self.vocab_processor.transform(x_1)))
            x_2 = np.asanyarray(list(self.vocab_processor.transform(x_2)))
            for i in range(len(y)):
                if random() > 0.5:
                    x_new.append((x_1[i], x_2[i]))
                else:
                    x_new.append((x_2[i], x_1[i]))
            return x_new, y

    def sentence_transform_and_padding(self, words, max_sentence_length, cfg):
        # 外部 word_index, 对句子进行 id map
        if len(words) > max_sentence_length:
            words = words[:max_sentence_length]
        indexes = [self.word_index_dic.get(w, UNKNOWN)
                   for w in words]
        if len(indexes) < max_sentence_length:
            indexes = indexes + [PAD, ] * (max_sentence_length - len(indexes))
        if cfg.use_pinyin:
            pinyins = pinyin(words)
            pinyin_idxs = [self.word_index_dic.get(w[0], UNKNOWN)
                       for w in pinyins]
            if len(pinyin_idxs) < max_sentence_length:
                pinyin_idxs = pinyin_idxs + [PAD, ] * (max_sentence_length - len(pinyin_idxs))
            # indexes.extend(pinyin_idxs)
            indexes = pinyin_idxs
        return indexes

    @staticmethod
    def data_split(data, valid_size=0.1, test_size=0.1):
        # 按照 valid_size, test_size 对数据进行切分
        x, y = data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=valid_size)

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    @staticmethod
    def batch_iter(data, batch_size, num_epochs=1, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        :parameter data:
        :parameter batch_size:
        :parameter num_epochs:
        :parameter shuffle:
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        if len(data) % batch_size == 0:
            num_batches_per_epoch -= 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    @staticmethod
    def save_test_data(pairs, path, seg='$$$'):
        """
         保存测试数据到文件中
         :parameter pairs: 三元组
         :parameter path: 文件路径
         :parameter seg: 分隔符

        """

        if not pairs or len(pairs) == 0:
            print("pairs is empty")
        with open(path, 'w') as f:
            for pair in pairs:
                text_1 = " ".join([str(ele) for ele in pair[0]])
                text_2 = " ".join([str(ele) for ele in pair[1]])
                label = str(pair[2])
                f.write(seg.join([text_1, text_2, label]) + '\n')

    @staticmethod
    def load_test_data(path, seg='$$$'):
        """
        加载测试数据路径
        :parameter path: 文件路径
        :parameter seg: 分隔符
        :return 相似文本对和标签
        """

        pairs = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                context = f.readlines()
                for line in context:
                    if line and line.strip():
                        pair = line.strip().split(seg)
                        text_1 = [int(ele) for ele in pair[0].split()]
                        text_2 = [int(ele) for ele in pair[1].split()]
                        label = int(pair[2])
                        pairs.append((text_1, text_2, label))
        return pairs

    @staticmethod
    def load_predict_data(path, seg="$$$"):
        """
        加载测试数据路径
        :parameter path: 文件路径
        :parameter seg:　句子分隔符
        :return 待计算相似度的文本对
        """
        pairs = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                context = f.readlines()
                for line in context:
                    if line and line.strip():
                        pair = line.strip().split(seg)
                        if len(pair) < 3:
                            continue
                        text_1 = list(jieba.cut(pair[1]))
                        text_2 = list(jieba.cut(pair[2]))
                        pairs.append((text_1, text_2))
        return pairs


if __name__ == '__main__':
    #DataHelper.data_split()
    print(f1(3228,3870,2532))
