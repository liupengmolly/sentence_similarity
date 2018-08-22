#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import numpy as np
import jieba
import os

reload(sys)
sys.setdefaultencoding('utf8')

from common.data_helper import DataHelper
from word2vec.word2vec_model import Word2vecModel
from data_test.ant import DATA_PATH


jieba.load_userdict(os.path.join(DATA_PATH, 'jieba_add_words.txt'))


class Embedding(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.word2vec = Word2vecModel.load(cfg.word2vec_file)
        self.word_id_map, self.id_vector_map = self.word2vec.generate_word_id_map(np.float32)

    def generate_sentence_token_ind(self,data):
        if self.cfg.feature_type == 'word':
            data=data[['words1','words2','label']]
        elif self.cfg.feature_type == 'char':
            data=data[['chars1','chars2','label']]
        elif self.cfg.feature_type == 'word+char':
            data=data[['words_chars1','words_chars2','label']]
        else:
            return ValueError('the argument feature_type must in [word,char,word+char],but your '
                              'feature_type is {}'.format(self.cfg.feature_type))
        data=list(np.array(data))
        data=[((x[0].split(),x[1].split()),int(x[2])) for x in data]
        data_help = DataHelper(zip(*data),
                               self.cfg.max_sentence_length,
                               word_index_dic=self.word_id_map,
                               by_word_index_dic=True)
        x,y=data_help.documents_transform_and_padding(self.cfg)
        return x, y


if __name__ == '__main__':
    pass