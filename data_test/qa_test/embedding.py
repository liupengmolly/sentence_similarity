#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from common.data_helper import DataHelper
from lib.model.configs import cfg
from word2vec.word2vec_model import Word2vecModel

# 加载外部的word2vec
word2vec = Word2vecModel.load_glove(cfg.word2vec_file)
# 生成词和索引，索引和向量的词典
word_id_map, id_vector_map = word2vec.generate_word_id_map(np.float32)


def generate_sentence_token_ind(data):
    data = list(np.array(data)[:, 1:4])
    data = [((str(x[0]).split(), str(x[1]).split()), int(x[2])) for x in data]
    data_help = DataHelper(zip(*data),
                           cfg.max_sentence_length,
                           word_index_dic=word_id_map,
                           by_word_index_dic=True)
    x, y = data_help.documents_transform_and_padding(cfg)
    return x, y

if __name__ == '__main__':
    pass