#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np


class Word2vecModel(object):

    def __init__(self, model):
        self.model = model
        self.dim = len(list(self.model.items())[0][1])

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            model = json.load(f)
        return Word2vecModel(model)

    @staticmethod
    def load_glove(path):
        model = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                word = line[0]
                vec = [float(i) for i in line[1:]]
                model[word] = vec
        return Word2vecModel(model)

    def transform(self, word):
        return self.model.get(word, [0.0001] * self.dim)

    def generate_word_id_map(self, dtype):
        # 根据已经训练的word2vec, 生成相应的word_id_map, id_vector_map

        word_id_map = {}
        id_vector_map = [[0.0001] * self.dim, [-0.0001] * self.dim]
        words = sorted(list(self.model.keys()))
        for id, word in enumerate(words):
            word_id_map[word] = id + 2
            id_vector_map.append(self.model[word])
        id_vector_map = np.array(id_vector_map, dtype=dtype)
        return word_id_map, id_vector_map


if __name__ == '__main__':
    pass