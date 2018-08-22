#! /usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
from collections import Counter


def train(sentences):
    model = Word2Vec(min_count=1, size=60)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.iter)
    w2v_dict = {}
    words = []
    for sentence in sentences:
        words.extend(sentence)
    words_filtered = [item[0] for item in Counter(words).items()]
    for w in words_filtered:
        w2v_dict[w] = [float(y) for y in list(model[w])]
    return w2v_dict
