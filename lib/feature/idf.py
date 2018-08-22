#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import sys

reload(sys)
sys.setdefaultencoding('utf8')


class Idf(object):
    def __init__(self, data):
        self.data = data
        self.words_idf = self.calculate()

    def calculate(self):
        words_dic = {}
        for s in self.data:
            for w in set(s):
                if w not in words_dic:
                    words_dic[w] = 0
                words_dic[w] += 1
        doc_count = len(self.data)
        words_idf = dict([(item[0], math.log((doc_count * 1.0) / item[1]))
                          for item in words_dic.items()])
        return words_idf


