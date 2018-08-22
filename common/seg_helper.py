#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba.posseg as pseg
from common.stop_word import stop_words


def cut(text, remove_stop=True, with_role=True):
    """
    split sentence to word
    :parameter text: text to split
    :parameter remove_stop: whether remove stop word
    :parameter with_role: whether with role, joined word and word role by '$'
    """

    if text is None:
        raise Exception("text is None")

    res = pseg.cut(text)

    if remove_stop:
        res = [x for x in res if x.word not in stop_words]

    if not with_role:
        res = [x.word for x in res]
    else:
        res = ["$".join([x.word, x.flag]) for x in res]
    return res


if __name__ == '__main__':
    print(cut("买什么跌什么啊..", remove_stop=False))
