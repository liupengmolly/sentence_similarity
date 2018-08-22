#! /usr/bin/env python
# -*- coding: utf-8 -*-

import json


def covert2json(path, output_path):

    f = open(path, path)
    temp = []

    for line in f:
        line = line.split()
        word = line[0]
        vec = [float(i) for i in list[1:]]
        temp.append(json.dumps())

if __name__ == '__main__':
    pass