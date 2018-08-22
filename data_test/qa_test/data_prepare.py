#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import pandas as pd
import numpy as np

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

from common.data_helper import DataHelper


def generate_data(data_path):
    data = pd.read_csv(data_path, sep='$', header=0)
    data = list(np.array(data)[:, 1:4])
    data = [((x[0], x[1]), x[2]) for x in data]
    data = zip(*data)
    x_train, y_train, x_valid, y_valid, x_test, y_test = DataHelper.data_split(
        data, valid_size=0.1, test_size=0.15
    )
    train_data = [(x_train[ind][0], x_train[ind][1], y_train[ind])
                  for ind in range(len(x_train))]
    valid_data = [(x_valid[ind][0], x_valid[ind][1], y_valid[ind])
                  for ind in range(len(x_valid))]
    test_data = [(x_test[ind][0], x_test[ind][1], y_test[ind])
                 for ind in range(len(x_test))]
    train_data_path = os.path.join(os.path.dirname(data_path), 'train_data.csv')
    pd.DataFrame(train_data).to_csv(train_data_path, sep='$')
    valid_data_path = os.path.join(os.path.dirname(data_path), 'valid_data.csv')
    pd.DataFrame(valid_data).to_csv(valid_data_path, sep='$')
    test_data_path = os.path.join(os.path.dirname(data_path), 'test_data.csv')
    pd.DataFrame(test_data).to_csv(test_data_path, sep='$')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='input data path')
    args = parser.parse_args()
    generate_data(args.data_path)
