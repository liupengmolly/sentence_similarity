#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import xlrd


def read_excel(path, head=True, sheet_index=0):
    """
    read excel
    :param path: excel file address
    :param head: excel file is have head
    :param sheet_index: which sheet to choose
    :return title, content
    """

    data = xlrd.open_workbook(path)
    table = data.sheets()[sheet_index]
    n_rows = table.nrows
    n_cols = table.ncols

    title = []
    if head:
        for i in xrange(0, n_cols):
            title.append(table.cell(0, i).value)
    content = []
    for i in xrange(1, n_rows):
        row = []
        for j in xrange(0, n_cols):
            row.append(table.cell(i, j).value)
        content.append(tuple(row))

    return title, content


def xls_2_json(path, out_file, sheet_index=0):
    """
    read excel
    :param path: excel file address
    :param out_file: the out put json file
    :param sheet_index: which sheet to choose
    """

    data = xlrd.open_workbook(path)
    table = data.sheets()[sheet_index]
    n_rows = table.nrows
    n_cols = table.ncols
    title = []
    for i in xrange(0, n_cols):
        title.append(table.cell(0, i).value)
    content = []
    for i in xrange(1, n_rows):
        ad = {}
        for j in xrange(0, n_cols):
            ad[title[j]] = table.cell(i, j).value
        content.append(json.dumps(ad))
    with open(out_file, 'wb+') as f:
        f.write("\n".join(content))


def save_dict_as_json(dic, path):
    """
    把字典存储为 json 文件
    Args:
        dic: dict
        path: 保存路径
    """

    if not dic or len(dic) == 0:
        print "the dict have nothing"
    with open(path, "w") as f:
        json.dump(dic, f)


if __name__ == '__main__':
    pass