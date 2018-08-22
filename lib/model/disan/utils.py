#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
from functools import reduce
from operator import mul

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_mat=None,
                           extra_trainable=False, scope=None):
    """
    generate embedding matrix for looking up
    :param dict_size: indices 0 and 1 corresponding to empty and unknown token
    :param emb_len:
    :param init_mat: init mat matching for [dict_size, emb_len]
    :param extra_mat: extra tensor [extra_dict_size, emb_len]
    :param extra_trainable:
    :param scope:
    :return: if extra_mat is None, return[dict_size+extra_dict_size,emb_len],
     else [dict_size,emb_len]
    """
    with tf.variable_scope(scope or 'gene_emb_mat'):
        emb_mat_ept_and_unk = tf.constant(
            value=0, dtype=tf.float32, shape=[2, emb_len])
        if init_mat is None:
            emb_mat_other = tf.get_variable(
                'emb_mat', [dict_size - 2, emb_len], tf.float32)
        else:
            emb_mat_other = tf.get_variable(
                "emb_mat", [dict_size - 2, emb_len], tf.float32,
                initializer=tf.constant_initializer(
                    init_mat[2:], dtype=tf.float32, verify_shape=True))
        emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)

        if extra_mat is not None:
            if extra_trainable:
                extra_mat_var = tf.get_variable(
                    "extra_emb_mat", extra_mat.shape, tf.float32,
                    initializer=tf.constant_initializer(
                        extra_mat, dtype=tf.float32, verify_shape=True))
                return tf.concat([emb_mat, extra_mat_var], 0)
            else:
                extra_mat_con = tf.constant(extra_mat, dtype=tf.float32)
                return tf.concat([emb_mat, extra_mat_con], 0)
        else:
            return emb_mat


def linear(args,
           output_size,
           bias,
           bias_start=0.0,
           scope=None,
           squeeze=False,
           wd=0.0,
           input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        # 如果args 是tensor,会进行这一步
        args = [args]

    # flat_args 一个二维矩阵构成的 数组, 这里的二维数组是 一个batch的所有句子的所有词的向量
    # 第一维是 句子数 * 句子长度, 第二维是词向量长度
    flat_args = [flatten(arg, 1) for arg in args]

    if input_keep_prob < 1.0:
        assert is_train is not None
        # tf.cond 函数的解释参见源码注释
        flat_args = [tf.cond(is_train,
                             lambda: tf.nn.dropout(arg, input_keep_prob),
                             lambda: arg)
                     for arg in flat_args]
    flat_out = _linear(
        flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    # out [句数, 句长, d(h)]
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out


def _linear(xs,
            output_size,
            bias,
            bias_start=0.,
            scope=None):
    # 函数做的事对应 论文中的公式(14)中神经元内部的部分
    with tf.variable_scope(scope or 'linear_layer'):
        # 所有的句子的向量矩阵 按照第二维进行 concat, 即所有句子的同位置词的向量连接了起来
        # x的第一维表示 所有的句子* 所有的句子,  第二维是词向量的长度
        x = tf.concat(xs, -1)
        # input_size 是 word_vector_size
        input_size = x.get_shape()[-1]
        # 论文中W 是一个 (d(h), 句长) 维的矩阵, 这边是 (词长, d(h)) 维
        # #理解错了 与论文一致 W 是[词长, d(h)]
        W = tf.get_variable(
            'W', shape=[input_size, output_size], dtype=tf.float32,
        )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(
                                       bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        # 输出是 二维矩阵, 一维是一个batch的句子数量 * 句子长度,
        # 二维是论文中的d(h)维,隐藏层词向量长度
        return out


def flatten(tensor, keep):
    """
    用于将多维的tensor 转化二维的 tensor
    :parameter tensor: 待转化的矩阵
    :parameter keep: 0-keep 维度转化成第一维, keep-n 转化成第二维
    """

    # 获取所有维度
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    # 前面 n-1 维度相乘
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i]
                        for i in range(start)])
    # 生成 shape [n-1 维乘积, n 维]
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i]
                          for i in range(start, len(fixed_shape))]
    # 将tensor 转化成上述shape的 tensor, 是一个二维的矩阵
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    # 这个函数主要是用来对 tensor 做shape变换, 将一个二维的矩阵变成一个三维的
    dim_reduced_keep = dim_reduced_keep or keep

    # ref 是一个batch所有句子所有词组成的三维矩阵, 一维是句子数量 二维是句子长度, 三维是词向量长度
    ref_shape = ref.get_shape().as_list() # original shape
    # tensor 一个二维矩阵 第一维是句数*句长 第二维是转化后的词长
    tensor_shape = tensor.get_shape().as_list() # current shape

    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    # 在这里 pre_shape 中的元素其实有2个, [句数, 句长]
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #

    # keep_shape 只有一个, 转化后的词长
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i]
                  for i in range(tensor_start, len(tensor_shape))] #

    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    # 返回的是一个三维的tensor [句数, 句长, 词长]
    return out


def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32),
                       name=name or 'mask_for_high_rank')


def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val,
                  (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')


def selu(x):
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))


def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter
if __name__ == '__main__':
    pass