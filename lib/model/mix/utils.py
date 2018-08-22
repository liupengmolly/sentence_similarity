#! /usr/bin/env python
# -*- coding: utf-8 -*-
from tensorflow.python.util import nest
import tensorflow as tf
from lib.model.configs import cfg
from tensorflow.contrib import rnn
from operator import mul

def BiLSTM(x,dropout,scope=None):
    # lstm_fw_cell = rnn.BasicLSTMCell(cfg.hidden_units_num)
    # lstm_bw_cell = rnn.BasicLSTMCell(cfg.hidden_units_num)
    stacked_lstm = []
    stacked_bw_lstm = []
    for i in range(1):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(cfg.hidden_units_num)
        lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell,output_keep_prob=dropout)
        lstm_cell_bw = tf.contrib.rnn.BasicLSTMCell(cfg.hidden_units_num)
        lstm_cell_bw = tf.contrib.rnn.DropoutWrapper(lstm_cell_bw, output_keep_prob = dropout)
        stacked_lstm.append(lstm_cell)
        stacked_bw_lstm.append(lstm_cell_bw)

    # 建立前向和后向的多层LSTM
    Gmcell = tf.contrib.rnn.MultiRNNCell(stacked_lstm)
    Gmcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_lstm)

    outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn([Gmcell], [Gmcell_bw],
                                                 x, dtype=tf.float32,scope=scope)
    return outputs

def BiGRU(x,dropout,scope=None):
    stacked_gru = []
    stacked_bw_gru = []
    for i in range(1):
        gru_cell = tf.contrib.rnn.GRUCell(cfg.hidden_units_num)
        gru_cell = tf.contrib.rnn.DropoutWrapper(gru_cell,output_keep_prob=dropout)
        gru_cell_bw = tf.contrib.rnn.GRUCell(cfg.hidden_units_num)
        gru_cell_bw = tf.contrib.rnn.DropoutWrapper(gru_cell_bw, output_keep_prob = dropout)
        stacked_gru.append(gru_cell)
        stacked_bw_gru.append(gru_cell_bw)

    # 建立前向和后向的多层LSTM
    Gmcell = tf.contrib.rnn.MultiRNNCell(stacked_gru)
    Gmcell_bw = tf.contrib.rnn.MultiRNNCell(stacked_bw_gru)

    outputs, _, _ = rnn.stack_bidirectional_dynamic_rnn([Gmcell], [Gmcell_bw],
                                                 x, dtype=tf.float32,scope=scope)
    return outputs

def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        # if keep_prob < 1.0:
        d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        out = tf.cond(is_train, lambda: d, lambda: x)
        return out

def conv1d(in_, filters, kernels, stride, padding, is_train=None,activation=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        out = tf.layers.conv1d(in_,filters,kernels,stride,padding,activation=activation)
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob, scope="conv1d_{}".format(height))
            outs.append(out)
        concat_out = tf.concat(outs, axis=2)
        return concat_out

def conv2d(in_, filters, kernels, stride, padding, is_train=None,activation=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv2d"):
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        out = tf.layers.conv2d(in_,filters,kernels,stride,padding,activation=activation)
        return out

def reduce(func,seq):
    if len(seq)==0:
        return  None
    result=seq[0]
    for i in seq[1:]:
        result=func(result,i)
    return result

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

def linear(args,
           output_size,
           bias,
           bias_start=0.0,
           scope=None,
           squeeze=False,
           wd=0.0,
           input_keep_prob=1.0,
           output_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        # 如果args 是tensor,会进行这一步
        args = [args]

    # flat_args 一个二维矩阵构成的 数组, 这里的二维数组是 一个batch的所有句子的所有词的向量
    # 第一维是 句子数 * 句子长度, 第二维是词向量长度
    flat_args = [flatten(arg, 1) for arg in args]

    flat_args = [tf.cond(is_train,
                        lambda: tf.nn.dropout(arg, input_keep_prob),
                        lambda: arg)
                for arg in flat_args]
    flat_out = _linear(
        flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    if output_keep_prob < 1.0 and is_train is True:
        flat_out = tf.layers.dropout(flat_out, rate = 1-cfg.dropout,training=is_train)
    # out [句数, 句长, d(h)]
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])

    if wd:
        add_reg_without_bias()

    return out

def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter

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

def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans*gate + in_val* (1.0- gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs

def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

def conv_highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len * passage_len, output_size])
    with tf.variable_scope(scope or "conv_highway_layer"):
        highway_w = tf.get_variable("conv_highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("conv_highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("conv_full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("conv_full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans*gate + in_val* (1.0- gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, passage_len, output_size])
    return outputs

def multi_conv_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'conv_highway_layer'
    if scope is not None: scope_name = scope
    for i in xrange(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = conv_highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val

