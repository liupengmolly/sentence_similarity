#!/usr/bin/env python
# -*- coding:utf-8 -*-
# df1 df2 df3 df4类型为: pandas.core.frame.DataFrame.分别引用输入桩数据
# topai(1, df1)函数把df1内容写入第一个输出桩
import argparse
import time
from random import random
import pandas as pd
from sklearn.metrics import log_loss


class Configs(object):
    def __init__(self):
        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--word2vec_file', type=str, help='word2vec file path')
        parser.add_argument(
            '--train_data', type=str, help='data file path')
        parser.add_argument(
            '--validate_data', type=str, help='validate_file save path'
        )
        parser.add_argument(
            '--model_directory', type=str, default='', help='mode directory'
        )
        parser.add_argument(
            '--model_index', type=int, help=''
        )
        parser.add_argument(
            "--models", type=str,
            help=
            '''
                model-5000, model-6000
                Load trained model checkpoint,
                separate by ,  (Default: None)
            '''
        )
        parser.add_argument('--gpu', type=str, help='gpu')
        parser.add_argument('--feature_type', type=str, default='word', help='word|char|both')
        parser.add_argument('--use_pinyin', type=bool, default=False, help='if use pinyin when '
                                                                           'embeddinng')
        parser.add_argument('--use_stacking', type=bool, default=False,
                            help='if using stacking when'
                                 'using fast_disan')
        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=500, help='max epoch number')
        parser.add_argument('--num_steps', type=int, default=1000, help='every steps to print')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
        parser.add_argument('--optimizer', type=str, default='adadelta',
                            help='choose an optimizer[adadelta|adam]')
        parser.add_argument('--learning_rate', type=float, default=0.1, help='Init Learning rate')
        parser.add_argument('--dy_lr', type=bool, default=True, help='if decay lr during training')
        parser.add_argument('--lr_decay', type=float, default=0.8, help='Learning rate decay')
        parser.add_argument('--dropout', type=float, default=0.95, help='dropout keep prob')
        parser.add_argument('--wd', type=float, default=1e-5,
                            help='weight decay factor/l2 decay factor')
        parser.add_argument('--var_decay', type=float, default=0.999, help='Learning rate')  # ema
        parser.add_argument('--decay', type=float, default=0.9, help='summary decay')  # ema
        parser.add_argument('--max_sentence_length', type=int, default=30,
                            help='the sentence max length')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--use_pre_trained', type=bool, default=True,
                            help='use or not use pre_trained w2v')
        # @ ----- Text Processing ----
        parser.add_argument('--word_embedding_length', type=int, default=300,
                            help='word embedding length')
        parser.add_argument('--lower_word', type=bool, default=True, help='')
        parser.add_argument('--data_clip_method', type=str, default='no_tree',
                            help='for space-efficiency[no_tree|]no_redundancy')
        parser.add_argument('--sent_len_rate', type=float,
                            default=0.97, help='delete too long sentences')
        timestamp = str(int(time.time()))
        parser.add_argument('--model_save_path', type=str,
                            default=timestamp, help=' file path name to save model')
        parser.add_argument('--log_name', type=str, default='', help='log file name')
        # @ ------neural network-----
        parser.add_argument('--use_char_emb', type=bool, default=False, help='abandoned')
        parser.add_argument('--use_token_emb', type=bool, default=True, help='abandoned')
        parser.add_argument('--char_embedding_length', type=int, default=8, help='(abandoned)')
        parser.add_argument('--char_out_size', type=int, default=150, help='(abandoned)')
        parser.add_argument('--out_channel_dims', type=str, default='50,50,50', help='(abandoned)')
        parser.add_argument('--filter_heights', type=str, default='1,3,5', help='(abandoned)')
        parser.add_argument('--highway_layer_num', type=int, default=2,
                            help='highway layer number(abandoned)')
        parser.add_argument('--hidden_units_num', type=int, default=300,
                            help='Hidden units number of Neural Network')
        parser.add_argument('--tree_hn', type=int, default=100, help='(abandoned)')

        parser.add_argument('--fine_tune', type=bool, default=False,
                            help='(abandoned, keep False)')  # ema

        # # emb_opt_direct_attn
        parser.add_argument('--batch_norm', type=bool, default=False,
                            help='(abandoned, keep False)')
        parser.add_argument('--activation', type=str, default='relu', help='(abandoned')

        parser.add_argument("--allow_soft_placement", type=bool, default=True,
                            help="Allow device soft device placement")
        parser.add_argument("--log_device_placement", type=bool, default=False,
                            help="Log placement of ops on devices")
        # --------------------------------@bimpm-------------------------------
        parser.add_argument('--suffix', type=str, default='normal',
                            help='Suffix of the model name.')
        parser.add_argument('--with_char', default=False,
                            help='With character-composed embeddings.',
                            action='store_true')
        parser.add_argument('--fix_word_vec', default=True,
                            help='Fix pre-trained word embeddings during training.',
                            action='store_true')
        parser.add_argument('--with_highway', default=True,
                            help='Utilize highway layers.', action='store_true')
        parser.add_argument('--with_match_highway', default=True,
                            help='Utilize highway layers for matching layer.',
                            action='store_true')
        parser.add_argument('--with_aggregation_highway', default=True,
                            help='Utilize highway layers for aggregation layer.',
                            action='store_true')
        parser.add_argument('--with_full_match', default=True,
                            help='With full matching.', action='store_true')
        parser.add_argument('--with_maxpool_match', default=False,
                            help='With maxpooling matching',
                            action='store_true')
        parser.add_argument('--with_attentive_match', default=True,
                            help='With attentive matching', action='store_true')
        parser.add_argument('--with_max_attentive_match', default=False,
                            help='With max attentive matching.',
                            action='store_true')
        parser.add_argument('--use_cudnn', type=bool, default=False, help='if use cudnn')
        parser.add_argument('--grad_clipper', type=float, default=10.0,
                            help='grad')
        parser.add_argument('--is_lower', type=bool, default=True,
                            help='is_lower')
        parser.add_argument('--with_cosine', type=bool, default=True,
                            help='with_cosine')
        parser.add_argument('--with_mp_cosine', type=bool, default=True,
                            help='map_cosine')
        parser.add_argument('--cosine_MP_dim', type=int, default=5, help='mp')
        parser.add_argument('--att_dim', type=int, default=50, help='att_dm')
        parser.add_argument('--att_type', type=str, default='symmetric',
                            help='att_type')
        parser.add_argument('--with_moving_average', type=bool, default=False,
                            help='moving_average')
        parser.add_argument('--lambda_l2', type=float, default=5e-5,
                            help='The coefficient of L2 regularizer.')
        parser.add_argument('--dropout_rate', type=float, default=0.2,
                            help='Dropout ratio.')
        parser.add_argument('--optimize_type', type=str, default='adam',
                            help='Optimizer type.')
        parser.add_argument('--context_lstm_dim', type=int, default=100,
                            help='Number of dimension for context representation layer.')
        parser.add_argument('--aggregation_lstm_dim', type=int, default=100,
                            help='Number of dimension for aggregation layer.')
        parser.add_argument('--aggregation_layer_num', type=int, default=1,
                            help='Number of LSTM layers for aggregation layer.')
        parser.add_argument('--context_layer_num', type=int, default=1,
                            help='Number of LSTM layers for context representation layer.')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['data_test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))


cfg = Configs()

from tensorflow.contrib import rnn
from operator import mul
from tensorflow.python.framework import tensor_util

def BiLSTM(x,dropout):
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
                                                 x, dtype=tf.float32)
    return outputs


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

import numpy as np

class ModelESIM:
    """
    ESIM模型，默认在训练过程中不会训练词向量
    """
    def __init__(self, id_vector_map, scope,graph=None):
        self.id_vector_map = id_vector_map
        self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
        self.sent2_token = tf.placeholder(tf.int32, [None, None], name='sent2_token')
        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        self.dropout = tf.placeholder_with_default(1.0,shape=(), name = 'dropout')

        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)

        self.global_step = tf.get_variable(
            'global_step', shape=[], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)
        self.hn = cfg.hidden_units_num
        self.output_class = 2
        self.scope = scope
        self.tensor_dict = {}

        self.logits = None
        self.predict = None
        self.loss = None
        self.accuracy = None
        self.probs = None

        #控制训练和验证测试时的不同逻辑
        self.var_ema = None
        self.ema = None
        self.learning_rate_value = cfg.learning_rate
        self.learning_rate_updated = True
        self.opt = None
        self.train_op = None
        self.update_tensor_add_ema_and_opt()

    def matmul_attention(self,sent1,sent2):
        bi_s2_t = tf.transpose(sent2, perm=[0, 2, 1])

        e_product = tf.matmul(sent1, bi_s2_t)
        return e_product

    def matadd_attention(self,sent1,sent2):
        sent1 = linear(sent1,cfg.word_embedding_length,False,scope='sent1',is_train=self.is_train)
        sent2 = linear(sent2,cfg.word_embedding_length,False,scope='sent2',is_train=self.is_train)
        sent1 = tf.expand_dims(sent1,2)
        sent2 = tf.expand_dims(sent2,1)
        f_bias = tf.get_variable(
            'f_bias', [cfg.word_embedding_length], tf.float32, tf.constant_initializer(0.))
        eproduct = linear(tf.nn.tanh(sent1+sent2+f_bias),1,True,is_train=self.is_train)
        eproduct = tf.reshape(eproduct,[-1, cfg.max_sentence_length,cfg.max_sentence_length])
        return eproduct

    def cross_representation(self,sent1,sent2,sent1_mask,sent2_mask, e_product):
        weight_s1 = tf.exp(e_product - tf.expand_dims(tf.reduce_max(e_product, 1), 1))
        weight_s2 = tf.exp(e_product - tf.expand_dims(tf.reduce_max(e_product, 2), 2))

        weight_s1 = tf.multiply(weight_s1, sent1_mask)
        weight_s2 = tf.multiply(weight_s2, tf.transpose(sent2_mask, [0, 2, 1]))

        soft_weight_s1 = weight_s1 / tf.expand_dims(tf.reduce_sum(weight_s1, 1), 1)
        soft_weight_s2 = weight_s2 / tf.expand_dims(tf.reduce_sum(weight_s2, 2), 2)

        e_a = tf.multiply(tf.expand_dims(soft_weight_s2, 3), tf.expand_dims(sent2, 1))
        alpha = tf.reduce_sum(e_a, 2)
        e_b = tf.multiply(tf.expand_dims(soft_weight_s1, 3), tf.expand_dims(sent1, 2))
        beta = tf.reduce_sum(e_b, 1)
        return alpha, beta

    def matching(self, sent1, sent2, sent1_mask, setn2_mask, embedding_length):
        # add_e = self.matadd_attention(sent1,sent2)
        mul_e = self.matmul_attention(sent1,sent2)
        # add_alpha,add_beta = self.cross_representation(sent1,sent2,sent1_mask,setn2_mask,add_e)
        mul_alpha,mul_beta = self.cross_representation(sent1,sent2,sent1_mask,setn2_mask,mul_e)

        # alpha = tf.concat([add_alpha,mul_alpha],2)
        # beta = tf.concat([add_beta,mul_beta],2)
        # alpha = tf.layers.dense(alpha,embedding_length,activation=tf.nn.elu)
        # beta = tf.layers.dense(beta,embedding_length,activation=tf.nn.elu)
        # alpha = tf.layers.dropout(alpha,rate=cfg.dropout_rate,training=self.is_train)
        # beta = tf.layers.dropout(beta,rate=cfg.dropout_rate,training = self.is_train)
        return mul_alpha, mul_beta

    def build_network(self):
        with tf.variable_scope('emb'):
            s1_emb = tf.nn.embedding_lookup(self.id_vector_map, self.sent1_token)
            s2_emb = tf.nn.embedding_lookup(self.id_vector_map, self.sent2_token)

            sent1_mask = tf.expand_dims(self.sent1_token_mask, axis=-1)
            sent2_mask = tf.expand_dims(self.sent2_token_mask, axis=-1)

            sent1_mask = tf.cast(sent1_mask, tf.float32)
            sent2_mask = tf.cast(sent2_mask, tf.float32)

        with tf.variable_scope('sent_enc',reuse=tf.AUTO_REUSE):
            s1_emb = s1_emb * sent1_mask
            s2_emb = s2_emb * sent2_mask

            s1_attn ,s2_attn = self.matching(s1_emb,s2_emb,sent1_mask,sent2_mask,cfg.word_embedding_length)
            enhance1, enhance2 = tf.concat([s1_emb, s1_attn,s1_emb*s1_attn,s1_emb-s1_attn],2),\
                                 tf.concat([s2_emb, s2_attn,s2_emb*s2_attn,s2_emb-s2_attn],2)

            bi_s1 = BiLSTM(s1_emb,self.dropout)
            bi_s2 = BiLSTM(s2_emb,self.dropout)

            bi_s1 = tf.multiply(bi_s1, sent1_mask)
            bi_s2 = tf.multiply(bi_s2, sent2_mask)

        with tf.variable_scope('inference',reuse=tf.AUTO_REUSE):
            alpha, beta = self.matching(bi_s1,bi_s2,sent1_mask,sent2_mask,2*cfg.hidden_units_num)

            # # --------------替换为A----------------------------
            inp1 = tf.concat([bi_s1, alpha, bi_s1*alpha, bi_s1-alpha],2)
            inp2 = tf.concat([bi_s2, beta, bi_s2*beta, bi_s2-beta],2)

            #-----------------------------res -------------------------------------------------
            inp1 = tf.concat([inp1, enhance1], 2)
            inp2 = tf.concat([inp2, enhance2], 2)
            #----------------------------------------------------------------------------------

            # inp1 = tf.layers.dense(inp1, cfg.hidden_units_num, activation=tf.nn.relu, name='project_inp1')
            # inp2 = tf.layers.dense(inp2, cfg.hidden_units_num, activation=tf.nn.relu, name='project_inp2')
            # inp1 = tf.layers.dropout(inp1, rate=1 - cfg.dropout, training=self.is_train)
            # inp2 = tf.layers.dropout(inp2, rate=1 - cfg.dropout, training=self.is_train)
            inp1 = tf.nn.elu(linear([inp1],cfg.hidden_units_num,True,0.,wd=cfg.wd,scope='inp1',
                                    output_keep_prob=cfg.dropout,is_train=self.is_train))
            inp2 = tf.nn.elu(linear([inp2],cfg.hidden_units_num,True,0.,wd=cfg.wd,scope='inp2',
                                    output_keep_prob=cfg.dropout,is_train=self.is_train))

        # # -------------------- A -----------------------------
        # with tf.variable_scope('fusion'):
        #     ivec = alpha.get_shape()[2]
        #
        #     o_bias1 = tf.get_variable(
        #         'o_bias1', [ivec], tf.float32, tf.constant_initializer(0.))
        #     fusion_gate1 = tf.nn.sigmoid(
        #         linear(bi_s1, ivec, True, 0., 'linear_fusion_i_1', False,
        #                cfg.wd, cfg.dropout, self.is_train) +
        #         linear(alpha, ivec, True, 0., 'linear_fusion_a_1', False,
        #                cfg.wd, cfg.dropout, self.is_train) + o_bias1)
        #     inp1 = fusion_gate1 * bi_s1 + (1-fusion_gate1) * alpha
        #
        #     o_bias2 = tf.get_variable(
        #         'o_bias2', [ivec], tf.float32, tf.constant_initializer(0.))
        #     fusion_gate2 = tf.nn.sigmoid(
        #         linear(bi_s2, ivec, True, 0., 'linear_fusion_i_2', False,
        #                cfg.wd, cfg.dropout, self.is_train) +
        #         linear(beta, ivec, True, 0., 'linear_fusion_a_2', False,
        #                cfg.wd, cfg.dropout, self.is_train) + o_bias2)
        #     inp2 = fusion_gate2 * bi_s2 + (1 - fusion_gate2) * beta
        #
        # # ----------------------------------------------------
        with tf.variable_scope('sent_dec',reuse=tf.AUTO_REUSE):
            ctx1 = BiLSTM(inp1,self.dropout)
            ctx2 = BiLSTM(inp2,self.dropout)

            avg1 = tf.reduce_sum(ctx1 * sent1_mask, 1) / tf.reduce_sum(sent1_mask, 1)
            max1 = tf.reduce_max(ctx1 * sent1_mask, 1)

            avg2 = tf.reduce_sum(ctx2 * sent2_mask, 1) / tf.reduce_sum(sent2_mask, 1)
            max2 = tf.reduce_max(ctx2 * sent2_mask, 1)
            ctx = tf.concat([avg1,max1,avg2,max2], -1)

        with tf.variable_scope('dense'):
            pre_output = tf.nn.elu(
                linear(
                    [ctx], cfg.hidden_units_num, True, 0.,
                    scope='pre_output', squeeze=False,
                    wd=cfg.wd, input_keep_prob=self.dropout,
                    output_keep_prob=cfg.dropout, is_train=self.is_train))
            logits = linear(
                [pre_output], 2, True, 0., scope='logits',
                squeeze=False, wd=cfg.wd,
                is_train=self.is_train)
        self.tensor_dict[logits] = logits
        self.probs = tf.nn.softmax(logits)
        return logits

    def build_loss(self):
        # weight_decay
        with tf.name_scope("weight_decay"):
            # 计算正则项的损失
            for var in set(tf.get_collection('reg_vars', self.scope)):
                # tf.multiply 两个数相乘
                weight_decay = tf.multiply(
                    tf.nn.l2_loss(var), cfg.wd,
                    name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)

        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        print('regularization var num: %d' % len(reg_vars))
        print('trainable var num: %d' % len(trainable_vars))
        # 样本的损失
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.logits
        )
        tf.add_to_collection(
            'losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))

        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.label
        )
        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):

        self.logits = self.build_network()

        self.logits = tf.cast(self.logits, tf.float32, name='predict_score')

        self.predict = tf.cast(tf.argmax(self.logits, -1),
                               tf.int32,
                               name='predict')
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()

        # ------------ema-------------
        if True:
            self.var_ema = tf.train.ExponentialMovingAverage(cfg.var_decay)
            self.build_var_ema()

        if cfg.mode == 'train':
            self.ema = tf.train.ExponentialMovingAverage(cfg.decay)
            self.build_ema()
        self.summary = tf.summary.merge_all()

        # ---------- optimization ---------
        if cfg.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)

        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        # trainable param num:
        # print params num
        all_params_num = 0
        for elem in trainable_vars:
            # elem.name
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'):
                continue
            params_num = 1
            for l in elem.get_shape().as_list():
                params_num *= l
            all_params_num += params_num
        print('Trainable Parameters Number: %d' % all_params_num)

        # minimize 函数集合了 compute_gradients 和 apply_gradients
        # global_step 在更新结束后会自增 1
        self.train_op = self.opt.minimize(
            self.loss, self.global_step,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))

    def build_var_ema(self):
        ema_op = self.var_ema.apply(tf.trainable_variables(), )
        with tf.control_dependencies([ema_op]):
            # 在control_dependencies的作用块下，需要增加一个新节点到 graph 中,
            # 表示在执行self.loss 之前, 先执行 ema_op
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)
        ema_op = self.ema.apply(tensors)
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def train_step(self, sess, batch_samples):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples)
        loss, train_op, accuracy, logits, predict, probs = sess.run(
            [self.loss, self.train_op, self.accuracy,
             self.logits, self.predict, self.probs],
            feed_dict=feed_dict)
        return loss, accuracy, predict, logits, probs

    def validate_step(self, sess, batch_samples):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'valid')
        accuracy, loss, predict, logits = sess.run(
            [self.accuracy, self.loss, self.predict, self.logits],
            feed_dict=feed_dict)
        return accuracy, loss, predict, logits


    def get_feed_dict(self, batch, mode = 'train'):
        x, y = zip(*batch)
        sentences1 = []
        sentences2 = []
        for ele in x:
            if random() > 0.5:
                sentences1.append(ele[0])
                sentences2.append(ele[1])
            else:
                sentences1.append(ele[1])
                sentences2.append(ele[0])
        feed_dict = {self.sent1_token: sentences1,
                     self.sent2_token: sentences2,
                     self.label: y,
                     self.is_train: True if mode == 'train' else False,
                     self.dropout: cfg.dropout if mode == 'train' else 1.0,
                     self.learning_rate: self.learning_rate_value}
        return feed_dict

    def update_learning_rate(self, global_step):
        if self.learning_rate_value < 5e-6:
            return
        self.learning_rate_value *= cfg.lr_decay


from tensorflow.contrib import learn


def tokenizer_word(iterator):
    for value in iterator:
        yield str(value).split()


class TfVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None):
        tokenizer_fn = tokenizer_word
        self.sup = super(TfVocabularyProcessor, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary,
                          tokenizer_fn)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids


import jieba
from sklearn.model_selection import train_test_split

UNKNOWN = 0  # 表示找不到的词
PAD = 1  # 表示向量的扩展位


class DataHelper(object):

    def __init__(self,
                 data=None,
                 max_document_length=50,
                 word_index_dic=None,
                 by_word_index_dic=False):

        self.data = data
        self.max_document_length = max_document_length
        self.by_word_index_dic = by_word_index_dic
        if self.by_word_index_dic:
            if not word_index_dic:
                raise Exception("请传入有效的 word_index_dic 或者 "
                                "初始化 by_trained_word2vec 为False")
            self.word_index_dic = word_index_dic
        else:
            x_1 = [" ".join(x[0]) for x in self.data[0]]
            x_2 = [" ".join(x[1]) for x in self.data[0]]
            documents = np.concatenate((x_1, x_2), axis=0)
            vocab_processor = TfVocabularyProcessor(max_document_length)
            vocab_processor.fit_transform(documents)
            print("Length of loaded vocabulary ={}".format(
                len(vocab_processor.vocabulary_)))
            self.vocab_processor = vocab_processor

    def set_by_trained_word2vec(self, value):
        self.by_word_index_dic = value

    def get_vocab_processor(self):
        return self.vocab_processor

    def documents_transform_and_padding(self, cfg):
        """
        将句子转换成index, unknow 0, 不足最大长度的位用 -1 填充
        """

        x, y = self.data
        x_new = []
        if self.by_word_index_dic and self.word_index_dic:
            for pair in x:
                pair_1, pair_2 = pair
                pair_1 = self.sentence_transform_and_padding(
                    pair_1, self.max_document_length, cfg)
                pair_2 = self.sentence_transform_and_padding(
                    pair_2, self.max_document_length, cfg)

                if random() > 0.5:
                    x_new.append((pair_1, pair_2))
                else:
                    x_new.append((pair_2, pair_1))
            return x_new, y
        else:
            x_1, x_2 = zip(*x)
            x_1 = [" ".join(ele) for ele in x_1]
            x_2 = [" ".join(ele) for ele in x_2]
            x_1 = np.asanyarray(list(self.vocab_processor.transform(x_1)))
            x_2 = np.asanyarray(list(self.vocab_processor.transform(x_2)))
            for i in range(len(y)):
                if random() > 0.5:
                    x_new.append((x_1[i], x_2[i]))
                else:
                    x_new.append((x_2[i], x_1[i]))
            return x_new, y

    def sentence_transform_and_padding(self, words, max_sentence_length, cfg):
        # 外部 word_index, 对句子进行 id map
        if len(words) > max_sentence_length:
            words = words[-max_sentence_length:]
        indexes = [self.word_index_dic.get(w, UNKNOWN)
                   for w in words]
        if len(indexes) < max_sentence_length:
            indexes = indexes + [PAD, ] * (max_sentence_length - len(indexes))
        return indexes

    @staticmethod
    def data_split(data, valid_size=0.1, test_size=0.1):
        # 按照 valid_size, test_size 对数据进行切分
        x, y = data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=valid_size)

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    @staticmethod
    def batch_iter(data, batch_size, num_epochs=1, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        :parameter data:
        :parameter batch_size:
        :parameter num_epochs:
        :parameter shuffle:
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        if len(data) % batch_size == 0:
            num_batches_per_epoch -= 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    @staticmethod
    def save_test_data(pairs, path, seg='$$$'):
        """
         保存测试数据到文件中
         :parameter pairs: 三元组
         :parameter path: 文件路径
         :parameter seg: 分隔符

        """

        if not pairs or len(pairs) == 0:
            print("pairs is empty")
        with open(path, 'w') as f:
            for pair in pairs:
                text_1 = " ".join([str(ele) for ele in pair[0]])
                text_2 = " ".join([str(ele) for ele in pair[1]])
                label = str(pair[2])
                f.write(seg.join([text_1, text_2, label]) + '\n')

    @staticmethod
    def load_test_data(path, seg='$$$'):
        """
        加载测试数据路径
        :parameter path: 文件路径
        :parameter seg: 分隔符
        :return 相似文本对和标签
        """

        pairs = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                context = f.readlines()
                for line in context:
                    if line and line.strip():
                        pair = line.strip().split(seg)
                        text_1 = [int(ele) for ele in pair[0].split()]
                        text_2 = [int(ele) for ele in pair[1].split()]
                        label = int(pair[2])
                        pairs.append((text_1, text_2, label))
        return pairs

    @staticmethod
    def load_predict_data(path, seg="$$$"):
        """
        加载测试数据路径
        :parameter path: 文件路径
        :parameter seg:　句子分隔符
        :return 待计算相似度的文本对
        """
        pairs = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                context = f.readlines()
                for line in context:
                    if line and line.strip():
                        pair = line.strip().split(seg)
                        if len(pair) < 3:
                            continue
                        text_1 = list(jieba.cut(pair[1]))
                        text_2 = list(jieba.cut(pair[2]))
                        pairs.append((text_1, text_2))
        return pairs


import json


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


custom_words = ['花呗', '借呗', '余额宝', '运费险', '销户',
           '健康果', '网商贷', '周周盈', '网银', '收钱码',
           '红包码', '乘车码', '收款码', '扫码', '定投',
           '飞月宝', '哈喽', '余利宝']
for word in custom_words:
    jieba.add_word(word)
class Embedding(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.word2vec = self.load_ray_word2vec(df3)
        self.stop_words = self.load_stop_word(df4)
        self.word_id_map, self.id_vector_map = self.word2vec.generate_word_id_map(np.float32)

    def transform(self, df):
        model = {}
        for i in np.array(df):
            model[i[0]] = [float(j) for j in i[1:]]
        return Word2vecModel(model)

    def load_ray_word2vec(self, df):
        model = {}
        word2vec_map = zip(df['word'], df['vec'])
        word2vec_map = [(x[0], x[1].split('_')) for x in word2vec_map]
        for item in word2vec_map:
            model[item[0]] = [float(x) for x in item[1]]
        return Word2vecModel(model)

    def load_stop_word(self, df):
        return set(list(df['word']))

    def generate_sentence_token_ind(self, data):
        data = list(np.array(data)[:, 1:4])
        method = sentences2char
        if self.cfg.feature_type == 'word':
            method = self.sentences2word
        data = [((method(x[0]), method(x[1])), int(x[2])) for x in data]
        data_help = DataHelper(zip(*data),
                               self.cfg.max_sentence_length,
                               word_index_dic=self.word_id_map,
                               by_word_index_dic=True)
        x, y = data_help.documents_transform_and_padding(self.cfg)
        return x, y

def sentences2word(self, sentence):
    words = list(jieba.cut(sentence))
    words = [w for w in words if w not in self.stop_words]
    return words


def sentences2char(sentence):
    return list(sentence)


import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
import time

start_time = time.localtime().tm_hour * 60 + time.localtime().tm_min
emb = Embedding(cfg)

train_data = df1
x_train, y_train = emb.generate_sentence_token_ind(train_data)
train_data_emb = list(zip(x_train, y_train))

valid_data = df2
x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
valid_data_emb = list(zip(x_valid, y_valid))

num_epoch = int(len(train_data) / cfg.batch_size) + 1

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)
    with sess.as_default():
        with tf.variable_scope("ant") as scope:
            model = ModelESIM(emb.id_vector_map, scope.name)
    sess.run(tf.global_variables_initializer())

    best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    last_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
    old_ckpt_path = model_dir + 'esim_res_match_v2.1'
    last_saver.restore(sess, old_ckpt_path)
    best_path = model_dir + 'esim_res_match_best_v3.1'
    last_path = model_dir + 'esim_res_match_v3.1'

    batches = DataHelper.batch_iter(
        train_data_emb, cfg.batch_size, cfg.max_epoch, shuffle=True)
    last_f1 = 0.0
    current_step = 0
    step = 0
    train_loss_array = []
    train_accuracy_array = []
    train_y = []
    train_predict_array = []
    logs = []
    for batch in batches:
        train_loss, train_acc, train_predict, _, _ = model.train_step(sess, batch)
        train_loss_array.append(train_loss)
        train_accuracy_array.extend(train_acc)
        train_predict_array.extend(train_predict)
        x, y = zip(*batch)
        train_y.extend(y)
        if step > 0 and step % 400 == 0:
            step_acc = np.mean(train_accuracy_array)
            step_loss = np.mean(train_loss_array)
            step_f1 = f1_score(train_y, train_predict_array)
            logs.append((step, step_acc, step_f1, 0))
            print(
                'current step: {}, train loss: {}, train_accuracy: {}, train f1: {}'.format(
                    step, step_loss, step_acc, step_f1))
            train_loss_array = []
            train_accuracy_array = []
            train_y = []
            train_predict_array = []
        if step > 0 and step % 1000 == 0:
            validate_batches = DataHelper.batch_iter(
                valid_data_emb, cfg.batch_size, shuffle=False)
            accuracy_array = []
            loss_array = []
            # predict_pos = 0
            valid_predict = []
            for validate_batch in validate_batches:
                accuracy, loss, predict, logits = model.validate_step(
                    sess, validate_batch)
                accuracy_array.extend(accuracy)
                loss_array.append(loss)
                valid_predict.extend(predict)
            accuracy = np.mean(accuracy_array)
            loss = np.mean(loss_array)

            if step % 6000 == 0:
                model.update_learning_rate(model.global_step)

            f1 = f1_score(y_valid, valid_predict)
            logs.append((step, accuracy, f1, 1))
            print("current step: %d" % step +
                  "validate data f1: %f" % f1)
            if f1 > last_f1 and step >= 2000:
                best_saver.save(sess, best_path)
                print('saved model')
                last_f1 = f1
                current_step = step
        if step % 100 == 0:
            if time.localtime().tm_hour * 60 + time.localtime().tm_min - start_time > 116:
                last_saver.save(sess, last_path)
                break
        step += 1
    logs = pd.DataFrame(logs, columns=['step', 'loss', 'accuracy', 'mode'])
    topai(1, logs)

# class Embedding(object):
#     def __init__(self, cfg):
#         self.cfg = cfg
#         self.word2vec = Word2vecModel.load(cfg.word2vec_file)
#         self.word_id_map, self.id_vector_map = self.word2vec.generate_word_id_map(np.float32)
#
#     def transform(self,df):
#         model={}
#         for i in np.array(df):
#             model[i[0]]=[float(j) for j in i[1:]]
#         return Word2vecModel(model)
#
#     def generate_sentence_token_ind(self, data):
#         data = list(np.array(data)[:, 1:4])
#         method = sentences2char
#         if self.cfg.feature_type == 'word':
#             method = sentences2word
#         data = [((method(x[0]), method(x[1])), int(x[2])) for x in data]
#         data_help = DataHelper(zip(*data),
#                                self.cfg.max_sentence_length,
#                                word_index_dic=self.word_id_map,
#                                by_word_index_dic=True)
#         x, y = data_help.documents_transform_and_padding(self.cfg)
#         return x, y
# def sentences2char(sentence):
#     return list(sentence)
#
# def sentences2word(sentence):
#     words = list(jieba.cut(sentence))
#     return words
#
# import os
# import numpy as np
# import tensorflow as tf
# from sklearn.metrics import f1_score
# import time
# import sys
# import logging
# root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# sys.path.append(root_path)
# logging.basicConfig(filename="ant_esim.log" + cfg.log_name,
#                     filemode="w",
#                     format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
#                     level=logging.INFO)
#
# os.environ['CUDA_VISIBLE_DEVICES']=cfg.gpu
# # start_time = time.localtime().tm_hour * 60 + time.localtime().tm_min
# emb = Embedding(cfg)
#
# train_data = pd.read_csv(cfg.train_data,'\t')
# x_train, y_train = emb.generate_sentence_token_ind(train_data)
# train_data_emb = list(zip(x_train, y_train))
#
# valid_data = pd.read_csv(cfg.validate_data,'\t')
# x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
# valid_data_emb = list(zip(x_valid, y_valid))
#
# num_epoch = int(len(train_data) / cfg.batch_size) + 1
#
# with tf.Graph().as_default():
#     #initializer = tf.random_uniform_initializer(-0.01, 0.01)
#     gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
#                                 allow_growth=True)
#
#     graph_config = tf.ConfigProto(
#         gpu_options=gpu_options, allow_soft_placement=True)
#     sess = tf.Session(config=graph_config)
#     with sess.as_default():
#         with tf.variable_scope("ant") as scope:
#             model = ModelESIM(emb.id_vector_map, scope.name)
#     sess.run(tf.global_variables_initializer())
#
#     checkpoint_dir = os.path.abspath(
#         os.path.join(os.path.curdir, "ant_esim_runs", cfg.model_save_path))
#
#     checkpoint_prefix = os.path.join(checkpoint_dir, "model")
#     if not os.path.exists(checkpoint_dir):
#         os.makedirs(checkpoint_dir)
#     last_f1 = 0.0
#     current_step = 0
#     saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
#     batches = DataHelper.batch_iter(
#         train_data_emb, cfg.batch_size, cfg.max_epoch, shuffle=True)
#     step = 0
#     train_loss_array = []
#     train_accuracy_array = []
#     train_y = []
#     train_predict_array = []
#     for batch in batches:
#         train_loss, train_acc, train_predict, _ ,_ = model.train_step(sess, batch)
#         train_loss_array.append(train_loss)
#         train_accuracy_array.extend(train_acc)
#         train_predict_array.extend(train_predict)
#         x, y = zip(*batch)
#         train_y.extend(y)
#         if step > 0 and step % 100 == 0:
#             logging.info(
#                 'current step: {}, train loss: {}, train_accuracy: {}, train f1: {}'.format(
#                     step,
#                     np.mean(train_loss_array),
#                     np.mean(train_accuracy_array),
#                     f1_score(train_y, train_predict_array)))
#
#             train_loss_array = []
#             train_accuracy_array = []
#             train_y = []
#             train_predict_array = []
#         if step > 0 and step % cfg.num_steps == 0:
#             validate_batches = DataHelper.batch_iter(
#                 valid_data_emb, cfg.batch_size, shuffle=False)
#             accuracy_array = []
#             loss_array = []
#             # predict_pos = 0
#             valid_predict = []
#             for validate_batch in validate_batches:
#                 accuracy, loss, predict, logits = model.validate_step(
#                     sess, validate_batch)
#                 accuracy_array.extend(accuracy)
#                 loss_array.append(loss)
#                 valid_predict.extend(predict)
#             accuracy = np.mean(accuracy_array)
#             loss = np.mean(loss_array)
#             if step % (3*cfg.num_steps)==0:
#                 model.update_learning_rate(model.global_step)
#
#             f1 = f1_score(y_valid, valid_predict)
#
#             logging.info("current step: %d" % step +
#                          "validate data f1: %f" % f1)
#             if f1 > last_f1 and step >= 2000:
#                 saver.save(sess,checkpoint_prefix,step)
#                 print('saved model')
#                 last_f1 = f1
#                 current_step = step
#         step += 1