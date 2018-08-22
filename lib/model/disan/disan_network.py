#! /usr/bin/env python
# -*- coding: utf-8 -*-

from lib.model.utils import *


# ---------------   DiSAN Interface  ----------------
def disan(rep_tensor,
          rep_mask,
          scope=None,
          keep_prob=1.,
          is_train=None,
          wd=0.,
          activation='elu',
          tensor_dict=None,
          name=''):
    """
    :param rep_tensor: 输入的多行样本, 已经转化成了词向量, 是一个三维矩阵
    :param rep_mask: 多行业本的token 对应的index, 并利用tf.cast([], tf.bool)进行了转化,
     即index不为0的都是true, 为0的是false
    :param scope:
    :param keep_prob:
    :param is_train:
    :param wd:
    :param activation:
    :param tensor_dict:
    :param name:
    :return:
    """
    with tf.variable_scope(scope or 'DiSAN'):
        with tf.variable_scope('ct_attn'):
            fw_res = directional_attention_with_dense(
                rep_tensor, rep_mask, 'forward', 'dir_attn_fw',
                keep_prob, is_train, wd, activation,
                tensor_dict=tensor_dict, name=name+'_fw_attn')
            bw_res = directional_attention_with_dense(
                rep_tensor, rep_mask, 'backward', 'dir_attn_bw',
                keep_prob, is_train, wd, activation,
                tensor_dict=tensor_dict, name=name+'_bw_attn')

            seq_rep = tf.concat([fw_res, bw_res], -1)

        with tf.variable_scope('sent_enc_attn'):
            sent_rep = multi_dimensional_attention(
                seq_rep, rep_mask, 'multi_dimensional_attention',
                keep_prob, is_train, wd, activation,
                tensor_dict=tensor_dict, name=name+'_attn')
            return sent_rep


# --------------- supporting networks ----------------
def directional_attention_with_dense(rep_tensor,
                                     rep_mask,
                                     direction=None,
                                     scope=None,
                                     keep_prob=1.,
                                     is_train=None,
                                     wd=0.,
                                     activation='elu',
                                     tensor_dict=None,
                                     name=None):
    def scaled_tanh(x, scale=5.):
        return scale * tf.nn.tanh(1./scale * x)
    # bs 数据集中句子的数量, sl 每个句子token的数量, vec 向量的维度
    bs, sl, vec = tf.shape(
        rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    # get_shape 返回的是Dimension(dim) 对象
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope
                           or 'directional_attention_%s' % direction or 'diag'):
        # mask generation
        sl_indices = tf.range(sl, dtype=tf.int32)
        # 用于下面 tf.greater 按照 行和列索引大小 生成 mask 矩阵
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)
        if direction is None:
            direct_mask = tf.cast(
                tf.diag(- tf.ones([sl], tf.int32)) + 1, tf.bool)
        else:
            if direction == 'forward':
                # 两个矩阵相同位置元素 如果sl_row > sl_col direct_task为true
                # 生成一个下三角为True 其他为False的矩阵
                direct_mask = tf.greater(sl_row, sl_col)
            else:
                direct_mask = tf.greater(sl_col, sl_row)

        direct_mask_tile = tf.tile(tf.expand_dims(direct_mask, 0), [bs, 1, 1])
        rep_mask_tile = tf.tile(tf.expand_dims(rep_mask, 1), [1, sl, 1])
        # attn_mask 的生成
        attn_mask = tf.logical_and(direct_mask_tile, rep_mask_tile)

        # rep_map 是进行过论文中公式(14)后的得到的 [句数, 句长, 词长]
        rep_map = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map',
                                 activation, False, wd, keep_prob, is_train)
        rep_map_tile = tf.tile(tf.expand_dims(rep_map, 1), [1, sl, 1, 1])
        rep_map_dp = dropout(rep_map, keep_prob, is_train)

        # attention
        with tf.variable_scope('attention'):  # bs,sl,sl,vec
            f_bias = tf.get_variable(
                'f_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            # dependent_etd 对应论文公式(15)中的W(1)H(i)
            # head_etd 对应论文公式(15)中的W(2)H(j)
            # f_bias 对应论文公式(15)中的b(1)
            dependent = linear(
                rep_map_dp, ivec, False, scope='linear_dependent')
            dependent_etd = tf.expand_dims(dependent, 1)
            head = linear(rep_map_dp, ivec, False, scope='linear_head')
            head_etd = tf.expand_dims(head, 2)
            # 对应论文中的公式15
            logits = scaled_tanh(dependent_etd + head_etd + f_bias, 5.0)
            logits_masked = exp_mask_for_high_rank(logits, attn_mask)

            # 对应论文中的公式13
            attn_score = tf.nn.softmax(logits_masked, 2)  # bs,sl,sl,vec
            attn_score = mask_for_high_rank(attn_score, attn_mask)
            attn_result = tf.reduce_sum(attn_score * rep_map_tile, 2)

        with tf.variable_scope('output'):
            o_bias = tf.get_variable(
                'o_bias', [ivec], tf.float32, tf.constant_initializer(0.))
            # input gate 对应公式19
            fusion_gate = tf.nn.sigmoid(
                linear(rep_map, ivec, True, 0., 'linear_fusion_i', False,
                       wd, keep_prob, is_train) +
                linear(attn_result, ivec, True, 0., 'linear_fusion_a', False,
                       wd, keep_prob, is_train) +
                o_bias)
            # 对应公式20
            output = fusion_gate * rep_map + (1-fusion_gate) * attn_result

            output = mask_for_high_rank(output, rep_mask)

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name + '_dependent'] = dependent
            tensor_dict[name + '_head'] = head
            tensor_dict[name] = attn_score
            tensor_dict[name + '_gate'] = fusion_gate
        return output


def multi_dimensional_attention(rep_tensor,
                                rep_mask,
                                scope=None,
                                keep_prob=1.,
                                is_train=None,
                                wd=0.,
                                activation='elu',
                                tensor_dict=None,
                                name=None):
    bs, sl, vec = tf.shape(
        rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1',
                              activation, False, wd, keep_prob, is_train)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec

        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft

        return attn_output


def bn_dense_layer(input_tensor,
                   hn,
                   bias,
                   bias_start=0.0,
                   scope=None,
                   activation='relu',
                   enable_bn=True,
                   wd=0.,
                   keep_prob=1.0,
                   is_train=None):
    if is_train is None:
        is_train = False

    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'bn_dense_layer'):

        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        # 是否进行BN操作
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True,
                is_training=is_train, scope='bn')
        return activation_func(linear_map)

def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


