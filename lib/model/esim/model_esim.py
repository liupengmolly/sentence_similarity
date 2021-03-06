#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import tensorflow as tf
from lib.model.configs import cfg
from lib.model.esim.utils import *
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

    def build_network(self):
        with tf.variable_scope('emb'):
            s1_emb = tf.nn.embedding_lookup(self.id_vector_map, self.sent1_token)
            s2_emb = tf.nn.embedding_lookup(self.id_vector_map, self.sent2_token)

        with tf.variable_scope('sent_enc',reuse=tf.AUTO_REUSE):
            bi_s1 = BiLSTM(s1_emb,self.dropout)
            bi_s2 = BiLSTM(s2_emb,self.dropout)

            sent1_mask = tf.expand_dims(self.sent1_token_mask, axis=-1)
            sent2_mask = tf.expand_dims(self.sent2_token_mask, axis=-1)

            sent1_mask = tf.cast(sent1_mask, tf.float32)
            sent2_mask = tf.cast(sent2_mask, tf.float32)

            bi_s1 = tf.multiply(bi_s1, sent1_mask)
            bi_s2 = tf.multiply(bi_s2, sent2_mask)

        with tf.variable_scope('inference',reuse=tf.AUTO_REUSE):
            bi_s2_t = tf.transpose(bi_s2, perm=[0,2,1])

            e_product = tf.matmul(bi_s1,bi_s2_t)

            weight_s1 = tf.exp(e_product - tf.expand_dims(tf.reduce_max(e_product,1),1))
            weight_s2 = tf.exp(e_product - tf.expand_dims(tf.reduce_max(e_product,2),2))

            weight_s1 = tf.multiply(weight_s1, sent1_mask)
            weight_s2 = tf.multiply(weight_s2, tf.transpose(sent2_mask,[0,2,1]))

            soft_weight_s1 = weight_s1 / tf.expand_dims(tf.reduce_sum(weight_s1,1),1)
            soft_weight_s2 = weight_s2 / tf.expand_dims(tf.reduce_sum(weight_s2,2),2)

            e_a = tf.multiply(tf.expand_dims(soft_weight_s2,3),tf.expand_dims(bi_s2,1))
            alpha = tf.reduce_sum(e_a, 2)
            e_b = tf.multiply(tf.expand_dims(soft_weight_s1,3),tf.expand_dims(bi_s1,2))
            beta = tf.reduce_sum(e_b, 1)

            # # --------------替换为A----------------------------
            inp1 = tf.concat([bi_s1, alpha, bi_s1*alpha, bi_s1-alpha],2)
            inp2 = tf.concat([bi_s2, beta, bi_s2*beta, bi_s2-beta],2)
            inp1 = tf.nn.elu(linear([inp1],cfg.hidden_units_num,True,0.,wd=cfg.wd,
                                    output_keep_prob=cfg.dropout,is_train=self.is_train))
            inp2 = tf.nn.elu(linear([inp2],cfg.hidden_units_num,True,0.,wd=cfg.wd,
                                    output_keep_prob=cfg.dropout,is_train=self.is_train))

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
            if random.random() > 0.5:
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



