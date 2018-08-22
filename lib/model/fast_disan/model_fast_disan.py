#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random

import tensorflow as tf

from lib.model.configs import cfg
from lib.model.fast_disan.fast_disan_network import fast_disan
from lib.model.fast_disan.utils import linear


# noinspection PyPackageRequirements
class ModelFastDiSAN:
    def __init__(self, id_vector_map, scope):
        self.id_vector_map = id_vector_map

        # sent1_token 句子列表, 每个句子中的token已经转化成了ind
        # sent1_char 第一维表示句子, 第二维是token, 第三维是char, tl是每个token的最大长度
        self.sent1_token = tf.placeholder(
            tf.int32, [None, None], name='sent1_token')

        self.sent2_token = tf.placeholder(
            tf.int32, [None, None], name='sent2_token')

        self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')

        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)

        self.tensor_dict = {}
        self.global_step = tf.get_variable(
            'global_step', shape=[], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)
        self.tel = cfg.word_embedding_length
        self.hn = cfg.hidden_units_num
        self.output_class = 2

        self.scope = scope
        self.logits = None
        self.predict = None
        self.loss = None
        self.accuracy = None
        self.var_ema = None
        self.ema = None
        self.summary = None
        self.previous_dev_loss = []
        self.learning_rate_value = cfg.learning_rate
        self.learning_rate_updated = False
        self.opt = None
        self.train_op = None
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        hn = self.hn

        with tf.variable_scope('emb'):
            if not cfg.use_pre_trained:
                self.id_vector_map = tf.Variable(
                    tf.random_uniform(
                        [len(self.id_vector_map), cfg.word_embedding_length], -1.0, 1.0),
                    trainable=True, name="W"
                )

            s1_emb = tf.nn.embedding_lookup(
                self.id_vector_map, self.sent1_token)
            # bs,sl2,tel
            s2_emb = tf.nn.embedding_lookup(
                self.id_vector_map, self.sent2_token)
            self.tensor_dict['s1_emb'] = s1_emb
            self.tensor_dict['s2_emb'] = s2_emb

        with tf.variable_scope('sent_enc'):
            s1_rep = fast_disan(
                s1_emb, self.sent1_token_mask, 'DiSAN', cfg.dropout,
                self.is_train, cfg.wd, 'elu', self.tensor_dict, 's1'
            )
            self.tensor_dict['s1_rep'] = s1_rep

            tf.get_variable_scope().reuse_variables()

            s2_rep = fast_disan(
                s2_emb, self.sent2_token_mask, 'DiSAN', cfg.dropout,
                self.is_train, cfg.wd, 'elu', self.tensor_dict, 's2'
            )
            self.tensor_dict['s2_rep'] = s2_rep

        with tf.variable_scope('output'):
            out_rep = tf.concat(
                [s1_rep, s2_rep, s1_rep - s2_rep, s1_rep * s2_rep], -1)
            pre_output = tf.nn.elu(
                linear(
                    [out_rep], hn, True, 0.,
                    scope='pre_output', squeeze=False,
                    wd=cfg.wd, input_keep_prob=cfg.dropout,
                    is_train=self.is_train))
            logits = linear(
                [pre_output], self.output_class, True, 0., scope='logits',
                squeeze=False, wd=cfg.wd, input_keep_prob=cfg.dropout,
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
            labels=self.gold_label,
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
            self.gold_label
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

        ema_op = self.var_ema.apply(tf.trainable_variables(),)
        with tf.control_dependencies([ema_op]):
            # 在control_dependencies的作用块下，需要增加一个新节点到 graph 中,
            # 表示在执行self.loss 之前, 先执行 ema_op
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)

        ema_op = self.ema.apply(tensors)
        # for var in tf.get_collection("ema/scalar", scope=self.scope):
        #     ema_var = self.ema.average(var)
        #     tf.summary.scalar(ema_var.op.name, ema_var)
        #
        # for var in tf.get_collection("ema/vector", scope=self.scope):
        #     ema_var = self.ema.average(var)
        #     tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def train_step(self, sess, batch_samples, logger, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples)
        logits = 0
        probs = 0
        # out_tensor_dict_1= sess.run(self.tensor_dict, feed_dict=feed_dict)
        if get_summary:
            loss, summary, train_op, accuracy, predict = sess.run(
                [self.loss, self.summary, self.train_op, self.accuracy, self.predict],
                feed_dict=feed_dict)

        else:
            loss, train_op, accuracy, logits, predict, probs= sess.run(
                [self.loss, self.train_op, self.accuracy,
                 self.logits, self.predict,self.probs],
                feed_dict=feed_dict)
        return loss, accuracy, predict,logits,probs

    def validate_step(self, sess, batch_samples):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples,'valid')
        accuracy, loss, predict, logits = sess.run(
            [self.accuracy, self.loss, self.predict, self.logits],
            feed_dict=feed_dict)
        return accuracy, loss, predict, logits

    def get_feed_dict(self, batch, data_type='train'):
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
                     self.gold_label: y,
                     self.is_train: True if data_type == 'train' else False,
                     self.learning_rate: self.learning_rate_value}
        return feed_dict

    def update_learning_rate(
            self, current_dev_loss, global_step, lr_decay_factor=0.7):
        if cfg.dy_lr:
            method = 1
            if method == 0:
                assert len(self.previous_dev_loss) >= 1
                self.previous_dev_loss.append(current_dev_loss)
                delta = []
                pre_loss = self.previous_dev_loss.pop(0)
                for loss in self.previous_dev_loss:
                    delta.append(pre_loss < loss)
                    pre_loss = loss

                do_decay = True
                for d in delta:
                    do_decay = do_decay and d

                if do_decay:
                    self.learning_rate_value *= lr_decay_factor
                    self.learning_rate_updated = True
                    print(
                        'found dev loss increases, decrease learning rate '
                        'to: %f' % self.learning_rate_value)

                if global_step % 10000 == 0:
                    if not self.learning_rate_updated:
                        self.learning_rate_value *= 0.9
                        print('decrease learning rate to:'
                              ' %f' % self.learning_rate_value)
                    else:
                        self.learning_rate_updated = False
            elif method == 1:
                if self.learning_rate_value < 5e-6:
                    return
                if global_step % 10000 == 0:
                    self.learning_rate_value *= lr_decay_factor
