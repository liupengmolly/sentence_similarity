#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random

import tensorflow as tf

from lib.model.configs import cfg
from lib.model.disan.disan_network import disan
from lib.model.fast_disan.utils import linear


# noinspection PyPackageRequirements
class ModelDiSAN:
    def __init__(self, id_vector_map, scope, pos_dic=None, char_id_vector_map=None,
                 pos_id_vector_map=None):
        self.id_vector_map = id_vector_map
        self.char_id_vector_map = char_id_vector_map
        if cfg.use_pos:
            self.pos_count = len(pos_dic) + 2
        # 词性 embedding
        self.pos_emb = pos_id_vector_map
        # 位置 embedding
        self.position_emb = None
        self.word_count = len(id_vector_map)

        # sent1_token 句子列表, 每个句子中的token已经转化成了ind
        # sent1_char 第一维表示句子, 第二维是token, 第三维是char, tl是每个token的最大长度
        self.sent1_token = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length], name='sent1_token')

        self.sent2_token = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length], name='sent2_token')

        self.char1 = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length, cfg.char_length], name='char1'
        )
        self.char2 = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length, cfg.char_length], name='char2'
        )

        self.sent1_pos = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length], name='sent1_pos'
        )
        self.sent2_pos = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length], name='sent2_pos'
        )
        self.sent1_idf = tf.placeholder(
            tf.float32, [None, cfg.max_sentence_length, 1], name='sent1_idf'
        )
        self.sent2_idf = tf.placeholder(
            tf.float32, [None, cfg.max_sentence_length, 1], name='sent2_idf'
        )
        self.sent1_token_position = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length, cfg.max_sentence_length],
            name='sent1_token_position')

        self.sent2_token_position = tf.placeholder(
            tf.int32, [None, cfg.max_sentence_length, cfg.max_sentence_length],
            name='sent2_token_position')

        self.extract_match_feature = tf.placeholder(
            tf.float32, [None, cfg.extract_match_feature_len], name='extract_match_feature'
        )
        self.sent1_token_match = tf.placeholder(
            tf.float32, [None, cfg.max_sentence_length, 1], name='sent1_token_match'
        )
        self.sent2_token_match = tf.placeholder(
            tf.float32, [None, cfg.max_sentence_length, 1], name='sent2_token_match'
        )

        # self.sent1_token_position = tf.placeholder(
        #     tf.int32, [None, cfg.max_sentence_length], name='sent1_token_position')
        #
        # self.sent2_token_position = tf.placeholder(
        #     tf.int32, [None, cfg.max_sentence_length], name='sent2_token_position')

        self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')

        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)

        self.tensor_dict = {}
        self.global_step = tf.get_variable(
            'global_step', shape=[], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)
        self.hn = cfg.hidden_units_num
        self.output_class = 2

        self.scope = scope
        self.logits = None
        self.predict = None
        self.sim_feature = None
        self.score = None
        self.loss = None
        self.probs = None
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
                        [self.word_count, cfg.word_embedding_length], -1.0, 1.0),
                    trainable=True, name="W"
                )

            s1_emb = tf.nn.embedding_lookup(
                self.id_vector_map, self.sent1_token)
            # bs,sl2,tel
            s2_emb = tf.nn.embedding_lookup(
                self.id_vector_map, self.sent2_token)

            if cfg.word2vec_sim:
                with tf.variable_scope('word2vec_sim_matrix'):
                    s2_emb_ = tf.transpose(s2_emb, perm=[0, 2, 1])
                    word2vec_sim_matrix = tf.matmul(s1_emb, s2_emb_)
                    s1_weight = self.sent1_idf
                    s2_weight = tf.transpose(self.sent2_idf, perm=[0, 2, 1])
                    attentions = tf.matmul(s1_weight, s2_weight)
                    sim_matrix = tf.multiply(word2vec_sim_matrix, attentions)
                    sim_matrix = tf.reshape(
                        sim_matrix, shape=[-1, cfg.max_sentence_length * cfg.max_sentence_length])
                    self.sim_feature = tf.nn.elu(
                        linear([sim_matrix], hn, True, 0., scope='sim_feature', squeeze=False,
                               wd=cfg.wd, input_keep_prob=cfg.dropout, is_train=self.is_train))

            if cfg.use_char_emb:
                with tf.variable_scope(name_or_scope='chars_emb', reuse=tf.AUTO_REUSE) as scope:
                    chars_emb1 = tf.nn.embedding_lookup(self.char_id_vector_map, self.char1)
                    chars_emb2 = tf.nn.embedding_lookup(self.char_id_vector_map, self.char2)

                    chars_emb1 = tf.reshape(
                        chars_emb1, shape=[-1, cfg.char_length * cfg.char_embedding_length]
                    )
                    chars_emb1 = tf.nn.elu(
                        linear([chars_emb1], cfg.char_embedding_length, True, 0., scope=scope,
                               squeeze=False, wd=cfg.wd, input_keep_prob=cfg.dropout,
                               is_train=self.is_train))

                    chars_emb1 = tf.reshape(
                        chars_emb1, shape=[-1, cfg.max_sentence_length, cfg.char_embedding_length]
                    )

                    chars_emb2 = tf.reshape(
                        chars_emb2, shape=[-1, cfg.char_length * cfg.char_embedding_length]
                    )

                    chars_emb2 = tf.nn.elu(
                        linear([chars_emb2], cfg.char_embedding_length, True, 0., scope=scope,
                               squeeze=False, wd=cfg.wd, input_keep_prob=cfg.dropout,
                               is_train=self.is_train))

                    chars_emb2 = tf.reshape(
                        chars_emb2, shape=[-1, cfg.max_sentence_length, cfg.char_embedding_length]
                    )

                    s1_emb = tf.concat([s1_emb, chars_emb1], axis=2)
                    s2_emb = tf.concat([s2_emb, chars_emb2], axis=2)

            if cfg.use_pos:
                if self.pos_emb is None:
                    self.pos_emb = tf.Variable(
                        tf.random_uniform(
                            [self.pos_count, cfg.pos_embedding_length], -1.0, 1.0
                        ), trainable=True, name='pos_emb'
                    )
                sent_pos1_emb = tf.nn.embedding_lookup(self.pos_emb, self.sent1_pos)
                sent_pos2_emb = tf.nn.embedding_lookup(self.pos_emb, self.sent2_pos)
                s1_emb = tf.concat([s1_emb, sent_pos1_emb], axis=2)
                s2_emb = tf.concat([s2_emb, sent_pos2_emb], axis=2)

            # if cfg.use_idf:
            #     s1_emb = tf.concat([s1_emb, self.sent1_idf], axis=2)
            #     s2_emb = tf.concat([s2_emb, self.sent2_idf], axis=2)

            if cfg.use_position:
                # self.position_emb = tf.Variable(
                #     tf.random_uniform(
                #         [cfg.max_sentence_length, cfg.position_embedding_length], -1.0, 1.0
                #     ), trainable=True, name='position_emb'
                # )
                # sent1_token_position = tf.nn.embedding_lookup(
                #     self.position_emb, self.sent1_token_position)
                # sent2_token_position = tf.nn.embedding_lookup(
                #     self.position_emb, self.sent2_token_position)

                s1_emb = tf.concat([s1_emb, self.sent1_token_position], axis=2)
                s2_emb = tf.concat([s2_emb, self.sent2_token_position], axis=2)

            if cfg.use_token_match:
                s1_emb = tf.concat([s1_emb, self.sent1_token_match], axis=2)
                s2_emb = tf.concat([s2_emb, self.sent2_token_match], axis=2)

            self.tensor_dict['s1_emb'] = s1_emb
            self.tensor_dict['s2_emb'] = s2_emb

        with tf.variable_scope('sent_enc'):
            print "s1_emb shape:", s1_emb.get_shape()
            print "self.sent1_token_mask: ", self.sent1_token_mask.get_shape()
            s1_rep = disan(
                s1_emb, self.sent1_token_mask, 'DiSAN', cfg.dropout,
                self.is_train, cfg.wd, 'elu', self.tensor_dict, 's1'
            )
            self.tensor_dict['s1_rep'] = s1_rep

            tf.get_variable_scope().reuse_variables()

            s2_rep = disan(
                s2_emb, self.sent2_token_mask, 'DiSAN', cfg.dropout,
                self.is_train, cfg.wd, 'elu', self.tensor_dict, 's2'
            )
            self.tensor_dict['s2_rep'] = s2_rep

        with tf.variable_scope('output'):
            out_rep = tf.concat(
                [s1_rep, s2_rep, s1_rep - s2_rep, s1_rep * s2_rep], -1)

            if cfg.use_extract_match:
                out_rep = tf.concat([out_rep, self.extract_match_feature], axis=1)

            pre_output = tf.nn.elu(
                linear(
                    [out_rep], hn, True, 0.,
                    scope='pre_output', squeeze=False,
                    wd=cfg.wd, input_keep_prob=cfg.dropout,
                    is_train=self.is_train))

            if cfg.word2vec_sim:
                pre_output = tf.concat([pre_output, self.sim_feature], axis=1)

            print 'pre_output: ',  pre_output.get_shape()

            logits = linear(
                [pre_output], self.output_class, True, 0., scope='logits',
                squeeze=False, wd=cfg.wd, input_keep_prob=cfg.dropout,
                is_train=self.is_train)

            self.tensor_dict[logits] = logits
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

        self.probs = tf.nn.softmax(self.logits)

        self.score = tf.cast(self.logits, tf.float32, name='predict_score')

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

        # out_tensor_dict_1= sess.run(self.tensor_dict, feed_dict=feed_dict)
        logits=None
        if get_summary:
            loss, summary, train_op, accuracy, predict = sess.run(
                [self.loss, self.summary, self.train_op, self.accuracy, self.predict],
                feed_dict=feed_dict)

        else:
            loss, train_op, accuracy, logits, predict = sess.run(
                [self.loss, self.train_op, self.accuracy,
                 self.logits, self.predict],
                feed_dict=feed_dict)
        return loss, accuracy, predict, logits

    def validate_step(self, sess, batch_samples):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, 'valid')
        accuracy, loss, predict, logits = sess.run(
            [self.accuracy, self.loss, self.predict, self.logits],
            feed_dict=feed_dict)
        return accuracy, loss, predict, logits

    def get_feed_dict(self, batch, data_type='train'):
        x, y = zip(*batch)
        features = zip(*x)
        sentences1 = []
        sentences2 = []
        pos1 = []
        pos2 = []
        idf1 = []
        idf2 = []
        position1 = []
        position2 = []
        extract_match_feature = []
        chars1 = []
        chars2 = []
        tokens1_match = []
        tokens2_match = []

        for ele in features[0]:
            sentences1.append(ele[0])
            sentences2.append(ele[1])

        if cfg.use_pos:
            if len(features) > 1:
                features = features[1:]
            for ele in features[0]:
                pos1.append(ele[0])
                pos2.append(ele[1])

        if cfg.use_idf:
            if len(features) > 1:
                features = features[1:]
            for ele in features[0]:
                idf1.append(ele[0])
                idf2.append(ele[1])

        if cfg.use_position:
            if len(features) > 1:
                features = features[1:]
            for ele in features[0]:
                position1.append(ele[0])
                position2.append(ele[1])

        if cfg.use_extract_match:
            if len(features) > 1:
                features = features[1:]
            for ele in features[0]:
                extract_match_feature.append(ele[0])

        if cfg.use_char_emb:
            if len(features) > 1:
                features = features[1:]
            for ele in features[0]:
                chars1.append(ele[0])
                chars2.append(ele[1])

        if cfg.use_token_match:
            if len(features) > 1:
                features = features[1:]
            for ele in features[0]:
                tokens1_match.append(ele[0])
                tokens2_match.append(ele[1])

        feed_dict = {self.sent1_token: sentences1,
                     self.sent2_token: sentences2,
                     self.gold_label: y,
                     self.is_train: True if data_type == 'train' else False,
                     self.learning_rate: self.learning_rate_value}
        if cfg.use_pos:
            feed_dict[self.sent1_pos] = pos1
            feed_dict[self.sent2_pos] = pos2

        if cfg.use_idf:
            feed_dict[self.sent1_idf] = idf1
            feed_dict[self.sent2_idf] = idf2

        if cfg.use_position:
            feed_dict[self.sent1_token_position] = position1
            feed_dict[self.sent2_token_position] = position2

        if cfg.use_extract_match:
            feed_dict[self.extract_match_feature] = extract_match_feature

        if cfg.use_char_emb:
            feed_dict[self.char1] = chars1
            feed_dict[self.char2] = chars2

        if cfg.use_token_match:
            import numpy as np
            print "tokens1_match shape is: ", np.array(tokens1_match).shape
            feed_dict[self.sent1_token_match] = tokens1_match
            feed_dict[self.sent2_token_match] = tokens2_match

        return feed_dict

    def update_learning_rate(
            self, global_step, lr_decay_factor=0.7):
        if cfg.dy_lr:
            if self.learning_rate_value < 5e-4:
                return
            if global_step >= 200000 and global_step % 10000 == 0:
                self.learning_rate_value *= lr_decay_factor
