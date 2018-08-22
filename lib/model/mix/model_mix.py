#! /usr/bin/env python
# -*- coding: utf-8 -*-

import random
import tensorflow as tf
from lib.model.configs import cfg
from lib.model.mix.utils import *
import numpy as np

class ModelMIX:
    def __init__(self, id_vector_map, scope,graph=None):
        self.id_vector_map = id_vector_map
        self.sent1_token = tf.placeholder(tf.int32, [None, None], name='sent1_token')
        self.sent2_token = tf.placeholder(tf.int32, [None, None], name='sent2_token')
        self.sent1_idf = tf.placeholder(tf.float32, [None,None], name = 'sent1_idf')
        self.sent2_idf = tf.placeholder(tf.float32, [None,None], name = 'sent2_idf')
        self.label = tf.placeholder(tf.int32, [None], name='label')
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')
        self.X1_len = tf.placeholder(tf.int32, name='X1_len', shape=(None,))
        self.X2_len = tf.placeholder(tf.int32, name='X2_len', shape=(None, ))
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        self.dropout = tf.placeholder_with_default(1.0,shape=(), name = 'dropout')
        self.dpool_index = tf.placeholder(tf.int32, name='dpool_index',
                                          shape=(None, cfg.max_sentence_length,
                                                 cfg.max_sentence_length,3))

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

    def interact_ngram(self,uni1,bi1,tri1,uni2,bi2,tri2):
        uni_uni = self.normalize(tf.einsum('abd,acd->abc',uni1,uni2))
        uni_bi = self.normalize(tf.einsum('abd,acd->abc',uni1,bi2))
        uni_tri = self.normalize(tf.einsum('abd,acd->abc',uni1,tri2))
        bi_uni = self.normalize(tf.einsum('abd,acd->abc',bi1,uni2))
        bi_bi = self.normalize(tf.einsum('abd,acd->abc',bi1,bi2))
        bi_tri = self.normalize(tf.einsum('abd,acd->abc',bi1,tri2))
        tri_uni = self.normalize(tf.einsum('abd,acd->abc',tri1,uni2))
        tri_bi = self.normalize(tf.einsum('abd,acd->abc',tri1,bi2))
        tri_tri = self.normalize(tf.einsum('abd,acd->abc',tri1,tri2))
        interact_img = tf.stack([uni_uni,uni_bi,uni_tri,bi_uni,bi_bi,bi_tri,tri_uni,tri_bi,tri_tri],-1)
        return interact_img

    def combine_feature(self,img,feature):
        feature = tf.expand_dims(feature,-1)
        return tf.concat([img,img*feature],-1)

    def build_network(self):
        with tf.variable_scope('emb'):
            s1_emb = tf.nn.embedding_lookup(self.id_vector_map, self.sent1_token)
            s2_emb = tf.nn.embedding_lookup(self.id_vector_map, self.sent2_token)

        with tf.variable_scope("enc"):
            s1_emb = BiLSTM(s1_emb, self.dropout,'bilstm1')
            s2_emb = BiLSTM(s2_emb, self.dropout,'bilstm2')

            sent1_mask = tf.expand_dims(self.sent1_token_mask, axis=-1)
            sent2_mask = tf.expand_dims(self.sent2_token_mask, axis=-1)

            sent1_mask = tf.cast(sent1_mask, tf.float32)
            sent2_mask = tf.cast(sent2_mask, tf.float32)

            s1_emb = tf.multiply(s1_emb, sent1_mask)
            s2_emb = tf.multiply(s2_emb, sent2_mask)

        with tf.variable_scope('ngram'):
            self.emb1_unigram = conv1d(s1_emb,cfg.word_embedding_length,1,1,'same',
                                       is_train=self.is_train,activation=tf.nn.relu,scope='uni1')
            self.emb1_bigram = conv1d(s1_emb,cfg.word_embedding_length,2,1,'same',
                                       is_train=self.is_train,activation=tf.nn.relu,scope='bi1')
            self.emb1_trigram = conv1d(s1_emb,cfg.word_embedding_length,3,1,'same',
                                       is_train=self.is_train,activation=tf.nn.relu,scope='tri1')
            self.emb2_unigram = conv1d(s2_emb,cfg.word_embedding_length,1,1,'same',
                                       is_train=self.is_train,activation=tf.nn.relu,scope='uni2')
            self.emb2_bigram = conv1d(s2_emb,cfg.word_embedding_length,2,1,'same',
                                      is_train=self.is_train,activation=tf.nn.relu,scope='bi2')
            self.emb2_trigram = conv1d(s2_emb,cfg.word_embedding_length,3,1,'same',
                                       is_train=self.is_train,activation=tf.nn.relu,scope='tri2')

        with tf.variable_scope('interact'):
            self.interact_img = self.interact_ngram(self.emb1_unigram,self.emb1_bigram,self.emb1_trigram,
                                                    self.emb2_unigram,self.emb2_bigram,self.emb2_trigram)

            self.interact_idf = tf.einsum('ab,ac->abc',self.sent1_idf,self.sent2_idf)

            self.interact_combine_feature = self.combine_feature(self.interact_img,self.interact_idf)

        with tf.variable_scope('conv'):
            self.conv1 = conv2d(self.interact_combine_feature,32,3,1,'same',is_train=self.is_train,
                                activation=tf.nn.relu,scope='conv1')

            sent1_mask = tf.cast(self.sent1_token_mask, tf.int32)
            sent2_mask = tf.cast(self.sent2_token_mask, tf.int32)
            # self.sent1_len = tf.reduce_sum(sent1_mask,1)
            # self.sent2_len = tf.reduce_sum(sent2_mask,1)
            # dpool_index = self.dynamic_pooling_index(self.sent1_len,self.sent2_len,
            #                                          cfg.max_sentence_length,cfg.max_sentence_length)
            self.dynamic_conv1 = tf.gather_nd(self.conv1, self.dpool_index)

            self.pool1 = tf.layers.max_pooling2d(self.dynamic_conv1,3,3)

            self.conv2 = conv2d(self.pool1,64,3,1,'same',is_train=self.is_train,
                                activation=tf.nn.relu,scope='conv2')
            self.pool2 = tf.layers.max_pooling2d(self.conv2,2,2)

            self.conv3 = conv2d(self.pool2, 64, 3, 1, 'same', is_train=self.is_train,
                                activation=tf.nn.relu, scope='conv3')
            self.pool3 = tf.layers.max_pooling2d(self.conv3, 3, 2)

        with tf.variable_scope('fc'):
            self.fc1 = tf.reshape(self.pool3, [-1, 2 * 2 * 64])

        with tf.variable_scope('dense'):
            pre_output = tf.nn.elu(
                linear(
                    [self.fc1], cfg.hidden_units_num, True, 0.,
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

    def get_valid_lengths(self,batches):
        x, y = zip(*batches)
        x1, x2 = zip(*x)
        x1_lens, x2_lens = [], []
        for i in range(len(x1)):
            x1_len = sum(np.array(x1[i]) != 1)
            x2_len = sum(np.array(x2[i]) != 1)
            x1_lens.append(x1_len)
            x2_lens.append(x2_len)
        return np.array(x1_lens), np.array(x2_lens)

    def dynamic_pooling_index(self, len1, len2, max_len1, max_len2):
        def dpool_index_(batch_idx, len1_one, len2_one, max_len1, max_len2):
            stride1 = 1.0 * max_len1 / len1_one
            stride2 = 1.0 * max_len2 / len2_one
            idx1_one = [int(i/stride1) for i in range(max_len1)]
            idx2_one = [int(i/stride2) for i in range(max_len2)]
            mesh1, mesh2 = np.meshgrid(idx1_one, idx2_one)
            index_one = np.transpose(np.stack([np.ones(mesh1.shape) * batch_idx, mesh1, mesh2]), (2,1,0))
            return index_one
        index = []
        for i in range(len(len1)):
            index.append(dpool_index_(i, len1[i], len2[i], max_len1, max_len2))
        return np.array(index)

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

    def train_step(self, sess, batch_samples, idf_map):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples,idf_map)
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len],
                                                                 feed_dict[self.X2_len],
                                                                 cfg.max_sentence_length,
                                                                 cfg.max_sentence_length)
        loss, train_op, accuracy, logits, predict, probs = sess.run(
            [self.loss, self.train_op, self.accuracy,
             self.logits, self.predict, self.probs],
            feed_dict=feed_dict)
        return loss, accuracy, predict, logits, probs

    def validate_step(self, sess, batch_samples, idf_map):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples, idf_map, 'valid')
        feed_dict[self.dpool_index] = self.dynamic_pooling_index(feed_dict[self.X1_len],
                                                                 feed_dict[self.X2_len],
                                                                 cfg.max_sentence_length,
                                                                 cfg.max_sentence_length)
        accuracy, loss, predict, logits = sess.run(
            [self.accuracy, self.loss, self.predict, self.logits],
            feed_dict=feed_dict)
        return accuracy, loss, predict, logits

    def normalize(self, x):
        mean = tf.reduce_mean(x,[1,2])
        mean = tf.expand_dims(tf.expand_dims(mean,1),2)
        std = tf.sqrt(tf.nn.moments(x,[1,2])[1])
        std = tf.expand_dims(tf.expand_dims(std,1),2)
        x = (x - mean)/(3*std)
        x = tf.where(tf.abs(x)<1.0, x, x/tf.abs(x))
        return x

    def get_idf(self,x,idf_map):
        sent1_idf, sent2_idf = zip(*[([idf_map[e] for e in ele[0]],
                                      [idf_map[e] for e in ele[1]]) for ele in x])
        sent1_idf, sent2_idf = np.array(sent1_idf),np.array(sent2_idf)
        #----------------max min normalization-----------------------
        # sent1_max, sent2_max = np.transpose(sent1_idf.max(-1),(-1,1)),\
        #                        np.transpose(sent2_idf.max(-1),(-1,1))
        # sent1_idf, sent1_idf = np.exp(sent1_idf - sent1_max),np.exp(sent2_idf -sent2_max)
        # sent1_sum, sent2_sum = np.transpose(sent1_idf.sum(-1),(-1,1)),\
        #                        np.transpose(sent2_idf.sum(-1),(-1,1))
        # sent1_idf, sent2_idf = sent1_idf/sent1_sum, sent2_idf/sent2_sum
        #----------------std mean normalization-----------------------
        sent1_mean, sent2_mean = np.reshape(sent1_idf.mean(-1),(-1,1)),\
                                 np.reshape(sent2_idf.mean(-1),(-1,1))
        sent1_std, sent2_std = np.reshape(sent1_idf.std(-1),(-1,1)),\
                               np.reshape(sent2_idf.std(-1),(-1,1))
        sent1_idf, sent2_idf = (sent1_idf-sent1_mean)/(3*sent1_std),\
                               (sent2_idf-sent2_mean)/(3*sent2_std)
        sent1_idf,sent2_idf = np.exp(np.where(np.abs(sent1_idf)<1.0,sent1_idf,sent1_idf/np.abs(sent1_idf))),\
                              np.exp(np.where(np.abs(sent2_idf)<1.0,sent2_idf,sent2_idf/np.abs(sent2_idf)))
        sent1_sum, sent2_sum = np.reshape(np.sum(sent1_idf,1),(-1,1)),np.reshape(np.sum(sent2_idf,1),(-1,1))
        sent1_idf, sent2_idf = sent1_idf.astype(np.float32)/sent1_sum,\
                               sent2_idf.astype(np.float32)/sent2_sum
        return sent1_idf,sent2_idf

    def get_feed_dict(self, batch, idf_map, mode = 'train'):
        x, y = zip(*batch)
        sent1_lens,sent2_lens = self.get_valid_lengths(batch)
        sent1_idf, sent2_idf = self.get_idf(x,idf_map)
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
                     self.sent1_idf: sent1_idf,
                     self.sent2_idf: sent2_idf,
                     self.X1_len: sent1_lens,
                     self.X2_len: sent2_lens,
                     self.label: y,
                     self.is_train: True if mode == 'train' else False,
                     self.dropout: cfg.dropout if mode == 'train' else 1.0,
                     self.learning_rate: self.learning_rate_value}
        return feed_dict

    def update_learning_rate(self, global_step):
        if self.learning_rate_value < 5e-6:
            return
        self.learning_rate_value *= cfg.lr_decay



