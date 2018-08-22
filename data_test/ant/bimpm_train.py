#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import f1_score

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from lib.model.configs import cfg
from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph
from data_test.ant.embedding import Embedding
from common.data_helper import DataHelper


os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

logging.basicConfig(filename="ant_bimpm.log" + cfg.log_name,
                    filemode="w",
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                    level=logging.INFO)

emb = Embedding(cfg)


def get_valid_lengths(batches):
    x, y = zip(*batches)
    x1, x2 = zip(*x)
    x1_lens, x2_lens = [], []
    for i in range(len(x1)):
        x1_len = sum(np.array(x1[i]) != 1)
        x2_len = sum(np.array(x2[i]) != 1)
        x1_lens.append(x1_len)
        x2_lens.append(x2_len)
    return np.array(x1_lens), np.array(x2_lens)


train_data = pd.read_csv(cfg.train_data, sep='\t')
x_train, y_train = emb.generate_sentence_token_ind(train_data)
train_data_emb = list(zip(x_train, y_train))
valid_data = pd.read_csv(cfg.validate_data, sep='\t')
x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
valid_data_emb = list(zip(x_valid, y_valid))

num_epoch = int(len(train_data) / cfg.batch_size) + 1

logging.info("starting graph: dropout: {},batch_size:{}".format(cfg.dropout_rate,cfg.batch_size))
with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-0.01, 0.01)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope("sentence_similarity", reuse=None, initializer=initializer) as scope:
        model = SentenceMatchModelGraph(2, emb.id_vector_map, is_training=True,
                                        options=cfg,
                                        global_step=global_step)
    with tf.variable_scope("sentence_similarity", reuse=True, initializer=initializer):
        valid_model = SentenceMatchModelGraph(2, emb.id_vector_map,
                                              is_training=False, options=cfg)

    initializer = tf.global_variables_initializer()

    sess = tf.Session(config=graph_config)
    sess.run(initializer)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.curdir, "ant_bimpm_runs", cfg.model_save_path))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    batches = DataHelper.batch_iter(
        train_data_emb, cfg.batch_size, cfg.max_epoch, shuffle=True)
    best_f1 = 0
    current_step = 0
    step = 0
    total_loss = 0
    total_correct = 0
    for batch in batches:
        q1_lens, q2_lens = get_valid_lengths(batch)
        x, y = zip(*batch)
        q1, q2 = zip(*x)
        y, q1, q2 = np.array(y), np.array(q1), np.array(q2)
        feed_dict = model.create_feed_dict(q1_lens, q2_lens, q1, q2, y)
        _, loss_value, cur_correct = sess.run([model.train_op, model.loss, model.eval_correct],
                                               feed_dict=feed_dict)
        total_loss += loss_value
        total_correct += cur_correct
        if step > 0 and step % 100 == 0:
            logging.info('step %d: loss = %.4f, accuracy=%.4f' %
                         (step, total_loss / 100.0, total_correct / float(len(batch)) / 100))
            total_loss = 0
            total_correct = 0
        # evaluation
        if step % cfg.num_steps == 0:
            correct = 0
            all_prediction = []
            for batch in DataHelper.batch_iter(valid_data_emb, cfg.batch_size,
                                               1, shuffle=False):
                q1_lens, q2_lens = get_valid_lengths(batch)
                x, y = zip(*batch)
                q1, q2 = zip(*x)
                y, q1, q2 = np.array(y), np.array(q1), np.array(q2)
                feed_dict = valid_model.create_feed_dict(q1_lens, q2_lens, q1, q2, y)
                cur_correct, probs, predictions = sess.run(
                    [valid_model.eval_correct, valid_model.prob, valid_model.predictions],
                    feed_dict=feed_dict)
                correct += cur_correct
                predictions = [int(x) for x in predictions]
                all_prediction += predictions
            f1 = f1_score(y_valid, all_prediction)

            logging.info("current step: %d" % step +
                         "validate data f1: %f" % f1)
            if f1 > best_f1 and step >= 2000:
                saver.save(sess, checkpoint_prefix, step)
                logging.info('saved model')
                best_f1 = f1
                current_step = step
        step += 1
