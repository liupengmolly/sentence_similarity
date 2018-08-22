#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from data_test.pai.embedding import Embedding
import logging
from common.data_helper import DataHelper
from lib.model.configs import cfg
from lib.model.fast_disan.model_fast_disan import ModelFastDiSAN
from data_test.pai.util import fast_disan_predict

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

emb = Embedding(cfg)

if cfg.test_data is None:
    logging.info("test_data is empty.")
    exit()

test_data = pd.read_csv(cfg.test_data)
x_test, y_test = emb.generate_sentence_token_ind(test_data)
x1_test, x2_test = zip(*x_test)

ckpt=tf.train.latest_checkpoint(cfg.model_directory)

def fast_disan_predict(sess, x_data, fast_disan_model, cfg, checkpoint_file=None):

    if checkpoint_file:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x1 = fast_disan_model.sent1_token
    input_x2 = fast_disan_model.sent2_token
    is_train = fast_disan_model.is_train
    scores = fast_disan_model.probs

    batches = DataHelper.batch_iter(list(x_data),
                                    2 * cfg.batch_size,
                                    1,
                                    shuffle=False)
    model_scores = []
    for db in batches:
        x1_dev_b, x2_dev_b = zip(*db)
        batch_score = sess.run(
            scores,
            feed_dict={input_x1: x1_dev_b,
                       input_x2: x2_dev_b,
                       is_train: False
                       })
        model_scores.extend(batch_score)
    return model_scores

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    init_scale = 0.01
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    sess = tf.Session(config=graph_config)
    with sess.as_default():
        with tf.variable_scope("ant") as scope:
            model = ModelFastDiSAN(emb.id_vector_map, scope.name)
    model_scores = fast_disan_predict(sess, x_test, model, cfg, ckpt)
    result = pd.DataFrame(model_scores, columns=['0', '1'])
    submit = result[['1']].rename(columns={'1': 'y_pre'})
    submit.to_csv('data_test/pai/data/result/submit/{}'.format('fastdisan_exp_1.7_word_0.8.csv'), index=False)
