#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from data_test.ant.embedding import Embedding
import logging
from lib.model.configs import cfg
from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph
from data_test.ant.util import bimpm_predict, get_model_list

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

emb = Embedding(cfg)

if cfg.test_data is None:
    logging.info("test_data is empty.")
    exit

test_data = pd.read_csv(cfg.test_data, sep='\t')
x_test, y_test = emb.generate_sentence_token_ind(test_data)
x1_test, x2_test = zip(*x_test)
test_data_emb = list(zip(x_test, y_test))

model_list = get_model_list(cfg.model_directory)
from datetime import datetime
print 'start time ... ', datetime.now()
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    init_scale = 0.01
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    sess = tf.Session(config=graph_config)
    with sess.as_default():
        with tf.variable_scope("sentence_similarity",
                               initializer=initializer) as scope:
            model = SentenceMatchModelGraph(2, emb.id_vector_map,
                                            is_training=False,
                                            options=cfg)
    for checkpoint_file in model_list:

        all_predictions = bimpm_predict(sess, x_test, model, cfg, checkpoint_file)[1]
        print 'end time ... ', datetime.now()
        labels = [0, 1]
        target_names = ['no', 'yes']
        print(" %s" % checkpoint_file)
        print(classification_report(y_pred=all_predictions,
                                    y_true=y_test,
                                    target_names=target_names,
                                    labels=labels))
        print("============")
