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
from sklearn.metrics import f1_score
from common.data_helper import DataHelper
from lib.model.configs import cfg
from lib.model.esim.model_esim_res import ModelESIMRES
from data_test.ant.util import get_model_list

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

emb = Embedding(cfg)

if cfg.test_data is None:
    print("test_data is empty.")
    exit()

test_data = pd.read_csv(cfg.test_data, sep = '\t')
x_test, y_test = emb.generate_sentence_token_ind(test_data)
x1_test, x2_test = zip(*x_test)

model_list = get_model_list(cfg.model_directory)

# print checkpoint_file
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)

    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    with tf.variable_scope("ant",reuse=tf.AUTO_REUSE) as scope:
        esim_model = ModelESIMRES(emb.id_vector_map, scope.name)

    sess = tf.Session(config=graph_config)
    report_list = []
    for model in model_list:
        from datetime import datetime

        checkpoint_file = model

        # Load the saved meta graph and restore variables
        # saver = tf.train.import_meta_graph(
        #     "{}.meta".format(checkpoint_file))
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x1 = esim_model.sent1_token
        input_x2 = esim_model.sent2_token
        input_y = esim_model.label
        is_train = esim_model.is_train
        predictions = esim_model.predict
        # scores = esim_model.logits

        # input_x1 = graph.get_operation_by_name("ant/sent1_token").outputs[0]
        # input_x2 = graph.get_operation_by_name("ant/sent2_token").outputs[0]
        # input_y = graph.get_operation_by_name("ant/gold_label").outputs[0]
        # is_train = graph.get_operation_by_name('ant/is_train').outputs[0]

        # Tensors we want to evaluate
        # predictions = graph.get_operation_by_name("ant/predict").outputs[0]
        # scores = graph.get_operation_by_name('ant/predict_score').outputs[0]
        print 'start now..', datetime.now()
        batches = DataHelper.batch_iter(list(zip(x1_test, x2_test)),
                                        2 * cfg.batch_size,
                                        1,
                                        shuffle=False)
        # Collect the predictions here
        all_predictions = []
        all_scores = []
        for db in batches:
            x1_dev_b, x2_dev_b = zip(*db)

            batch_predictions = sess.run(
                predictions,
                feed_dict={input_x1: x1_dev_b,
                           input_x2: x2_dev_b,
                           is_train: False
                           })

            all_predictions.extend(batch_predictions)
        print 'end now..', datetime.now()
        labels = [0, 1]
        target_names = ["no", "yes"]
        print(" %s" % model)
        print(classification_report(y_pred=all_predictions,
                                    y_true=y_test,
                                    target_names=target_names,
                                    labels=labels))
        print("============")
        print('f1: {}'.format(f1_score(y_test,all_predictions)))
        print('')
