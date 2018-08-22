#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

from data_test.ant.embedding import Embedding

GPU = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

from common.data_helper import DataHelper
from lib.model.configs import cfg
from data_test.ant.util import get_model_list
# Parameters
# ==================================================

emb = Embedding(cfg)

if cfg.test_data is None:
    print("test_data is empty.")
    exit()

test_data = pd.read_csv(cfg.test_data, sep='\t')
x_test, y_test = emb.generate_sentence_token_ind(test_data)
x1_test, x2_test = zip(*x_test)

model_list = get_model_list(cfg.model_directory)

# print checkpoint_file
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    with sess.as_default(), tf.device("/gpu:%s" % GPU):
        report_list = []
        for model in model_list:
            checkpoint_file = model
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph(
                "{}.meta".format(checkpoint_file))
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = \
                graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            predictions = \
            graph.get_operation_by_name("output/distance").outputs[0]

            accuracy = graph.get_operation_by_name(
                "accuracy/accuracy").outputs[0]

            sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

            batches = DataHelper.batch_iter(list(zip(x1_test, x2_test, y_test)),
                                            2 * 96,
                                            1,
                                            shuffle=False)
            # Collect the predictions here
            all_predictions = []
            all_d = []
            for db in batches:
                x1_dev_b, x2_dev_b, y_dev_b = zip(*db)
                batch_predictions, batch_acc, batch_sim = sess.run(
                    [predictions, accuracy, sim],
                    {input_x1: x1_dev_b, input_x2: x2_dev_b, input_y: y_dev_b,
                     dropout_keep_prob: 1.0})
                all_predictions = np.concatenate(
                    [all_predictions, batch_predictions])
                all_d = np.concatenate([all_d, batch_sim])
            correct_predictions = float(np.mean(all_d == y_test))
            labels = [0, 1]
            target_names = ["no", "yes"]
            print(" %s" % model)
            print(classification_report(y_pred=all_d,
                                        y_true=y_test,
                                        target_names=target_names,
                                        labels=labels))
            print("============")
            print('')
