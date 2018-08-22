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

from common.data_helper import DataHelper
from lib.model.configs import cfg
from lib.model.fast_disan.model_fast_disan import ModelFastDiSAN
from data_test.ant.util import get_model_list

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

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
        with tf.variable_scope("ant") as scope:
            disan_model = ModelFastDiSAN(emb.id_vector_map, scope.name)

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
            input_x1 = disan_model.sent1_token
            input_x2 = disan_model.sent2_token
            input_y = disan_model.gold_label
            is_train = disan_model.is_train
            predictions = disan_model.predict
            # scores = disan_model.logits

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
            print('')