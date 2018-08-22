#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

from data_test.qa_test.embedding import *
GPU = '6'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

test_data = pd.read_csv(cfg.test_data, sep='$')
x_test, y_test = generate_sentence_token_ind(test_data)
x1_test, x2_test = zip(*x_test)

model_list = []

if os.path.isdir(cfg.model_directory):
    file_list = os.listdir(cfg.model_directory)
    file_list = [model_name.split(".")[0] for model_name in file_list
                 if model_name.find(".index") > 0]
    model_list = [os.path.join(cfg.model_directory, model_name)
                  for model_name in file_list]
else:
    raise Exception("directory not exists...")

model_list = sorted(model_list, key=lambda x: int(x.split("-")[1]))
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
            input_x1 = graph.get_operation_by_name(
                "quora/sent1_token").outputs[0]
            input_x2 = graph.get_operation_by_name(
                "quora/sent2_token").outputs[0]
            input_y = graph.get_operation_by_name(
                "quora/gold_label").outputs[0]
            is_train = graph.get_operation_by_name(
                'quora/is_train').outputs[0]


            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name(
                "quora/predict").outputs[0]

            batches = DataHelper.batch_iter(list(zip(x1_test, x2_test)),
                                            2 * cfg.batch_size,
                                            1,
                                            shuffle=False)
            # Collect the predictions here
            all_predictions = []
            for db in batches:
                x1_dev_b, x2_dev_b = zip(*db)
                batch_predictions = sess.run(
                    predictions,
                    feed_dict={input_x1: x1_dev_b,
                               input_x2: x2_dev_b,
                               is_train: False
                               })
                all_predictions = np.concatenate(
                    [all_predictions, batch_predictions])

            labels = [0, 1]
            target_names = ["no", "yes"]
            print(" %s" % model)
            print(classification_report(y_pred=all_predictions,
                                        y_true=y_test,
                                        target_names=target_names,
                                        labels=labels))
            print("============")
            print('')
