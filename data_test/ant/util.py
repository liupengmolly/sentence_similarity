#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
import numpy as np
from common.data_helper import DataHelper


def disan_predict(sess, x_data, disan_model, cfg, checkpoint_file=None):

    if checkpoint_file:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x1 = disan_model.sent1_token
    input_x2 = disan_model.sent2_token
    is_train = disan_model.is_train
    scores = disan_model.logits

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


def get_valid_lengths(batches):
    x1, x2 = zip(*batches)
    x1_lens, x2_lens = [], []
    for i in range(len(x1)):
        x1_len = sum(np.array(x1[i]) != 1)
        x2_len = sum(np.array(x2[i]) != 1)
        x1_lens.append(x1_len)
        x2_lens.append(x2_len)
    return np.array(x1_lens), np.array(x2_lens)


def bimpm_predict(sess, x_data, bimpm_model, cfg, checkpoint_file=None):
    if checkpoint_file:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)
    global_step = sess.run(bimpm_model.global_step)
    print(global_step)
    all_predictions = []
    model_scores = []
    for batch in DataHelper.batch_iter(x_data, cfg.batch_size, 1,
                                       shuffle=False):
        q1_lens, q2_lens = get_valid_lengths(batch)
        q1, q2 = zip(*batch)
        y, q1, q2 = np.array([1] * len(x_data)), np.array(q1), np.array(q2)
        feed_dict = bimpm_model.create_feed_dict(q1_lens, q2_lens, q1, q2, y)
        probs, predictions = sess.run(
            [bimpm_model.prob, bimpm_model.predictions],
            feed_dict=feed_dict)
        all_predictions.extend(predictions)
        model_scores.extend(probs)

    return model_scores, all_predictions

# def bimpm_predict(sess, x_data, bimpm_model, cfg, checkpoint_file=None):
#     if checkpoint_file:
#         saver = tf.train.import_meta_graph('ant_bimpm_runs/bimpm_enhanced_res_allmatch_bn_v2/model-7000.meta')
#         saver.restore(sess, checkpoint_file)
#         sess.run(tf.global_variables_initializer())
#         graph = tf.get_default_graph()
#     #for name in graph.get_operations():
#     #    print(name.name)
#     global_step = tf.train.get_global_step(graph=graph)
#     print(sess.run(global_step))
#     all_predictions = []
#     model_scores = []
#     for batch in DataHelper.batch_iter(x_data, cfg.batch_size, 1,
#                                        shuffle=False):
#         q1_lens, q2_lens = get_valid_lengths(batch)
#         q1, q2 = zip(*batch)
#         y, q1, q2 = np.array([1] * len(x_data)), np.array(q1), np.array(q2)
#         q_len = graph.get_tensor_by_name('sentence_similarity/ql:0')
#         p_len = graph.get_tensor_by_name('sentence_similarity/pl:0')
#         q = graph.get_tensor_by_name('sentence_similarity/q:0')
#         p = graph.get_tensor_by_name("sentence_similarity/p:0")
#         truth = graph.get_tensor_by_name('sentence_similarity/truth:0')
#         feed_dict = {
#             q_len: q1_lens,
#             p_len: q2_lens,
#             q: q1,
#             p: q2,
#             truth: y
#         }
#
#         probs, predictions = sess.run([graph.get_tensor_by_name('sentence_similarity/prob:0'),graph.get_tensor_by_name('sentence_similarity/predictions:0')],feed_dict=feed_dict)
#         all_predictions.extend(predictions)
#         model_scores.extend(probs)
#
#     return model_scores, all_predictions


# def disan_predict_(param):
#     model_path = param['model_path']
#     sess = param['sess']
#     x_data = param['x_data']
#     disan_model = param['disan_model']
#     scores = disan_predict(model_path, sess, x_data, cfg, disan_model)
#     scores = [(float(x[0]), float(x[1])) for x in scores]
#     return scores


def get_model_list(model_directory):
    if os.path.isdir(model_directory):
        file_list = os.listdir(model_directory)
        file_list = [model_name.split(".")[0] for model_name in file_list
                     if model_name.find(".index") > 0]
        model_list = [os.path.join(model_directory, model_name)
                      for model_name in file_list]
    else:
        raise Exception("directory not exists...")

    model_list = sorted(model_list, key=lambda x: int(x.split("-")[1]))
    return model_list
