#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import pandas as pd
import tensorflow as tf

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph
from data_test.qa_test.embedding import *
os.environ["CUDA_VISIBLE_DEVICES"] =cfg.gpu

def get_valid_lengths(batches):
    x,y=zip(*batches)
    x1,x2=zip(*x)
    x1_lens,x2_lens=[],[]
    for i in range(len(x1)):
        x1_len=sum(np.array(x1[i])!=1)
        x2_len=sum(np.array(x2[i])!=1)
        x1_lens.append(x1_len)
        x2_lens.append(x2_len)
    return np.array(x1_lens),np.array(x2_lens)

test_data = pd.read_csv(cfg.test_data, sep='$')
x_test, y_test = generate_sentence_token_ind(test_data)
x1_test, x2_test = zip(*x_test)
test_data_emb=list(zip(x_test,y_test))

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)
    global_step = tf.train.get_or_create_global_step()
    with sess.as_default(), tf.device("/gpu:%s" % cfg.gpu):
        with tf.variable_scope("sentence_similarity") as scope:
            model = SentenceMatchModelGraph(2, id_vector_map,
                                            is_training=False,
                                            options=cfg,
                                            global_step=global_step)
    saver = tf.train.Saver()
    print('Getting Latest Checkpoint')
    latest_checkpoint = tf.train.latest_checkpoint(cfg.model_directory)
    saver.restore(sess, latest_checkpoint)
    print('successfully load {}'.format(latest_checkpoint))

    total = 0
    correct = 0
    for batch in DataHelper.batch_iter(test_data_emb, cfg.batch_size, 1,
                                       shuffle=False):
        total += cfg.batch_size
        q1_lens, q2_lens = get_valid_lengths(batch)
        x, y = zip(*batch)
        q1, q2 = zip(*x)
        y, q1, q2 = np.array(y), np.array(q1), np.array(q2)
        feed_dict = model.create_feed_dict(q1_lens, q2_lens, q1, q2, y)
        [cur_correct, probs, predictions] = sess.run(
            [model.eval_correct, model.prob, model.predictions],
            feed_dict=feed_dict)
        correct += cur_correct
    acc = correct / float(total) * 100
    print('Accuracy: {}\nDone'.format(acc))