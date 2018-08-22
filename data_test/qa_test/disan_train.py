#! /usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import sys
import time

import pandas as pd
import tensorflow as tf

root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

from lib.model.disan.model_disan import ModelDiSAN

GPU = '6'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

from data_test.qa_test.embedding import *
logging.basicConfig(filename="quora_disan.log",
                    filemode="w",
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                    level=logging.INFO)

train_data = pd.read_csv(cfg.train_data, sep='$')
x_train, y_train = generate_sentence_token_ind(train_data)
train_data_emb = list(zip(x_train, y_train))
valid_data = pd.read_csv(cfg.validate_data, sep='$')
x_valid, y_valid = generate_sentence_token_ind(valid_data)
valid_data_emb = list(zip(x_valid, y_valid))

num_epoch = int(len(train_data) / cfg.batch_size) + 1

logging.info("starting graph ")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)

    sess = tf.Session(config=graph_config)
    with sess.as_default(), tf.device("/gpu:%s" % GPU):
        with tf.variable_scope("quaro") as scope:
            model = ModelDiSAN(id_vector_map, scope.name)
    sess.run(tf.global_variables_initializer())
    timestamp = str(int(time.time()))

    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.curdir, "quora_disan_runs", timestamp))

    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    last_accuracy = 0.0
    current_step = 0
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    batches = DataHelper.batch_iter(
        train_data_emb, cfg.batch_size, cfg.max_epoch, shuffle=True)
    step = 0
    for batch in batches:
        model.train_step(sess, batch, logging)

        if step > 0 and step % cfg.num_steps == 0:
            validate_batches = DataHelper.batch_iter(
                valid_data_emb, cfg.batch_size, shuffle=False)
            accuracy_array = []
            loss_array = []
            for validate_batch in validate_batches:
                accuracy, loss, predict, logits \
                    = model.validate_step(sess, validate_batch)
                accuracy_array.extend(accuracy)
                loss_array.append(loss)
            accuracy = np.mean(accuracy_array)
            loss = np.mean(loss_array)
            model.update_learning_rate(loss, model.global_step, cfg.lr_decay)
            logging.info("current step: %d" % step +
                         "validate data accuracy: %f" % accuracy)
            if step >= 10000 and accuracy > last_accuracy:
                saver.save(sess, checkpoint_prefix, step)
                last_accuracy = accuracy
                current_step = step
        if step - current_step > 30000 and current_step >= 10000:
            break
        step += 1

if __name__ == '__main__':
    pass
