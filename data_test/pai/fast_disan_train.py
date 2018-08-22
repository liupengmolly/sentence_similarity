#! /usr/bin/env python
# -*- coding: utf-8 -*-
import time
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
from lib.model.fast_disan.model_fast_disan import ModelFastDiSAN
from data_test.pai.embedding import Embedding
from common.data_helper import DataHelper
from sklearn.metrics import log_loss

def softmax(a):
    a=np.exp(a)
    a_sum=np.sum(a,axis=1)
    for i,x in enumerate(a):
        if a_sum[i]==0.0:
            a_sum[i]=1e-10
        a[i]=a[i]/a_sum[i]
    return a

emb = Embedding(cfg)

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU


logging.basicConfig(filename="log_pai_fast_disan_" + cfg.log_name,
                    filemode="w",
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                    level=logging.INFO)

train_data = pd.read_csv(cfg.train_data)
x_train, y_train = emb.generate_sentence_token_ind(train_data)
train_data_emb = list(zip(x_train, y_train))

valid_data = pd.read_csv(cfg.validate_data)
x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
valid_data_emb = list(zip(x_valid, y_valid))

num_epoch = int(len(train_data) / cfg.batch_size) + 1

logging.info("starting graph ")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)

    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)
    with sess.as_default():
        with tf.variable_scope("ant") as scope:
            model = ModelFastDiSAN(emb.id_vector_map, scope.name)
    sess.run(tf.global_variables_initializer())

    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.curdir, "pai_fast_disan_runs", cfg.model_save_path))

    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    last_logloss = 100.0
    current_step = 0

    if cfg.model_directory:
        saver = tf.train.Saver()
        latest_model =tf.train.latest_checkpoint(cfg.model_directory)
        saver.restore(sess,latest_model)
    else:
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

    batches = DataHelper.batch_iter(
        train_data_emb, cfg.batch_size, cfg.max_epoch, shuffle=True)
    step = 0
    train_loss_array = []
    train_accuracy_array = []
    train_y = []
    train_predict_array = []
    train_probs_array = []
    for batch in batches:
        train_loss, train_acc, train_predict, train_logits, train_probs = model.train_step(sess, batch, logging)
        # for i,l in enumerate(train_logits):
        #     if np.nan in l:
        #         logging.info('logit: {}'.format(l))
        #         logging.info('predout: {}'.format(preds[i]))
        #         time.sleep(1000000)

        train_loss_array.append(train_loss)
        train_accuracy_array.extend(train_acc)
        train_predict_array.extend(train_predict)
        train_probs_array.extend(train_probs)
        _, y = zip(*batch)
        train_y.extend(y)
        if step > 0 and step % 100 == 0:
            logging.info(
                'current step: {}, train log_loss: {}, train_accuracy: {}, train f1: {}'.format(
                    step,
                    log_loss(train_y,train_probs_array),
                    np.mean(train_accuracy_array),
                    f1_score(train_y, train_predict_array)))
            train_probs_array = []
            train_loss_array = []
            train_accuracy_array = []
            train_y = []
            train_predict_array = []
        if step > 0 and step % cfg.num_steps == 0:
            validate_batches = DataHelper.batch_iter(
                valid_data_emb, cfg.batch_size, shuffle=False)
            accuracy_array = []
            loss_array = []
            # predict_pos = 0
            valid_predict = []
            probs_array=[]
            x,y=zip(*valid_data_emb)
            for validate_batch in validate_batches:
                accuracy, loss, predict, logits = model.validate_step(
                    sess, validate_batch)
                probs=softmax(logits)
                accuracy_array.extend(accuracy)
                loss_array.append(loss)
                valid_predict.extend(predict)
                probs_array.extend(probs)
            accuracy = np.mean(accuracy_array)
            loss = np.mean(loss_array)
            logloss=log_loss(y,probs_array)

            model.update_learning_rate(loss, model.global_step, cfg.lr_decay)

            f1 = f1_score(y_valid, valid_predict)

            logging.info("current step: %d" % step +
                         "validate data f1: %f" % f1+
                         "validate data log_loss: %f" % logloss)
            # logging.info("current step: %d" % step +
            #              "validate data predict pos: %d" % predict_pos)
            if logloss < last_logloss:
                saver.save(sess, checkpoint_prefix, step)
                last_logloss=logloss
                current_step = step
        if step - current_step > 30*num_epoch and current_step >= 10000:
            break
        step += 1


if __name__ == '__main__':
    pass
