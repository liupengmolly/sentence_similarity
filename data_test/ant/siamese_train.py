#! /usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

GPU = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

import datetime
import gc

import numpy as np
import time
from random import random
import pandas as pd
import tensorflow as tf
import logging

from lib.model.configs import cfg
from data_test.ant.embedding import Embedding

from common.data_helper import DataHelper
from lib.model.siamese.siamese_network import SiameseLSTM
from lib.model.siamese.siamese_network_semantic import SiameseLSTMw2v

emb = Embedding(cfg)

if cfg.train_data == None:
    print("Input Files List is empty. use --training_files argument.")
    exit()

max_document_length = 50


logging.basicConfig(filename="ant_siamese_pretrained.log",
                    filemode="w",
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                    level=logging.INFO)

trainableEmbeddings = True

train_data = pd.read_csv(cfg.train_data, sep='\t')
x_train, y_train = emb.generate_sentence_token_ind(train_data)
train_data_emb = list(zip(x_train, y_train))
valid_data = pd.read_csv(cfg.validate_data, sep='\t')
x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
valid_data_emb = list(zip(x_valid, y_valid))

sum_no_of_batches = int(len(y_train) / cfg.batch_size)

# Training
# ==================================================
logging.info("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False)
    sess = tf.Session(config=session_conf)
    logging.info("started session")
    with sess.as_default(), tf.device("/gpu:%s" % GPU):
        if not cfg.use_pre_trained:
            siameseModel = SiameseLSTM(
                sequence_length=max_document_length,
                vocab_size=len(emb.id_vector_map),
                embedding_size=cfg.word_embedding_length,
                hidden_units=cfg.hidden_units_num,
                l2_reg_lambda=cfg.wd,
                batch_size=cfg.batch_size
            )
        else:
            siameseModel = SiameseLSTMw2v(
                sequence_length=max_document_length,
                vocab_size=len(emb.id_vector_map),
                embedding_size=emb.word2vec.dim,
                hidden_units=cfg.hidden_units_num,
                l2_reg_lambda=cfg.wd,
                batch_size=cfg.batch_size,
                trainableEmbeddings=trainableEmbeddings
            )
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        logging.info("initialized siameseModel object")

    grads_and_vars = optimizer.compute_gradients(siameseModel.loss)

    # global_step 会自增 1
    tr_op_set = optimizer.apply_gradients(grads_and_vars,
                                          global_step=global_step)
    logging.info("defined training_ops")
    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    logging.info("Writing to {}\n".format(out_dir))

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    if cfg.use_pre_trained:
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25, 0.25,
                                  (len(emb.word_id_map), emb.word2vec.dim))
        # load any vectors from the word2vec
        logging.info("initializing initW with pre-trained word2vec embeddings")

        for i in range(len(emb.word_id_map)):
            initW[i] = np.asarray(emb.id_vector_map[i]).astype(np.float32)

        gc.collect()
        sess.run(siameseModel.W.assign(emb.id_vector_map))

    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random() > 0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 0.7,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 0.7,
            }
        _, step, loss, accuracy, dist, sim = sess.run(
            [tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy,
             siameseModel.distance, siameseModel.temp_sim],
            feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logging.info(
            "TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step,
                                                            loss,
                                                            accuracy))

    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random() > 0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        step, loss, accuracy, sim = sess.run(
            [global_step, siameseModel.loss, siameseModel.accuracy,
             siameseModel.temp_sim], feed_dict)
        return accuracy

    # Generate batches
    batches = DataHelper.batch_iter(train_data_emb,
                                    cfg.batch_size,
                                    cfg.max_epoch)

    step = 0
    max_validation_acc = 0.0
    max_step = 0

    for batch in batches:

        if len(batch) < 1:
            continue
        x_batch, y_batch = zip(*batch)
        x1_batch, x2_batch = zip(*x_batch)
        if len(y_batch) < 1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        acc_array = []
        if current_step % cfg.num_steps == 0:
            logging.info("\nEvaluation:")
            dev_batches = DataHelper.batch_iter(
                valid_data_emb, cfg.batch_size, shuffle=False)
            for db in dev_batches:
                if len(db) < 1:
                    continue
                x_dev_b, y_dev_b = zip(*db)
                x1_dev_b, x2_dev_b = zip(*x_dev_b)
                if len(y_dev_b) < 1:
                    continue
                acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
                acc_array.append(acc)
            time_str = datetime.datetime.now().isoformat()
            logging.info(
                    "validate {}: step {}, acc {:g}".format(
                        time_str, step, np.mean(acc_array)))

            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                max_step = current_step
                saver.save(
                    sess, checkpoint_prefix, global_step=current_step)
            if current_step - max_step >= 100000:
                exit()
        step += 1
