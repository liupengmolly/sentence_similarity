#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import gc

import numpy as np
from sklearn.externals import joblib
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from data_test.ant.embedding import Embedding

from data_test.ant.util import disan_predict, bimpm_predict

from lib.model.configs import cfg
from lib.model.disan.model_disan import ModelDiSAN
from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph
from data_test.ant import DATA_PATH


from data_test.ant.util import get_model_list

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

if cfg.test_data is None:
    print("test_data is empty.")
    exit()

emb = Embedding(cfg)
train_data = pd.read_csv(cfg.train_data, sep='\t')
x_train, y_train = emb.generate_sentence_token_ind(train_data)

test_data = pd.read_csv(cfg.test_data, sep='\t')
x_test, y_test = emb.generate_sentence_token_ind(test_data)

disan_model_path = os.path.join(root_path, 'model/disan_models')
bimpm_model_path = os.path.join(root_path, 'model/bimpm_models')
bimpm_pinyin_model_path = os.path.join(root_path, 'model/bimpm_pinyin_models')
disan_pingying_model_path = os.path.join(root_path, 'model/disan_pinyin_models')

disan_models = get_model_list(disan_model_path)[1:2]
bimpm_models = get_model_list(bimpm_model_path)[1:3]
bimpm_pinyin_models = get_model_list(bimpm_pinyin_model_path)
disan_pingying_models = get_model_list(disan_pingying_model_path)

models_count = len(disan_models) + len(bimpm_models)\
               + len(bimpm_pinyin_models) + len(disan_pingying_models)

# print checkpoint_file
all_train_scores = []
all_test_scores = []
cfg.dropout = 1
cfg.dropout_rate = 0.0

graph = tf.Graph()
with graph.as_default():
    if len(disan_models) > 0:
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default(), tf.device("/gpu:%s" % GPU):
            with tf.variable_scope("ant") as scope:
                disan_model = ModelDiSAN(emb.id_vector_map, scope.name)

            for model in disan_models:

                train_scores = disan_predict(sess, x_train, disan_model, cfg, model)
                train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
                all_train_scores.append(train_scores)

                test_scores = disan_predict(sess, x_test, disan_model, cfg)
                test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
                all_test_scores.append(test_scores)

tf.reset_default_graph()

with tf.Graph().as_default():
    if len(bimpm_models) > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                    allow_growth=True)
        graph_config = tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True)
        init_scale = 0.01
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        sess = tf.Session(config=graph_config)
        with sess.as_default():
            with tf.variable_scope("sentence_similarity", reuse=False,
                                   initializer=initializer) as scope:
                bimpm_model = SentenceMatchModelGraph(
                    2, emb.id_vector_map, is_training=False, options=cfg)

            for model in bimpm_models:
                train_scores = bimpm_predict(sess, x_train, bimpm_model, cfg, model)[0]
                train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
                all_train_scores.append(train_scores)

                test_scores = bimpm_predict(sess, x_test, bimpm_model, cfg)[0]
                test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
                all_test_scores.append(test_scores)
tf.reset_default_graph()

cfg.use_pinyin = True
cfg.feature_type = 'char'
cfg.max_sentence_length = 70
cfg.word2vec_file = os.path.join(DATA_PATH, 'atec_w2v_with_pinyin_60d.json')
emb = Embedding(cfg)
train_data = pd.read_csv(cfg.train_data, sep='\t')
x_train, y_train = emb.generate_sentence_token_ind(train_data)

test_data = pd.read_csv(cfg.test_data, sep='\t')
x_test, y_test = emb.generate_sentence_token_ind(test_data)
graph = tf.Graph()

with tf.Graph().as_default():
    if len(bimpm_pinyin_models) > 0:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                    allow_growth=True)
        graph_config = tf.ConfigProto(
            gpu_options=gpu_options, allow_soft_placement=True)
        init_scale = 0.01
        initializer = tf.random_uniform_initializer(-init_scale, init_scale)
        sess = tf.Session(config=graph_config)
        with sess.as_default():
            with tf.variable_scope("sentence_similarity", reuse=False,
                                   initializer=initializer) as scope:
                bimpm_model = SentenceMatchModelGraph(
                    2, emb.id_vector_map, is_training=False, options=cfg)

            for model in bimpm_pinyin_models:
                train_scores = bimpm_predict(sess, x_train, bimpm_model, cfg, model)[0]
                train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
                all_train_scores.append(train_scores)

                test_scores = bimpm_predict(sess, x_test, bimpm_model, cfg)[0]
                test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
                all_test_scores.append(test_scores)


cfg.max_sentence_length = 70
cfg.use_pinyin = True
cfg.feature_type = 'char'
cfg.batch_size = 50
cfg.word2vec_file = os.path.join(DATA_PATH, 'atec_w2v_with_pinyin.json')
emb = Embedding(cfg)
train_data = pd.read_csv(cfg.train_data, sep='\t')
x_train, y_train = emb.generate_sentence_token_ind(train_data)

test_data = pd.read_csv(cfg.test_data, sep='\t')
x_test, y_test = emb.generate_sentence_token_ind(test_data)
graph = tf.Graph()
with graph.as_default():
    if len(disan_pingying_models) > 0:
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default(), tf.device("/gpu:%s" % GPU):
            with tf.variable_scope("ant") as scope:
                disan_model = ModelDiSAN(emb.id_vector_map, scope.name)

            for model in disan_pingying_models:

                train_scores = disan_predict(sess, x_train, disan_model, cfg, model)
                train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
                all_train_scores.append(train_scores)

                test_scores = disan_predict(sess, x_test, disan_model, cfg)
                test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
                all_test_scores.append(test_scores)
#

all_train_scores = np.transpose(all_train_scores, axes=[1, 0, 2])
all_train_scores = np.reshape(all_train_scores, [len(all_train_scores), models_count * 2])

all_test_scores = np.transpose(all_test_scores, axes=[1, 0, 2])
all_test_scores = np.reshape(all_test_scores, [len(all_test_scores), models_count * 2])

print "开始xgboost模型训练..."

model = XGBClassifier()
max_depth = [3, ]
n_estimators = [80, 110, 140, 170]
learning_rate = [0.1, ]
scale_pos_weight = [1, ]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,
                  max_depth=max_depth, scale_pos_weight=scale_pos_weight)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="f1", n_jobs=1, cv=kfold)
grid_result = grid_search.fit(all_train_scores, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = grid_search.best_estimator_

model_path = os.path.join(root_path, 'model/xgb.model')
joblib.dump(model, model_path)

model_restored = joblib.load(model_path)
y_predict = model_restored.predict(all_test_scores)
labels = [0, 1]
target_names = ["no", "yes"]

print '-------------------测试集----------------------'
print(classification_report(y_pred=y_predict,
                            y_true=y_test,
                            target_names=target_names,
                            labels=labels))
print('\n\n')
y_predict = model_restored.predict(all_train_scores)
print '-------------------训练集----------------------'
print(classification_report(y_pred=y_predict,
                            y_true=y_train,
                            target_names=target_names,
                            labels=labels))
