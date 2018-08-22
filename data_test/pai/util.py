#! /usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
from common.data_helper import DataHelper
from data_test.pai.embedding import Embedding
import os
from lib.model.fast_disan.model_fast_disan import ModelFastDiSAN
from lib.model.disan.model_disan import ModelDiSAN
from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph
import pickle
import time
import xgboost as xgb
from sklearn.externals import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC,SVR
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss


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

def fast_disan_predict(sess, x_data, fast_disan_model, cfg, checkpoint_file=None):

    if checkpoint_file:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x1 = fast_disan_model.sent1_token
    input_x2 = fast_disan_model.sent2_token
    is_train = fast_disan_model.is_train
    scores = fast_disan_model.logits

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

def disan_for_ensemble(root_path, model_name,cfg):
    disan_model_path = os.path.join(root_path,'pai_disan_runs/{}'.format(model_name))
    model = tf.train.latest_checkpoint(disan_model_path)
    emb = Embedding(cfg)
    train_data = pd.read_csv(cfg.train_data)
    x_train, y_train = emb.generate_sentence_token_ind(train_data)
    valid_data = pd.read_csv(cfg.validate_data)
    x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
    test_data = pd.read_csv(cfg.test_data)
    x_test, y_test = emb.generate_sentence_token_ind(test_data)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default(), tf.device("/gpu:%s" % cfg.gpu):
            with tf.variable_scope("ant") as scope:
                disan_model = ModelDiSAN(emb.id_vector_map, scope.name)
            train_scores = disan_predict(sess, x_train, disan_model, cfg, model)
            train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
            valid_scores = disan_predict(sess, x_valid, disan_model, cfg)
            valid_scores = [(float(x[0]), float(x[1])) for x in valid_scores]
            test_scores = disan_predict(sess, x_test, disan_model, cfg)
            test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
    return train_scores, valid_scores, test_scores

def bimpm_for_ensemble(root_path, model_name,cfg):
    bimpm_model_path = os.path.join(root_path, 'pai_bimpm_runs/{}'.format(model_name))
    model = tf.train.latest_checkpoint(bimpm_model_path)
    emb = Embedding(cfg)
    train_data = pd.read_csv(cfg.train_data)
    x_train, y_train = emb.generate_sentence_token_ind(train_data)
    valid_data = pd.read_csv(cfg.validate_data)
    x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
    test_data=pd.read_csv(cfg.test_data)
    x_test, y_test = emb.generate_sentence_token_ind(test_data)
    with tf.Graph().as_default():
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
            train_scores = bimpm_predict(sess, x_train, bimpm_model, cfg, model)[0]
            train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
            valid_scores = bimpm_predict(sess, x_valid, bimpm_model, cfg)[0]
            valid_scores = [(float(x[0]), float(x[1])) for x in valid_scores]
            test_scores = bimpm_predict(sess, x_test, bimpm_model, cfg)[0]
            test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
    return  train_scores, valid_scores, test_scores

def fast_disan_for_ensemble(root_path,model_name,cfg):
    fast_disan_model_path = os.path.join(root_path, 'pai_fast_disan_runs/{}'.format(model_name))
    model = tf.train.latest_checkpoint(fast_disan_model_path)
    emb = Embedding(cfg)
    train_data = pd.read_csv(cfg.train_data)
    x_train, y_train = emb.generate_sentence_token_ind(train_data)
    valid_data = pd.read_csv(cfg.validate_data)
    x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
    test_data = pd.read_csv(cfg.test_data)
    x_test, y_test = emb.generate_sentence_token_ind(test_data)

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default(), tf.device("/gpu:%s" % cfg.gpu):
            with tf.variable_scope("ant") as scope:
                fast_disan_model = ModelFastDiSAN(emb.id_vector_map, scope.name)
            train_scores = fast_disan_predict(sess, x_train, fast_disan_model, cfg, model)
            train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
            valid_scores = fast_disan_predict(sess, x_valid, fast_disan_model, cfg)
            valid_scores = [(float(x[0]), float(x[1])) for x in valid_scores]
            test_scores = fast_disan_predict(sess, x_test, fast_disan_model, cfg)
            test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
    return train_scores, valid_scores, test_scores

def xgb_predict(x_train,y_train,x_valid,x_test,dump_model=False, dump_path=None):
    print('开始xgboost训练')
    start = time.clock()
    model = XGBClassifier(**{'tree_method': 'gpu_hist'})
    max_depth = [3, ]
    n_estimators = [200]
    learning_rate = [0.1, ]
    scale_pos_weight = [1.0]
    subsample = [0.9]
    min_child_weight = [2.5]
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,
                      min_child_weight=min_child_weight,
                      max_depth=max_depth, scale_pos_weight=scale_pos_weight, subsample=subsample)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    end = time.clock()
    print('use time :{}'.format(end - start))
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    model = grid_search.best_estimator_
    if dump_model:
        pickle.dump(model,open(dump_path,'w'))
    y_train_predict = model.predict(x_train)
    y_valid_predict = model.predict(x_valid)
    y_train_predict_probs = model.predict_proba(x_train)
    y_valid_predict_probs = model.predict_proba(x_valid)
    y_test_predict_probs = model.predict_proba(x_test)
    return y_train_predict, y_valid_predict, y_train_predict_probs, \
           y_valid_predict_probs, y_test_predict_probs

def xgb_feature(x_train,y_train,x_valid,x_test,num_rounds=1000):
    model = XGBClassifier(**{'tree_method': 'gpu_hist'})
    max_depth = [3, ]
    n_estimators = [200]
    learning_rate = [0.1, ]
    scale_pos_weight = [1.0]
    subsample = [0.9]
    min_child_weight = [2.5]
    param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,
                      min_child_weight=min_child_weight,
                      max_depth=max_depth, scale_pos_weight=scale_pos_weight, subsample=subsample)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    model = grid_search.best_estimator_
    joblib.dump(model, 'feature_model/xgb_model')
    train_predict_probs=model.predict(x_valid)
    valid_predict_probs=model.predict(x_test)
    return train_predict_probs,valid_predict_probs

# def lgb_feature(x_train,y_train,x_valid,x_test,num_rounds=10):
#     param = {'num_leaves': 127,
#              'num_trees': 100,
#              'learning_rate':0.01,
#              'num_iterations':1000,
#              'objective': 'binary',
#              'metric': 'binary_logloss'}
#     dtrain = lgb.Dataset(x_train, label=list(y_train))
#     dvalid = lgb.Dataset(x_valid)
#     dtest = lgb.Dataset(x_test)
#     model=lgb.train(param,dtrain,num_rounds)
#     joblib.dump(model, 'feature_model/lgb_model')
#     train_predict_probs=model.predict(x_train)
#     valid_predict_probs=model.predict(x_valid)
#     return train_predict_probs,valid_predict_probs

def lgb_feature(x_train,y_train,x_valid,x_test,num_rounds=10):
    model=LGBMClassifier()
    num_leaves = [127,191,255]
    num_trees = [100,150]
    max_bin=[255,127,512]
    param_grid = dict(num_leaves=num_leaves,num_trees=num_trees,max_bin=max_bin)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    model = grid_search.best_estimator_
    joblib.dump(model, 'feature_model/lgb_model')
    train_predict_probs=model.predict(x_train)
    valid_predict_probs=model.predict(x_valid)
    return train_predict_probs,valid_predict_probs

def lr_feature(x_train,y_train,x_valid,x_test):
    cls=LogisticRegression(penalty='l1')
    C=[1,0.3,0.1,0.05,0.01]
    param_grid = dict(C=C)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(cls, param_grid, scoring="neg_log_loss", cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    model = grid_search.best_estimator_
    joblib.dump(model, 'feature_model/lr_model')
    train_predict_probs=cls.predict_proba(x_train)
    valid_predict_probs = cls.predict_proba(x_valid)
    return train_predict_probs,valid_predict_probs
#
# def lr_feature(x_train,y_train,x_valid,x_test):
#     cls=LogisticRegression(penalty='l1',C=0.05)
#     cls.fit(x_train,y_train)
#     joblib.dump(cls, 'feature_model/lr_model')
#     train_predict_probs=cls.predict_proba(x_train)
#     valid_predict_probs = cls.predict_proba(x_valid)
#     return train_predict_probs,valid_predict_probs

def svc_feature(x_train,y_train,x_valid,x_test):
    cls=SVC(probability=True,C=0.1)
    cls.fit(x_train,y_train)
    train_predict_probs=cls.predict(x_train)
    valid_predict_probs = cls.predict(x_valid)
    return train_predict_probs,valid_predict_probs

def rf_feature(x_train,y_train,x_valid,x_test):
    cls=RandomForestClassifier(n_estimators=1000)
    max_depth=[5,7,9,11,13,15,17,19,21,23]
    param_grid = dict(max_depth=max_depth)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    grid_search = GridSearchCV(cls, param_grid, scoring="neg_log_loss", cv=kfold)
    grid_result = grid_search.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    model = grid_search.best_estimator_
    joblib.dump(model, 'feature_model/rf_model')
    train_predict_probs=cls.predict_proba(x_train)
    valid_predict_probs = cls.predict_proba(x_valid)
    return train_predict_probs,valid_predict_probs

# def rf_feature(x_train,y_train,x_valid,x_test):
#     cls=RandomForestClassifier(n_estimators=1000,max_depth=27)
#     cls.fit(x_train,y_train)
#     joblib.dump(cls, 'feature_model/rf_model')
#     train_predict_probs=cls.predict_proba(x_train)
#     valid_predict_probs = cls.predict_proba(x_valid)
#     return train_predict_probs,valid_predict_probs

def gp_feature(x_train,y_train,x_valid,x_test):
    gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
    gp.fit(x_train,y_train)
    train_predict_probs=gp.predict_proba(x_train)
    valid_predict_probs = gp.predict_proba(x_valid)
    return train_predict_probs,valid_predict_probs

def gpc_feature(x_train, y_train,x_valid, x_test):
    gpc = GaussianProcessClassifier()
    gpc.fit(x_train,y_train)
    train_predict_probs=gpc.predict_proba(x_train)
    valid_predict_probs = gpc.predict_proba(x_valid)
    return train_predict_probs,valid_predict_probs





