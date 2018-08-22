#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import sys
import gc
import pickle
import numpy as np
from sklearn.externals import joblib
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from data_test.pai.util import *
from data_test.pai.embedding import Embedding
from data_test.pai.util import fast_disan_predict, disan_predict, bimpm_predict,xgb_feature
from data_test.pai.util import disan_for_ensemble,fast_disan_for_ensemble,bimpm_for_ensemble
from lib.model.configs import cfg
from lib.model.disan.model_disan import ModelDiSAN
from lib.model.fast_disan.model_fast_disan import ModelFastDiSAN
from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph
from data_test.ant.util import get_model_list
from lib.feature.features import get_feature


nn_name='exp1.7_d0.85_dw0.85_dc0.85_b0.2_bcompc0.2_bwam0.2_fd0.8_fdw0.8_train2'
name = 'feature_ensemble_0710'
models_count = 8
load_feature = True

def softmax(a):
    a=np.exp(a)
    a_sum=np.sum(a,axis=1)
    for i,x in enumerate(a):
        a[i]=a[i]/a_sum[i]
    return a

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_char_embed.json')
cfg.max_sentence_length = 80
cfg.feature_type = 'word+char'
emb = Embedding(cfg)

train_data = pd.read_csv(cfg.train_data)
x_train, y_train = emb.generate_sentence_token_ind(train_data)

valid_data = pd.read_csv(cfg.validate_data)
x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)

test_data = pd.read_csv(cfg.test_data)
x_test, y_test = emb.generate_sentence_token_ind(test_data)

if not load_feature:
    train_F, valid_F, test_F = get_feature(train_data,x_train), get_feature(valid_data,x_valid), \
                           get_feature(test_data, x_test)
    pickle.dump(train_F, open('data_test/pai/data/cache/features/train_tfidf_features.pickle','w'))
    pickle.dump(valid_F, open('data_test/pai/data/cache/features/valid_tfidf_features.pickle','w'))
    pickle.dump(test_F, open('data_test/pai/data/cache/features/test_tfidf_features.pickle','w'))
else:
    train_F = pickle.load(open('data_test/pai/data/cache/features/sin_train_features.pickle','r'))
    valid_F = pickle.load(open('data_test/pai/data/cache/features/valid_rep_features.pickle','r'))
    test_F = pickle.load(open('data_test/pai/data/cache/features/test_rep_features.pickle','r'))

all_train=train_F
all_valid=valid_F
all_test=test_F
# all_train_scores=pickle.load(open("data_test/pai/data/cache/{}/all_train_scores.pickle".
#                                   format(nn_name),'r'))
# all_valid_scores=pickle.load(open("data_test/pai/data/cache/{}/all_valid_scores.pickle".
#                                   format(nn_name),'r'))
# all_test_scores=pickle.load(open('data_test/pai/data/cache/{}/all_test_scores.pickle'.
#                                  format(nn_name),'r'))
#
# for i in range(models_count):
#     all_train_scores[i],all_valid_scores[i],all_test_scores[i] = \
#         softmax(all_train_scores[i]), softmax(all_valid_scores[i]), softmax(all_test_scores[i])
#
# all_train_scores = np.transpose(all_train_scores, axes=[1, 0, 2])
# all_train_scores = np.reshape(all_train_scores, [len(all_train_scores), models_count * 2])
#
# all_valid_scores = np.transpose(all_valid_scores, axes=[1, 0, 2])
# all_valid_scores = np.reshape(all_valid_scores, [len(all_valid_scores), models_count * 2])
#
# all_test_scores = np.transpose(all_test_scores, axes=[1, 0, 2])
# all_test_scores = np.reshape(all_test_scores, [len(all_test_scores), models_count * 2])
#
# all_train = np.concatenate((train_F, all_train_scores), axis=1)
# all_valid = np.concatenate((valid_F, all_valid_scores), axis=1)
# all_test = np.concatenate((test_F, all_test_scores), axis=1)


# start=time.clock()
# model = XGBClassifier(**{'tree_method':'gpu_hist'})
# max_depth = [3,5]
# n_estimators = [200]
# learning_rate = [0.1,]
# scale_pos_weight = [1.0 ]
# subsample=[0.9]
# min_child_weight=[2.5]
# param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,min_child_weight=min_child_weight,
#                   max_depth=max_depth, scale_pos_weight=scale_pos_weight,subsample=subsample)
# kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", cv=kfold)
# grid_result = grid_search.fit(all_train, y_train)
# end=time.clock()
# print('use time :{}'.format(end-start))
# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
#
# model = grid_search.best_estimator_
# #joblib.dump(model,'xgbmodel/{}'.format(name))
#
# y_valid_predict = model.predict(all_valid)
train_predict_probs,valid_predict_probs = \
    xgb_feature(all_train,y_train,all_valid,all_test,3)

labels = [0, 1]
target_names = ["no", "yes"]

# print '-------------------测试集----------------------'
# print(classification_report(y_pred=y_valid_predict,
#                             y_true=y_valid,
#                             target_names=target_names,
#                             labels=labels))
# print('\n\n')
# # y_train_predict = model.predict(all_train)
# print '-------------------训练集----------------------'
# print(classification_report(y_pred=y_train_predict,
#                             y_true=y_train,
#                             target_names=target_names,
#                             labels=labels))
print('\n\n')
# valid_predict_probs = model.predict_proba(all_valid)
print('logloss: {}'.format(log_loss(y_valid,valid_predict_probs)))

print('\n\n')
# train_predict_probs = model.predict_proba(all_train)
print('logloss: {}'.format(log_loss(y_train,train_predict_probs)))








