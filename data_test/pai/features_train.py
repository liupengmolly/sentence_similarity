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
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)

from data_test.pai.embedding import Embedding
from lib.model.configs import cfg
from lib.feature.features import get_feature

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

train_F, valid_F, test_F = get_feature(train_data,x_train), get_feature(valid_data,x_valid),\
                   get_feature(test_data, x_test)

pickle.dump(train_F, open('data_test/pai/data/cache/features/train_features.pickle','w'))
pickle.dump(valid_F, open('data_test/pai/data/cache/features/valid_features.pickle','w'))
pickle.dump(test_F, open('data_test/pai/data/cache/features/test_features.pickle','w'))

model = XGBClassifier(**{'tree_method':'gpu_hist'})
max_depth = [3,5]
n_estimators = [200]
learning_rate = [0.1,]
scale_pos_weight = [1.0 ]
subsample=[0.9]
min_child_weight=[2.5]
param_grid = dict(learning_rate=learning_rate, n_estimators=n_estimators,min_child_weight=min_child_weight,
                  max_depth=max_depth, scale_pos_weight=scale_pos_weight,subsample=subsample)
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", cv=kfold)
grid_result = grid_search.fit(train_F, y_train)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = grid_search.best_estimator_
joblib.dump(model, 'feature_model/0710_v1')

y_valid_predict = model.predict(valid_F)
labels = [0, 1]
target_names = ["no", "yes"]

print '-------------------测试集----------------------'
print(classification_report(y_pred=y_valid_predict,
                            y_true=y_valid,
                            target_names=target_names,
                            labels=labels))
print('\n\n')
y_train_predict = model.predict(train_F)
print '-------------------训练集----------------------'
print(classification_report(y_pred=y_train_predict,
                            y_true=y_train,
                            target_names=target_names,
                            labels=labels))
print('\n\n')
y_predict_probs = model.predict_proba(valid_F)
print('logloss: {}'.format(log_loss(y_valid,y_predict_probs)))

print('\n\n')
y_predict_probs = model.predict_proba(train_F)
print('logloss: {}'.format(log_loss(y_train,y_predict_probs)))

# y_test_predict_probs=model.predict_proba(test_F)
# pickle.dump(y_test_predict_probs,open('data_test/pai/data/tmp/feature_prob.pickle','w'))
# result=pd.DataFrame(y_test_predict_probs,columns=['0','1'])
# result.to_csv('data_test/pai/data/tmp/feature_prob.csv',index=False)


