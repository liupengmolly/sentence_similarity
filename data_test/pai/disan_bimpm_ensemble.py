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

from data_test.pai.embedding import Embedding
from data_test.pai.util import fast_disan_predict, disan_predict, bimpm_predict, xgb_predict
from data_test.pai.util import disan_for_ensemble,fast_disan_for_ensemble,bimpm_for_ensemble
from lib.model.configs import cfg
from lib.model.disan.model_disan import ModelDiSAN
from lib.model.fast_disan.model_fast_disan import ModelFastDiSAN
from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph
from lib.feature.features import get_feature

GPU = cfg.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = GPU
def softmax(a):
    a=np.exp(a)
    a_sum=np.sum(a,axis=1)
    for i,x in enumerate(a):
        a[i]=a[i]/a_sum[i]
    return a

# control params
load_data= False
load_model= False
models_count = 8
name='exp1.7_ft_d0.85_dw0.85_dc0.85_b0.2_bcompc0.2_bw0.2am_fd0.8_fdw0.8_train2'
old_name = 'exp1.7_d0.85_dw0.85_dc0.85_b0.2_bcompc0.2_bw0.2am_fd0.8_fdw0.8_train2'
submit_name='result_20180710_v0.csv'

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

if not load_data:
    all_train_scores = []
    all_valid_scores = []
    all_test_scores = []

    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_char_embed.json')
    # cfg.batch_size = 32
    # cfg.max_sentence_length = 80
    # cfg.feature_type = 'word+char'
    # train_scores, valid_scores, test_scores = disan_for_ensemble(root_path, 'disan_exp_1.7_0.85',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()
    #
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/char_embed.json')
    # cfg.batch_size = 64
    # cfg.max_sentence_length = 50
    # cfg.feature_type = 'char'
    # train_scores, valid_scores, test_scores = \
    #     disan_for_ensemble(root_path, 'disan_exp_1.7_char_0.85',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()

    feature_model=joblib.load('feature_model/0710_v1')
    train_F = pickle.load(open('data_test/pai/data/cache/features/train_features.pickle','r'))
    valid_F = pickle.load(open('data_test/pai/data/cache/features/valid_features.pickle','r'))
    test_F = pickle.load(open('data_test/pai/data/cache/features/test_features.pickle','r'))
    train_scores = feature_model.predict_proba(train_F)
    train_scores = [(float(x[0]), float(x[1])) for x in train_scores]
    valid_scores = feature_model.predict_proba(valid_F)
    valid_scores = [(float(x[0]), float(x[1])) for x in valid_scores]
    test_scores = feature_model.predict_proba(test_F)
    test_scores = [(float(x[0]), float(x[1])) for x in test_scores]
    all_train_scores.append(train_scores)
    all_valid_scores.append(valid_scores)
    all_test_scores.append(test_scores)

    # tf.reset_default_graph()

    # cfg.feature_type = 'word+char'
    # cfg.batch_size = 64
    # cfg.max_sentence_length = 80
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_char_embed.json')
    # train_scores, valid_scores, test_scores = \
    #     bimpm_for_ensemble(root_path, 'bimpm_exp_1.7_0.2',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()
    #
    # cfg.feature_type = 'word'
    # cfg.max_sentence_length = 30
    # cfg.with_maxpool_match=True
    # cfg.with_max_attentive_match=True
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_embed.json')
    # train_scores, valid_scores, test_scores = \
    #     bimpm_for_ensemble(root_path, 'bimpm_exp_1.7_word_0.15_allmatch',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()

    # cfg.feature_type = 'word+char'
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_char_embed.json')
    # cfg.with_maxpool_match = True
    # cfg.with_max_attentive_match = True
    # train_scores, valid_scores, test_scores = \
    #     bimpm_for_ensemble(root_path, 'bimpm_exp_1.7_0.2_allmatch',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()

    # cfg.feature_type = 'char'
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/char_embed.json')
    # train_scores, valid_scores, test_scores = \
    #     bimpm_for_ensemble(root_path, 'bimpm_exp_1.7_char_0.2_allmatch', cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()

    # cfg.feature_type = 'word+char'
    # cfg.max_sentence_length = 80
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_char_embed.json')
    # train_scores, valid_scores, test_scores = \
    #     fast_disan_for_ensemble(root_path, 'fastdisan_exp_1.7_0.8',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()
    #
    # cfg.feature_type='word'
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_embed.json')
    # train_scores, valid_scores, test_scores = \
    #     fast_disan_for_ensemble(root_path, 'fastdisan_exp_1.7_word_0.8',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()
    #
    # cfg.feature_type='word'
    # cfg.word2vec_file = os.path.join(root_path, 'data_test/pai/data/word_embed.json')
    # train_scores, valid_scores, test_scores = \
    #     fast_disan_for_ensemble(root_path, 'fastdisan_exp_comp_1.7_word_0.8',cfg)
    # all_train_scores.append(train_scores)
    # all_valid_scores.append(valid_scores)
    # all_test_scores.append(test_scores)
    # tf.reset_default_graph()

    # if not os.path.exists('data_test/pai/data/cache/{}'.format(name)):
    #     os.makedirs('data_test/pai/data/cache/{}'.format(name))
    # pickle.dump(all_train_scores,open("data_test/pai/data/cache/{}/all_train_scores.pickle".format(name),'w'))
    # pickle.dump(all_valid_scores,open("data_test/pai/data/cache/{}/all_valid_scores.pickle".format(name),'w'))
    # pickle.dump(all_test_scores,open("data_test/pai/data/cache/{}/all_test_scores.pickle".format(name),'w'))
else:
    all_train_scores=pickle.load(open("data_test/pai/data/cache/{}/all_train_scores.pickle".format(name),'r'))
    all_valid_scores=pickle.load(open("data_test/pai/data/cache/{}/all_valid_scores.pickle".format(name),'r'))
    all_test_scores=pickle.load(open('data_test/pai/data/cache/{}/all_test_scores.pickle'.format(name),'r'))

    # train_F, valid_F, test_F = get_feature(train_data), get_feature(valid_data), get_feature(test_data)
    # all_train_scores.append(train_F)
    # all_valid_scores.append(valid_F)
    # all_test_scores.append(test_F)

labels = [0, 1]
target_names = ["no", "yes"]

#计算神经网络模型的集成结果，方便和后续传统模型融合
train_scores=pickle.load(open("data_test/pai/data/cache/{}/all_train_scores.pickle".format(old_name),'r'))
valid_scores=pickle.load(open("data_test/pai/data/cache/{}/all_valid_scores.pickle".format(old_name),'r'))
test_scores=pickle.load(open('data_test/pai/data/cache/{}/all_test_scores.pickle'.format(old_name),'r'))

train_scores=train_scores[1:8]
valid_scores=valid_scores[1:8]
test_scores=test_scores[1:8]

# for i in range(6):
#     train_scores[i],valid_scores[i],test_scores[i] = \
#         softmax(train_scores[i]), softmax(valid_scores[i]), softmax(test_scores[i])
#
# train_scores = np.transpose(train_scores, axes=[1, 0, 2])
# train_scores = np.reshape(train_scores, [len(train_scores), (models_count-1) * 2])
#
# valid_scores = np.transpose(valid_scores, axes=[1, 0, 2])
# valid_scores = np.reshape(valid_scores, [len(valid_scores), (models_count-1) * 2])
#
# test_scores = np.transpose(test_scores, axes=[1, 0, 2])
# test_scores = np.reshape(test_scores, [len(test_scores), (models_count-1) * 2])
#
# nn_y_train_predict, nn_y_valid_predict, nn_y_train_probs, nn_y_valid_probs, nn_y_test_probs = \
#     xgb_predict(train_scores,y_train,valid_scores,test_scores)
#
# print '-------------------测试集----------------------'
# print(classification_report(y_pred=nn_y_valid_predict,
#                             y_true=y_valid,
#                             target_names=target_names,
#                             labels=labels))
# print('\n\n')
# print '-------------------训练集----------------------'
# print(classification_report(y_pred=nn_y_train_predict,
#                             y_true=y_train,
#                             target_names=target_names,
#                             labels=labels))
# print('\n\n')
# print('logloss: {}'.format(log_loss(y_valid,nn_y_valid_probs)))
#
# print('\n\n')
# print('logloss: {}'.format(log_loss(y_train,nn_y_train_probs)))

#将神经网络的结果和传统模型的结果集成
# all_train_scores.append(nn_y_train_probs)
# all_valid_scores.append(nn_y_valid_probs)
# all_test_scores.append(nn_y_test_probs)
all_train_scores.extend(train_scores)
all_valid_scores.extend(valid_scores)
all_test_scores.extend(test_scores)

all_train_scores = np.transpose(all_train_scores, axes=[1, 0, 2])
all_train_scores = np.reshape(all_train_scores, [len(all_train_scores), models_count * 2])

all_valid_scores = np.transpose(all_valid_scores, axes=[1, 0, 2])
all_valid_scores = np.reshape(all_valid_scores, [len(all_valid_scores), models_count * 2])

all_test_scores = np.transpose(all_test_scores, axes=[1, 0, 2])
all_test_scores = np.reshape(all_test_scores, [len(all_test_scores), models_count * 2])

y_train_predict, y_valid_predict, y_train_probs, y_valid_probs, y_test_probs = \
    xgb_predict(all_train_scores,y_train,all_valid_scores,all_test_scores)


print '-------------------测试集----------------------'
print(classification_report(y_pred=y_valid_predict,
                            y_true=y_valid,
                            target_names=target_names,
                            labels=labels))
print('\n\n')
print '-------------------训练集----------------------'
print(classification_report(y_pred=y_train_predict,
                            y_true=y_train,
                            target_names=target_names,
                            labels=labels))
print('\n\n')
print('logloss: {}'.format(log_loss(y_valid,y_valid_probs)))

print('\n\n')
print('logloss: {}'.format(log_loss(y_train,y_train_probs)))

result=pd.DataFrame(y_test_probs,columns=['0','1'])
result.to_csv("data_test/pai/data/result/predict_{}.csv".
                                     format(name),index=False)
submit=result[['1']].rename(columns={'1':'y_pre'})
submit.to_csv('data_test/pai/data/result/submit/{}'.format(submit_name),index=False)
submit.to_csv('/DeepLearning/peng_liu/pai/pai_result/{}'.format(submit_name),index=False)
import subprocess as sp
sp.call(['ln', '-s', submit_name, 'data_test/pai/data/result/submit/submit_{}.csv'.format(name)])

