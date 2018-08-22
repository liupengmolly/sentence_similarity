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

# train_data = pd.read_csv(cfg.train_data)
# x_train, y_train = emb.generate_sentence_token_ind(train_data)
valid_data = pd.read_csv(cfg.validate_data)
x_valid, y_valid = emb.generate_sentence_token_ind(valid_data)
test_data = pd.read_csv(cfg.test_data)
x_test, y_test = emb.generate_sentence_token_ind(test_data)

# train_F = get_feature(train_data,x_train)
valid_F=get_feature(valid_data,x_valid)
test_F = get_feature(test_data, x_test)

model=joblib.load(cfg.model_directory)

# train_predict=model.predict_proba(train_F)
# train_predict=[(float(x[0]),float(x[1])) for x in train_predict]
valid_predict=model.predict_proba(valid_F)
valid_predict=[(float(x[0]),float(x[1])) for x in valid_predict]
test_predict=model.predict_proba(test_F)
test_predict=[(float(x[0]),float(x[1])) for x in test_predict]
# pickle.dump(train_predict,open('data_test/pai/data/tmp/train__pred.pickle','w'))
pickle.dump(valid_predict,open('data_test/pai/data/tmp/valid_pred.pickle','w'))
pickle.dump(test_predict,open('data_test/pai/data/tmp/test_pred.pickle','w'))