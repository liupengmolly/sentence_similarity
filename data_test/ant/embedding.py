#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import random
import sys

import numpy as np

reload(sys)
sys.setdefaultencoding('utf8')
import jieba

import jieba.posseg as pos

from common.data_helper import DataHelper
from word2vec.word2vec_model import Word2vecModel
from data_test.ant import DATA_PATH

jieba.load_userdict(os.path.join(DATA_PATH, 'jieba_add_words.txt'))
# POS_FILE = os.path.join(DATA_PATH, 'pos_id.json')
pos2vec_file = os.path.join(DATA_PATH, 'ant_data_pos2vec.json')


class Embedding(object):

    def __init__(self, cfg, logger):
        print cfg.max_sentence_length
        print cfg.word2vec_file
        self.cfg = cfg
        self.logger = logger
        self.word2vec = Word2vecModel.load(cfg.word2vec_file)
        self.word_id_map, self.id_vector_map = self.word2vec.generate_word_id_map(np.float32)
        self.pos_id_map = None
        if cfg.use_pos:
            self.pos2vec = Word2vecModel.load(pos2vec_file)
            self.pos_id_map, self.pos_id_vector_map = self.pos2vec.generate_word_id_map(np.float32)
            # self.pos_id_map = json.load(open(POS_FILE, 'r'))
        if cfg.use_idf:
            self.id_idf_weight_map = dict(enumerate(json.load(open(self.cfg.idf_file, 'r'))))

        self.char_id_map = None
        self.char_vector_map = None
        if self.cfg.use_char_emb:
            self.char2vec = Word2vecModel.load(cfg.char2vec_file)
            self.char_id_map, self.char_vector_map = self.char2vec.generate_word_id_map(np.float32)

    def generate_sentence_token_ind(self, data):

        data = list(np.array(data)[:, 1:4])
        method = sentences2char

        if self.cfg.feature_type == 'word':
            method = sentences2word
        data = [((method(x[0]), method(x[1])), int(x[2])) for x in data]
        data_ = []
        for i in xrange(len(data)):
            if random.random() >= 0.5:
                data_.append(data[i])
            else:
                data_.append(((data[i][0][1], data[i][0][0]), data[i][1]))

        x, y = zip(*data)
        x1, x2 = zip(*x)
        token1, pos1 = zip(*x1)
        token2, pos2 = zip(*x2)

        data = zip(zip(token1, token2), y)

        data_help = DataHelper(zip(*data),
                               self.cfg.max_sentence_length,
                               word_index_dic=self.word_id_map,
                               by_word_index_dic=True)
        x, y = data_help.documents_transform_and_padding(self.cfg)
        feature = [x]
        if self.cfg.use_pos:
            self.logger.info('use pos emb...')
            pos_pair = zip(pos1, pos2)
            pos_pair_ind = self.generate_sentence_pos_ind(pos_pair)
            feature.append(pos_pair_ind)
        if self.cfg.use_idf:
            self.logger.info('use idf emb...')
            tokens_idf = self.generate_sentence_word_idf(x)
            feature.append(tokens_idf)
        if self.cfg.use_position:
            self.logger.info('use position emb...')
            tokens_position = self.generate_token_position(x)
            feature.append(tokens_position)
        if self.cfg.use_extract_match:
            self.logger.info('use extract match...')
            extract_match_feature = self.generate_extract_match_feature(zip(x1, x2))
            feature.append(extract_match_feature)

        if self.cfg.use_char_emb:
            self.logger.info('use char emb...')
            sent_token_char_feature = self.generate_char_feature(zip(token1, token2))
            feature.append(sent_token_char_feature)
        if self.cfg.use_token_match:
            self.logger.info('use token match...')
            token_match_feature = self.generate_token_match_feature(zip(token1, token2))
            feature.append(token_match_feature)

        return zip(*feature), y

    def generate_sentence_pos_ind(self, pos_pair):

        def generate_sentence_pos_ind_(s):

            if len(s) >= self.cfg.max_sentence_length:
                s = s[-self.cfg.max_sentence_length:]
            res = [self.pos_id_map.get(ele, 0) for ele in s]
            if len(res) < self.cfg.max_sentence_length:
                res = res + [1] * (self.cfg.max_sentence_length - len(res))
            return res

        pos_ind = []
        for pair in pos_pair:
            s1, s2 = pair
            s1_ind = generate_sentence_pos_ind_(s1)
            s2_ind = generate_sentence_pos_ind_(s2)
            pos_ind.append((s1_ind, s2_ind))

        return pos_ind

    def generate_sentence_word_idf(self, tokens_pair):

        def normalize_(weight_list):
            total = sum(weight_list)
            weight_list = [x / total for x in weight_list]
            return weight_list

        tokens_idf_weight = []
        for pair in tokens_pair:
            s1_tokens, s2_tokens = pair
            s1_idf = [self.id_idf_weight_map.get(ind) for ind in s1_tokens]

            s1_idf = normalize_(s1_idf)

            s1_idf = [(x, ) for x in s1_idf]
            s2_idf = [self.id_idf_weight_map.get(ind) for ind in s2_tokens]
            s2_idf = normalize_(s2_idf)
            s2_idf = [(x, ) for x in s2_idf]
            tokens_idf_weight.append((s1_idf, s2_idf))
        return tokens_idf_weight

    def generate_token_position(self, tokens_pair):

        tokens_position = []
        for i in xrange(len(tokens_pair)):
            s_token_position = np.eye(self.cfg.max_sentence_length)
            # s1_token_position = range(self.cfg.max_sentence_length)
            # s2_token_position = range(self.cfg.max_sentence_length)
            tokens_position.append((s_token_position, s_token_position))
        return tokens_position

    @staticmethod
    def generate_extract_match_feature(pairs):
        pairs_feature = []
        for pair in pairs:
            pair_feature = []
            tag_label = ['nz', 'Ag', 'Ng', 'an', 'f', 'i', 'n', 's', 'tg', 't', 'q']
            for tag in tag_label:
                tokens1 = set([x[0] for x in pair[0] if x[1] == tag])
                tokens2 = set([x[0] for x in pair[1] if x[1] == tag])
                if tokens1 & tokens2:
                    pair_feature.append(1)
                else:
                    pair_feature.append(0)
            # 此处填充一个None, 是为了保持格式一致，在feed_dict中可以方便解析
            pairs_feature.append((pair_feature, None))
        return pairs_feature

    def generate_char_feature(self, pairs):

        def generate_tokens_chars_ind_matrix(tokens):
            if len(tokens) > self.cfg.max_sentence_length:
                tokens = tokens[-self.cfg.max_sentence_length:]
            chars_ind_matrix = [[1] * self.cfg.char_length] * self.cfg.max_sentence_length
            for token_ind, token in enumerate(tokens):
                if len(token) > self.cfg.char_length:
                    token = token[:self.cfg.char_length]
                for char_ind, char in enumerate(token):
                    chars_ind_matrix[token_ind][char_ind] = self.char_id_map.get(char)
            return chars_ind_matrix

        batch_tokens_chars_ind = []
        for pair in pairs:
            tokens1, tokens2 = pair
            tokens1_chars_ind = generate_tokens_chars_ind_matrix(tokens1)
            tokens2_chars_ind = generate_tokens_chars_ind_matrix(tokens2)
            batch_tokens_chars_ind.append((tokens1_chars_ind, tokens2_chars_ind))

        return batch_tokens_chars_ind

    def generate_token_match_feature(self, pairs):

        def find_match_token_ind(sent1, sent2):
            if len(sent1) > self.cfg.max_sentence_length:
                sent1 = sent1[-self.cfg.max_sentence_length:]
            if len(sent2) > 50:
                sent2 = sent2[-self.cfg.max_sentence_length:]

            match_vector = [[0]] * self.cfg.max_sentence_length
            tokens2 = set(sent2)
            for ind, token in enumerate(sent1):
                if token in tokens2:
                    match_vector[ind][0] = 1
            return match_vector
        batch_tokens_match = []
        for pair in pairs:
            match_vec1 = find_match_token_ind(pair[0], pair[1])
            match_vec2 = find_match_token_ind(pair[1], pair[1])
            batch_tokens_match.append((match_vec1, match_vec2))
        return batch_tokens_match


def sentences2char(sentence):
    if isinstance(sentence, str):
        sentence = unicode(sentence)
    if isinstance(sentence, unicode):
        return list(sentence), None
    else:
        raise Exception('data error !!!')


def sentences2word(sentence):

    if isinstance(sentence, (str, unicode)):
        words = list(pos.cut(sentence))
        words_ = [x.word for x in words]
        poses = [x.flag for x in words]
        return words_, poses
    else:
        raise Exception('data error !!!')


if __name__ == '__main__':
    pass