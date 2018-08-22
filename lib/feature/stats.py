import scipy
import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_path)
from fuzzywuzzy import fuzz
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, \
    minkowski, braycurtis
from lib.feature.util import *
import pandas as pd
import  numpy as np


def get_basic_feature(data):
    data['len_q1'] = data.words_chars1.apply(lambda x: len(str(x).split()))
    data['len_q2'] = data.words_chars2.apply(lambda x: len(str(x).split()))
    data['diff_len'] = data.len_q1 - data.len_q2
    data['len_char_q1'] = data.chars1.apply(lambda x: len(str(x).split()))
    data['len_char_q2'] = data.chars2.apply(lambda x: len(str(x).split()))
    data['len_word_q1'] = data.words1.apply(lambda x: len(str(x).split()))
    data['len_word_q2'] = data.words2.apply(lambda x: len(str(x).split()))
    data['common_words'] = data.apply(lambda x: len(
        set(str(x['words1']).lower().split()).intersection(
            set(str(x['words2']).lower().split()))), axis=1)
    data['common_chars'] = data.apply(lambda x: len(
        set(str(x['chars1']).lower().split()).intersection(
            set(str(x['chars2']).lower().split()))), axis=1)
    return data


def get_distance_features(data,emb):
    data['cosine_distance'] = pd.Series([cosine(x, y) for x, y in emb])
    data['cityblock_distance'] = pd.Series([cityblock(x, y) for x, y in emb])
    data['jaccard_distance'] = pd.Series([jaccard(x, y) for x, y in emb])
    data['canberra_distance'] = pd.Series([canberra(x, y) for x, y in emb])
    data['euclidean_distance'] = pd.Series([euclidean(x, y) for x, y in emb])
    data['minkowski_distance'] = pd.Series([minkowski(x, y, 3) for x, y in emb])
    data['braycurtis_distance'] = pd.Series([braycurtis(x, y) for x, y in emb])
    return data

def get_stats_features(data,emb):
    question1_vectors=[x[0] for x in emb]
    question2_vectors=[x[1] for x in emb]
    data['skew_q1vec'] = pd.Series([skew(x) for x in np.nan_to_num(question1_vectors)])
    data['skew_q2vec'] = pd.Series([skew(x) for x in np.nan_to_num(question2_vectors)])
    data['kur_q1vec'] = pd.Series([kurtosis(x) for x in np.nan_to_num(question1_vectors)])
    data['kur_q2vec'] = pd.Series([kurtosis(x) for x in np.nan_to_num(question2_vectors)])
    return data

def get_fuzzy_feature(data):
    data['fuzz_qratio_words'] = data.apply(
        lambda x: fuzz.QRatio(str(x['words1']), str(x['words2'])), axis=1)
    data['fuzz_qratio_chars'] = data.apply(
        lambda x: fuzz.QRatio(str(x['chars1']), str(x['chars2'])), axis=1)
    data['fuzz_WRatio_words'] = data.apply(
        lambda x: fuzz.WRatio(str(x['words1']), str(x['words2'])), axis=1)
    data['fuzz_WRatio_chars'] = data.apply(
        lambda x: fuzz.WRatio(str(x['chars1']), str(x['chars2'])), axis=1)
    data['fuzz_partial_ratio_words'] = data.apply(
        lambda x: fuzz.partial_ratio(str(x['words1']), str(x['words2'])), axis=1)
    data['fuzz_partiald_datad_ratio_chars'] = data.apply(
        lambda x: fuzz.partial_ratio(str(x['chars1']), str(x['chars2'])), axis=1)
    data['fuzz_partial_token_set_ratio_words'] = data.apply(
        lambda x: fuzz.partial_token_set_ratio(str(x['words1']), str(x['words2'])), axis=1)
    data['fuzz_partial_token_set_ratio_chars'] = data.apply(
        lambda x: fuzz.partial_token_set_ratio(str(x['chars1']), str(x['chars2'])), axis=1)
    data['fuzz_partial_token_sort_ratio_words'] = data.apply(
        lambda x: fuzz.partial_token_sort_ratio(str(x['words1']), str(x['words2'])), axis=1)
    data['fuzz_partial_token_sort_ratio_chars'] = data.apply(
        lambda x: fuzz.partial_token_sort_ratio(str(x['chars1']), str(x['chars2'])), axis=1)
    data['fuzz_token_set_ratio_words'] = data.apply(
        lambda x: fuzz.token_set_ratio(str(x['words1']), str(x['words2'])), axis=1)
    data['fuzz_token_set_ratio_chars'] = data.apply(
        lambda x: fuzz.token_set_ratio(str(x['chars1']), str(x['chars2'])), axis=1)
    data['fuzz_token_sort_ratio_words'] = data.apply(
        lambda x: fuzz.token_sort_ratio(str(x['words1']), str(x['words2'])), axis=1)
    data['fuzz_token_sort_ratio_chars'] = data.apply(
        lambda x: fuzz.token_sort_ratio(str(x['chars1']), str(x['chars2'])), axis=1)
    return data

def get_pword(data,thresh_num=500,thresh_rate=0.9):
    pword=load_powerful_word(os.path.join(root_path,'data_test/pai/data/powerful_words.txt'))
    pword_dside=init_powerful_word_dside(pword,thresh_num,thresh_rate)
    data['pword_dside']=data.apply(lambda x: extract_row_dside(x,pword_dside),axis=1)
    data['pword_oside']=data.apply(lambda x: extract_row_oside(x,pword_dside),axis=1)
    return data

def get_pword_rate(data):
    pword_dict=dict(load_powerful_word(os.path.join(root_path,
                                                    'data_test/pai/data/powerful_words.txt')))
    data['pword_rate_dside']=data.apply(lambda x: extract_row_rate_dside(x,pword_dict),axis=1)
    data['pword_rate_oside']=data.apply(lambda x: extract_row_rate_oside(x,pword_dict),axis=1)
    return data


