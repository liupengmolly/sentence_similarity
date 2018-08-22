#! /usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from os.path import join
import time
from data_test.ant import DATA_PATH


class Configs(object):
    def __init__(self):
        self.project_dir = os.getcwd()
        self.dataset_dir = join(self.project_dir, 'dataset')

        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        word2vec_path = os.path.join(DATA_PATH, 'ant_data_w2v.json')
        parser.add_argument(
            '--word2vec_file', type=str, default=word2vec_path, help='word2vec file path')
        parser.add_argument(
            '--train_data', type=str, help='data file path')
        parser.add_argument(
            '--validate_data', type=str, help='validate_file save path'
        )
        parser.add_argument(
            '--test_data', type=str, help='test_data save path'
        )
        parser.add_argument(
            '--model_directory', type=str, help='mode directory'
        )
        parser.add_argument(
            '--model_index', type=int, help=''
        )

        parser.add_argument(
            "--models", type=str,
            help=
            '''
                model-5000, model-6000
                Load trained model checkpoint,
                separate by ,  (Default: None)
            '''
        )
        parser.add_argument('--feature_type', type=str, default='word', help='word|char|both')
        parser.add_argument('--use_pinyin', type=bool, default=False, help='if use pinyin when '
                                                                           'embeddinng')
        parser.add_argument('--use_stacking', type=bool, default=False,
                            help='if using stacking when using fast_disan')

        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=500, help='max epoch number')
        parser.add_argument('--num_steps', type=int, default=1000, help='every steps to print')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
        parser.add_argument('--optimizer', type=str, default='adadelta',
                            help='choose an optimizer[adadelta|adam]')
        parser.add_argument('--learning_rate', type=float, default=0.1, help='Init Learning rate')
        parser.add_argument('--dy_lr', type=bool, default=False, help='if decay lr during training')
        parser.add_argument('--lr_decay', type=float, default=0.8, help='Learning rate decay')
        parser.add_argument('--dropout', type=float, default=0.75, help='dropout keep prob')
        parser.add_argument('--wd', type=float, default=5e-5,
                            help='weight decay factor/l2 decay factor')
        parser.add_argument('--var_decay', type=float, default=0.999, help='Learning rate')  # ema
        parser.add_argument('--decay', type=float, default=0.9, help='summary decay')  # ema
        parser.add_argument('--max_sentence_length', type=int, default=30,
                            help='the sentence max length')
        parser.add_argument('--mode', type=str, default='train')
        parser.add_argument('--use_pre_trained', type=bool, default=False,
                            help='use or not use pre_trained w2v')
        # @ ----- Text Processing ----
        parser.add_argument('--word_embedding_length', type=int, default=100,
                            help='word embedding length')
        parser.add_argument('--pos_embedding_length', type=int, default=10,
                            help='pos embedding length')

        parser.add_argument('--lower_word', type=bool, default=True, help='')
        parser.add_argument('--data_clip_method', type=str, default='no_tree',
                            help='for space-efficiency[no_tree|]no_redundancy')
        parser.add_argument('--sent_len_rate', type=float,
                            default=0.97, help='delete too long sentences')
        timestamp = str(int(time.time()))
        parser.add_argument('--model_save_path', type=str,
                            default=timestamp, help=' file path name to save model')
        parser.add_argument('--log_name', type=str, default='', help='log file name')
        parser.add_argument('--gpu', type=str, default='8', help='gpu')
        # @ ------neural network-----
        parser.add_argument('--use_char_emb', type=bool, default=False, help='abandoned')
        parser.add_argument('--use_token_emb', type=bool, default=True, help='abandoned')

        parser.add_argument('--highway_layer_num', type=int, default=1,
                            help='highway layer number(abandoned)')

        parser.add_argument('--hidden_units_num', type=int, default=300,
                            help='Hidden units number of Neural Network')
        parser.add_argument('--tree_hn', type=int, default=100, help='(abandoned)')

        parser.add_argument('--fine_tune', type=bool, default=False,
                            help='(abandoned, keep False)')  # ema

        # # emb_opt_direct_attn
        parser.add_argument('--batch_norm', type=bool, default=False,
                            help='(abandoned, keep False)')
        parser.add_argument('--activation', type=str, default='relu', help='(abandoned')

        parser.add_argument("--allow_soft_placement", type=bool, default=True,
                            help="Allow device soft device placement")
        parser.add_argument("--log_device_placement", type=bool, default=False,
                            help="Log placement of ops on devices")

        # ---------------ant nlp config ---------------
        parser.add_argument('--input_path', type=str, help='input data path')
        parser.add_argument('--output_path', type=str, help='out data path')

        # @-----------bimpm----------
        parser.add_argument('--suffix', type=str, default='normal',
                            help='Suffix of the model name.')
        parser.add_argument('--with_char', default=False,
                            help='With character-composed embeddings.',
                            action='store_true')
        parser.add_argument('--fix_word_vec', default=True,
                            help='Fix pre-trained word embeddings during training.',
                            action='store_true')
        parser.add_argument('--with_highway', default=True,
                            help='Utilize highway layers.', action='store_true')
        parser.add_argument('--with_match_highway', default=True,
                            help='Utilize highway layers for matching layer.',
                            action='store_true')
        parser.add_argument('--with_aggregation_highway', default=True,
                            help='Utilize highway layers for aggregation layer.',
                            action='store_true')
        parser.add_argument('--with_full_match', default=True, type=bool,
                            help='With full matching.')
        parser.add_argument('--with_maxpool_match', default=False, type=bool,
                            help='With maxpooling matching')
        parser.add_argument('--with_attentive_match', default=True, type=bool,
                            help='With attentive matching')
        parser.add_argument('--with_max_attentive_match', default=False, type=bool,
                            help='With max attentive matching.')
        parser.add_argument('--use_cudnn', type=bool, default=False, help='if use cudnn')
        parser.add_argument('--grad_clipper', type=float, default=10.0,
                            help='grad')
        parser.add_argument('--is_lower', type=bool, default=True,
                            help='is_lower')
        parser.add_argument('--with_cosine', type=bool, default=True,
                            help='with_cosine')
        parser.add_argument('--with_mp_cosine', type=bool, default=True,
                            help='map_cosine')
        parser.add_argument('--cosine_MP_dim', type=int, default=5, help='mp')
        parser.add_argument('--att_dim', type=int, default=50, help='att_dm')
        parser.add_argument('--att_type', type=str, default='symmetric',
                            help='att_type')
        parser.add_argument('--with_moving_average', type=bool, default=True,
                            help='moving_average')
        parser.add_argument('--lambda_l2', type=float, default=5e-5,
                            help='The coefficient of L2 regularizer.')
        parser.add_argument('--dropout_rate', type=float, default=0.2,
                            help='Dropout ratio.')
        parser.add_argument('--optimize_type', type=str, default='adam',
                            help='Optimizer type.')
        parser.add_argument('--context_lstm_dim', type=int, default=100,
                            help='Number of dimension for context representation layer.')
        parser.add_argument('--aggregation_lstm_dim', type=int, default=100,
                            help='Number of dimension for aggregation layer.')
        parser.add_argument('--aggregation_layer_num', type=int, default=1,
                            help='Number of LSTM layers for aggregation layer.')
        parser.add_argument('--context_layer_num', type=int, default=1,
                            help='Number of LSTM layers for context representation layer.')
        # --------------------matchpyramid------------------
        parser.add_argument('--pool_size', type=int, default=4, help='the pooling size')
        parser.add_argument('--window_size', type=int, default=4, help='the pooling size')

        #  ------------- mix --------------------------
        parser.add_argument('--attention_type', type=str, default='idf', help='idf|type|all')

        idf_file = os.path.join(DATA_PATH, 'id_idf.json')
        parser.add_argument('--channal4', type=bool, default=True, help='')
        parser.add_argument('--channal9', type=bool, default=False, help='')
        parser.add_argument('--use_pos', type=bool, default=False, help='')
        parser.add_argument('--pos_count', type=int, default=55, help='')
        parser.add_argument('--use_idf', type=bool, default=False, help='')
        parser.add_argument('--idf_file', type=str, default=idf_file, help='')
        parser.add_argument('--use_position', type=bool, default=False, help='')
        parser.add_argument('--position_embedding_length', type=int, default=10,
                            help='pos embedding length')

        parser.add_argument(
            '--use_extract_match', type=bool, default=False, help='pairs have same part')
        parser.add_argument('--extract_match_feature_len', type=int, default=11,
                            help='feature count')
        parser.add_argument(
            '--word2vec_sim', type=bool, default=False,
            help='sum word2vec as sent2vec, calculate sim of sentence'
        )
        parser.add_argument(
            '--char2vec_file', type=str, default='', help='char2vec file stored as json file'
        )
        parser.add_argument(
            '--char_embedding_length', type=int, default=60, help='char embedding length'
        )
        parser.add_argument('--char_length', type=int, default=5, help='(abandoned)')
        parser.add_argument('--use_token_match', type=bool, default=False, help='use token')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['data_test', 'shuffle']:
                exec ('self.%s = self.args.%s' % (key, key))

        # # ------- name --------
        # self.train_data_name = 'snli_1.0_train.jsonl'
        # self.dev_data_name = 'snli_1.0_dev.jsonl'
        # self.test_data_name = 'snli_1.0_test.jsonl'
        #
        # self.processed_name = 'processed' + self.get_params_str(['lower_word', 'use_glove_unk_token',
        #                                                          'glove_corpus', 'word_embedding_length',
        #                                                          'sent_len_rate',
        #                                                          'data_clip_method']) + '.pickle'
        # self.dict_name = 'dicts' + self.get_params_str(['lower_word', 'use_glove_unk_token',
        #                                                 ])
        #
        # if not self.network_type == 'data_test':
        #     params_name_list = ['network_type', 'dropout', 'glove_corpus',
        #                         'word_embedding_length', 'fine_tune', 'char_out_size', 'sent_len_rate',
        #                         'hidden_units_num', 'wd', 'optimizer', 'learning_rate', 'dy_lr', 'lr_decay']
        #     self.model_name = self.get_params_str(params_name_list)
        # else:
        #     self.model_name = self.network_type
        # self.model_ckpt_name = 'modelfile.ckpt'
        #
        # # ---------- dir -------------
        # self.data_dir = join(self.dataset_dir, 'snli_1.0')
        # self.glove_dir = join(self.dataset_dir, 'glove')
        # self.result_dir = self.make_dir(self.project_dir, 'result')
        # self.standby_log_dir = self.make_dir(self.result_dir, 'log')
        # self.dict_dir = self.make_dir(self.result_dir, 'dict')
        # self.processed_dir = self.make_dir(self.result_dir, 'processed_data')
        #
        # self.log_dir = None
        # self.all_model_dir = self.make_dir(self.result_dir, 'model')
        # self.model_dir = self.make_dir(self.all_model_dir, self.model_dir_suffix + self.model_name)
        # self.log_dir = self.make_dir(self.model_dir, 'log_files')
        # self.summary_dir = self.make_dir(self.model_dir, 'summary')
        # self.ckpt_dir = self.make_dir(self.model_dir, 'ckpt')
        # self.answer_dir = self.make_dir(self.model_dir, 'answer')
        #
        # # -------- path --------
        # self.train_data_path = join(self.data_dir, self.train_data_name)
        # self.dev_data_path = join(self.data_dir, self.dev_data_name)
        # self.test_data_path = join(self.data_dir, self.test_data_name)
        #
        # self.processed_path = join(self.processed_dir, self.processed_name)
        # self.dict_path = join(self.dict_dir, self.dict_name)
        # self.ckpt_path = join(self.ckpt_dir, self.model_ckpt_name)
        #
        # self.extre_dict_path = join(self.dict_dir,
        #                             'extra_dict'+self.get_params_str(['data_clip_method'])+'.json')
        #
        # # dtype
        # self.floatX = 'float32'
        # self.intX = 'int32'
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

    @staticmethod
    def get_params_str(params):
        def abbreviation(name):
            words = name.strip().split('_')
            abb = ''
            for word in words:
                abb += word[0]
            return abb

        abbreviations = map(abbreviation, params)
        model_params_str = ''
        for params_str, abb in zip(params, abbreviations):
            model_params_str += '_' + abb + '_' + str(
                eval('self.args.' + params_str))
        return model_params_str

    @staticmethod
    def make_dir(*args):
        dir_path = join(*args)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        return dir_path

    @staticmethod
    def get_file_name_from_path(path):
        assert isinstance(path, str)
        file_name = '.'.join((path.split('/')[-1]).split('.')[:-1])
        return file_name


cfg = Configs()
