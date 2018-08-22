import argparse
import time
import random
import pandas as pd
class Configs(object):
    def __init__(self):
        # ------parsing input arguments"--------
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--model_directory', type=str, default='',help='mode directory'
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
        parser.add_argument('--use_pinyin',type=bool,default=False,help='if use pinyin when '
                                                                        'embeddinng')
        parser.add_argument('--use_stacking',type=bool,default=False,help='if using stacking when'
                                                                          'using fast_disan')
        # @ ----------training ------
        parser.add_argument('--max_epoch', type=int, default=500, help='max epoch number')
        parser.add_argument('--num_steps', type=int, default=2000, help='every steps to print')
        parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
        parser.add_argument('--optimizer', type=str, default='adadelta',
                            help='choose an optimizer[adadelta|adam]')
        parser.add_argument('--learning_rate', type=float, default=0.0005, help='Init Learning rate')
        parser.add_argument('--dy_lr', type=bool, default=False, help='if decay lr during training')
        parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay')
        parser.add_argument('--dropout', type=float, default=0.8, help='dropout keep prob')
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
        parser.add_argument('--lower_word', type=bool, default=True, help='')
        parser.add_argument('--data_clip_method', type=str, default='no_tree',
                            help='for space-efficiency[no_tree|]no_redundancy')
        parser.add_argument('--sent_len_rate', type=float,
                            default=0.97, help='delete too long sentences')
        timestamp = str(int(time.time()))
        parser.add_argument('--model_save_path', type=str,
                            default=timestamp, help=' file path name to save model')
        parser.add_argument('--log_name', type=str, default='', help='log file name')
        # @ ------neural network-----
        parser.add_argument('--use_char_emb', type=bool, default=False, help='abandoned')
        parser.add_argument('--use_token_emb', type=bool, default=True, help='abandoned')
        parser.add_argument('--char_embedding_length', type=int, default=8, help='(abandoned)')
        parser.add_argument('--char_out_size', type=int, default=150, help='(abandoned)')
        parser.add_argument('--out_channel_dims', type=str, default='50,50,50', help='(abandoned)')
        parser.add_argument('--filter_heights', type=str, default='1,3,5', help='(abandoned)')
        parser.add_argument('--highway_layer_num', type=int, default=2,
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
        # --------------------------------@bimpm-------------------------------
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
        parser.add_argument('--with_full_match', default=True,
                            help='With full matching.', action='store_true')
        parser.add_argument('--with_maxpool_match', default=False,
                            help='With maxpooling matching',
                            action='store_true')
        parser.add_argument('--with_attentive_match', default=True,
                            help='With attentive matching', action='store_true')
        parser.add_argument('--with_max_attentive_match', default=False,
                            help='With max attentive matching.',
                            action='store_true')
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
        parser.add_argument('--with_moving_average', type=bool, default=False,
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
        parser.add_argument('--max_sent_length', type=int, default = 30,
                            help='Maximum number of words within each sentence.')
        parser.add_argument('--aggregation_layer_num', type=int, default=1,
                            help='Number of LSTM layers for aggregation layer.')
        parser.add_argument('--context_layer_num', type=int, default=1,
                            help='Number of LSTM layers for context representation layer.')

        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['data_test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))
cfg = Configs()


import numpy as np
import tensorflow as tf
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


def my_lstm_layer(input_reps, lstm_dim, input_lengths=None,
                  scope_name=None, reuse=False, is_training=True,
                  dropout_rate=0.2, use_cudnn=False):
    '''
    :param inputs: [batch_size, seq_len, feature_dim]
    :param lstm_dim:
    :param scope_name:
    :param reuse:
    :param is_training:
    :param dropout_rate:
    :return:
    '''
    # def state_shape():
    #     return ([2, cfg.batch_size, lstm_dim],
    #         [2, cfg.batch_size, lstm_dim])
    #
    # def _zero_state():
    #     res = []
    #     for sp in state_shape():
    #         res.append(array_ops.zeros(sp, dtype=dtypes.float32))
    #     return tuple(res)

    input_reps = dropout_layer(input_reps, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name, reuse=reuse):
        context_lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
        context_lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(lstm_dim)
        if is_training:
            context_lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(
                context_lstm_cell_fw, output_keep_prob=(1 - dropout_rate))
            context_lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(
                context_lstm_cell_bw, output_keep_prob=(1 - dropout_rate))
        context_lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_fw])
        context_lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([context_lstm_cell_bw])

        (f_rep, b_rep), _ = tf.nn.bidirectional_dynamic_rnn(
            context_lstm_cell_fw, context_lstm_cell_bw, input_reps, dtype=tf.float32,
            sequence_length=input_lengths)  # [batch_size, question_len, context_lstm_dim]
        outputs = tf.concat(axis=2, values=[f_rep, b_rep])
    return (f_rep, b_rep, outputs)


def dropout_layer(input_reps, dropout_rate, is_training=True):
    if is_training:
        output_repr = tf.nn.dropout(input_reps, (1 - dropout_rate))
    else:
        output_repr = input_reps
    return output_repr


def cosine_distance1(y1,y2, cosine_norm=True, eps=1e-6):
    # cosine_norm = True
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    if not cosine_norm:
        return tf.tanh(cosine_numerator)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm


def euclidean_distance(y1, y2, eps=1e-6):
    distance = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1 - y2), axis=-1), eps))
    return distance


def cross_entropy(logits, truth, mask=None):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]
    if mask is not None: logits = tf.multiply(logits, mask)
    xdev = tf.subtract(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.subtract(xdev, tf.expand_dims(tf.log(tf.reduce_sum(tf.exp(xdev),-1)),-1))
    result = tf.multiply(truth, log_predictions) # [batch_size, passage_len]
    if mask is not None: result = tf.multiply(result, mask) # [batch_size, passage_len]
    return tf.multiply(-1.0,tf.reduce_sum(result, -1)) # [batch_size]


def projection_layer(in_val, input_size, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, input_size])
    with tf.variable_scope(scope or "projection_layer"):
        full_w = tf.get_variable("full_w", [input_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        outputs = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs # [batch_size, passage_len, output_size]


def highway_layer(in_val, output_size, activation_func=tf.tanh, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
#     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size], dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size], dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size], dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = activation_func(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = tf.add(tf.multiply(trans, gate), tf.multiply(in_val, tf.subtract(1.0, gate)), "y")
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


def multi_highway_layer(in_val, output_size, num_layers,
                        activation_func=tf.tanh, scope_name=None, reuse=False):
    with tf.variable_scope(scope_name, reuse=reuse):
        for i in range(num_layers):
            cur_scope_name = scope_name + "-{}".format(i)
            in_val = highway_layer(
                in_val, output_size, activation_func=activation_func, scope=cur_scope_name)
    return in_val


def collect_representation(representation, positions):
    # representation: [batch_size, node_num, feature_dim]
    # positions: [batch_size, neigh_num]
    return collect_probs(representation, positions)


def collect_final_step_of_lstm(lstm_representation, lengths):
    # lstm_representation: [batch_size, passsage_length, dim]
    # lengths: [batch_size]
    lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

    batch_size = tf.shape(lengths)[0]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    indices = tf.stack((batch_nums, lengths), axis=1) # shape (batch_size, 2)
    result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
    return result # [batch_size, dim]


def collect_probs(probs, positions):
    # probs [batch_size, chunks_size]
    # positions [batch_size, pair_size]
    batch_size = tf.shape(probs)[0]
    pair_size = tf.shape(positions)[1]
    batch_nums = tf.range(0, limit=batch_size) # shape (batch_size)
    batch_nums = tf.reshape(batch_nums, shape=[-1, 1]) # [batch_size, 1]
    batch_nums = tf.tile(batch_nums, multiples=[1, pair_size]) # [batch_size, pair_size]

    indices = tf.stack((batch_nums, positions), axis=2) # shape (batch_size, pair_size, 2)
    pair_probs = tf.gather_nd(probs, indices)
    # pair_probs = tf.reshape(pair_probs, shape=[batch_size, pair_size])
    return pair_probs


def calcuate_attention(in_value_1, in_value_2, feature_dim1, feature_dim2, scope_name='att',
                       att_type='symmetric', att_dim=20, remove_diagnoal=False, mask1=None,
                       mask2=None, is_training=False, dropout_rate=0.2):
    input_shape = tf.shape(in_value_1)
    batch_size = input_shape[0]
    len_1 = input_shape[1]
    len_2 = tf.shape(in_value_2)[1]

    in_value_1 = dropout_layer(in_value_1, dropout_rate, is_training=is_training)
    in_value_2 = dropout_layer(in_value_2, dropout_rate, is_training=is_training)
    with tf.variable_scope(scope_name):
        # calculate attention ==> a: [batch_size, len_1, len_2]
        atten_w1 = tf.get_variable("atten_w1", [feature_dim1, att_dim], dtype=tf.float32)
        if feature_dim1 == feature_dim2: atten_w2 = atten_w1
        else: atten_w2 = tf.get_variable("atten_w2", [feature_dim2, att_dim], dtype=tf.float32)
        # [batch_size*len_1, feature_dim]
        atten_value_1 = tf.matmul(
            tf.reshape(in_value_1, [batch_size * len_1, feature_dim1]), atten_w1)
        atten_value_1 = tf.reshape(atten_value_1, [batch_size, len_1, att_dim])
        # [batch_size*len_2, feature_dim]
        atten_value_2 = tf.matmul(
            tf.reshape(in_value_2, [batch_size * len_2, feature_dim2]), atten_w2)
        atten_value_2 = tf.reshape(atten_value_2, [batch_size, len_2, att_dim])

        if att_type == 'additive':
            atten_b = tf.get_variable("atten_b", [att_dim], dtype=tf.float32)
            atten_v = tf.get_variable("atten_v", [1, att_dim], dtype=tf.float32)
            # [batch_size, len_1, 'x', feature_dim]
            atten_value_1 = tf.expand_dims(atten_value_1, axis=2, name="atten_value_1")
            # [batch_size, 'x', len_2, feature_dim]
            atten_value_2 = tf.expand_dims(atten_value_2, axis=1, name="atten_value_2")
            # + tf.expand_dims(tf.expand_dims(tf.expand_dims(atten_b, axis=0), axis=0), axis=0)
            atten_value = atten_value_1 + atten_value_2
            atten_value = nn_ops.bias_add(atten_value, atten_b)
            # [batch_size, len_1, len_2, feature_dim]
            atten_value = tf.tanh(atten_value)
            # tf.expand_dims(atten_v, axis=0) # [batch_size*len_1*len_2, feature_dim]
            atten_value = tf.reshape(atten_value, [-1, att_dim]) * atten_v
            atten_value = tf.reduce_sum(atten_value, axis=-1)
            atten_value = tf.reshape(atten_value, [batch_size, len_1, len_2])
        else:
            atten_value_1 = tf.tanh(atten_value_1)
            # atten_value_1 = tf.nn.relu(atten_value_1)
            atten_value_2 = tf.tanh(atten_value_2)
            # atten_value_2 = tf.nn.relu(atten_value_2)
            diagnoal_params = tf.get_variable("diagnoal_params", [1, 1, att_dim], dtype=tf.float32)
            atten_value_1 = atten_value_1 * diagnoal_params
            # [batch_size, len_1, len_2]
            atten_value = tf.matmul(atten_value_1, atten_value_2, transpose_b=True)

        # normalize
        if remove_diagnoal:
            diagnoal = tf.ones([len_1], tf.float32)  # [len1]
            diagnoal = 1.0 - tf.diag(diagnoal)  # [len1, len1]
            diagnoal = tf.expand_dims(diagnoal, axis=0)  # ['x', len1, len1]
            atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
        atten_value = tf.nn.softmax(atten_value, name='atten_value')  # [batch_size, len_1, len_2]
        if remove_diagnoal: atten_value = atten_value * diagnoal
        if mask1 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask1, axis=-1))
        if mask2 is not None: atten_value = tf.multiply(atten_value, tf.expand_dims(mask2, axis=1))
    return atten_value

def weighted_sum(atten_scores, in_values):
    '''

    :param atten_scores: # [batch_size, len1, len2]
    :param in_values: [batch_size, len2, dim]
    :return:
    '''
    return tf.matmul(atten_scores, in_values)


def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    # [batch_size, 1, question_len, dim]
    in_question_repres_tmp = tf.expand_dims(in_question_repres, 1)
    # [batch_size, passage_len, 1, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres, 2)
    # [batch_size, passage_len, question_len]
    relevancy_matrix = cosine_distance1(in_question_repres_tmp,in_passage_repres_tmp)
    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    if question_mask is not None:
        relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix, tf.expand_dims(passage_mask, 2))
    return relevancy_matrix


def compute_gradients(tensor, var_list):
  grads = tf.gradients(tensor, var_list)
  return [grad if grad is not None else tf.zeros_like(var) for var, grad in zip(var_list, grads)]


eps = 1e-6


def cosine_distance2(y1, y2):
    # y1 [....,a, 1, d]
    # y2 [....,1, b, d]
    cosine_numerator = tf.reduce_sum(tf.multiply(y1, y2), axis=-1)
    y1_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y1), axis=-1), eps))
    y2_norm = tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(y2), axis=-1), eps))
    return cosine_numerator / y1_norm / y2_norm


def cal_relevancy_matrix(in_question_repres, in_passage_repres):
    in_question_repres_tmp = tf.expand_dims(in_question_repres,
                                            1)  # [batch_size, 1, question_len, dim]
    in_passage_repres_tmp = tf.expand_dims(in_passage_repres,
                                           2)  # [batch_size, passage_len, 1, dim]
    relevancy_matrix = cosine_distance2(in_question_repres_tmp,
                                       in_passage_repres_tmp)  # [batch_size, passage_len, question_len]
    return relevancy_matrix


def mask_relevancy_matrix(relevancy_matrix, question_mask, passage_mask):
    # relevancy_matrix: [batch_size, passage_len, question_len]
    # question_mask: [batch_size, question_len]
    # passage_mask: [batch_size, passsage_len]
    relevancy_matrix = tf.multiply(relevancy_matrix,
                                   tf.expand_dims(question_mask, 1))
    relevancy_matrix = tf.multiply(relevancy_matrix,
                                   tf.expand_dims(passage_mask, 2))
    return relevancy_matrix


def multi_perspective_expand_for_3D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor,
                               axis=2)  # [batch_size, passage_len, 'x', dim]
    decompose_params = tf.expand_dims(tf.expand_dims(decompose_params, axis=0),
                                      axis=0)  # [1, 1, decompse_dim, dim]
    return tf.multiply(in_tensor,
                       decompose_params)  # [batch_size, passage_len, decompse_dim, dim]


def multi_perspective_expand_for_2D(in_tensor, decompose_params):
    in_tensor = tf.expand_dims(in_tensor, axis=1)  # [batch_size, 'x', dim]
    decompose_params = tf.expand_dims(decompose_params,
                                      axis=0)  # [1, decompse_dim, dim]
    return tf.multiply(in_tensor,
                       decompose_params)  # [batch_size, decompse_dim, dim]


def cal_maxpooling_matching(passage_rep, question_rep, decompose_params):
    # passage_representation: [batch_size, passage_len, dim]
    # qusetion_representation: [batch_size, question_len, dim]
    # decompose_params: [decompose_dim, dim]

    def singel_instance(x):
        p = x[0]
        q = x[1]
        # p: [pasasge_len, dim], q: [question_len, dim]
        p = multi_perspective_expand_for_2D(p,
                                            decompose_params)  # [pasasge_len, decompose_dim, dim]
        q = multi_perspective_expand_for_2D(q,
                                            decompose_params)  # [question_len, decompose_dim, dim]
        p = tf.expand_dims(p, 1)  # [pasasge_len, 1, decompose_dim, dim]
        q = tf.expand_dims(q, 0)  # [1, question_len, decompose_dim, dim]
        return cosine_distance2(p, q)  # [passage_len, question_len, decompose]

    elems = (passage_rep, question_rep)
    matching_matrix = tf.map_fn(singel_instance, elems,
                                dtype=tf.float32)  # [batch_size, passage_len, question_len, decompse_dim]
    return tf.concat(axis=2, values=[tf.reduce_max(matching_matrix, axis=2),
                                     tf.reduce_mean(matching_matrix,
                                                    axis=2)])  # [batch_size, passage_len, 2*decompse_dim]


def cross_entropy(logits, truth, mask):
    # logits: [batch_size, passage_len]
    # truth: [batch_size, passage_len]
    # mask: [batch_size, passage_len]

    #     xdev = x - x.max()
    #     return xdev - T.log(T.sum(T.exp(xdev)))
    logits = tf.multiply(logits, mask)
    xdev = tf.sub(logits, tf.expand_dims(tf.reduce_max(logits, 1), -1))
    log_predictions = tf.sub(xdev, tf.expand_dims(
        tf.log(tf.reduce_sum(tf.exp(xdev), -1)), -1))
    #     return -T.sum(targets * log_predictions)
    result = tf.multiply(tf.multiply(truth, log_predictions),
                         mask)  # [batch_size, passage_len]
    return tf.multiply(-1.0, tf.reduce_sum(result, -1))  # [batch_size]


def highway_layer(in_val, output_size, scope=None):
    # in_val: [batch_size, passage_len, dim]
    input_shape = tf.shape(in_val)
    batch_size = input_shape[0]
    passage_len = input_shape[1]
    #     feat_dim = input_shape[2]
    in_val = tf.reshape(in_val, [batch_size * passage_len, output_size])
    with tf.variable_scope(scope or "highway_layer"):
        highway_w = tf.get_variable("highway_w", [output_size, output_size],
                                    dtype=tf.float32)
        highway_b = tf.get_variable("highway_b", [output_size],
                                    dtype=tf.float32)
        full_w = tf.get_variable("full_w", [output_size, output_size],
                                 dtype=tf.float32)
        full_b = tf.get_variable("full_b", [output_size], dtype=tf.float32)
        trans = tf.nn.tanh(tf.nn.xw_plus_b(in_val, full_w, full_b))
        gate = tf.nn.sigmoid(tf.nn.xw_plus_b(in_val, highway_w, highway_b))
        outputs = trans * gate + in_val * (1.0 - gate)
    outputs = tf.reshape(outputs, [batch_size, passage_len, output_size])
    return outputs


def multi_highway_layer(in_val, output_size, num_layers, scope=None):
    scope_name = 'highway_layer'
    if scope is not None: scope_name = scope
    for i in range(num_layers):
        cur_scope_name = scope_name + "-{}".format(i)
        in_val = highway_layer(in_val, output_size, scope=cur_scope_name)
    return in_val


def cal_max_question_representation(question_representation, atten_scores):
    atten_positions = tf.argmax(atten_scores, axis=2,
                                output_type=tf.int32)  # [batch_size, passage_len]
    max_question_reps = collect_representation(
        question_representation, atten_positions)
    return max_question_reps


def multi_perspective_match(feature_dim, repres1, repres2, is_training=True,
                            dropout_rate=0.2,
                            options=None, scope_name='mp-match', reuse=False):
    '''
        :param repres1: [batch_size, len, feature_dim]
        :param repres2: [batch_size, len, feature_dim]
        :return:
    '''
    input_shape = tf.shape(repres1)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    matching_result = []
    with tf.variable_scope(scope_name, reuse=reuse):
        match_dim = 0
        if options.with_cosine:
            cosine_value = cosine_distance1(repres1, repres2,
                                                       cosine_norm=False)
            cosine_value = tf.reshape(cosine_value, [batch_size, seq_length, 1])
            matching_result.append(cosine_value)
            match_dim += 1

        if options.with_mp_cosine:
            mp_cosine_params = tf.get_variable("mp_cosine",
                                               shape=[options.cosine_MP_dim,
                                                      feature_dim],
                                               dtype=tf.float32)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            mp_cosine_params = tf.expand_dims(mp_cosine_params, axis=0)
            repres1_flat = tf.expand_dims(repres1, axis=2)
            repres2_flat = tf.expand_dims(repres2, axis=2)
            mp_cosine_matching = cosine_distance1(
                tf.multiply(repres1_flat, mp_cosine_params),
                repres2_flat, cosine_norm=False)
            matching_result.append(mp_cosine_matching)
            match_dim += options.cosine_MP_dim

    matching_result = tf.concat(axis=2, values=matching_result)
    return (matching_result, match_dim)


def match_passage_with_question(passage_reps, question_reps, passage_mask,
                                question_mask, passage_lengths,
                                question_lengths,
                                context_lstm_dim, scope=None,
                                with_full_match=True, with_maxpool_match=True,
                                with_attentive_match=True,
                                with_max_attentive_match=True,
                                is_training=True, options=None, dropout_rate=0,
                                forward=True):
    passage_reps = tf.multiply(passage_reps, tf.expand_dims(passage_mask, -1))
    question_reps = tf.multiply(question_reps,
                                tf.expand_dims(question_mask, -1))
    all_question_aware_representatins = []
    dim = 0
    with tf.variable_scope(scope or "match_passage_with_question"):
        relevancy_matrix = cal_relevancy_matrix(question_reps, passage_reps)
        relevancy_matrix = mask_relevancy_matrix(relevancy_matrix,
                                                 question_mask, passage_mask)
        # relevancy_matrix = calcuate_attention(passage_reps, question_reps, context_lstm_dim, context_lstm_dim,
        #             scope_name="fw_attention", att_type=options.att_type, att_dim=options.att_dim,
        #             remove_diagnoal=False, mask1=passage_mask, mask2=question_mask, is_training=is_training, dropout_rate=dropout_rate)

        all_question_aware_representatins.append(
            tf.reduce_max(relevancy_matrix, axis=2, keep_dims=True))
        all_question_aware_representatins.append(
            tf.reduce_mean(relevancy_matrix, axis=2, keep_dims=True))
        dim += 2
        if with_full_match:
            if forward:
                question_full_rep = collect_final_step_of_lstm(
                    question_reps, question_lengths - 1)
            else:
                question_full_rep = question_reps[:, 0, :]

            passage_len = tf.shape(passage_reps)[1]
            question_full_rep = tf.expand_dims(question_full_rep, axis=1)
            question_full_rep = tf.tile(question_full_rep, [1, passage_len,
                                                            1])  # [batch_size, pasasge_len, feature_dim]

            (attentive_rep, match_dim) = multi_perspective_match(
                context_lstm_dim,
                passage_reps, question_full_rep, is_training=is_training,
                dropout_rate=options.dropout_rate,
                options=options, scope_name='mp-match-full-match')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_maxpool_match:
            maxpooling_decomp_params = tf.get_variable(
                "maxpooling_matching_decomp",
                shape=[options.cosine_MP_dim, context_lstm_dim],
                dtype=tf.float32)
            maxpooling_rep = cal_maxpooling_matching(passage_reps,
                                                     question_reps,
                                                     maxpooling_decomp_params)
            all_question_aware_representatins.append(maxpooling_rep)
            dim += 2 * options.cosine_MP_dim

        if with_attentive_match:
            atten_scores = calcuate_attention(passage_reps,
                                                          question_reps,
                                                          context_lstm_dim,
                                                          context_lstm_dim,
                                                          scope_name="attention",
                                                          att_type=options.att_type,
                                                          att_dim=options.att_dim,
                                                          remove_diagnoal=False,
                                                          mask1=passage_mask,
                                                          mask2=question_mask,
                                                          is_training=is_training,
                                                          dropout_rate=dropout_rate)
            att_question_contexts = tf.matmul(atten_scores, question_reps)
            (attentive_rep, match_dim) = multi_perspective_match(
                context_lstm_dim,
                passage_reps, att_question_contexts, is_training=is_training,
                dropout_rate=options.dropout_rate,
                options=options, scope_name='mp-match-att_question')
            all_question_aware_representatins.append(attentive_rep)
            dim += match_dim

        if with_max_attentive_match:
            max_att = cal_max_question_representation(question_reps,
                                                      relevancy_matrix)
            (max_attentive_rep, match_dim) = multi_perspective_match(
                context_lstm_dim,
                passage_reps, max_att, is_training=is_training,
                dropout_rate=options.dropout_rate,
                options=options, scope_name='mp-match-max-att')
            all_question_aware_representatins.append(max_attentive_rep)
            dim += match_dim

        all_question_aware_representatins = tf.concat(axis=2,
                                                      values=all_question_aware_representatins)
    return (all_question_aware_representatins, dim)


def bilateral_match_func(in_question_repres, in_passage_repres,
                         question_lengths, passage_lengths, question_mask,
                         passage_mask, input_dim, is_training, options=None):
    question_aware_representatins = []
    question_aware_dim = 0
    passage_aware_representatins = []
    passage_aware_dim = 0

    # ====word level matching======
    (match_reps, match_dim) = match_passage_with_question(in_passage_repres,
                                                          in_question_repres,
                                                          passage_mask,
                                                          question_mask,
                                                          passage_lengths,
                                                          question_lengths,
                                                          input_dim,
                                                          scope="word_match_forward",
                                                          with_full_match=False,
                                                          with_maxpool_match=options.with_maxpool_match,
                                                          with_attentive_match=options.with_attentive_match,
                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                          is_training=is_training,
                                                          options=options,
                                                          dropout_rate=options.dropout_rate,
                                                          forward=True)
    question_aware_representatins.append(match_reps)
    question_aware_dim += match_dim

    (match_reps, match_dim) = match_passage_with_question(in_question_repres,
                                                          in_passage_repres,
                                                          question_mask,
                                                          passage_mask,
                                                          question_lengths,
                                                          passage_lengths,
                                                          input_dim,
                                                          scope="word_match_backward",
                                                          with_full_match=False,
                                                          with_maxpool_match=options.with_maxpool_match,
                                                          with_attentive_match=options.with_attentive_match,
                                                          with_max_attentive_match=options.with_max_attentive_match,
                                                          is_training=is_training,
                                                          options=options,
                                                          dropout_rate=options.dropout_rate,
                                                          forward=False)
    passage_aware_representatins.append(match_reps)
    passage_aware_dim += match_dim

    with tf.variable_scope('context_MP_matching'):
        for i in range(
                options.context_layer_num):  # support multiple context layer
            with tf.variable_scope('layer-{}'.format(i)):
                # contextual lstm for both passage and question
                in_question_repres = tf.multiply(in_question_repres,
                                                 tf.expand_dims(question_mask,
                                                                axis=-1))
                in_passage_repres = tf.multiply(in_passage_repres,
                                                tf.expand_dims(passage_mask,
                                                               axis=-1))
                (question_context_representation_fw,
                 question_context_representation_bw,
                 in_question_repres) = my_lstm_layer(
                    in_question_repres, options.context_lstm_dim,
                    input_lengths=question_lengths,
                    scope_name="context_represent",
                    reuse=False, is_training=is_training,
                    dropout_rate=options.dropout_rate,
                    use_cudnn=options.use_cudnn)
                (passage_context_representation_fw,
                 passage_context_representation_bw,
                 in_passage_repres) = my_lstm_layer(
                    in_passage_repres, options.context_lstm_dim,
                    input_lengths=passage_lengths,
                    scope_name="context_represent",
                    reuse=True, is_training=is_training,
                    dropout_rate=options.dropout_rate,
                    use_cudnn=options.use_cudnn)

                # Multi-perspective matching
                with tf.variable_scope('left_MP_matching'):
                    (match_reps, match_dim) = match_passage_with_question(
                        passage_context_representation_fw,
                        question_context_representation_fw, passage_mask,
                        question_mask, passage_lengths,
                        question_lengths, options.context_lstm_dim,
                        scope="forward_match",
                        with_full_match=options.with_full_match,
                        with_maxpool_match=options.with_maxpool_match,
                        with_attentive_match=options.with_attentive_match,
                        with_max_attentive_match=options.with_max_attentive_match,
                        is_training=is_training, options=options,
                        dropout_rate=options.dropout_rate, forward=True)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim
                    (match_reps, match_dim) = match_passage_with_question(
                        passage_context_representation_bw,
                        question_context_representation_bw, passage_mask,
                        question_mask, passage_lengths,
                        question_lengths, options.context_lstm_dim,
                        scope="backward_match",
                        with_full_match=options.with_full_match,
                        with_maxpool_match=options.with_maxpool_match,
                        with_attentive_match=options.with_attentive_match,
                        with_max_attentive_match=options.with_max_attentive_match,
                        is_training=is_training, options=options,
                        dropout_rate=options.dropout_rate, forward=False)
                    question_aware_representatins.append(match_reps)
                    question_aware_dim += match_dim

                with tf.variable_scope('right_MP_matching'):
                    (match_reps, match_dim) = match_passage_with_question(
                        question_context_representation_fw,
                        passage_context_representation_fw, question_mask,
                        passage_mask, question_lengths,
                        passage_lengths, options.context_lstm_dim,
                        scope="forward_match",
                        with_full_match=options.with_full_match,
                        with_maxpool_match=options.with_maxpool_match,
                        with_attentive_match=options.with_attentive_match,
                        with_max_attentive_match=options.with_max_attentive_match,
                        is_training=is_training, options=options,
                        dropout_rate=options.dropout_rate, forward=True)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim
                    (match_reps, match_dim) = match_passage_with_question(
                        question_context_representation_bw,
                        passage_context_representation_bw, question_mask,
                        passage_mask, question_lengths,
                        passage_lengths, options.context_lstm_dim,
                        scope="backward_match",
                        with_full_match=options.with_full_match,
                        with_maxpool_match=options.with_maxpool_match,
                        with_attentive_match=options.with_attentive_match,
                        with_max_attentive_match=options.with_max_attentive_match,
                        is_training=is_training, options=options,
                        dropout_rate=options.dropout_rate, forward=False)
                    passage_aware_representatins.append(match_reps)
                    passage_aware_dim += match_dim

    question_aware_representatins = tf.concat(axis=2,
                                              values=question_aware_representatins)  # [batch_size, passage_len, question_aware_dim]
    passage_aware_representatins = tf.concat(axis=2,
                                             values=passage_aware_representatins)  # [batch_size, question_len, question_aware_dim]

    if is_training:
        question_aware_representatins = tf.nn.dropout(
            question_aware_representatins, (1 - options.dropout_rate))
        passage_aware_representatins = tf.nn.dropout(
            passage_aware_representatins, (1 - options.dropout_rate))

    # ======Highway layer======
    if options.with_match_highway:
        with tf.variable_scope("left_matching_highway"):
            question_aware_representatins = multi_highway_layer(
                question_aware_representatins, question_aware_dim,
                options.highway_layer_num)
        with tf.variable_scope("right_matching_highway"):
            passage_aware_representatins = multi_highway_layer(
                passage_aware_representatins, passage_aware_dim,
                options.highway_layer_num)

    # ========Aggregation Layer======
    aggregation_representation = []
    aggregation_dim = 0

    qa_aggregation_input = question_aware_representatins
    pa_aggregation_input = passage_aware_representatins
    with tf.variable_scope('aggregation_layer'):
        for i in range(
                options.aggregation_layer_num):  # support multiple aggregation layer
            qa_aggregation_input = tf.multiply(qa_aggregation_input,
                                               tf.expand_dims(passage_mask,
                                                              axis=-1))
            (fw_rep, bw_rep,
             cur_aggregation_representation) = my_lstm_layer(
                qa_aggregation_input, options.aggregation_lstm_dim,
                input_lengths=passage_lengths,
                scope_name='left_layer-{}'.format(i),
                reuse=False, is_training=is_training,
                dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            fw_rep = collect_final_step_of_lstm(fw_rep,
                                                            passage_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2 * options.aggregation_lstm_dim
            qa_aggregation_input = cur_aggregation_representation  # [batch_size, passage_len, 2*aggregation_lstm_dim]

            pa_aggregation_input = tf.multiply(pa_aggregation_input,
                                               tf.expand_dims(question_mask,
                                                              axis=-1))
            (fw_rep, bw_rep,
             cur_aggregation_representation) = my_lstm_layer(
                pa_aggregation_input, options.aggregation_lstm_dim,
                input_lengths=question_lengths,
                scope_name='right_layer-{}'.format(i),
                reuse=False, is_training=is_training,
                dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
            fw_rep = collect_final_step_of_lstm(fw_rep,
                                                            question_lengths - 1)
            bw_rep = bw_rep[:, 0, :]
            aggregation_representation.append(fw_rep)
            aggregation_representation.append(bw_rep)
            aggregation_dim += 2 * options.aggregation_lstm_dim
            pa_aggregation_input = cur_aggregation_representation  # [batch_size, passage_len, 2*aggregation_lstm_dim]

    aggregation_representation = tf.concat(axis=1,
                                           values=aggregation_representation)  # [batch_size, aggregation_dim]

    # ======Highway layer======
    if options.with_aggregation_highway:
        with tf.variable_scope("aggregation_highway"):
            agg_shape = tf.shape(aggregation_representation)
            batch_size = agg_shape[0]
            aggregation_representation = tf.reshape(aggregation_representation,
                                                    [1, batch_size,
                                                     aggregation_dim])
            aggregation_representation = multi_highway_layer(
                aggregation_representation, aggregation_dim,
                options.highway_layer_num)
            aggregation_representation = tf.reshape(aggregation_representation,
                                                    [batch_size,
                                                     aggregation_dim])

    return (aggregation_representation, aggregation_dim)


class SentenceMatchModelGraph(object):
    def __init__(self, num_classes, word_vocab=None, char_vocab=None,
                 is_training=True, options=None, global_step=None):
        self.options = options
        self.create_placeholders()
        self.word_embedding = None
        self.create_model_graph(num_classes, word_vocab, char_vocab,
                                is_training, global_step=global_step)

    def create_placeholders(self):
        self.question_lengths = tf.placeholder(tf.int32, [None])
        self.passage_lengths = tf.placeholder(tf.int32, [None])
        self.truth = tf.placeholder(tf.int32, [None])  # [batch_size]
        self.in_question_words = tf.placeholder(tf.int32, [None,
                                                           None])  # [batch_size, question_len]
        self.in_passage_words = tf.placeholder(tf.int32, [None,
                                                          None])  # [batch_size, passage_len]

        if self.options.with_char:
            self.question_char_lengths = tf.placeholder(tf.int32, [None,
                                                                   None])  # [batch_size, question_len]
            self.passage_char_lengths = tf.placeholder(tf.int32, [None,
                                                                  None])  # [batch_size, passage_len]
            self.in_question_chars = tf.placeholder(tf.int32, [None, None,
                                                               None])  # [batch_size, question_len, q_char_len]
            self.in_passage_chars = tf.placeholder(tf.int32, [None, None,
                                                              None])  # [batch_size, passage_len, p_char_len]

    def create_feed_dict(self, question_lengths, passage_lengths,
                         in_question_words, in_passage_words, label_truth):
        feed_dict = {
            self.question_lengths: question_lengths,
            self.passage_lengths: passage_lengths,
            self.in_question_words: in_question_words,
            self.in_passage_words: in_passage_words,
            self.truth: label_truth,
        }
        return feed_dict

    def create_model_graph(self, num_classes, word_vocab=None, char_vocab=None,
                           is_training=True, global_step=None):
        options = self.options
        # ======word representation layer======
        in_question_repres = []
        in_passage_repres = []
        input_dim = 0
        if word_vocab is not None:
            word_vec_trainable = True

            if options.fix_word_vec:
                word_vec_trainable = False

            self.word_embedding = tf.get_variable(
                "word_embedding", trainable=word_vec_trainable,
                initializer=tf.constant(
                    word_vocab), dtype=tf.float32)

            in_question_word_repres = tf.nn.embedding_lookup(
                self.word_embedding,
                self.in_question_words)  # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding,
                                                            self.in_passage_words)  # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += len(word_vocab[0])

        if options.with_char and char_vocab is not None:
            input_shape = tf.shape(self.in_question_chars)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            q_char_len = input_shape[2]
            input_shape = tf.shape(self.in_passage_chars)
            passage_len = input_shape[1]
            p_char_len = input_shape[2]
            char_dim = char_vocab.word_dim
            self.char_embedding = tf.get_variable("char_embedding",
                                                  initializer=tf.constant(
                                                      char_vocab.word_vecs),
                                                  dtype=tf.float32)

            in_question_char_repres = tf.nn.embedding_lookup(
                self.char_embedding,
                self.in_question_chars)  # [batch_size, question_len, q_char_len, char_dim]
            in_question_char_repres = tf.reshape(in_question_char_repres,
                                                 shape=[-1, q_char_len,
                                                        char_dim])
            question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
            quesiton_char_mask = tf.sequence_mask(question_char_lengths,
                                                  q_char_len,
                                                  dtype=tf.float32)  # [batch_size*question_len, q_char_len]
            in_question_char_repres = tf.multiply(in_question_char_repres,
                                                  tf.expand_dims(
                                                      quesiton_char_mask,
                                                      axis=-1))

            in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding,
                                                            self.in_passage_chars)  # [batch_size, passage_len, p_char_len, char_dim]
            in_passage_char_repres = tf.reshape(in_passage_char_repres,
                                                shape=[-1, p_char_len,
                                                       char_dim])
            passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
            passage_char_mask = tf.sequence_mask(passage_char_lengths,
                                                 p_char_len,
                                                 dtype=tf.float32)  # [batch_size*passage_len, p_char_len]
            in_passage_char_repres = tf.multiply(in_passage_char_repres,
                                                 tf.expand_dims(
                                                     passage_char_mask,
                                                     axis=-1))

            (question_char_outputs_fw, question_char_outputs_bw,
             _) = my_lstm_layer(in_question_char_repres,
                                            options.char_lstm_dim,
                                            input_lengths=question_char_lengths,
                                            scope_name="char_lstm", reuse=False,
                                            is_training=is_training,
                                            dropout_rate=options.dropout_rate,
                                            use_cudnn=options.use_cudnn)
            question_char_outputs_fw = collect_final_step_of_lstm(
                question_char_outputs_fw, question_char_lengths - 1)
            question_char_outputs_bw = question_char_outputs_bw[:, 0, :]
            question_char_outputs = tf.concat(axis=1,
                                              values=[question_char_outputs_fw,
                                                      question_char_outputs_bw])
            question_char_outputs = tf.reshape(question_char_outputs,
                                               [batch_size, question_len,
                                                2 * options.char_lstm_dim])

            (passage_char_outputs_fw, passage_char_outputs_bw,
             _) = my_lstm_layer(in_passage_char_repres,
                                            options.char_lstm_dim,
                                            input_lengths=passage_char_lengths,
                                            scope_name="char_lstm", reuse=True,
                                            is_training=is_training,
                                            dropout_rate=options.dropout_rate,
                                            use_cudnn=options.use_cudnn)
            passage_char_outputs_fw = collect_final_step_of_lstm(
                passage_char_outputs_fw, passage_char_lengths - 1)
            passage_char_outputs_bw = passage_char_outputs_bw[:, 0, :]
            passage_char_outputs = tf.concat(axis=1,
                                             values=[passage_char_outputs_fw,
                                                     passage_char_outputs_bw])
            passage_char_outputs = tf.reshape(passage_char_outputs,
                                              [batch_size, passage_len,
                                               2 * options.char_lstm_dim])

            in_question_repres.append(question_char_outputs)
            in_passage_repres.append(passage_char_outputs)

            input_dim += 2 * options.char_lstm_dim

        in_question_repres = tf.concat(axis=2,
                                       values=in_question_repres)  # [batch_size, question_len, dim]
        in_passage_repres = tf.concat(axis=2,
                                      values=in_passage_repres)  # [batch_size, passage_len, dim]

        if is_training:
            in_question_repres = tf.nn.dropout(in_question_repres,
                                               (1 - options.dropout_rate))
            in_passage_repres = tf.nn.dropout(in_passage_repres,
                                              (1 - options.dropout_rate))

        mask = tf.sequence_mask(self.passage_lengths, passage_len,
                                dtype=tf.float32)  # [batch_size, passage_len]
        question_mask = tf.sequence_mask(self.question_lengths, question_len,
                                         dtype=tf.float32)  # [batch_size, question_len]

        # ======Highway layer======
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = multi_highway_layer(
                    in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = multi_highway_layer(
                    in_passage_repres, input_dim, options.highway_layer_num)

        # in_question_repres = tf.multiply(in_question_repres, tf.expand_dims(question_mask, axis=-1))
        # in_passage_repres = tf.multiply(in_passage_repres, tf.expand_dims(mask, axis=-1))

        # ========Bilateral Matching=====
        (match_representation, match_dim) = bilateral_match_func(
            in_question_repres, in_passage_repres,
            self.question_lengths, self.passage_lengths, question_mask, mask,
            input_dim, is_training, options=options)

        # ========Prediction Layer=========
        # match_dim = 4 * self.options.aggregation_lstm_dim
        w_0 = tf.get_variable("w_0", [match_dim, match_dim / 2],
                              dtype=tf.float32)
        b_0 = tf.get_variable("b_0", [match_dim / 2], dtype=tf.float32)
        w_1 = tf.get_variable("w_1", [match_dim / 2, num_classes],
                              dtype=tf.float32)
        b_1 = tf.get_variable("b_1", [num_classes], dtype=tf.float32)

        # if is_training: match_representation = tf.nn.dropout(match_representation, (1 - options.dropout_rate))
        logits = tf.matmul(match_representation, w_0) + b_0
        logits = tf.tanh(logits)
        if is_training: logits = tf.nn.dropout(logits,
                                               (1 - options.dropout_rate))
        logits = tf.matmul(logits, w_1) + b_1

        self.prob = tf.nn.softmax(logits)

        gold_matrix = tf.one_hot(self.truth, num_classes, dtype=tf.float32)
        self.loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                    labels=gold_matrix))

        correct = tf.nn.in_top_k(logits, self.truth, 1)
        self.eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        self.predictions = tf.argmax(self.prob, 1)

        if not is_training: return

        tvars = tf.trainable_variables()
        if self.options.lambda_l2 > 0.0:
            l2_loss = tf.add_n(
                [tf.nn.l2_loss(v) for v in tvars if v.get_shape().ndims > 1])
            self.loss += self.options.lambda_l2 * l2_loss

        if self.options.optimize_type == 'adadelta':
            optimizer = tf.train.AdadeltaOptimizer(
                learning_rate=self.options.learning_rate)
        elif self.options.optimize_type == 'adam':
            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.options.learning_rate)

        grads = compute_gradients(self.loss, tvars)
        grads, _ = tf.clip_by_global_norm(grads, self.options.grad_clipper)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=global_step)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        if self.options.with_moving_average:
            # Track the moving averages of all trainable variables.
            MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
            variable_averages = tf.train.ExponentialMovingAverage(
                MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(
                tf.trainable_variables())
            train_ops = [self.train_op, variables_averages_op]
            self.train_op = tf.group(*train_ops)


from tensorflow.contrib import learn


def tokenizer_word(iterator):
    for value in iterator:
        yield str(value).split()

class TfVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 vocabulary=None):
        tokenizer_fn = tokenizer_word
        self.sup = super(TfVocabularyProcessor, self)
        self.sup.__init__(max_document_length, min_frequency, vocabulary,
                          tokenizer_fn)

    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids

import jieba
from random import random
from sklearn.model_selection import train_test_split
UNKNOWN = 0  # 表示找不到的词
PAD = 1  # 表示向量的扩展位

class DataHelper(object):

    def __init__(self,
                 data=None,
                 max_document_length=50,
                 word_index_dic=None,
                 by_word_index_dic=False):

        self.data = data
        self.max_document_length = max_document_length
        self.by_word_index_dic = by_word_index_dic
        if self.by_word_index_dic:
            if not word_index_dic:
                raise Exception("请传入有效的 word_index_dic 或者 "
                                "初始化 by_trained_word2vec 为False")
            self.word_index_dic = word_index_dic
        else:
            x_1 = [" ".join(x[0]) for x in self.data[0]]
            x_2 = [" ".join(x[1]) for x in self.data[0]]
            documents = np.concatenate((x_1, x_2), axis=0)
            vocab_processor = TfVocabularyProcessor(max_document_length)
            vocab_processor.fit_transform(documents)
            print("Length of loaded vocabulary ={}".format(
                len(vocab_processor.vocabulary_)))
            self.vocab_processor = vocab_processor

    def set_by_trained_word2vec(self, value):
        self.by_word_index_dic = value

    def get_vocab_processor(self):
        return self.vocab_processor

    def documents_transform_and_padding(self, cfg):
        """
        将句子转换成index, unknow 0, 不足最大长度的位用 -1 填充
        """

        x, y = self.data
        x_new = []
        if self.by_word_index_dic and self.word_index_dic:
            for pair in x:
                pair_1, pair_2 = pair
                pair_1 = self.sentence_transform_and_padding(
                    pair_1, self.max_document_length, cfg)
                pair_2 = self.sentence_transform_and_padding(
                    pair_2, self.max_document_length, cfg)

                if random() > 0.5:
                    x_new.append((pair_1, pair_2))
                else:
                    x_new.append((pair_2, pair_1))
            return x_new, y
        else:
            x_1, x_2 = zip(*x)
            x_1 = [" ".join(ele) for ele in x_1]
            x_2 = [" ".join(ele) for ele in x_2]
            x_1 = np.asanyarray(list(self.vocab_processor.transform(x_1)))
            x_2 = np.asanyarray(list(self.vocab_processor.transform(x_2)))
            for i in range(len(y)):
                if random() > 0.5:
                    x_new.append((x_1[i], x_2[i]))
                else:
                    x_new.append((x_2[i], x_1[i]))
            return x_new, y

    def sentence_transform_and_padding(self, words, max_sentence_length, cfg):
        # 外部 word_index, 对句子进行 id map
        if len(words) > max_sentence_length:
            words = words[:max_sentence_length]
        indexes = [self.word_index_dic.get(w, UNKNOWN)
                   for w in words]
        if len(indexes) < max_sentence_length:
            indexes = indexes + [PAD, ] * (max_sentence_length - len(indexes))
        if cfg.use_pinyin:
            pinyins = pinyin(words)
            pinyin_idxs = [self.word_index_dic.get(w[0], UNKNOWN)
                           for w in pinyins]
            if len(pinyin_idxs) < max_sentence_length:
                pinyin_idxs = pinyin_idxs + [PAD, ] * (max_sentence_length - len(pinyin_idxs))
            # indexes.extend(pinyin_idxs)
            indexes = pinyin_idxs
        return indexes

    @staticmethod
    def data_split(data, valid_size=0.1, test_size=0.1):
        # 按照 valid_size, test_size 对数据进行切分
        x, y = data
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size)
        x_train, x_valid, y_train, y_valid = train_test_split(
            x_train, y_train, test_size=valid_size)

        return x_train, y_train, x_valid, y_valid, x_test, y_test

    @staticmethod
    def batch_iter(data, batch_size, num_epochs=1, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        :parameter data:
        :parameter batch_size:
        :parameter num_epochs:
        :parameter shuffle:
        """
        data = np.asarray(data)
        data_size = len(data)
        num_batches_per_epoch = int(len(data) / batch_size) + 1
        if len(data) % batch_size == 0:
            num_batches_per_epoch -= 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

    @staticmethod
    def save_test_data(pairs, path, seg='$$$'):
        """
         保存测试数据到文件中
         :parameter pairs: 三元组
         :parameter path: 文件路径
         :parameter seg: 分隔符

        """

        if not pairs or len(pairs) == 0:
            print("pairs is empty")
        with open(path, 'w') as f:
            for pair in pairs:
                text_1 = " ".join([str(ele) for ele in pair[0]])
                text_2 = " ".join([str(ele) for ele in pair[1]])
                label = str(pair[2])
                f.write(seg.join([text_1, text_2, label]) + '\n')

    @staticmethod
    def load_test_data(path, seg='$$$'):
        """
        加载测试数据路径
        :parameter path: 文件路径
        :parameter seg: 分隔符
        :return 相似文本对和标签
        """

        pairs = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                context = f.readlines()
                for line in context:
                    if line and line.strip():
                        pair = line.strip().split(seg)
                        text_1 = [int(ele) for ele in pair[0].split()]
                        text_2 = [int(ele) for ele in pair[1].split()]
                        label = int(pair[2])
                        pairs.append((text_1, text_2, label))
        return pairs

    @staticmethod
    def load_predict_data(path, seg="$$$"):
        """
        加载测试数据路径
        :parameter path: 文件路径
        :parameter seg:　句子分隔符
        :return 待计算相似度的文本对
        """
        pairs = []
        if os.path.exists(path):
            with open(path, 'r') as f:
                context = f.readlines()
                for line in context:
                    if line and line.strip():
                        pair = line.strip().split(seg)
                        if len(pair) < 3:
                            continue
                        text_1 = list(jieba.cut(pair[1]))
                        text_2 = list(jieba.cut(pair[2]))
                        pairs.append((text_1, text_2))
        return pairs

import json
class Word2vecModel(object):

    def __init__(self, model):
        self.model = model
        self.dim = len(list(self.model.items())[0][1])

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            model = json.load(f)
        return Word2vecModel(model)

    @staticmethod
    def load_glove(path):
        model = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                word = line[0]
                vec = [float(i) for i in line[1:]]
                model[word] = vec
        return Word2vecModel(model)

    def transform(self, word):
        return self.model.get(word, [0.0001] * self.dim)

    def generate_word_id_map(self, dtype):
        # 根据已经训练的word2vec, 生成相应的word_id_map, id_vector_map

        word_id_map = {}
        id_vector_map = [[0.0001] * self.dim, [-0.0001] * self.dim]
        words = sorted(list(self.model.keys()))
        for id, word in enumerate(words):
            word_id_map[word] = id + 2
            id_vector_map.append(self.model[word])
        id_vector_map = np.array(id_vector_map, dtype=dtype)
        return word_id_map, id_vector_map

class Embedding(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.word2vec = self.transform(df3)
        self.word_id_map, self.id_vector_map = self.word2vec.generate_word_id_map(np.float32)

    def transform(self,df):
        model={}
        for i in np.array(df):
            model[i[0]]=[float(j) for j in i[1:]]
        return Word2vecModel(model)

    def generate_sentence_token_ind(self, data):
        data = list(np.array(data)[:, 1:4])
        method = sentences2char
        if self.cfg.feature_type == 'word':
            method = sentences2word
        data = [((method(x[0]), method(x[1])), int(x[2])) for x in data]
        data_help = DataHelper(zip(*data),
                               self.cfg.max_sentence_length,
                               word_index_dic=self.word_id_map,
                               by_word_index_dic=True)
        x, y = data_help.documents_transform_and_padding(self.cfg)
        return x, y

def sentences2char(sentence):
    return list(sentence)

def sentences2word(sentence):
    words = list(jieba.cut(sentence))
    return words
import os
import numpy as np
import pandas as pd
import tensorflow as tf

emb = Embedding(cfg)

test_data = df1
df1['label']=pd.Series([1]*len(df1))
x_test, y_test = emb.generate_sentence_token_ind(test_data)
x1_test, x2_test = zip(*x_test)

ckpt = '/nasfile/models/user_0010214548/' + 'bimpm_remove_stopwords'


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
    for batch in DataHelper.batch_iter(x_data, cfg.batch_size, 1,
                                       shuffle=False):
        q1_lens, q2_lens = get_valid_lengths(batch)
        q1, q2 = zip(*batch)
        y, q1, q2 = np.array([1] * len(x_data)), np.array(q1), np.array(q2)
        feed_dict = bimpm_model.create_feed_dict(q1_lens, q2_lens, q1, q2, y)
        predictions = sess.run(
            [bimpm_model.predictions],
            feed_dict=feed_dict)
        all_predictions.extend(predictions[0])

    return all_predictions

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    init_scale = 0.01
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    sess = tf.Session(config=graph_config)
    with sess.as_default():
        with tf.variable_scope("sentence_similarity") as scope:
            model = SentenceMatchModelGraph(2, emb.id_vector_map,
                                            is_training=False,
                                            options=cfg)
    predictions = bimpm_predict(sess, x_test, model, cfg, ckpt)
    result = pd.DataFrame(predictions, columns=['label'])
    result['id'] = df1['id']
    result=result[['id','label']]
    topai(1,result)
