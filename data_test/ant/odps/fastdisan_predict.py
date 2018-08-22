import argparse
import time
import random

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
        parser.add_argument('--learning_rate', type=float, default=0.1, help='Init Learning rate')
        parser.add_argument('--dy_lr', type=bool, default=False, help='if decay lr during training')
        parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay')
        parser.add_argument('--dropout', type=float, default=0.8, help='dropout keep prob')
        parser.add_argument('--wd', type=float, default=5e-5,
                            help='weight decay factor/l2 decay factor')
        parser.add_argument('--var_decay', type=float, default=0.999, help='Learning rate')  # ema
        parser.add_argument('--decay', type=float, default=0.9, help='summary decay')  # ema
        parser.add_argument('--max_sentence_length', type=int, default=50,
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
        parser.add_argument('--gpu', type=str, default='8', help='gpu')
        # @ ------neural network-----
        parser.add_argument('--use_char_emb', type=bool, default=False, help='abandoned')
        parser.add_argument('--use_token_emb', type=bool, default=True, help='abandoned')
        parser.add_argument('--char_embedding_length', type=int, default=8, help='(abandoned)')
        parser.add_argument('--char_out_size', type=int, default=150, help='(abandoned)')
        parser.add_argument('--out_channel_dims', type=str, default='50,50,50', help='(abandoned)')
        parser.add_argument('--filter_heights', type=str, default='1,3,5', help='(abandoned)')
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
        parser.set_defaults(shuffle=True)
        self.args = parser.parse_args()

        ## ---- to member variables -----
        for key, value in self.args.__dict__.items():
            if key not in ['data_test', 'shuffle']:
                exec('self.%s = self.args.%s' % (key, key))
cfg = Configs()


from operator import mul
VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER

def reduce(func,seq):
    if len(seq)==0:
        return  None
    result=seq[0]
    for i in seq[1:]:
        result=func(result,i)
    return result

def generate_embedding_mat(dict_size, emb_len, init_mat=None, extra_mat=None,
                           extra_trainable=False, scope=None):
    """
    generate embedding matrix for looking up
    :param dict_size: indices 0 and 1 corresponding to empty and unknown token
    :param emb_len:
    :param init_mat: init mat matching for [dict_size, emb_len]
    :param extra_mat: extra tensor [extra_dict_size, emb_len]
    :param extra_trainable:
    :param scope:
    :return: if extra_mat is None, return[dict_size+extra_dict_size,emb_len],
     else [dict_size,emb_len]
    """
    with tf.variable_scope(scope or 'gene_emb_mat'):
        emb_mat_ept_and_unk = tf.constant(
            value=0, dtype=tf.float32, shape=[2, emb_len])
        if init_mat is None:
            emb_mat_other = tf.get_variable(
                'emb_mat', [dict_size - 2, emb_len], tf.float32)
        else:
            emb_mat_other = tf.get_variable(
                "emb_mat", [dict_size - 2, emb_len], tf.float32,
                initializer=tf.constant_initializer(
                    init_mat[2:], dtype=tf.float32, verify_shape=True))
        emb_mat = tf.concat([emb_mat_ept_and_unk, emb_mat_other], 0)

        if extra_mat is not None:
            if extra_trainable:
                extra_mat_var = tf.get_variable(
                    "extra_emb_mat", extra_mat.shape, tf.float32,
                    initializer=tf.constant_initializer(
                        extra_mat, dtype=tf.float32, verify_shape=True))
                return tf.concat([emb_mat, extra_mat_var], 0)
            else:
                extra_mat_con = tf.constant(extra_mat, dtype=tf.float32)
                return tf.concat([emb_mat, extra_mat_con], 0)
        else:
            return emb_mat

def linear(args,
           output_size,
           bias,
           bias_start=0.0,
           scope=None,
           squeeze=False,
           wd=0.0,
           input_keep_prob=1.0,
           is_train=None):
    if args is None or (isinstance(args, (tuple, list)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (tuple, list)):
        # 如果args 是tensor,会进行这一步
        args = [args]

    # flat_args 一个二维矩阵构成的 数组, 这里的二维数组是 一个batch的所有句子的所有词的向量
    # 第一维是 句子数 * 句子长度, 第二维是词向量长度
    flat_args = [flatten(arg, 1) for arg in args]

    if input_keep_prob < 1.0:
        assert is_train is not None
        # tf.cond 函数的解释参见源码注释
        flat_args = [tf.cond(is_train,
                             lambda: tf.nn.dropout(arg, input_keep_prob),
                             lambda: arg)
                     for arg in flat_args]
    flat_out = _linear(
        flat_args, output_size, bias, bias_start=bias_start, scope=scope)
    # out [句数, 句长, d(h)]
    out = reconstruct(flat_out, args[0], 1)
    if squeeze:
        out = tf.squeeze(out, [len(args[0].get_shape().as_list())-1])
    if wd:
        add_reg_without_bias()
    return out

def _linear(xs,
            output_size,
            bias,
            bias_start=0.,
            scope=None):
    # 函数做的事对应 论文中的公式(14)中神经元内部的部分
    with tf.variable_scope(scope or 'linear_layer'):
        # 所有的句子的向量矩阵 按照第二维进行 concat, 即所有句子的同位置词的向量连接了起来
        # x的第一维表示 所有的句子* 所有的句子,  第二维是词向量的长度
        x = tf.concat(xs, -1)
        # input_size 是 word_vector_size
        input_size = x.get_shape()[-1]
        # 论文中W 是一个 (d(h), 句长) 维的矩阵, 这边是 (词长, d(h)) 维
        # #理解错了 与论文一致 W 是[词长, d(h)]
        W = tf.get_variable(
            'W', shape=[input_size, output_size], dtype=tf.float32,
        )
        if bias:
            bias = tf.get_variable('bias', shape=[output_size],
                                   dtype=tf.float32,
                                   initializer=tf.constant_initializer(
                                       bias_start))
            out = tf.matmul(x, W) + bias
        else:
            out = tf.matmul(x, W)
        # 输出是 二维矩阵, 一维是一个batch的句子数量 * 句子长度,
        # 二维是论文中的d(h)维,隐藏层词向量长度
        return out

def flatten(tensor, keep):
    """
    用于将多维的tensor 转化二维的 tensor
    :parameter tensor: 待转化的矩阵
    :parameter keep: 0-keep 维度转化成第一维, keep-n 转化成第二维
    """

    # 获取所有维度
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    # 前面 n-1 维度相乘
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i]
                        for i in range(start)])
    # 生成 shape [n-1 维乘积, n 维]
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i]
                          for i in range(start, len(fixed_shape))]
    # 将tensor 转化成上述shape的 tensor, 是一个二维的矩阵
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep, dim_reduced_keep=None):
    # 这个函数主要是用来对 tensor 做shape变换, 将一个二维的矩阵变成一个三维的
    dim_reduced_keep = dim_reduced_keep or keep

    # ref 是一个batch所有句子所有词组成的三维矩阵, 一维是句子数量 二维是句子长度, 三维是词向量长度
    ref_shape = ref.get_shape().as_list() # original shape
    # tensor 一个二维矩阵 第一维是句数*句长 第二维是转化后的词长
    tensor_shape = tensor.get_shape().as_list() # current shape

    ref_stop = len(ref_shape) - keep # flatten dims list
    tensor_start = len(tensor_shape) - dim_reduced_keep  # start
    # 在这里 pre_shape 中的元素其实有2个, [句数, 句长]
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)] #

    # keep_shape 只有一个, 转化后的词长
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i]
                  for i in range(tensor_start, len(tensor_shape))] #

    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    # 返回的是一个三维的tensor [句数, 句长, 词长]
    return out

def mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.multiply(val, tf.cast(val_mask, tf.float32),
                       name=name or 'mask_for_high_rank')

def exp_mask_for_high_rank(val, val_mask, name=None):
    val_mask = tf.expand_dims(val_mask, -1)
    return tf.add(val,
                  (1 - tf.cast(val_mask, tf.float32)) * VERY_NEGATIVE_NUMBER,
                  name=name or 'exp_mask_for_high_rank')

def selu(x):
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x>=0.0, x, alpha*tf.nn.elu(x))

def add_reg_without_bias(scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    counter = 0
    for var in variables:
        if len(var.get_shape().as_list()) <= 1: continue
        tf.add_to_collection('reg_vars', var)
        counter += 1
    return counter

N_INF = -1e12
import math
N_INF = -1e12

# ---------------   DiSAN Interface  ----------------
def fast_disan(rep_tensor,
               rep_mask,
               scope=None,
               keep_prob=1.,
               is_train=None,
               wd=0.,
               activation='elu',
               tensor_dict=None,
               name=''):
    """
    :param rep_tensor: 输入的多行样本, 已经转化成了词向量, 是一个三维矩阵
    :param rep_mask: 多行业本的token 对应的index, 并利用tf.cast([], tf.bool)进行了转化,
     即index不为0的都是true, 为0的是false
    :param scope:
    :param keep_prob:
    :param is_train:
    :param wd:
    :param activation:
    :param tensor_dict:
    :param name:
    :return:
    """
    with tf.variable_scope(scope or 'FastDiSAN'):
        if cfg.use_stacking:
            with tf.variable_scope('ct_attn'):
                fw_res = stacking_fast_directional_self_attention(
                    rep_tensor, cfg.word_embedding_length, is_train=is_train, head_num=10,
                    residual_keep_prob=keep_prob, attn_keep_prob=keep_prob,
                    dense_keep_prob=keep_prob, wd=wd, dot_activation_name='sigmoid',
                    activation_func_name=activation, layer_num=5, scope= 'dir_attn_fw')
                bw_res = stacking_fast_directional_self_attention(
                    rep_tensor, rep_mask, cfg.word_embedding_length, is_train=is_train, head_num=10,
                    residual_keep_prob=keep_prob, attn_keep_prob=keep_prob,
                    dense_keep_prob=keep_prob, wd=wd, dot_activation_name='sigmoid',
                    activation_func_name=activation, layer_num=5, scope='dir_attn_bw')
                seq_rep = tf.concat([fw_res, bw_res], -1)
        else:
            with tf.variable_scope('ct_attn'):
                fw_res = fast_directional_self_attention(
                    rep_tensor, rep_mask,cfg.word_embedding_length, is_train=is_train,
                    attn_keep_prob=keep_prob,dense_keep_prob=keep_prob, wd=wd,
                    dot_activation_name='sigmoid',activation_func_name=activation, scope= 'dir_attn_fw')
                bw_res = fast_directional_self_attention(
                    rep_tensor, rep_mask, cfg.word_embedding_length, is_train=is_train,
                    attn_keep_prob=keep_prob, dense_keep_prob=keep_prob, wd=wd,
                    dot_activation_name='sigmoid', activation_func_name=activation, scope='dir_attn_bw')
                seq_rep = tf.concat([fw_res, bw_res], -1)

        with tf.variable_scope('sent_enc_attn'):
            sent_rep = multi_dimensional_attention(
                seq_rep, rep_mask, 'multi_dimensional_attention',
                keep_prob, is_train, wd, activation,
                tensor_dict=tensor_dict, name=name+'_attn')
            return sent_rep


def multi_dimensional_attention(rep_tensor,
                                rep_mask,
                                scope=None,
                                keep_prob=1.,
                                is_train=None,
                                wd=0.,
                                activation='elu',
                                tensor_dict=None,
                                name=None):
    bs, sl, vec = tf.shape(
        rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = origin_bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1',
                                     activation, False, wd, keep_prob, is_train)
        map2 = origin_bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                                     False, wd, keep_prob, is_train)
        map2_masked = exp_mask_for_high_rank(map2, rep_mask)
        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec
        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft
        return attn_output


def origin_bn_dense_layer(input_tensor,
                          hn,
                          bias,
                          bias_start=0.0,
                          scope=None,
                          activation='relu',
                          enable_bn=True,
                          wd=0.,
                          keep_prob=1.0,
                          is_train=None):
    if is_train is None:
        is_train = False
    # activation
    if activation == 'linear':
        activation_func = tf.identity
    elif activation == 'relu':
        activation_func = tf.nn.relu
    elif activation == 'elu':
        activation_func = tf.nn.elu
    elif activation == 'selu':
        activation_func = selu
    else:
        raise AttributeError('no activation function named as %s' % activation)

    with tf.variable_scope(scope or 'origin_bn_dense_layer'):

        linear_map = linear(input_tensor, hn, bias, bias_start, 'linear_map',
                            False, wd, keep_prob, is_train)
        # 是否进行BN操作
        if enable_bn:
            linear_map = tf.contrib.layers.batch_norm(
                linear_map, center=True, scale=True,
                is_training=is_train, scope='bn')
        return activation_func(linear_map)


def origin_dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "origin_dropout"):
        assert is_train is not None
        if keep_prob < 1.0:
            d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
            out = tf.cond(is_train, lambda: d, lambda: x)
            return out
        return x


def stacking_fast_directional_self_attention(
        rep_tensor, rep_mask, hn, head_num=8,
        is_train=None, residual_keep_prob=.8, attn_keep_prob=.8, dense_keep_prob=.9, wd=0.,  # dropout and L2
        use_direction=True, attn_self=False,
        activation_func_name='relu', dot_activation_name='exp',
        layer_num=10, scope=None
):
    """
    stacked Fast-DiSA
    :param rep_tensor: same as that in Fast-DiSA;
    :param rep_mask: same as that in Fast-DiSA;
    :param hn: same as that in Fast-DiSA;
    :param head_num: same as that in Fast-DiSA;
    :param is_train: same as that in Fast-DiSA;
    :param residual_keep_prob: float-[], dropout keep probability for residual connection;
    :param attn_keep_prob: same as that in Fast-DiSA;
    :param dense_keep_prob: same as that in Fast-DiSA;
    :param wd: same as that in Fast-DiSA;
    :param use_direction: same as that in Fast-DiSA;
    :param attn_self: same as that in Fast-DiSA;
    :param activation_func_name: same as that in Fast-DiSA;
    :param dot_activation_name: same as that in Fast-DiSA;
    :param layer_num: int-[], the number of layer stacked;
    :param scope: soc
    :return:
    """
    with tf.variable_scope(scope or 'stacking_fast_disa'):
        final_mask_ft = mask_ft_generation(rep_mask, head_num, use_direction, attn_self)
        x = rep_tensor
        for layer_idx in range(layer_num):
            with tf.variable_scope('layer_%d' % layer_idx):
                # ffn
                y = bn_dense_layer(
                    x, hn, True, 0., 'ffn', activation_func_name, False, wd, dense_keep_prob, is_train)
                x = residual_connection(x, y, is_train, residual_keep_prob, 'res_con_1')
                # self-attn
                y = fast_directional_self_attention(
                    x, rep_mask, hn, head_num, is_train, attn_keep_prob, dense_keep_prob, wd, use_direction,
                    attn_self=attn_self, use_fusion_gate=False, final_mask_ft=final_mask_ft,
                    dot_activation_name=dot_activation_name, use_input_for_attn=True, add_layer_for_multi=False,
                    activation_func_name=activation_func_name, apply_act_for_v=False, input_hn=None,
                    output_hn=hn, accelerate=True, merge_var=False, scope='fast_disa'
                )
                x = residual_connection(x, y, is_train, residual_keep_prob, 'res_con_2')
    return x


def fast_directional_self_attention(
        rep_tensor, rep_mask, hn, head_num=2,
        is_train=None, attn_keep_prob=1., dense_keep_prob=1., wd=0.,  # dropout and L2
        use_direction=True, attn_self=False, use_fusion_gate=True, final_mask_ft=None,  # direction & fusion gate
        dot_activation_name='exp', use_input_for_attn=False, add_layer_for_multi=True,
        activation_func_name='relu', apply_act_for_v=True, input_hn=None, output_hn=None,  # non-linearity
        accelerate=False, merge_var=False,
        scope=None
):
    """
    The general API for Fast Self-Attention Attention mechanism for context fusion.
    :param rep_tensor: tf.float32-[batch_size,seq_len,channels], input sequence tensor;
    :param rep_mask: tf.bool-[batch_size,seq_len], mask to indicate padding or not for "rep_tensor";
    :param hn: int32-[], hidden unit number for this attention module;
    :param head_num: int32-[]; multi-head number, if "use_direction" is set to True, this must be set to a even number,
    i.e., half for forward and remaining for backward;
    :param is_train: tf.bool-[]; This arg must be a Placehold or Tensor of Tensorflow. This may be useful if you build
    a graph for both training and testing, and you can create a Placehold to indicate training(True) or testing(False)
    and pass the Placehold into this method;
    :param attn_keep_prob: float-[], the value must be in [0.0 ,1.0] and this keep probability is for attenton dropout;
    :param dense_keep_prob: float-[], the value must be in [0.0 ,1.0] and this probability is for dense-layer dropout;
    :param wd: float-[], if you use L2-reg, set this value to be greater than 0., which will result in that the
    trainable parameters (without biases) are added to a tensorflow collection named as "reg_vars";
    :param use_direction: bool-[], for mask generation, use forward and backward direction masks or not;
    :param attn_self: bool-[], for mask generation, include attention over self or not
    :param use_fusion_gate: bool-[], use a fusion gate to dynamically combine attention results with input or not.
    :param final_mask_ft: None/tf.float-[head_num,batch_size,seq_len,seq_len], the value is whether 0 (disabled) or
    1 (enabled), set to None if you only use single layer of this method; use *mask_generation* method
    to generate one and pass it into this method if you want to stack this module for computation resources saving;
    :param dot_activation_name: str-[], "exp" or "sigmoid", the activation function name for dot product
    self-attention logits;
    :param use_input_for_attn: bool-[], if True, use *rep_tensor* to compute dot-product and s2t multi-dim self-attn
    alignment score; if False, use a tensor obtained by applying a dense layer to the *rep_tensor*, which can add the
    non-linearity for this layer;
    :param add_layer_for_multi: bool-[], if True, add a dense layer with activation func -- "activation_func_name"
    to calculate the s2t multi-dim self-attention alignment score;
    :param activation_func_name: str-[], activation function name, commonly-used: "relu", "elu", "selu";
    :param apply_act_for_v: bool-[], if or not apply the non-linearity activation function ("activation_func_name") to
    value map (same as the value map in multi-head attention);
    :param apply_act_for_v: bool-[], if apply an activation function to v in the attention;
    :param input_hn: None/int32-[], if not None, add an extra dense layer (unit num is "input_hn") with
    activation function ("activation_func_name") before attention without consideration of multi-head.
    :param output_hn: None/int32-[], if not None, add an extra dense layer (unit num is "output_hn") with
    activation function ("activation_func_name") after attention without consideration of multi-head.
    :param accelerate: bool-[], for model acceleration, we optimize and combined some matrix multiplication if using
    the accelerate (i.e., set as True), which may effect the dropout-sensitive models or tasks.
    :param merge_var: bool-[], because the batch matmul is used for parallelism of multi-head attention, if True, the
    trainable variables are declared and defined together, otherwise them are defined separately and combined together.
    :param scope: None/str-[], variable scope name.
    :return: tf.float32-[batch_size, sequence_length, out_hn], if output_hn is not None, the out_hn = "output_hn"
    otherwise out_hn = "hn"
    """
    with tf.variable_scope(scope or 'proposed_self_attention'):
        # parameters inspection
        assert hn % head_num == 0, "hn (%d) must be divisible by the number of " \
                                   "attention heads (%d)." % (hn, head_num)
        if use_direction:
            assert head_num % 2 == 0, "attention heads (%d) must be a even number when using direction." % head_num

        # input non-linearity
        if input_hn is not None:
            with tf.variable_scope("input_non_linearity"):
                rep_tensor = bn_dense_layer(
                    rep_tensor, input_hn, True, 0., 'linear_input',
                    activation_func_name, False, wd, dense_keep_prob, is_train, 1, merge_var
                )

        # position mask generate [num,bs,sl,sl]
        if final_mask_ft is None:
            final_mask_ft = mask_ft_generation(rep_mask, head_num, use_direction, attn_self)

        # dimension/channel number for each head
        head_dim = int(hn / head_num)

        if not accelerate:
            # input preparation for each head. tiling here is to make the dropout different for each head.
            rep_tensor_tl = tf.tile(tf.expand_dims(rep_tensor, 0), [head_num, 1, 1, 1])  # num,bs,sl,xx

            # calculate value map
            v = multi_head_dense_layer(  # num,bs,sl,dim
                rep_tensor_tl, head_dim, True, 0., 'v_transform',
                'linear' if not apply_act_for_v else activation_func_name,
                False, wd, dense_keep_prob, is_train, 1, merge_var
            )

            # choose the source for both dot-product self-attention and s2t multi-dim self-attention
            for_attn_score = rep_tensor_tl if use_input_for_attn else v

            q = multi_head_dense_layer(
                for_attn_score, head_dim, False, 0., 'q_transform', 'linear',
                False, wd, dense_keep_prob, is_train, 1, merge_var)
            k = multi_head_dense_layer(
                for_attn_score, head_dim, False, 0., 'k_transform', 'linear',
                False, wd, dense_keep_prob, is_train, 1, merge_var)
        else:  # use_input_for_attn, apply_act_for_v
            for_attn_score = None
            if use_input_for_attn:
                qkv_combine = bn_dense_layer(
                    rep_tensor, hn, False, 0., 'qkv_combine', 'linear',
                    False, wd, dense_keep_prob, is_train, 3, merge_var)
                q, k, v = tf.split(qkv_combine, 3, -1)
                q = split_head(q, head_num)
                k = split_head(k, head_num)
                v = split_head(v, head_num)  # num,bs,sl,dim
                if apply_act_for_v:
                    v = activation_name_to_func(activation_func_name)(v)
            else:
                v = bn_dense_layer(  # num,bs,sl,dim
                    rep_tensor, hn, True, 0., 'v_transform',
                    'linear' if not apply_act_for_v else activation_func_name,
                    False, wd, dense_keep_prob, is_train, 1, merge_var
                )
                v = split_head(v, head_num)  # num,bs,sl,dim
                if apply_act_for_v:
                    v = activation_name_to_func(activation_func_name)(v)
                qk_combine = multi_head_dense_layer(
                    v, head_dim, False, 0., 'qk_combine', 'linear',
                    False, wd, dense_keep_prob, is_train, 2, merge_var
                )
                q, k = tf.split(qk_combine, 2, -1)  # num,bs,sl,dim

        # dot-product (multi-head) self-attention
        with tf.name_scope("dot_product_attention"):
            # calculate the logits
            dot_logits = tf.matmul(q, tf.transpose(k, [0, 1, 3, 2])) / math.sqrt(head_dim)  # num,bs,sl,sl
            # apply activation function and positional mask to logits from the attention
            e_dot_logits = final_mask_ft * activation_name_to_func(dot_activation_name)(dot_logits)

        # s2t multi-dim self-attention
        with tf.variable_scope("s2t_multi_dim_attention"):
            if not accelerate:
                assert for_attn_score is not None
                # Add an extra dense layer with activation func
                if add_layer_for_multi:
                    tensor4multi = multi_head_dense_layer(
                        for_attn_score, head_dim, True, 0., 'tensor4multi', activation_func_name,
                        False, wd, dense_keep_prob, is_train, 1, merge_var)
                else:
                    tensor4multi = for_attn_score
                # calculate the logits
                multi_logits = multi_head_dense_layer(
                    tensor4multi, head_dim, True, 0., 'multi_logits', 'linear', False,
                    wd, dense_keep_prob, is_train, 1, merge_var)
            else:  # use_input_for_attn, add_layer_for_multi
                if use_input_for_attn:
                    tensor4multi = bn_dense_layer(
                        rep_tensor, hn, True, 0., 'tensor4multi', 'linear', False,
                        wd, dense_keep_prob, is_train, 1, merge_var
                    )
                    tensor4multi = split_head(tensor4multi, head_num)
                else:
                    tensor4multi = multi_head_dense_layer(
                        v, head_dim, True, 0., 'tensor4multi', 'linear', False,
                        wd, dense_keep_prob, is_train, 1, merge_var
                    )
                if add_layer_for_multi:
                    multi_logits = multi_head_dense_layer(
                        activation_name_to_func(activation_func_name)(tensor4multi), head_dim,
                        True, 0., 'multi_logits', 'linear', False, wd, dense_keep_prob, is_train, 1, merge_var
                    )
                else:
                    multi_logits = tf.identity(tensor4multi, name='multi_logits')

            # apply exponent to the logits
            e_multi_logits = mask_v2(tf.exp(multi_logits), rep_mask, multi_head=True, high_dim=True)  # num,bs,sl,dim

        # combine both calculated attention logists, i.e., alignment scores, and perform attention procedures
        with tf.name_scope("hybrid_attn"):
            # Z: softmax normalization term in attention probabilities calculation
            accum_z_deno = tf.matmul(e_dot_logits, e_multi_logits)  # num,bs,sl,dim
            accum_z_deno = tf.where(  # in case of NaN and Inf
                tf.greater(accum_z_deno, tf.zeros_like(accum_z_deno)),
                accum_z_deno,
                tf.ones_like(accum_z_deno)
            )
            # attention dropout
            e_dot_logits = dropout(e_dot_logits, math.sqrt(attn_keep_prob), is_train)
            e_multi_logits = dropout(e_multi_logits, math.sqrt(attn_keep_prob), is_train)
            # sum of exp(logits) \multiply attention target sequence
            rep_mul_score = v * e_multi_logits
            accum_rep_mul_score = tf.matmul(e_dot_logits, rep_mul_score)
            # calculate the final attention results
            attn_res = accum_rep_mul_score / accum_z_deno

        # using a fusion gate to dynamically combine the attention results and attention input sequence
        if use_fusion_gate:
            with tf.variable_scope('context_fusion_gate'):
                fusion_gate = multi_head_dense_layer(
                    tf.concat([v, attn_res], -1), head_dim, True, 0.,
                    'linear_fusion_gate', 'sigmoid', False, wd, dense_keep_prob, is_train, 1, merge_var
                )  # num,bs,sl,dim
                attn_res = fusion_gate * v + (1 - fusion_gate) * attn_res

        # concatenate the channels from different heads
        attn_res = combine_head(attn_res)  # bs,sl,hn

        # output non-linearity
        if output_hn is not None:
            with tf.variable_scope("output_non_linearity"):
                attn_res = bn_dense_layer(
                    attn_res, output_hn, True, 0., 'linear_output',
                    activation_func_name, False, wd, dense_keep_prob, is_train, 1, merge_var
                )
        # set un-mask sequence terms to zeros
        output = mask_v2(attn_res, rep_mask, False, True)
        return output

# ===============================================
def mask_ft_generation(rep_mask, head_num, use_direction, attn_self):
    return tf.cast(mask_generation(rep_mask, head_num, use_direction, attn_self), tf.float32)

def mask_generation(rep_mask, head_num, use_direction, attn_self):
    with tf.name_scope('mask_generation'):
        bs, sl = tf.shape(rep_mask)[0], tf.shape(rep_mask)[1]
        # regular mask
        rep_mask_epd1 = tf.expand_dims(rep_mask, 1)  # bs,1,sl
        rep_mask_epd2 = tf.expand_dims(rep_mask, 2)  # bs,sl,1
        rep_mask_mat = tf.logical_and(rep_mask_epd1, rep_mask_epd2)  # bs,sl,sl

        # position mask
        sl_indices = tf.range(sl, dtype=tf.int32)
        sl_col, sl_row = tf.meshgrid(sl_indices, sl_indices)

        if use_direction:
            comp_func = tf.greater_equal if attn_self else tf.greater
            fw_mask = comp_func(sl_row, sl_col)  # sl,sl
            bw_mask = comp_func(sl_col, sl_row)  # sl,sl
            direct_mask = tf.stack([fw_mask, bw_mask], 0)  # 2,sl,sl
            direct_mask = tf.reshape(  # num,sl,sl
                tf.tile(tf.expand_dims(direct_mask, 1), [1, int(head_num / 2), 1, 1]),  # 2,4,sl,sl
                [head_num, sl, sl])
        else:
            if not attn_self:
                direct_mask = tf.tile(tf.expand_dims(tf.not_equal(sl_row, sl_col), 0), [head_num, 1, 1])  # n,sl,sl
            else:
                raise(ValueError, "A attention overself must be avoided without fw/bw information")
        final_mask = tf.logical_and(  # num,bs,sl,sl
            tf.expand_dims(rep_mask_mat, 0),
            tf.expand_dims(direct_mask, 1))
        return final_mask

def split_head(inp_tensor, head_num, name=None):
    bs, sl = tf.shape(inp_tensor)[0], tf.shape(inp_tensor)[1]
    ivec = inp_tensor.get_shape().as_list()[-1]
    head_dim = int(ivec // head_num)
    with tf.name_scope(name or 'split_head'):
        inp_rsp = tf.reshape(inp_tensor, [bs, sl, head_num, head_dim])
        return tf.transpose(inp_rsp, [2, 0, 1, 3])  # num, bs, sl, dim

def combine_head(inp_tensor, name=None):
    head_num, head_dim = inp_tensor.get_shape().as_list()[0], inp_tensor.get_shape().as_list()[-1]
    bs, sl = tf.shape(inp_tensor)[1], tf.shape(inp_tensor)[2]
    with tf.name_scope(name or 'combine_head'):
        inp_trans = tf.transpose(inp_tensor, [1, 2, 0, 3])
        return tf.reshape(inp_trans, [bs, sl, head_num*head_dim])

def exp_mask_v2(val, m, multi_head=False, high_dim=False, name=None):
    with tf.name_scope(name or "new_exp_mask"):
        if multi_head:
            m = tf.expand_dims(m, 0)
        if high_dim:
            m = tf.expand_dims(m, -1)
        m_flt = tf.cast(m, tf.float32)
        return val + (1. - m_flt) * N_INF

def mask_v2(val, m, multi_head=False, high_dim=False, name=None):
    with tf.name_scope(name or "new_exp_mask"):
        if multi_head:
            m = tf.expand_dims(m, 0)
        if high_dim:
            m = tf.expand_dims(m, -1)
        m_flt = tf.cast(m, tf.float32)
        return val * m_flt

def multi_head_dense_layer(
        input_tensor, hn, bias, bias_start=0.0, scope=None, activation='relu',
        enable_bn=False, wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False):
    assert not enable_bn
    # activation
    activation_func = activation_name_to_func(activation)

    hd_num = input_tensor.get_shape().as_list()[0]
    bs = tf.shape(input_tensor)[1]
    sl = tf.shape(input_tensor)[2]
    hd_dim = input_tensor.get_shape().as_list()[3]

    with tf.variable_scope(scope or 'multi_head_dense_layer'):
        # dropout
        input_tensor = dropout(input_tensor, keep_prob, is_train)

        if merge_var:
            weight = tf.get_variable('W', shape=[hd_num, hd_dim, hn*dup_num])
        else:
            weight_list = []
            for i in range(hd_num):
                sub_weight_list = []
                for j in range(dup_num):
                    sub_weight_list.append(tf.get_variable('W_%d_%d' % (i, j), shape=[hd_dim, hn]))
                weight_list.append(tf.concat(sub_weight_list, -1) if dup_num > 1 else sub_weight_list[0])
            weight = tf.stack(weight_list, 0)

        input_tensor_rsp = tf.reshape(input_tensor, [hd_num, bs*sl, hd_dim])  # hd_num, bs*sl, hd_dim
        out_rsp = tf.matmul(input_tensor_rsp, weight)  # hd_num, bs*sl, hn
        if bias:
            if merge_var:
                bias_val = tf.get_variable(
                    'bias', shape=[hd_num, 1, hn], dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start))
            else:
                bias_list = []
                for i in range(hd_num):
                    sub_bias_list = []
                    for j in range(dup_num):
                        sub_bias_list.append(
                            tf.get_variable(
                                'bias_%d_%d' % (i, j), shape=[1, hn], dtype=tf.float32,
                                initializer=tf.constant_initializer(bias_start)))
                    bias_list.append(tf.concat(sub_bias_list, -1) if dup_num > 1 else sub_bias_list[0])
                bias_val = tf.stack(bias_list, 0)
            out_rsp = out_rsp + bias_val   # hd_num, bs*sl, hn
        out = tf.reshape(out_rsp, [hd_num, bs, sl, hn*dup_num])
        if wd:
            tf.add_to_collection('reg_vars', weight)
        return activation_func(out)

def selu(x):
    with tf.name_scope('elu'):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale*tf.where(x >= 0.0, x, alpha*tf.nn.elu(x))

def activation_name_to_func(activation_name):
    assert isinstance(activation_name, str)

    if activation_name == 'linear':
        activation_func = tf.identity
    elif activation_name == 'relu':
        activation_func = tf.nn.relu
    elif activation_name == 'elu':
        activation_func = tf.nn.elu
    elif activation_name == 'selu':
        activation_func = selu
    elif activation_name == 'sigmoid':
        activation_func = tf.nn.sigmoid
    elif activation_name == 'tanh':
        activation_func = tf.nn.tanh
    elif activation_name == 'exp':
        activation_func = tf.exp
    elif activation_name == 'log':
        activation_func = tf.log
    else:
        raise AttributeError('no activation function named as %s' % activation_name)
    return activation_func

def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        if is_train is None:
            if keep_prob < 1.0:
                return tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        else:
            if keep_prob < 1.0:
                out = tf.cond(
                    is_train,
                    lambda: tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed),
                    lambda: x
                )
                return out
        return x

# =========linear==============
def bn_dense_layer(
        input_tensor, hn, bias, bias_start=0.0, scope=None,
        activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False
):
    assert len(input_tensor.get_shape().as_list()) == 3
    # activation
    activation_func = activation_name_to_func(activation)
    with tf.variable_scope(scope or 'bn_dense_layer'):
        bs, sl = tf.shape(input_tensor)[0], tf.shape(input_tensor)[1]
        input_dim = input_tensor.get_shape().as_list()[2]

        input_tensor = dropout(input_tensor, keep_prob, is_train)
        input_tensor_rsp = tf.reshape(input_tensor, [bs*sl, input_dim])

        if merge_var:
            weight = tf.get_variable('W', shape=[input_dim, hn * dup_num], dtype=tf.float32)
        else:
            weight_list = []
            for i in range(dup_num):
                weight_list.append(tf.get_variable('W_%d' % i, shape=[input_dim, hn]))
            weight = tf.concat(weight_list, -1)
        output_rsp = tf.matmul(input_tensor_rsp, weight)

        if bias:
            if merge_var or dup_num == 1:
                bias_val = tf.get_variable(
                    'bias', shape=[hn * dup_num], dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start)
                )
            else:
                bias_list = []
                for i in range(dup_num):
                    bias_list.append(
                        tf.get_variable(
                            'bias_%d' % i, shape=[hn], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
                    )
                bias_val = tf.concat(bias_list, -1)
            output_rsp += bias_val
        output = tf.reshape(output_rsp, [bs, sl, hn*dup_num])
        if enable_bn:
            output = tf.contrib.layers.batch_norm(
                output, center=True, scale=True, is_training=is_train,
                updates_collections=None,  decay=0.9,
                scope='bn')
        if wd:
            tf.add_to_collection('reg_vars', weight)
        return activation_func(output)


def bn_dense_layer_conv(
        input_tensor, hn, bias, bias_start=0.0, scope=None,
        activation='relu', enable_bn=False,
        wd=0., keep_prob=1.0, is_train=None, dup_num=1, merge_var=False
):
    assert len(input_tensor.get_shape().as_list()) == 3
    # activation
    activation_func = activation_name_to_func(activation)
    with tf.variable_scope(scope or 'bn_dense_layer'):
        input_dim = input_tensor.get_shape().as_list()[2]

        # dropout
        input_tensor = dropout(input_tensor, keep_prob, is_train)

        if merge_var:
            weight = tf.get_variable('W', shape=[input_dim, hn * dup_num], dtype=tf.float32)
        else:
            weight_list = []
            for i in range(dup_num):
                weight_list.append(tf.get_variable('W_%d' % i, shape=[input_dim, hn]))
            weight = tf.concat(weight_list, -1)

        matrix = tf.expand_dims(tf.expand_dims(weight, 0), 1)
        input_tensor_epd = tf.expand_dims(input_tensor, 1)
        output_epd = tf.nn.convolution(
            input_tensor_epd, matrix, "VALID", data_format="NHWC")
        output = tf.squeeze(output_epd, [1])

        if bias:
            if merge_var or dup_num == 1:
                bias_val = tf.get_variable(
                    'bias', shape=[hn * dup_num], dtype=tf.float32,
                    initializer=tf.constant_initializer(bias_start)
                )
            else:
                bias_list = []
                for i in range(dup_num):
                    bias_list.append(
                        tf.get_variable(
                            'bias_%d' % i, shape=[hn], dtype=tf.float32,
                            initializer=tf.constant_initializer(bias_start))
                    )
                bias_val = tf.concat(bias_list, -1)
            output += bias_val
        if enable_bn:
            output = tf.contrib.layers.batch_norm(
                output, center=True, scale=True, is_training=is_train,
                updates_collections=None, decay=0.9,
                scope='bn')
        if wd:
            tf.add_to_collection('reg_vars', weight)
        return activation_func(output)

# ------------------ multi-dim Source2token ------------
def multi_dim_souce2token_self_attn(rep_tensor, rep_mask, scope=None,
                                    keep_prob=1., is_train=None, wd=0., activation='elu',
                                    tensor_dict=None, name=None):
    bs, sl, vec = tf.shape(rep_tensor)[0], tf.shape(rep_tensor)[1], tf.shape(rep_tensor)[2]
    ivec = rep_tensor.get_shape().as_list()[2]
    with tf.variable_scope(scope or 'multi_dimensional_attention'):
        map1 = bn_dense_layer(rep_tensor, ivec, True, 0., 'bn_dense_map1', activation,
                              False, wd, keep_prob, is_train, 1, False)
        map2 = bn_dense_layer(map1, ivec, True, 0., 'bn_dense_map2', 'linear',
                              False, wd, keep_prob, is_train, 1, False)
        map2_masked = exp_mask_v2(map2, rep_mask, high_dim=True)

        soft = tf.nn.softmax(map2_masked, 1)  # bs,sl,vec
        attn_output = tf.reduce_sum(soft * rep_tensor, 1)  # bs, vec
        # save attn
        if tensor_dict is not None and name is not None:
            tensor_dict[name] = soft
        return attn_output

# --------------- residual connection -------------
def residual_connection(x, y, is_train=None, residual_keep_prob=1., scope=None):
    with tf.variable_scope(scope or 'residual_connection'):
        y = dropout(y, residual_keep_prob, is_train)
        return layer_norm(x + y, scope='layer_norm')


def layer_norm(inputs, epsilon=1e-6, scope=None):
    with tf.variable_scope(scope or "layer_norm"):
        channel_size = inputs.get_shape().as_list()[-1]
        scale = tf.get_variable("scale", shape=[channel_size],
                                initializer=tf.ones_initializer())
        offset = tf.get_variable("offset", shape=[channel_size],
                                 initializer=tf.zeros_initializer())
        mean = tf.reduce_mean(inputs, axis=-1, keep_dims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=-1,
                                  keep_dims=True)
        norm_inputs = (inputs - mean) * tf.rsqrt(variance + epsilon)
        return norm_inputs * scale + offset

class ModelFastDiSAN:
    def __init__(self, id_vector_map, scope):
        self.id_vector_map = id_vector_map

        # sent1_token 句子列表, 每个句子中的token已经转化成了ind
        # sent1_char 第一维表示句子, 第二维是token, 第三维是char, tl是每个token的最大长度
        self.sent1_token = tf.placeholder(
            tf.int32, [None, None], name='sent1_token')

        self.sent2_token = tf.placeholder(
            tf.int32, [None, None], name='sent2_token')

        self.gold_label = tf.placeholder(tf.int32, [None], name='gold_label')
        self.is_train = tf.placeholder(tf.bool, [], name='is_train')
        self.learning_rate = tf.placeholder(tf.float32, [], 'learning_rate')

        self.sent1_token_mask = tf.cast(self.sent1_token, tf.bool)
        self.sent2_token_mask = tf.cast(self.sent2_token, tf.bool)

        self.tensor_dict = {}
        self.global_step = tf.get_variable(
            'global_step', shape=[], dtype=tf.int32,
            initializer=tf.constant_initializer(0), trainable=False)
        self.tel = cfg.word_embedding_length
        self.hn = cfg.hidden_units_num
        self.output_class = 2

        self.scope = scope
        self.logits = None
        self.predict = None
        self.score = None
        self.loss = None
        self.accuracy = None
        self.var_ema = None
        self.ema = None
        self.summary = None
        self.previous_dev_loss = []
        self.learning_rate_value = cfg.learning_rate
        self.learning_rate_updated = False
        self.opt = None
        self.train_op = None
        self.update_tensor_add_ema_and_opt()

    def build_network(self):
        hn = self.hn

        with tf.variable_scope('emb'):
            if not cfg.use_pre_trained:
                self.id_vector_map = tf.Variable(
                    tf.random_uniform(
                        [len(self.id_vector_map), cfg.word_embedding_length], -1.0, 1.0),
                    trainable=True, name="W"
                )

            s1_emb = tf.nn.embedding_lookup(
                self.id_vector_map, self.sent1_token)
            # bs,sl2,tel
            s2_emb = tf.nn.embedding_lookup(
                self.id_vector_map, self.sent2_token)
            self.tensor_dict['s1_emb'] = s1_emb
            self.tensor_dict['s2_emb'] = s2_emb

        with tf.variable_scope('sent_enc'):
            s1_rep = fast_disan(
                s1_emb, self.sent1_token_mask, 'DiSAN', cfg.dropout,
                self.is_train, cfg.wd, 'elu', self.tensor_dict, 's1'
            )
            self.tensor_dict['s1_rep'] = s1_rep

            tf.get_variable_scope().reuse_variables()

            s2_rep = fast_disan(
                s2_emb, self.sent2_token_mask, 'DiSAN', cfg.dropout,
                self.is_train, cfg.wd, 'elu', self.tensor_dict, 's2'
            )
            self.tensor_dict['s2_rep'] = s2_rep

        with tf.variable_scope('output'):
            out_rep = tf.concat(
                [s1_rep, s2_rep, s1_rep - s2_rep, s1_rep * s2_rep], -1)
            pre_output = tf.nn.elu(
                linear(
                    [out_rep], hn, True, 0.,
                    scope='pre_output', squeeze=False,
                    wd=cfg.wd, input_keep_prob=cfg.dropout,
                    is_train=self.is_train))
            logits = linear(
                [pre_output], self.output_class, True, 0., scope='logits',
                squeeze=False, wd=cfg.wd, input_keep_prob=cfg.dropout,
                is_train=self.is_train)
            self.tensor_dict[logits] = logits
            self.probs = tf.nn.softmax(logits)
        return logits

    def build_loss(self):
        # weight_decay
        with tf.name_scope("weight_decay"):
            # 计算正则项的损失
            for var in set(tf.get_collection('reg_vars', self.scope)):
                # tf.multiply 两个数相乘
                weight_decay = tf.multiply(
                    tf.nn.l2_loss(var), cfg.wd,
                    name="{}-wd".format('-'.join(str(var.op.name).split('/'))))
                tf.add_to_collection('losses', weight_decay)
        reg_vars = tf.get_collection('losses', self.scope)

        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        print('regularization var num: %d' % len(reg_vars))
        print('trainable var num: %d' % len(trainable_vars))
        # 样本的损失
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.gold_label,
            logits=self.logits
        )
        tf.add_to_collection(
            'losses', tf.reduce_mean(losses, name='xentropy_loss_mean'))

        loss = tf.add_n(tf.get_collection('losses', self.scope), name='loss')
        tf.summary.scalar(loss.op.name, loss)
        tf.add_to_collection('ema/scalar', loss)
        return loss

    def build_accuracy(self):
        correct = tf.equal(
            tf.cast(tf.argmax(self.logits, -1), tf.int32),
            self.gold_label
        )
        return tf.cast(correct, tf.float32)

    def update_tensor_add_ema_and_opt(self):

        self.logits = self.build_network()

        self.score = tf.cast(self.logits, tf.float32, name='predict_score')

        self.predict = tf.cast(tf.argmax(self.logits, -1),
                               tf.int32,
                               name='predict')
        self.loss = self.build_loss()
        self.accuracy = self.build_accuracy()

        # ------------ema-------------
        if True:
            self.var_ema = tf.train.ExponentialMovingAverage(cfg.var_decay)
            self.build_var_ema()

        if cfg.mode == 'train':
            self.ema = tf.train.ExponentialMovingAverage(cfg.decay)
            self.build_ema()
        self.summary = tf.summary.merge_all()

        # ---------- optimization ---------
        if cfg.optimizer.lower() == 'adadelta':
            self.opt = tf.train.AdadeltaOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'adam':
            self.opt = tf.train.AdamOptimizer(self.learning_rate)
        elif cfg.optimizer.lower() == 'rmsprop':
            self.opt = tf.train.RMSPropOptimizer(self.learning_rate)
        else:
            raise AttributeError('no optimizer named as \'%s\'' % cfg.optimizer)

        trainable_vars = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        # trainable param num:
        # print params num
        all_params_num = 0
        for elem in trainable_vars:
            # elem.name
            var_name = elem.name.split(':')[0]
            if var_name.endswith('emb_mat'):
                continue
            params_num = 1
            for l in elem.get_shape().as_list():
                params_num *= l
            all_params_num += params_num
        print('Trainable Parameters Number: %d' % all_params_num)

        # minimize 函数集合了 compute_gradients 和 apply_gradients
        # global_step 在更新结束后会自增 1
        self.train_op = self.opt.minimize(
            self.loss, self.global_step,
            var_list=tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, self.scope))

    def build_var_ema(self):

        ema_op = self.var_ema.apply(tf.trainable_variables(),)
        with tf.control_dependencies([ema_op]):
            # 在control_dependencies的作用块下，需要增加一个新节点到 graph 中,
            # 表示在执行self.loss 之前, 先执行 ema_op
            self.loss = tf.identity(self.loss)

    def build_ema(self):
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + \
                  tf.get_collection("ema/vector", scope=self.scope)

        ema_op = self.ema.apply(tensors)
        # for var in tf.get_collection("ema/scalar", scope=self.scope):
        #     ema_var = self.ema.average(var)
        #     tf.summary.scalar(ema_var.op.name, ema_var)
        #
        # for var in tf.get_collection("ema/vector", scope=self.scope):
        #     ema_var = self.ema.average(var)
        #     tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def train_step(self, sess, batch_samples, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples)
        logits = 0
        probs = 0
        # out_tensor_dict_1= sess.run(self.tensor_dict, feed_dict=feed_dict)
        if get_summary:
            loss, summary, train_op, accuracy, predict = sess.run(
                [self.loss, self.summary, self.train_op, self.accuracy, self.predict],
                feed_dict=feed_dict)

        else:
            loss, train_op, accuracy, logits, predict, probs= sess.run(
                [self.loss, self.train_op, self.accuracy,
                 self.logits, self.predict,self.probs],
                feed_dict=feed_dict)
        return loss, accuracy, predict,logits,probs

    def validate_step(self, sess, batch_samples):
        assert isinstance(sess, tf.Session)
        feed_dict = self.get_feed_dict(batch_samples,'valid')
        accuracy, loss, predict, logits = sess.run(
            [self.accuracy, self.loss, self.predict, self.logits],
            feed_dict=feed_dict)
        return accuracy, loss, predict, logits

    def get_feed_dict(self, batch, data_type='train'):
        x, y = zip(*batch)
        sentences1 = []
        sentences2 = []
        for ele in x:
            if random() > 0.5:
                sentences1.append(ele[0])
                sentences2.append(ele[1])
            else:
                sentences1.append(ele[1])
                sentences2.append(ele[0])
        feed_dict = {self.sent1_token: sentences1,
                     self.sent2_token: sentences2,
                     self.gold_label: y,
                     self.is_train: True if data_type == 'train' else False,
                     self.learning_rate: self.learning_rate_value}
        return feed_dict

    def update_learning_rate(
            self, current_dev_loss, global_step, lr_decay_factor=0.7):
        if cfg.dy_lr:
            method = 1
            if method == 0:
                assert len(self.previous_dev_loss) >= 1
                self.previous_dev_loss.append(current_dev_loss)
                delta = []
                pre_loss = self.previous_dev_loss.pop(0)
                for loss in self.previous_dev_loss:
                    delta.append(pre_loss < loss)
                    pre_loss = loss

                do_decay = True
                for d in delta:
                    do_decay = do_decay and d

                if do_decay:
                    self.learning_rate_value *= lr_decay_factor
                    self.learning_rate_updated = True
                    print(
                        'found dev loss increases, decrease learning rate '
                        'to: %f' % self.learning_rate_value)

                if global_step % 10000 == 0:
                    if not self.learning_rate_updated:
                        self.learning_rate_value *= 0.9
                        print('decrease learning rate to:'
                              ' %f' % self.learning_rate_value)
                    else:
                        self.learning_rate_updated = False
            elif method == 1:
                if self.learning_rate_value < 5e-6:
                    return
                if global_step % 10000 == 0:
                    self.learning_rate_value *= lr_decay_factor

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

ckpt = '/nasfile/models/user_0010214548/' + 'fastdisan'

def fast_disan_predict(sess, x_data, fast_disan_model, cfg, checkpoint_file=None):

    if checkpoint_file:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, checkpoint_file)

    # Get the placeholders from the graph by name
    input_x1 = fast_disan_model.sent1_token
    input_x2 = fast_disan_model.sent2_token
    is_train = fast_disan_model.is_train
    predict = fast_disan_model.predict

    batches = DataHelper.batch_iter(list(x_data),
                                    2 * cfg.batch_size,
                                    1,
                                    shuffle=False)
    predictions = []
    for db in batches:
        x1_dev_b, x2_dev_b = zip(*db)
        batch_score = sess.run(
            predict,
            feed_dict={input_x1: x1_dev_b,
                       input_x2: x2_dev_b,
                       is_train: False
                       })
        predictions.extend(batch_score)
    return predictions

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    init_scale = 0.01
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)
    sess = tf.Session(config=graph_config)
    with sess.as_default():
        with tf.variable_scope("ant") as scope:
            model = ModelFastDiSAN(emb.id_vector_map, scope.name)
    predictions = fast_disan_predict(sess, x_test, model, cfg, ckpt)
    result = pd.DataFrame(predictions, columns=['label'])
    result['id'] = df1['id']
    result=result[['id','label']]
    topai(1,result)