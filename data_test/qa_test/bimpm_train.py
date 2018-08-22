import logging
import os
import sys
import time

import pandas as pd
import tensorflow as tf
root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(root_path)

from lib.model.configs import cfg
from lib.model.bimpm.model_bimpm import SentenceMatchModelGraph

def get_valid_lengths(batches):
    x,y=zip(*batches)
    x1,x2=zip(*x)
    x1_lens,x2_lens=[],[]
    for i in range(len(x1)):
        x1_len=sum(np.array(x1[i])!=1)
        x2_len=sum(np.array(x2[i])!=1)
        x1_lens.append(x1_len)
        x2_lens.append(x2_len)
    return np.array(x1_lens),np.array(x2_lens)


os.environ["CUDA_VISIBLE_DEVICES"] =cfg.gpu

from data_test.qa_test.embedding import *
logging.basicConfig(filename="quora_bimpm.log",
                    filemode="w",
                    format="%(asctime)s-%(name)s-%(levelname)s-%(message)s",
                    level=logging.INFO)

train_data = pd.read_csv(cfg.train_data, sep='$')
x_train, y_train = generate_sentence_token_ind(train_data)
train_data_emb = list(zip(x_train, y_train))
valid_data = pd.read_csv(cfg.validate_data, sep='$')
x_valid, y_valid = generate_sentence_token_ind(valid_data)
valid_data_emb = list(zip(x_valid, y_valid))

num_epoch = int(len(train_data) / cfg.batch_size) + 1

logging.info("starting graph ")
with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96,
                                allow_growth=True)
    graph_config = tf.ConfigProto(
        gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=graph_config)
    global_step = tf.train.get_or_create_global_step()
    with sess.as_default(), tf.device("/gpu:%s" % cfg.gpu):
        with tf.variable_scope("sentence_similarity") as scope:
            model = SentenceMatchModelGraph(2, id_vector_map, is_training=True,
                                            options=cfg,
                                            global_step=global_step)
    sess.run(tf.global_variables_initializer())
    checkpoint_dir = os.path.abspath(
        os.path.join(os.path.curdir, "bimpm_disan_runs", cfg.model_save_path))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    total_loss = 0
    saver = tf.train.Saver(tf.global_variables())
    batches = DataHelper.batch_iter(
        train_data_emb, cfg.batch_size, cfg.max_epoch, shuffle=True)
    best_accuracy = -1
    step = 0
    current_step=0
    for batch in batches:
        q1_lens, q2_lens = get_valid_lengths(batch)
        x, y = zip(*batch)
        q1, q2 = zip(*x)
        y, q1, q2 = np.array(y), np.array(q1), np.array(q2)
        feed_dict = model.create_feed_dict(q1_lens, q2_lens, q1, q2, y)
        _, loss_value = sess.run([model.train_op, model.loss],
                                 feed_dict=feed_dict)
        total_loss += loss_value
        if step>0 and step % 100 == 0:
            logging.info('{} '.format(step))
            sys.stdout.flush()
            logging.info('\nstep %d: loss = %.4f ' % (
                step, total_loss / step))
        # evaluation
        if step % num_epoch == 0:
            total = 0
            correct = 0
            for batch in DataHelper.batch_iter(valid_data_emb, cfg.batch_size,
                                               1, shuffle=False):
                total += cfg.batch_size
                q1_lens, q2_lens = get_valid_lengths(batch)
                x, y = zip(*batch)
                q1, q2 = zip(*x)
                y, q1, q2 = np.array(y), np.array(q1), np.array(q2)
                feed_dict = model.create_feed_dict(q1_lens, q2_lens, q1, q2, y)
                [cur_correct, probs, predictions] = sess.run(
                    [model.eval_correct, model.prob, model.predictions],
                    feed_dict=feed_dict)
                correct += cur_correct
            acc = correct / float(total) * 100
            logging.info("Accuracy: %.2f" % acc)
            if acc > best_accuracy:
                saver.save(sess, checkpoint_prefix, step)
                logging.info('saved model')
                best_accuracy = acc
                current_step=step
        if step-current_step>3*num_epoch and current_step>=num_epoch:
            break
        step += 1
    logging.info('training done')