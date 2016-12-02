from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import logging

from seqModel import SeqModel

import pandas as pd
import configparser
import env

sys.path.insert(0, '../utils')
import embed_attribute
from xing_data import data_read
import data_iterator
from data_iterator import DataIterator
from best_buckets import *

tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.83,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate.")

tf.app.flags.DEFINE_boolean("withAdagrad", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("fromScratch", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("saveCheckpoint", False,
                            "save Model at each checkpoint.")
tf.app.flags.DEFINE_integer("batch_size", 4,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 4, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("user_vocab_size", 150000, "User vocabulary size.")
tf.app.flags.DEFINE_integer("item_vocab_size", 50000, "Item vocabulary size.")
tf.app.flags.DEFINE_integer("n_sampled", 1024, "sampled softmax/warp loss.")
tf.app.flags.DEFINE_string("data_dir", "../mf/data0", "Data directory")

tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")

tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")

tf.app.flags.DEFINE_integer("n_epoch", 20,
                            "How many epochs to train.")

tf.app.flags.DEFINE_integer("patience", 5,"exit if the model can't improve for $patence evals")

tf.app.flags.DEFINE_integer("L", 5,"max length")



FLAGS = tf.app.flags.FLAGS

_buckets = []

                
def split_buckets(array,buckets):
    """
    array : [(user,[items])]
    return:
    d : [[(user, [items])]]
    """
    d = [[] for i in xrange(len(buckets))]
    for u, items in array:
        index = get_buckets_id(len(items), buckets)
        if index >= 0:
            d[index].append((u,items))
    return d

def get_buckets_id(l, buckets):
    id = -1
    for i in xrange(len(buckets)):
        if l <= buckets[i]:
             id = i
             break

    return id

def form_sequence(data):
    """
    Args:
      data = [(u,i,week)]
    Return:
      d : [(user_id, [item_id])]
    """
    # 
    # return users, items

    users = []
    items = []
    d = {} # d[u] = [(i,week)]
    for u,i,week in data:
        if not u in d:
            d[u] = []
        d[u].append((i,week))
    
    
    dd = []
        
    for u in d:
        tmp = sorted(d[u],key = lambda x: x[1])
        tmp =  [x[0] for x in tmp]
        dd.append((u,tmp))
            
    return dd

def read_data():
    (data_tr, data_va, u_attr, i_attr, item_ind2logit_ind, logit_ind2item_ind) = data_read(FLAGS.data_dir, _submit = 0, ta = 1, logits_size_tr=FLAGS.item_vocab_size)

    # remove unk
    data_tr = [p for p in data_tr if (p[1] in item_ind2logit_ind)]
    data_va = [p for p in data_va if (p[1] in item_ind2logit_ind)]
    
    # UNK and START
    START_ID = i_attr.get_item_last_index()
    seq_tr = form_sequence(data_tr)
    seq_va = form_sequence(data_va)

    # calculate buckets
    global _buckets
    _buckets = calculate_buckets(seq_tr+seq_va, 10, 10)
    _buckets = sorted(_buckets)


    # split_buckets
    seq_tr = split_buckets(seq_tr,_buckets)
    seq_va = split_buckets(seq_va,_buckets)

    # create embedAttr
    u_attr.set_model_size(FLAGS.size)
    i_attr.set_model_size(FLAGS.size)

    embAttr = embed_attribute.EmbeddingAttribute(u_attr, i_attr, FLAGS.batch_size, FLAGS.n_sampled, _buckets[-1], False, item_ind2logit_ind, logit_ind2item_ind)



    return seq_tr, seq_va, embAttr, START_ID


def create_model(session,embAttr,START_ID):
    dtype = tf.float32
    model = SeqModel(_buckets,
                     FLAGS.size,
                     FLAGS.num_layers,
                     FLAGS.max_gradient_norm,
                     FLAGS.batch_size,
                     FLAGS.learning_rate,
                     FLAGS.learning_rate_decay_factor,
                     embAttr,
                     withAdagrad = FLAGS.withAdagrad,
                     num_samples = FLAGS.n_sampled,
                     dropoutRate = FLAGS.keep_prob,
                     START_ID = START_ID,
                     dtype = dtype
                     )

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if (not FLAGS.fromScratch) and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        log_it("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        log_it("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def log_it(msg):
    print(msg)
    logging.info(msg)

def show_all_variables():
    all_vars = tf.all_variables()
    for var in all_vars:
        log_it(var.name)


def train():

    # Read Data
    log_it("Reading Data...")
    train_set, dev_set, embAttr, START_ID = read_data()
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size for i in xrange(len(train_bucket_sizes))]
    dev_bucket_sizes = [len(dev_set[b]) for b in xrange(len(_buckets))]
    dev_total_size = int(sum(dev_bucket_sizes))


    # steps
    batch_size = FLAGS.batch_size
    n_epoch = FLAGS.n_epoch
    steps_per_epoch = int(train_total_size / batch_size)
    steps_per_dev = int(dev_total_size / batch_size)

    steps_per_checkpoint = 200 #steps_per_dev * 4
    total_steps = steps_per_epoch * n_epoch

    # reports
    log_it(_buckets)
    log_it("Train:")
    log_it("total: {}".format(train_total_size))
    log_it("buckets: {}".format(train_bucket_sizes))
    log_it("Dev:")
    log_it("total: {}".format(dev_total_size))
    log_it("buckets: {}".format(dev_bucket_sizes))
    log_it("")
    log_it("Steps_per_epoch: {}".format(steps_per_epoch))
    log_it("Total_steps:{}".format(total_steps))
    log_it("Steps_per_checkpoint: {}".format(steps_per_checkpoint))


    with tf.Session() as sess:
        
        log_it("Creating Model")
        model = create_model(sess, embAttr, START_ID)
        show_all_variables()
    
        # Data Iterators
        dite = DataIterator(model, train_set, len(train_buckets_scale), batch_size, train_buckets_scale)
        
        iteType = 0
        if iteType == 0:
            log_it("withRandom")
            ite = dite.next_random()
        elif iteType == 1:
            log_it("withSequence")
            ite = dite.next_sequence()
        
        # statistics during training
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        his = []
        low_ppx = 10000000
        low_ppx_step = 0

        while current_step < total_steps:
            
            # start
            start_time = time.time()
            
            # data and train
            users, inputs, outputs, weights, bucket_id = ite.next()
            L = model.step(sess, users, inputs, outputs, weights, bucket_id)
            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint
            
            loss += L / (steps_per_checkpoint * batch_size)
            current_step += 1
        
            if current_step % steps_per_checkpoint == 0:
                log_it("--------------------"+"TRAIN"+str(current_step)+"-------------------")
                # Print statistics for the previous epoch.
                
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                log_it("global step %d learning rate %.4f step-time %.2f perplexity " "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                
                train_ppx = perplexity
                
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")
                if FLAGS.saveCheckpoint:
                    log_it("Saving model....")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                                
                # dev data
                log_it("--------------------" + "DEV" + str(current_step) + "-------------------")
                eval_loss, eval_ppx = evaluate(sess, model, dev_set)
                log_it("dev: ppx: {}".format(eval_ppx))

                his.append([current_step, train_ppx, eval_ppx])
                if eval_ppx < low_ppx:
                    low_ppx = eval_ppx
                    low_ppx_step = current_step

                sys.stdout.flush()
                # Decrease learning rate if current eval ppl is larger
                if len(previous_losses) > FLAGS.patience and eval_ppx > max(previous_losses[-5:]):
                    break
                    #sess.run(model.learning_rate_decay_op)
                previous_losses.append(eval_ppx)

        theone = his[low_ppx_step]
        log_it("Step: {} Train/Dev: {:2f}/{:2f}".format(theone[0],theone[1],theone[2]))

        df = pd.DataFrame(his)
        df.columns=["step""Train_ppx","Dev_ppx"]
        df.to_csv(os.path.join(FLAGS.train_dir,"log.csv"))

def evaluate(sess, model, data_set):
    # Run evals on development set and print their perplexity.
    dropoutRateRaw = FLAGS.keep_prob
    
    sess.run(model.dropoutRate.assign(1.0))


    start_id = 0
    loss = 0.0
    n_steps = 0
    batch_size = FLAGS.batch_size
    
    dite = DataIterator(model, data_set, len(_buckets), batch_size, None)
    ite = dite.next_sequence(stop = True)

    for users, inputs, outputs, weights, bucket_id in ite:
        L = model.step(sess, users, inputs, outputs, weights, bucket_id, forward_only = True)
        loss += L
        n_steps += 1
        if n_steps > 50:
            break
            
    loss = loss/(n_steps * batch_size)
    ppx = math.exp(loss) if loss < 300 else float("inf")


    sess.run(model.dropoutRate.assign(dropoutRateRaw))

    return loss, ppx

def main(_):
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    log_path = os.path.join(FLAGS.train_dir,"log.txt")
    logging.basicConfig(filename=log_path,level=logging.DEBUG)
    train()
    
if __name__ == "__main__":
    tf.app.run()
