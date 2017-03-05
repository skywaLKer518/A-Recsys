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

# import pandas as pd
# import configparser
# import env

sys.path.insert(0, '../utils')
sys.path.insert(0, '../attributes')


import embed_attribute
from input_attribute import read_data as read_attributed_data


import data_iterator
from data_iterator import DataIterator
from best_buckets import *
from tensorflow.python.client import timeline
from prepare_train import positive_items, item_frequency, sample_items, to_week

# datasets, paths, and preprocessing
tf.app.flags.DEFINE_string("dataset", "xing", "dataset name")
tf.app.flags.DEFINE_string("raw_data", "../raw_data", "input data directory")
tf.app.flags.DEFINE_string("data_dir", "./cache0/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train", "Training directory.")
tf.app.flags.DEFINE_boolean("test", True, "fix to be True as we split non-test part by users.")
tf.app.flags.DEFINE_string("combine_att", 'mix', "method to combine attributes: het or mix")
tf.app.flags.DEFINE_boolean("use_item_feature", True, "RT")
tf.app.flags.DEFINE_boolean("use_user_feature", True, "RT")
tf.app.flags.DEFINE_integer("item_vocab_size", 50000, "Item vocabulary size.")
tf.app.flags.DEFINE_integer("vocab_min_thresh", 2, "filter inactive tokens.")

# tuning hypers
tf.app.flags.DEFINE_string("loss", 'ce', "loss function: ce, warp")
tf.app.flags.DEFINE_float("learning_rate", 0.5, "Learning rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.83,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate.")
tf.app.flags.DEFINE_float("power", 0.5, "related to sampling rate.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training/evaluation.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("n_epoch", 500,
                            "Maximum number of epochs in training.")
tf.app.flags.DEFINE_integer("L", 30,"max length")
tf.app.flags.DEFINE_integer("n_bucket", 10,
                            "num of buckets to run.")
tf.app.flags.DEFINE_integer("patience", 10,"exit if the model can't improve for $patence evals")
#tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,"How many training steps to do per checkpoint.")

# recommendation
tf.app.flags.DEFINE_boolean("recommend", False, "Set to True for recommending.")
tf.app.flags.DEFINE_boolean("recommend_new", False, "TODO.")
tf.app.flags.DEFINE_integer("topk", 100, "recommend items with the topk values")

# for ensemble 
tf.app.flags.DEFINE_boolean("ensemble", False, "to ensemble")
tf.app.flags.DEFINE_string("ensemble_suffix", "", "multiple models suffix: 1,2,3,4,5")
tf.app.flags.DEFINE_integer("seed", 0, "dev split random seed.")

# attribute model variants
tf.app.flags.DEFINE_integer("output_feat", 1, "0: no use, 1: use, mean-mulhot, 2: use, max-pool")
tf.app.flags.DEFINE_boolean("use_sep_item", False, "use separate embedding parameters for output items.")
tf.app.flags.DEFINE_boolean("no_input_item_feature", False, "not using attributes at input layer")
tf.app.flags.DEFINE_boolean("use_concat", False, "use concat or mean")
tf.app.flags.DEFINE_boolean("no_user_id", True, "use user id or not")

# devices
tf.app.flags.DEFINE_string("N", "000", "GPU layer distribution: [input_embedding, lstm, output_embedding]")

# 
tf.app.flags.DEFINE_boolean("withAdagrad", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("fromScratch", True,
                            "withAdagrad.")
tf.app.flags.DEFINE_boolean("saveCheckpoint", False,
                            "save Model at each checkpoint.")
tf.app.flags.DEFINE_boolean("profile", False, "False = no profile, True = profile")

# others...
tf.app.flags.DEFINE_integer("ta", 1, "part = 1, full = 0")
tf.app.flags.DEFINE_float("user_sample", 1.0, "user sample rate.")

tf.app.flags.DEFINE_boolean("after40", False,
                            "whether use items after week 40 only.")
tf.app.flags.DEFINE_string("split", "last", "last: last maxlen only; overlap: overlap 1 / 3 of maxlen")

tf.app.flags.DEFINE_integer("n_sampled", 1024, "sampled softmax/warp loss.")
tf.app.flags.DEFINE_integer("n_resample", 30, "iterations before resample.")

tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_boolean("old_att", False, "tmp: use attribute_0.8.csv")

FLAGS = tf.app.flags.FLAGS

_buckets = []

def mylog(msg):
    print(msg)
    sys.stdout.flush()
    logging.info(msg)

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

def form_sequence_prediction(data, uids, maxlen, START_ID):
    """
    Args:
      data = [(user_id,[item_id])]
      uids = [user_id]
     Return:
      d : [(user_id,[item_id])]
    """
    d = []
    m = {}
    for uid, items in data:
        m[uid] = items
    for uid in uids:
        if uid in m:
            items = [START_ID] + m[uid][-(maxlen-1):]
        else:
            items = [START_ID]
        d.append((uid, items))

    return d

def form_sequence(data, maxlen = 100):
    """
    Args:
      data = [(u,i,week)]
    Return:
      d : [(user_id, [item_id])]
    """

    users = []
    items = []
    d = {} # d[u] = [(i,week)]
    for u,i,week in data:
        if not u in d:
            d[u] = []
        d[u].append((i,week))
    
    dd = []
    n_all_item = 0
    n_rest_item = 0
    for u in d:
        tmp = sorted(d[u],key = lambda x: x[1])
        n_all_item += len(tmp)
        while True:
            new_tmp =  [x[0] for x in tmp][:maxlen]
            n_rest_item += len(new_tmp)
            # make sure every sequence has at least one item 
            if len(new_tmp) > 0:
                dd.append((u,new_tmp))
            if len(tmp) <= maxlen:
                break
            else:
                if len(tmp) - maxlen <=7:
                    tmp = tmp[maxlen-10:]
                else:
                    tmp = tmp[maxlen:]

    # count below not valid any more
    # mylog("All item: {} Rest item: {} Remove item: {}".format(n_all_item, n_rest_item, n_all_item - n_rest_item))

    return dd

def prepare_warp(embAttr, data_tr, data_va):
    pos_item_list, pos_item_list_val = {}, {}
    for t in data_tr:
        u, i_list = t
        pos_item_list[u] = list(set(i_list))
    for t in data_va:
        u, i_list = t
        pos_item_list_val[u] = list(set(i_list))
    embAttr.prepare_warp(pos_item_list, pos_item_list_val) 

def get_device_address(s):
    add = []
    if s == "":
        for i in xrange(3):
            add.append("/cpu:0")
    else:
        add = ["/gpu:{}".format(int(x)) for x in s]
    print(add)
    return add

def split_train_dev(seq_all, ratio = 0.05):
    random.seed(FLAGS.seed)
    seq_tr, seq_va = [],[]
    for item in seq_all:
        r = random.random()
        if r < ratio:
            seq_va.append(item)
        else:
            seq_tr.append(item)
    return seq_tr, seq_va


def get_data(raw_data, data_dir=FLAGS.data_dir, combine_att=FLAGS.combine_att, 
    logits_size_tr=FLAGS.item_vocab_size, thresh=FLAGS.vocab_min_thresh, 
    use_user_feature=FLAGS.use_user_feature, test=FLAGS.test, mylog=mylog,
    use_item_feature=FLAGS.use_item_feature, recommend = False):

    (data_tr, data_va, u_attr, i_attr, item_ind2logit_ind, logit_ind2item_ind, 
        user_index, item_index) = read_attributed_data(
        raw_data_dir=raw_data, 
        data_dir=data_dir, 
        combine_att=combine_att, 
        logits_size_tr=logits_size_tr, 
        thresh=thresh, 
        use_user_feature=use_user_feature,
        use_item_feature=use_item_feature, 
        test=test, 
        mylog=mylog)

    # remove unk
    data_tr = [p for p in data_tr if (p[1] in item_ind2logit_ind)]

    # remove items before week 40 
    if FLAGS.after40:
        data_tr = [p for p in data_tr if (to_week(p[2]) >= 40)]

    # item frequency (for sampling)
    item_population, p_item = item_frequency(data_tr, FLAGS.power)

    # UNK and START
    # print(len(item_ind2logit_ind))
    # print(len(logit_ind2item_ind))
    # print(len(item_index))
    START_ID = len(item_index)
    # START_ID = i_attr.get_item_last_index()
    item_ind2logit_ind[START_ID] = 0
    seq_all = form_sequence(data_tr, maxlen = FLAGS.L)
    seq_tr0, seq_va0 = split_train_dev(seq_all,ratio=0.05)
    
    
    # calculate buckets
    global _buckets
    _buckets = calculate_buckets(seq_tr0+seq_va0, FLAGS.L, FLAGS.n_bucket)
    _buckets = sorted(_buckets)

    # split_buckets
    seq_tr = split_buckets(seq_tr0,_buckets)
    seq_va = split_buckets(seq_va0,_buckets)
    
    # get test data
    if recommend:
        from evaluate import Evaluation as Evaluate
        evaluation = Evaluate(raw_data, test=test)
        uids = evaluation.get_uinds()
        seq_test = form_sequence_prediction(seq_all, uids, FLAGS.L, START_ID)
        _buckets = calculate_buckets(seq_test, FLAGS.L, FLAGS.n_bucket)
        _buckets = sorted(_buckets)
        seq_test = split_buckets(seq_test,_buckets)
    else:
        seq_test = []
        evaluation = None
        uids = []

    # create embedAttr

    devices = get_device_address(FLAGS.N)
    with tf.device(devices[0]):
        u_attr.set_model_size(FLAGS.size)
        i_attr.set_model_size(FLAGS.size)

        # if not FLAGS.use_item_feature:
        #     mylog("NOT using item attributes")
        #     i_attr.num_features_cat = 1
        #     i_attr.num_features_mulhot = 0 

        # if not FLAGS.use_user_feature:
        #     mylog("NOT using user attributes")
        #     u_attr.num_features_cat = 1
        #     u_attr.num_features_mulhot = 0 

        embAttr = embed_attribute.EmbeddingAttribute(u_attr, i_attr, FLAGS.batch_size, FLAGS.n_sampled, _buckets[-1], FLAGS.use_sep_item, item_ind2logit_ind, logit_ind2item_ind, devices=devices)

        if FLAGS.loss in ["warp", 'mw']:
            prepare_warp(embAttr, seq_tr0, seq_va0)

    return seq_tr, seq_va, seq_test, embAttr, START_ID, item_population, p_item, evaluation, uids, user_index, item_index, logit_ind2item_ind


def create_model(session,embAttr,START_ID, run_options, run_metadata):
    devices = get_device_address(FLAGS.N)
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
                     loss = FLAGS.loss,
                     dtype = dtype,
                     devices = devices,
                     use_concat = FLAGS.use_concat,
                     no_user_id=FLAGS.no_user_id,
                     output_feat = FLAGS.output_feat,
                     no_input_item_feature = FLAGS.no_input_item_feature,
                     topk_n = FLAGS.topk,
                     run_options = run_options,
                     run_metadata = run_metadata
                     )

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    # if FLAGS.recommend or (not FLAGS.fromScratch) and ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    if FLAGS.recommend or (not FLAGS.fromScratch) and ckpt:
        mylog("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        mylog("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model


def show_all_variables():
    all_vars = tf.global_variables()
    for var in all_vars:
        mylog(var.name)


def train(raw_data=FLAGS.raw_data):

    # Read Data
    mylog("Reading Data...")
    train_set, dev_set, test_set, embAttr, START_ID, item_population, p_item, _, _, _, _, _ = get_data(raw_data,data_dir=FLAGS.data_dir)
    n_targets_train = np.sum([np.sum([len(items) for uid, items in x]) for x in train_set])
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

    steps_per_checkpoint = int(steps_per_epoch / 2)
    total_steps = steps_per_epoch * n_epoch

    # reports
    mylog(_buckets)
    mylog("Train:")
    mylog("total: {}".format(train_total_size))
    mylog("bucket sizes: {}".format(train_bucket_sizes))
    mylog("Dev:")
    mylog("total: {}".format(dev_total_size))
    mylog("bucket sizes: {}".format(dev_bucket_sizes))
    mylog("")
    mylog("Steps_per_epoch: {}".format(steps_per_epoch))
    mylog("Total_steps:{}".format(total_steps))
    mylog("Steps_per_checkpoint: {}".format(steps_per_checkpoint))

    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement = False, device_count={'CPU':8, 'GPU':1})) as sess:
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)) as sess:
        
        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model.. (this can take a few minutes)")
        model = create_model(sess, embAttr, START_ID, run_options, run_metadata)
        show_all_variables()
    
        # Data Iterators
        dite = DataIterator(model, train_set, len(train_buckets_scale), batch_size, train_buckets_scale)
        
        iteType = 0
        if iteType == 0:
            mylog("withRandom")
            ite = dite.next_random()
        elif iteType == 1:
            mylog("withSequence")
            ite = dite.next_sequence()
        
        # statistics during training
        step_time, loss = 0.0, 0.0
        current_step = 0
        previous_losses = []
        his = []
        low_ppx = float("inf")
        low_ppx_step = 0
        steps_per_report = 30
        n_targets_report = 0
        report_time = 0
        n_valid_sents = 0
        patience = FLAGS.patience
        item_sampled, item_sampled_id2idx = None, None
        
        while current_step < total_steps:
            
            # start
            start_time = time.time()
            
            # re-sample every once a while
            if FLAGS.loss in ['mw', 'mce'] and current_step % FLAGS.n_resample == 0 :
                item_sampled, item_sampled_id2idx = sample_items(item_population, FLAGS.n_sampled, p_item)
            else:
                item_sampled = None

            # data and train
            users, inputs, outputs, weights, bucket_id = ite.next()

            L = model.step(sess, users, inputs, outputs, weights, bucket_id, item_sampled=item_sampled, item_sampled_id2idx=item_sampled_id2idx)
            
            # loss and time
            step_time += (time.time() - start_time) / steps_per_checkpoint

            loss += L
            current_step += 1
            n_valid_sents += np.sum(np.sign(weights[0]))

            # for report
            report_time += (time.time() - start_time)
            n_targets_report += np.sum(weights)

            if current_step % steps_per_report == 0:                
                mylog("--------------------"+"Report"+str(current_step)+"-------------------")
                mylog("StepTime: {} Speed: {} targets / sec in total {} targets".format(report_time/steps_per_report, n_targets_report*1.0 / report_time, n_targets_train))

                report_time = 0
                n_targets_report = 0

                # Create the Timeline object, and write it to a json
                if FLAGS.profile:
                    tl = timeline.Timeline(run_metadata.step_stats)
                    ctf = tl.generate_chrome_trace_format()
                    with open('timeline.json', 'w') as f:
                        f.write(ctf)
                    exit()



            if current_step % steps_per_checkpoint == 0:
                mylog("--------------------"+"TRAIN"+str(current_step)+"-------------------")
                # Print statistics for the previous epoch.
 
                loss = loss / n_valid_sents
                perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
                mylog("global step %d learning rate %.4f step-time %.2f perplexity " "%.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
                
                train_ppx = perplexity
                
                # Save checkpoint and zero timer and loss.
                step_time, loss, n_valid_sents = 0.0, 0.0, 0
                                
                # dev data
                mylog("--------------------" + "DEV" + str(current_step) + "-------------------")
                eval_loss, eval_ppx = evaluate(sess, model, dev_set, item_sampled_id2idx=item_sampled_id2idx)
                mylog("dev: ppx: {}".format(eval_ppx))

                his.append([current_step, train_ppx, eval_ppx])

                if eval_ppx < low_ppx:
                    patience = FLAGS.patience
                    low_ppx = eval_ppx
                    low_ppx_step = current_step
                    checkpoint_path = os.path.join(FLAGS.train_dir, "best.ckpt")
                    mylog("Saving best model....")
                    s = time.time()
                    model.saver.save(sess, checkpoint_path, global_step=0, write_meta_graph = False)
                    mylog("Best model saved using {} sec".format(time.time()-s))
                else:
                    patience -= 1

                if patience <= 0:
                    mylog("Training finished. Running out of patience.")
                    break

                sys.stdout.flush()

def evaluate(sess, model, data_set, item_sampled_id2idx=None):
    # Run evals on development set and print their perplexity/loss.
    dropoutRateRaw = FLAGS.keep_prob
    sess.run(model.dropout10_op)

    start_id = 0
    loss = 0.0
    n_steps = 0
    n_valids = 0
    batch_size = FLAGS.batch_size
    
    dite = DataIterator(model, data_set, len(_buckets), batch_size, None)
    ite = dite.next_sequence(stop = True)

    for users, inputs, outputs, weights, bucket_id in ite:
        L = model.step(sess, users, inputs, outputs, weights, bucket_id, forward_only = True)
        loss += L
        n_steps += 1
        n_valids += np.sum(np.sign(weights[0]))

    loss = loss/(n_valids)
    ppx = math.exp(loss) if loss < 300 else float("inf")

    sess.run(model.dropoutAssign_op)

    return loss, ppx

def recommend(raw_data=FLAGS.raw_data):
    
    # Read Data
    mylog("recommend")
    mylog("Reading Data...")
    _, _, test_set, embAttr, START_ID, _, _, evaluation, uids, user_index, item_index, logit_ind2item_ind = get_data(
        raw_data, data_dir=FLAGS.data_dir, recommend =True)
    test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog(_buckets)
    mylog("Test:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))
    
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement = False)) as sess:

        # runtime profile
        if FLAGS.profile:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
        else:
            run_options = None
            run_metadata = None

        mylog("Creating Model")
        model = create_model(sess, embAttr, START_ID, run_options, run_metadata)
        show_all_variables()
        
        sess.run(model.dropoutRate.assign(1.0))

        start_id = 0
        n_steps = 0
        batch_size = FLAGS.batch_size
    
        dite = DataIterator(model, test_set, len(_buckets), batch_size, None)
        ite = dite.next_sequence(stop = True, recommend = True)

        n_total_user = len(uids)
        n_recommended = 0
        uid2rank = {}
        for r, uid in enumerate(uids):
            uid2rank[uid] = r
        rec = np.zeros((n_total_user,FLAGS.topk), dtype = int)
        rec_value = np.zeros((n_total_user,FLAGS.topk), dtype = float)
        start = time.time()

        for users, inputs, positions, valids, bucket_id in ite:
            results = model.step_recommend(sess, users, inputs, positions, bucket_id)
            for i, valid in enumerate(valids):
                if valid == 1:
                    n_recommended += 1
                    if n_recommended % 1000 == 0:
                        mylog("Evaluating n {} bucket_id {}".format(n_recommended, bucket_id))
                    uid, topk_values, topk_indexes = results[i]
                    rank= uid2rank[uid]
                    rec[rank,:] = topk_indexes                
                    rec_value[rank,:] = topk_values
            n_steps += 1
        end = time.time()
        mylog("Time used {} sec for {} steps {} users ".format(end-start, n_steps, n_recommended))

        ind2id = {}
        for iid in item_index:
            uind = item_index[iid]
            assert(uind not in ind2id)
            ind2id[uind] = iid

        R = {}
        for i in xrange(n_total_user):
            uid = uids[i]
            R[uid] = [ind2id[logit_ind2item_ind[v]] for v in list(rec[i, :])]

        evaluation.eval_on(R)

        scores_self, scores_ex = evaluation.get_scores()
        mylog("====evaluation scores (NDCG, RECALL, PRECISION, MAP) @ 2,5,10,20,30====")
        mylog("METRIC_FORMAT (self): {}".format(scores_self))
        mylog("METRIC_FORMAT (ex  ): {}".format(scores_ex))
        
        # save the two matrix
        np.save(os.path.join(FLAGS.train_dir,"top{}_index.npy".format(FLAGS.topk)),rec)
        np.save(os.path.join(FLAGS.train_dir,"top{}_value.npy".format(FLAGS.topk)),rec_value)


def ensemble(raw_data=FLAGS.raw_data):
    # Read Data
    mylog("Ensemble {} {}".format(FLAGS.train_dir, FLAGS.ensemble_suffix))
    mylog("Reading Data...")
    # task = Task(FLAGS.dataset)
    _, _, test_set, embAttr, START_ID, _, _, evaluation, uids, user_index, item_index, logit_ind2item_ind = get_data(
        raw_data, data_dir=FLAGS.data_dir, recommend =True)
    test_bucket_sizes = [len(test_set[b]) for b in xrange(len(_buckets))]
    test_total_size = int(sum(test_bucket_sizes))

    # reports
    mylog(_buckets)
    mylog("Test:")
    mylog("total: {}".format(test_total_size))
    mylog("buckets: {}".format(test_bucket_sizes))
    
    # load top_index, and top_value
    suffixes = FLAGS.ensemble_suffix.split(',')
    top_indexes = []
    top_values = []
    for suffix in suffixes:
        # dir_path = FLAGS.train_dir+suffix
        dir_path = FLAGS.train_dir.replace('seed', 'seed' + suffix)
        mylog("Loading results from {}".format(dir_path))
        index_path = os.path.join(dir_path,"top{}_index.npy".format(FLAGS.topk))
        value_path = os.path.join(dir_path,"top{}_value.npy".format(FLAGS.topk))
        top_index = np.load(index_path)
        top_value = np.load(value_path)
        top_indexes.append(top_index)
        top_values.append(top_value)

    # ensemble
    rec = np.zeros(top_indexes[0].shape)
    for row in xrange(rec.shape[0]):
        v = {}
        for i in xrange(len(suffixes)):
            for col in xrange(rec.shape[1]):
                index = top_indexes[i][row,col]
                value = top_values[i][row,col]
                if index not in v:
                    v[index] = 0
                v[index] += value
        items = [(index,v[index]/len(suffixes)) for index in v ]
        items = sorted(items,key = lambda x: -x[1])
        rec[row:] = [x[0] for x in items][:FLAGS.topk]
        if row % 1000 == 0:
            mylog("Ensembling n {}".format(row))
        
    ind2id = {}
    for iid in item_index:
        uind = item_index[iid]
        assert(uind not in ind2id)
        ind2id[uind] = iid

    R = {}
    for i in xrange(n_total_user):
        uid = uids[i]
        R[uid] = [ind2id[logit_ind2item_ind[v]] for v in list(rec[i, :])]

    evaluation.eval_on(R)

    scores_self, scores_ex = evaluation.get_scores()
    mylog("====evaluation scores (NDCG, RECALL, PRECISION, MAP) @ 2,5,10,20,30====")
    mylog("METRIC_FORMAT (self): {}".format(scores_self))
    mylog("METRIC_FORMAT (ex  ): {}".format(scores_ex))
        

def main(_):
    print("V 2017-02-28 12:23")
    if FLAGS.test:
        if FLAGS.data_dir[-1] == '/':
            FLAGS.data_dir = FLAGS.data_dir[:-1] + '_test'
        else:
            FLAGS.data_dir = FLAGS.data_dir + '_test'

    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)
    
    if FLAGS.ensemble:
        suffixes = FLAGS.ensemble_suffix.split(',')
        # log_path = os.path.join(FLAGS.train_dir+suffixes[0],"log.ensemble.txt.{}".format(FLAGS.topk))
        log_path = os.path.join(FLAGS.train_dir.replace('seed', 'seed'+suffixes[0]),"log.ensemble.txt.{}".format(FLAGS.topk))
        logging.basicConfig(filename=log_path,level=logging.DEBUG, filemode ="w")
        ensemble()
        return 

    if FLAGS.recommend:
        log_path = os.path.join(FLAGS.train_dir,"log.recommend.txt.{}".format(FLAGS.topk))
        logging.basicConfig(filename=log_path,level=logging.DEBUG, filemode ="w")
        recommend()
    else:
        log_path = os.path.join(FLAGS.train_dir,"log.txt")
        if FLAGS.fromScratch:
            filemode = "w"
        else:
            filemode = "a"
        logging.basicConfig(filename=log_path,level=logging.DEBUG, filemode = filemode)
        train()
    
if __name__ == "__main__":
    tf.app.run()
