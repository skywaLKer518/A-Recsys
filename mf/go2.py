from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math, os, shutil, sys
import random, time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
sys.path.insert(0, '../utils')

from xing_data import data_read
# from ml_data import data_read
from xing_eval import *
from xing_submit import *
from prepare_train import positive_items, item_frequency, sample_items

# in order to profile
from tensorflow.python.client import timeline

tf.app.flags.DEFINE_float("learning_rate", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "dropout rate.")
tf.app.flags.DEFINE_float("power", 0.5, "related to sampling rate.")
tf.app.flags.DEFINE_float("learning_rate_decay_factor", 1.0,
                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 64,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 20, "Size of each model layer.")
tf.app.flags.DEFINE_integer("hidden_size", 500, "when nonlinear proj used")
tf.app.flags.DEFINE_integer("n_resample", 50, "iterations before resample.")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("user_vocab_size", 150000, "User vocabulary size.")
tf.app.flags.DEFINE_integer("item_vocab_size", 50000, "Item vocabulary size.")
tf.app.flags.DEFINE_integer("n_sampled", 1024, "sampled softmax/warp loss.")
tf.app.flags.DEFINE_string("data_dir", "./data0", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./test0", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("patience", 20,
                            "exit if the model can't improve for $patience evals")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 4000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("use_user_feature", True, "RT")
tf.app.flags.DEFINE_boolean("use_item_feature", True, "RT")
tf.app.flags.DEFINE_boolean("recommend", False,
                            "Set to True for recommend items.")
tf.app.flags.DEFINE_boolean("recommend_new", False,
                            "Set to True for recommend new items that were not used to train.")
tf.app.flags.DEFINE_boolean("device_log", False,
                            "Set to True for logging device usages.")
tf.app.flags.DEFINE_boolean("eval", True,
                            "Set to True for evaluation.")
tf.app.flags.DEFINE_boolean("use_more_train", False,
                            "Set true if use non-appearred items to train.")
tf.app.flags.DEFINE_boolean("profile", False, "False = no profile, True = profile")

tf.app.flags.DEFINE_string("loss", 'ce',
                            "loss function")
tf.app.flags.DEFINE_string("model_option", 'loss',
                            "model to evaluation")
tf.app.flags.DEFINE_string("log", 'log/log0', "logfile")
tf.app.flags.DEFINE_string("nonlinear", 'linear', "nonlinear activation")
tf.app.flags.DEFINE_integer("gpu", -1, "gpu card number")

# Xing related
tf.app.flags.DEFINE_integer("ta", 0, "target_active")
tf.app.flags.DEFINE_integer("top_N_items", 30,
                            "number of items output")

FLAGS = tf.app.flags.FLAGS

def mylog(msg):
  print(msg)
  logging.info(msg)
  return

def read_data():
  (data_tr, data_va, u_attr, i_attr, item_ind2logit_ind, 
    logit_ind2item_ind) = data_read(FLAGS.data_dir, _submit = 0, ta = FLAGS.ta, 
    logits_size_tr=FLAGS.item_vocab_size)
  print('length of item_ind2logit_ind', len(item_ind2logit_ind))
  return (data_tr, data_va, u_attr, i_attr, item_ind2logit_ind, 
    logit_ind2item_ind)

def create_model(session, u_attributes=None, i_attributes=None, 
  item_ind2logit_ind=None, logit_ind2item_ind=None, 
  loss = FLAGS.loss, logit_size_test=None, ind_item = None):  
  gpu = None if FLAGS.gpu == -1 else FLAGS.gpu
  n_sampled = FLAGS.n_sampled if FLAGS.loss in ['mw', 'mce'] else None
  import mf_model2 
  model = mf_model2.LatentProductModel(FLAGS.user_vocab_size, 
    FLAGS.item_vocab_size, FLAGS.size, FLAGS.num_layers, 
    FLAGS.batch_size, FLAGS.learning_rate, 
    FLAGS.learning_rate_decay_factor, u_attributes, i_attributes, 
    item_ind2logit_ind, logit_ind2item_ind, loss_function = loss, GPU=gpu, 
    logit_size_test=logit_size_test, nonlinear=FLAGS.nonlinear, 
    dropout=FLAGS.keep_prob, n_sampled=n_sampled, indices_item=ind_item, 
    hidden_size=FLAGS.hidden_size)

  if not os.path.isdir(FLAGS.train_dir):
    os.mkdir(FLAGS.train_dir)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if FLAGS.recommend and ckpt:
    print("%s" % ckpt.model_checkpoint_path)
    if FLAGS.model_option == 'loss':
      f = os.path.join(FLAGS.train_dir, 'go.ckpt-best')
    elif FLAGS.model_option == 'auc':
      f = os.path.join(FLAGS.train_dir, 'go.ckpt-best_auc')
    else:
      print("no such models %s" % FLAGS.model_option)
      exit()
    ckpt.model_checkpoint_path = f
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)

  elif ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    logging.info("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model

def train():
  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
    log_device_placement=FLAGS.device_log)) as sess:
    run_options = None
    run_metadata = None
    if FLAGS.profile:
      run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
      run_metadata = tf.RunMetadata()
      FLAGS.steps_per_checkpoint = 30
    
    print("reading data")
    logging.info("reading data")
    (data_tr, data_va, u_attributes, i_attributes,item_ind2logit_ind, 
      logit_ind2item_ind) = read_data()

    print("train/dev size: %d/%d" %(len(data_tr),len(data_va)))
    logging.info("train/dev size: %d/%d" %(len(data_tr),len(data_va)))

    '''
    remove some rare items in both train and valid set
    this helps make train/valid set distribution similar 
    to each other
    '''
    print("original train/dev size: %d/%d" %(len(data_tr),len(data_va)))
    logging.info("original train/dev size: %d/%d" %(len(data_tr),len(data_va)))
    data_tr = [p for p in data_tr if (p[1] in item_ind2logit_ind)]
    data_va = [p for p in data_va if (p[1] in item_ind2logit_ind)]
    print("new train/dev size: %d/%d" %(len(data_tr),len(data_va)))
    logging.info("new train/dev size: %d/%d" %(len(data_tr),len(data_va)))

    power = FLAGS.power
    item_pop, p_item = item_frequency(data_tr, power)

    if FLAGS.use_more_train:
      item_population = range(len(item_ind2logit_ind))
    else:
      item_population = item_pop

    if not FLAGS.use_item_feature:
      mylog("NOT using item attributes")
      i_attributes.num_features_cat = 1
      i_attributes.num_features_mulhot = 0
    if not FLAGS.use_user_feature:
      mylog("NOT using user attributes")
      u_attributes.num_features_cat = 1
      u_attributes.num_features_mulhot = 0

    model = create_model(sess, u_attributes, i_attributes, item_ind2logit_ind,
      logit_ind2item_ind, loss=FLAGS.loss, ind_item=item_population)

    pos_item_list, pos_item_list_val = None, None
    if FLAGS.loss in ['warp', 'mw', 'bbpr']:
      pos_item_list, pos_item_list_val = positive_items(data_tr, data_va)
      model.prepare_warp(pos_item_list, pos_item_list_val)

    mylog('started training')
    step_time, loss, current_step, auc = 0.0, 0.0, 0, 0.0
    
    repeat = 5 if FLAGS.loss.startswith('bpr') else 1   
    patience = FLAGS.patience

    if os.path.isfile(os.path.join(FLAGS.train_dir, 'auc_train.npy')):
      auc_train = list(np.load(os.path.join(FLAGS.train_dir, 'auc_train.npy')))
      auc_dev = list(np.load(os.path.join(FLAGS.train_dir, 'auc_dev.npy')))
      previous_losses = list(np.load(os.path.join(FLAGS.train_dir, 
        'loss_train.npy')))
      losses_dev = list(np.load(os.path.join(FLAGS.train_dir, 'loss_dev.npy')))
      best_auc = max(auc_dev)
      best_loss = min(losses_dev)
    else:
      previous_losses, auc_train, auc_dev, losses_dev = [], [], [], []
      best_auc, best_loss = -1, 1000000

    item_sampled, item_sampled_id2idx = None, None
    while True:
      ranndom_number_01 = np.random.random_sample()
      start_time = time.time()
      (user_input, item_input, neg_item_input) = model.get_batch(data_tr, 
        loss=FLAGS.loss)
      
      if FLAGS.loss in ['mw', 'mce'] and current_step % FLAGS.n_resample == 0:
        item_sampled, item_sampled_id2idx = sample_items(item_population, 
          FLAGS.n_sampled, p_item)
      else:
        item_sampled = None

      step_loss = model.step(sess, user_input, item_input, 
        neg_item_input, item_sampled, item_sampled_id2idx, loss=FLAGS.loss,run_op=run_options, run_meta=run_metadata)

      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      # auc += step_auc / FLAGS.steps_per_checkpoint
      current_step += 1
      
      if current_step % FLAGS.steps_per_checkpoint == 0:

        if FLAGS.loss in ['ce', 'mce']:
          perplexity = math.exp(loss) if loss < 300 else float('inf')
          mylog("global step %d learning rate %.4f step-time %.4f perplexity %.2f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, perplexity))
        else:
          mylog("global step %d learning rate %.4f step-time %.4f loss %.3f" % (model.global_step.eval(), model.learning_rate.eval(), step_time, loss))
        if FLAGS.profile:
          # Create the Timeline object, and write it to a json
          tl = timeline.Timeline(run_metadata.step_stats)
          ctf = tl.generate_chrome_trace_format()
          with open('timeline.json', 'w') as f:
              f.write(ctf)
          exit()

        # Decrease learning rate if no improvement was seen over last 3 times.
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op) 
        previous_losses.append(loss)
        auc_train.append(auc)
        step_time, loss, auc = 0.0, 0.0, 0.0

        if not FLAGS.eval:
          continue
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "go.ckpt")
        current_model = model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        
        # Run evals on development set and print their loss/auc.
        l_va = len(data_va)
        eval_loss, eval_auc = 0.0, 0.0
        count_va = 0
        start_time = time.time()
        for idx_s in range(0, l_va, FLAGS.batch_size):
          idx_e = idx_s + FLAGS.batch_size
          if idx_e > l_va:
            break
          lt = data_va[idx_s:idx_e]
          user_va = [x[0] for x in lt]
          item_va = [x[1] for x in lt]
          for _ in range(repeat):
            # item_va_neg = model.get_eval_batch(FLAGS.loss, user_va, item_va, 
            #   hist_withval)
            item_va_neg = None
            the_loss = 'warp' if FLAGS.loss == 'mw' else FLAGS.loss
            eval_loss0 = model.step(sess, user_va, item_va, item_va_neg,
              None, None, forward_only=True, 
              loss=the_loss)
            eval_loss += eval_loss0
            # eval_auc += auc0
            count_va += 1
        eval_loss /= count_va
        eval_auc /= count_va
        step_time = (time.time() - start_time) / count_va
        if FLAGS.loss in ['ce', 'mce']:
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          mylog("  eval: perplexity %.2f eval_auc %.4f step-time %.4f" % (
            eval_ppx, eval_auc, step_time))
        else:
          mylog("  eval: loss %.3f eval_auc %.4f step-time %.4f" % (eval_loss, 
            eval_auc, step_time))
        sys.stdout.flush()

        # if eval_auc > best_auc:
        #   best_auc = eval_auc
        #   new_filename = os.path.join(FLAGS.train_dir, "go.ckpt-best_auc")
        #   shutil.copy(current_model, new_filename)
        #   patience = FLAGS.patience
        
        if eval_loss < best_loss:
          best_loss = eval_loss
          new_filename = os.path.join(FLAGS.train_dir, "go.ckpt-best")
          shutil.copy(current_model, new_filename)
          patience = FLAGS.patience

        if eval_loss > best_loss: # and eval_auc < best_auc:
          patience -= 1

        auc_dev.append(eval_auc)
        losses_dev.append(eval_loss)

        np.save(os.path.join(FLAGS.train_dir, 'auc_train'), auc_train)
        np.save(os.path.join(FLAGS.train_dir, 'auc_dev'), auc_dev)
        np.save(os.path.join(FLAGS.train_dir, 'loss_train'), previous_losses)
        np.save(os.path.join(FLAGS.train_dir, 'loss_dev'), losses_dev)

        if patience < 0:
          print("no improvement for too long.. terminating..")
          print("best auc %.4f" % best_auc)
          print("best loss %.4f" % best_loss)
          logging.info("no improvement for too long.. terminating..")
          logging.info("best auc %.4f" % best_auc)
          logging.info("best loss %.4f" % best_loss)
          sys.stdout.flush()
          break
  return

def recommend():

  with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, 
    log_device_placement=FLAGS.device_log)) as sess:
    print("reading data")
    (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
      logit_ind2item_ind) = read_data()
    
    print("train/dev size: %d/%d" %(len(data_tr),len(data_va)))
  
    evaluation = Evaluate(logit_ind2item_ind, ta=FLAGS.ta)
    
    model = create_model(sess, u_attributes, i_attributes, item_ind2logit_ind,
      logit_ind2item_ind, loss=FLAGS.loss, ind_item=None)

    Uinds = evaluation.get_uinds()
    N = len(Uinds)
    print("N = %d" % N)
    rec = np.zeros((N, 30), dtype=int)
    count = 0
    time_start = time.time()
    for idx_s in range(0, N, FLAGS.batch_size):
      count += 1
      if count % 100 == 0:
        print("idx: %d, c: %d" % (idx_s, count))
        
      idx_e = idx_s + FLAGS.batch_size
      if idx_e <= N:
        users = Uinds[idx_s: idx_e]
        recs = model.step(sess, users, None, None, forward_only=True, 
          recommend = True, recommend_new = FLAGS.recommend_new)
        rec[idx_s:idx_e, :] = recs
      else:
        users = range(idx_s, N) + [0] * (idx_e - N)
        users = [Uinds[t] for t in users]
        recs = model.step(sess, users, None, None, forward_only=True, 
          recommend = True, recommend_new = FLAGS.recommend_new)
        idx_e = N
        rec[idx_s:idx_e, :] = recs[:(idx_e-idx_s),:]
    # return rec: i:  uinds[i] --> logid

    time_end = time.time()
    mylog("Time used %.1f" % (time_end - time_start))

    R = evaluation.gen_rec(rec, FLAGS.recommend_new)

    evaluation.eval_on(R)
    s1, s2, s3, r20, p5 = evaluation.get_scores()
    mylog('scores: \n\t{}\n\t{}\n\t{}\n\t{}\n\t{}'.format(s1, s2, s3, r20, p5))
  return

def main(_):
  
  # logging.debug('This message should go to the log file')
  # logging.info('So should this')
  # logging.warning('And this, too')

  if not FLAGS.recommend:
    logging.basicConfig(filename=FLAGS.log,level=logging.DEBUG)
    train()
  else:
    logging.basicConfig(filename=FLAGS.log+'_rec',level=logging.DEBUG)
    recommend()
  return

if __name__ == "__main__":
    tf.app.run()

