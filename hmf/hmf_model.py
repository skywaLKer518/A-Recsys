
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import time
import sys
import itertools

sys.path.insert(0, '../attributes')
import embed_attribute

class LatentProductModel(object):
  def __init__(self, user_size, item_size, size,
               num_layers, batch_size, learning_rate,
               learning_rate_decay_factor, user_attributes=None, 
               item_attributes=None, item_ind2logit_ind=None, 
               logit_ind2item_ind=None, loss_function='ce', GPU=None, 
               logit_size_test=None, nonlinear=None, dropout=1.0, 
               n_sampled=None, indices_item=None, dtype=tf.float32, 
               top_N_items=100, hidden_size=500):

    self.user_size = user_size
    self.item_size = item_size
    self.top_N_items = top_N_items

    if user_attributes is not None:
      user_attributes.set_model_size(size)
      self.user_attributes = user_attributes
    if item_attributes is not None:
      item_attributes.set_model_size(size)
      self.item_attributes = item_attributes

    self.item_ind2logit_ind = item_ind2logit_ind
    self.logit_ind2item_ind = logit_ind2item_ind
    if logit_ind2item_ind is not None:
      self.logit_size = len(logit_ind2item_ind)
    if indices_item is not None:
      self.indices_item = indices_item
    else:
      self.indices_item = range(self.logit_size)
    self.logit_size_test = logit_size_test

    self.nonlinear = nonlinear
    self.loss_function = loss_function
    self.n_sampled = n_sampled
    self.batch_size = batch_size
    
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    self.att_emb = None
    self.dtype=dtype
    
    self.data_length = None
    self.train_permutation = None
    self.start_index = None

    mb = self.batch_size
    ''' this is mapped item target '''
    self.item_target = tf.placeholder(tf.int32, shape = [mb], name = "item")
    self.item_id_target = tf.placeholder(tf.int32, shape = [mb], name = "item_id")

    self.dropout = dropout
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    m = embed_attribute.EmbeddingAttribute(user_attributes, item_attributes, mb, 
      self.n_sampled, 0, False, item_ind2logit_ind, logit_ind2item_ind)
    self.att_emb = m
    embedded_user, user_b = m.get_batch_user(self.keep_prob, False)
    
    if self.nonlinear in ['relu', 'tanh']:
      act = tf.nn.relu if self.nonlinear == 'relu' else tf.tanh
      w1 = tf.get_variable('w1', [size, hidden_size], dtype=self.dtype)
      b1 = tf.get_variable('b1', [hidden_size], dtype=self.dtype)
      w2 = tf.get_variable('w2', [hidden_size, size], dtype=self.dtype)
      b2 = tf.get_variable('b2', [size], dtype=self.dtype)

      embedded_user, user_b = m.get_batch_user(1.0, False)
      h0 = tf.nn.dropout(act(embedded_user), self.keep_prob)
      
      h1 = act(tf.matmul(h0, w1) + b1)
      h1 = tf.nn.dropout(h1, self.keep_prob)

      h2 = act(tf.matmul(h1, w2) + b2)
      embedded_user = tf.nn.dropout(h2, self.keep_prob)

    pos_embs_item, pos_item_b = m.get_batch_item('pos', batch_size)
    pos_embs_item = tf.reduce_mean(pos_embs_item, 0)

    neg_embs_item, neg_item_b = m.get_batch_item('neg', batch_size)
    neg_embs_item = tf.reduce_mean(neg_embs_item, 0)
    # print('debug: user, item dim', embedded_user.get_shape(), neg_embs_item.get_shape())

    print("construct postive/negative items/scores \n(for bpr loss, AUC)")
    self.pos_score = tf.reduce_sum(tf.mul(embedded_user, pos_embs_item), 1) + pos_item_b
    self.neg_score = tf.reduce_sum(tf.mul(embedded_user, neg_embs_item), 1) + neg_item_b
    neg_pos = self.neg_score - self.pos_score
    self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))

    # mini batch version
    if self.n_sampled is not None:
      print("sampled prediction")
      sampled_logits = m.get_prediction(embedded_user, 'sampled')
      # embedded_item, item_b = m.get_sampled_item(self.n_sampled)
      # sampled_logits = tf.matmul(embedded_user, tf.transpose(embedded_item)) + item_b
      target_score = m.get_target_score(embedded_user, self.item_id_target)

    print("non-sampled prediction")
    logits = m.get_prediction(embedded_user)

    loss = self.loss_function
    if loss in ['warp', 'ce', 'bbpr']:
      batch_loss = m.compute_loss(logits, self.item_target, loss)
    elif loss in ['mw']:
      # batch_loss = m.compute_loss(sampled_logits, self.pos_score, loss)
      batch_loss = m.compute_loss(sampled_logits, target_score, loss)
      batch_loss_eval = m.compute_loss(logits, self.item_target, 'warp')

    elif loss in ['bpr', 'bpr-hinge']:
      batch_loss = m.compute_loss(neg_pos, self.item_target, loss)
    else:
      print("not implemented!")
      exit(-1)
    if loss in ['warp', 'mw', 'bbpr']:
      self.set_mask, self.reset_mask = m.get_warp_mask()

    self.loss = tf.reduce_mean(batch_loss)
    self.loss_eval = tf.reduce_mean(batch_loss_eval) if loss == 'mw' else self.loss
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    opt = tf.train.AdagradOptimizer(self.learning_rate)
    # opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = tf.gradients(self.loss, params)
    self.updates = opt.apply_gradients(
      zip(gradients, params), global_step=self.global_step)

    self.output = logits
    values, self.indices= tf.nn.top_k(self.output, self.top_N_items, sorted=True)
    # self.saver = tf.train.Saver(tf.global_variables())
    self.saver = tf.train.Saver(tf.all_variables())

  def prepare_warp(self, pos_item_set, pos_item_set_eval):
    self.att_emb.prepare_warp(pos_item_set, pos_item_set_eval)
    return 

  def step(self, session, user_input, item_input, neg_item_input=None, 
    item_sampled = None, item_sampled_id2idx = None,
    forward_only=False, recommend=False, recommend_new = False, loss=None, 
    run_op=None, run_meta=None):
    input_feed = {}
    if forward_only or recommend:
      input_feed[self.keep_prob.name] = 1.0
    else:
      input_feed[self.keep_prob.name] = self.dropout
        
    if recommend == False:
      targets = self.att_emb.target_mapping([item_input])
      input_feed[self.item_target.name] = targets[0]
      if loss in ['mw']:
        input_feed[self.item_id_target.name] = item_input
      
    # if loss in ['mw', 'mce'] and recommend == False:
      # input_feed[self.item_target.name] = [item_sampled_id2idx[v] for v in item_input]

    if self.att_emb is not None:
      (update_sampled, input_feed_sampled, 
        input_feed_warp) = self.att_emb.add_input(input_feed, user_input, 
        item_input, neg_item_input=neg_item_input, 
        item_sampled = item_sampled, item_sampled_id2idx = item_sampled_id2idx, 
        forward_only=forward_only, recommend=recommend, loss = loss)

    if not recommend:
      if not forward_only:
        # output_feed = [self.updates, self.loss, self.auc]
        output_feed = [self.updates, self.loss]
        # output_feed = [self.embedded_user, self.pos_embs_item]
      else:
        # output_feed = [self.loss_eval, self.auc]
        output_feed = [self.loss_eval]
    else:
      if recommend_new:
        output_feed = [self.indices_test]
      else:
        output_feed = [self.indices]

    if item_sampled is not None and loss in ['mw', 'mce']:
      session.run(update_sampled, input_feed_sampled)

    if (loss in ['warp', 'bbpr', 'mw']) and recommend is False:
      session.run(self.set_mask[loss], input_feed_warp)

    if run_op is not None and run_meta is not None:
      outputs = session.run(output_feed, input_feed, options=run_op, run_metadata=run_meta)
    else:
      outputs = session.run(output_feed, input_feed)

    if (loss in ['warp', 'bbpr', 'mw']) and recommend is False:
      session.run(self.reset_mask[loss], input_feed_warp)

    if not recommend:
      if not forward_only:
        return outputs[1]#, outputs[2]#, outputs[3] #, outputs[3], outputs[4]
      else:
        return outputs[0]#, outputs[1]
    else:
      return outputs[0]

  def get_batch(self, data, loss = 'ce', hist = None):
    batch_user_input, batch_item_input = [], []
    batch_neg_item_input = []

    count = 0
    while count < self.batch_size:
      u, i, _ = random.choice(data)
      batch_user_input.append(u)
      batch_item_input.append(i)
      count += 1
        
    return batch_user_input, batch_item_input, batch_neg_item_input

  def get_permuted_batch(self, data):
    batch_user_input, batch_item_input = [], []
    if self.data_length == None:
      self.data_length = len(data)
      self.start_index = 0
      self.train_permutation = np.random.permutation(self.data_length)
    if self.start_index + self.batch_size >= self.data_length:
      self.start_index = 0
      self.train_permutation = np.random.permutation(self.data_length)

    indices = range(self.start_index, self.start_index + self.batch_size)
    indices = self.train_permutation[indices]
    self.start_index += self.batch_size
    for j in indices:
      u, i , _  = data[j]
      batch_user_input.append(u)
      batch_item_input.append(i)
    return batch_user_input, batch_item_input, None


  # def get_eval_batch(self, loss, users, items, hist = None):
  #   neg_items = []
  #   l, i = len(users), 0
  #   while i < l:
  #     u = users[i]
  #     i2 = random.choice(self.indices_item)
  #     while i2 in hist[u]:
  #       i2 = random.choice(self.indices_item)
  #     neg_items.append(i2)
  #     i += 1
        
  #   return neg_items #, None, None