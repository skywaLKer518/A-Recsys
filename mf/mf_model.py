
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random, math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops

import time
import attribute
import itertools

class LatentProductModel(object):
  def __init__(self, user_size, item_size, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, user_attributes=None, 
               item_attributes=None, item_ind2logit_ind=None, 
               logit_ind2item_ind=None, loss_function='ce', GPU=None, 
               logit_size_test=None, nonlinear=None, dropout=1.0, 
               n_sampled=None, indices_item=None):

    self.user_size = user_size
    self.item_size = item_size

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

    mb = self.batch_size
    self.user_input = tf.placeholder(tf.int32, shape = [mb], name = "user")
    self.item_target = tf.placeholder(tf.int32, shape = [mb], name = "item")
    # self.neg_item_input = tf.placeholder(tf.int32, shape = [mb], 
    #   name = "neg_item")
    
    self.dropout = dropout
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    self.input_att_map = True
    self.use_user_bias = False # not sure if need to be True

    m = attribute.EmbeddingAttribute(user_attributes, item_attributes, mb, 
      self.n_sampled, item_ind2logit_ind, logit_ind2item_ind)
    self.att_emb = m
    embedded_user, user_b = m.get_batch_user(self.keep_prob)
    user_model_size = m.get_user_model_size()
    proj_user_drops = m.get_user_proj(embedded_user, self.keep_prob, 
      user_model_size, self.nonlinear)
    
    pos_embs_item, pos_item_b = m.get_batch_item('pos', batch_size)
    neg_embs_item, neg_item_b = m.get_batch_item('neg', batch_size)

    print("construct postive/negative items/scores (for bpr los, AUC")
    pos_scores, neg_scores = [],[]
    for i in xrange(m.num_item_features):
      pos_scores.append(tf.reduce_sum(tf.mul(proj_user_drops[i], 
        pos_embs_item[i]), 1))
      neg_scores.append(tf.reduce_sum(tf.mul(proj_user_drops[i], 
        neg_embs_item[i]), 1))
    self.pos_score = tf.reduce_sum(pos_scores, 0) + pos_item_b
    self.neg_score = tf.reduce_sum(neg_scores, 0) + neg_item_b
    neg_pos = self.neg_score - self.pos_score
    self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))

    ''' mini batch version. not ready'''
    # print("construct mini-batch item candicate pool")
    # embedded_item, item_b = m.get_batch_items('target', self.n_sampled)
    # # compute sampled 'logits'/'scores'
    # innerps = []
    # for i in xrange(m.num_item_features):
    #   innerps.append(tf.matmul(embedded_item[i], 
    #     tf.transpose(proj_user_drops[i])))
    # sampled_logits = tf.transpose(tf.reduce_sum(innerps, 0)) + item_b

    print("prediction")
    logits = m.get_prediction(proj_user_drops)

    batch_loss = m.compute_loss(logits, self.item_target, 
      self.loss_function)
    self.loss = tf.reduce_mean(batch_loss)
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    opt = tf.train.AdagradOptimizer(self.learning_rate)
    # opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = tf.gradients(self.loss, params)
    self.updates = opt.apply_gradients(
      zip(gradients, params), global_step=self.global_step)
    self.output = logits
    values, self.indices= tf.nn.top_k(self.output, 30, sorted=True)
    self.saver = tf.train.Saver(tf.all_variables())

  def step(self, session, user_input, item_input, neg_item_input=None, 
    item_sampled = None, item_sampled_id2idx = None,
    forward_only=False, recommend=False, recommend_new = False, loss=None, 
    run_op=None, run_meta=None):
    input_feed = {}
    if forward_only or recommend:
      input_feed[self.keep_prob.name] = 1.0
    else:
      input_feed[self.keep_prob.name] = self.dropout
    
    if self.att_emb is not None:
      self.att_emb.add_input(input_feed, user_input, item_input, 
        neg_item_input=neg_item_input, item_sampled = item_sampled, 
        item_sampled_id2idx = item_sampled_id2idx, 
        forward_only=forward_only, recommend=recommend, 
        recommend_new = recommend_new, loss = loss)

    if not recommend:
      if not forward_only:
        # output_feed = [self.updates, self.loss, self.auc]
        output_feed = [self.updates, self.loss, self.auc] #, self.neg_score, self.pos_score]
      else:
        output_feed = [self.loss, self.auc]
    else:
      if recommend_new:
        output_feed = [self.indices_test]
      else:
        output_feed = [self.indices]

    if run_op is not None and run_meta is not None:
      outputs = session.run(output_feed, input_feed, options=run_op, run_metadata=run_meta)
    else:
      outputs = session.run(output_feed, input_feed)

    if not recommend:
      if not forward_only:
        return outputs[1], outputs[2] #, outputs[3], outputs[4]
      else:
        return outputs[0], outputs[1]
    else:
      return outputs[0]

  def get_batch(self, data, loss = 'ce', hist = None):
    batch_user_input, batch_item_input = [], []
    batch_neg_item_input = []

    count = 0
    while count < self.batch_size:
      u, i = random.choice(data)
      batch_user_input.append(u)
      batch_item_input.append(i)
      
      i2 = random.choice(self.indices_item)
      while i2 in hist[u]:
        i2 = random.choice(self.indices_item)
      batch_neg_item_input.append(i2)
      count += 1

    if loss in  ['mw', 'mce']:
      pos_item_set = set(batch_item_input)
      l0 = len(pos_item_set)
      pos_item_set.union(random.sample(self.indices_item, self.n_sampled - l0))
      l = len(pos_item_set)
      while l < self.n_sampled:
        pos_item_set.add(random.choice(self.indices_item))
        l = len(pos_item_set)
      item_sampled = list(pos_item_set)
      id2idx = {}
      i = 0
      for item in item_sampled:
        id2idx[item] = i
        i += 1
      return (batch_user_input, batch_item_input, batch_neg_item_input, 
        item_sampled, id2idx)
        
    return batch_user_input, batch_item_input, batch_neg_item_input, None, None

  def get_eval_batch(self, loss, users, items, hist = None):
    neg_items = []
    l, i = len(users), 0
    while i < l:
      u = users[i]
      i2 = random.choice(self.indices_item)
      while i2 in hist[u]:
        i2 = random.choice(self.indices_item)
      neg_items.append(i2)
      i += 1
    if loss in ['mw', 'mce']:
      pos_item_set = set(items)
      l0 = len(pos_item_set)
      pos_item_set.union(random.sample(self.indices_item, self.n_sampled - l0))
      l = len(pos_item_set)
      while l < self.n_sampled:
        pos_item_set.add(random.choice(self.indices_item))
        l = len(pos_item_set)
      item_sampled = list(pos_item_set)
      id2idx, i = {}, 0
      for item in item_sampled:
        id2idx[item] = i
        i += 1
      return (neg_items, item_sampled, id2idx)
        
    return neg_items, None, None