
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
import sys
import itertools
sys.path.insert(0, '../utils')
import embed_attribute


class LatentProductModel(object):
  def __init__(self, user_size, item_size, size,
               num_layers, batch_size, learning_rate,
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
    self.item_target = tf.placeholder(tf.int32, shape = [mb], name = "item")
    # self.user_input = tf.placeholder(tf.int32, shape = [mb], name = "user")    
    # self.neg_item_input = tf.placeholder(tf.int32, shape = [mb], 
    #   name = "neg_item")
    
    self.dropout = dropout
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    m = embed_attribute.EmbeddingAttribute(user_attributes, item_attributes, mb, 
      self.n_sampled, 0, False, item_ind2logit_ind, logit_ind2item_ind)
    self.att_emb = m
    embedded_user, user_b = m.get_batch_user(self.keep_prob)
    user_model_size = m.get_user_model_size()
    proj_user_drops = self._get_user_proj(embedded_user, self.keep_prob, 
      user_model_size, self.nonlinear)
    
    pos_embs_item, pos_item_b = m.get_batch_item('pos', batch_size)
    neg_embs_item, neg_item_b = m.get_batch_item('neg', batch_size)

    print("construct postive/negative items/scores \n(for bpr loss, AUC)")
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

    # mini batch version
    # print("sampled prediction")
    # if self.n_sampled is not None:
    #   embedded_item, item_b = m.get_batch_item('sampled', self.n_sampled)
    #   # compute sampled 'logits'/'scores'
    #   innerps = []
    #   for i in xrange(m.num_item_features):
    #     innerps.append(tf.matmul(embedded_item[i], 
    #       tf.transpose(proj_user_drops[i])))
    #   sampled_logits = tf.transpose(tf.reduce_sum(innerps, 0)) + item_b

    print("sampled prediction")
    if self.n_sampled is not None:
      embedded_item, item_b = m.get_batch_item('sampled', self.n_sampled)
      # compute sampled 'logits'/'scores'
      innerps = []
      for i in xrange(m.num_item_features):
        innerps.append(tf.matmul(proj_user_drops[i], 
          tf.transpose(embedded_item[i])))
      sampled_logits = tf.reduce_sum(innerps, 0) + item_b

    print("non-sampled prediction")
    logits = m.get_prediction(proj_user_drops)

    loss = self.loss_function
    if loss in ['warp', 'ce']:
      batch_loss = m.compute_loss(logits, self.item_target, loss)
    elif loss in ['mw', 'mce']:
      batch_loss = m.compute_loss(sampled_logits, self.item_target, loss)
    elif loss in ['bpr', 'bpr-hinge']:
      batch_loss = m.compute_loss(neg_pos, self.item_target, loss)
    else:
      print("not implemented!")
      exit(-1)
    if loss in ['warp', 'mw']:
      self.set_mask, self.reset_mask = m.get_warp_mask()

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

  def _get_user_proj(self, embedded_user, keep_prob, user_model_size, nonlinear):
    projs_cat, projs_mulhot, projs_cat_b, projs_mulhot_b = self._user_proj(
      self.item_attributes, hidden_size=user_model_size)
    # prepare projected version of user embedding
    proj_user_drops_cat, proj_user_drops_mulhot = [],[]
    for i in xrange(self.item_attributes.num_features_cat):
      proj_user = tf.matmul(embedded_user, projs_cat[i]) + projs_cat_b[i] # mb by d_f
      if nonlinear == 'relu':
        proj_user = tf.nn.relu(proj_user)
      elif nonlinear == 'tanh':
        proj_user = tf.tanh(proj_user)
      proj_user_drops_cat.append(tf.nn.dropout(proj_user, keep_prob))
    for i in xrange(self.item_attributes.num_features_mulhot):  
      proj_user = tf.matmul(embedded_user, projs_mulhot[i]) + projs_mulhot_b[i]
      if nonlinear == 'relu':
        proj_user = tf.nn.relu(proj_user)
      elif nonlinear == 'tanh':
        proj_user = tf.tanh(proj_user)
      proj_user_drops_mulhot.append(tf.nn.dropout(proj_user, keep_prob))
    return  proj_user_drops_cat + proj_user_drops_mulhot

  def _user_proj(self, attributes, hidden_size):
    biases_cat, biases_mulhot = [], []
    projs_cat, projs_mulhot = [], []
    
    for i in range(attributes.num_features_cat):
      size = attributes._embedding_size_list_cat[i]
      w = tf.get_variable("out_proj_cat_{0}".format(i), [hidden_size, size], 
        dtype=tf.float32)
      projs_cat.append(w)
      b = tf.get_variable("out_proj_cat_b_{0}".format(i), [size], 
        dtype=tf.float32)
      biases_cat.append(b)
    for i in range(attributes.num_features_mulhot):
      size = attributes._embedding_size_list_mulhot[i]
      w = tf.get_variable("out_proj_mulhot_{0}".format(i), 
        [hidden_size, size], dtype=tf.float32)
      projs_mulhot.append(w)
      b = tf.get_variable("out_proj_mulhot_b_{0}".format(i), [size], 
        dtype=tf.float32)
      biases_mulhot.append(b)
    return projs_cat, projs_mulhot, biases_cat, biases_mulhot

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
    
    if loss in ['warp', 'ce'] and recommend == False:
      input_feed[self.item_target.name] = [self.item_ind2logit_ind[v] for v in item_input]
    if loss in ['mw', 'mce'] and recommend == False:
      input_feed[self.item_target.name] = [item_sampled_id2idx[v] for v in item_input]

    if self.att_emb is not None:
      input_feed_warp = self.att_emb.add_input(input_feed, user_input, 
        item_input, neg_item_input=neg_item_input, item_sampled = item_sampled, 
        item_sampled_id2idx = item_sampled_id2idx, 
        forward_only=forward_only, recommend=recommend, 
        recommend_new = recommend_new, loss = loss)

    if not recommend:
      if not forward_only:
        output_feed = [self.updates, self.loss, self.auc]
      else:
        output_feed = [self.loss, self.auc]
    else:
      if recommend_new:
        output_feed = [self.indices_test]
      else:
        output_feed = [self.indices]

    if (loss == 'warp' or loss =='mw') and recommend is False:
      session.run(self.set_mask, input_feed_warp)

    if run_op is not None and run_meta is not None:
      outputs = session.run(output_feed, input_feed, options=run_op, run_metadata=run_meta)
    else:
      outputs = session.run(output_feed, input_feed)

    if (loss == 'warp' or loss =='mw') and recommend is False:
      session.run(self.reset_mask, input_feed_warp)

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
      u, i, _ = random.choice(data)
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