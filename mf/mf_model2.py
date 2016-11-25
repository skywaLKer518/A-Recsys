
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
# from mul_index import ravel_multi_index
# from sparse_map import create_sparse_map
import itertools

class LatentProductModel(object):
  def __init__(self, user_size, item_size, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, user_attributes=None, 
               item_attributes=None, item_ind2logit_ind=None, 
               logit_ind2item_ind=None, loss_function='ce', GPU=None, 
               logit_size_test=None, nonlinear=None, dropout=1.0):

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
      self.indices_item = range(self.logit_size) # for sample
    self.logit_size_test = logit_size_test

    self.nonlinear = nonlinear
    self.loss_function = loss_function
    self.batch_size = batch_size
    self.reuse_item_tr = None

    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    mb = self.batch_size
    self.user_input = tf.placeholder(tf.int32, shape = [mb], name = "user")
    self.item_input = tf.placeholder(tf.int32, shape = [mb], name = "item")
    self.neg_item_input = tf.placeholder(tf.int32, shape = [mb], 
      name = "neg_item")
    
    self.dropout = dropout
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    self.input_att_map = True
    self.use_user_bias = False # not sure if need to be True

    '''
    user embeddings:
      variables -
        embedding parameters (each type of attributes)
        attribute mapping (from user ind (continuous) to embed ind)
      tensors -
        mini batch user representation
    '''
    if user_attributes is None:
      embedded_user = self.EmbeddingLayer(self.user_input, size)
      user_model_size = size
    else:
      user_embs_cat, user_embs_mulhot = self.embedded(user_attributes, 
        prefix='user')
      self.u_cat_indices, self.u_mulhot_indices, self.u_mulhot_segids, self.u_mulhot_lengths = [],[], [], []
      for i in xrange(user_attributes.num_features_cat):
        self.u_cat_indices.append(tf.placeholder(tf.int32, shape = [mb], 
          name = "u_cat_ind{0}".format(i)))
      for i in xrange(user_attributes.num_features_mulhot):
        self.u_mulhot_indices.append(tf.placeholder(tf.int32, shape = [None], 
          name = "u_mulhot_ind{0}".format(i)))
        self.u_mulhot_segids.append(tf.placeholder(tf.int32, shape = [None], 
          name = "u_mulhot_seg{0}".format(i)))
        self.u_mulhot_lengths.append(tf.placeholder(tf.float32, shape= [mb, 1], 
          name = "u_mulhot_len{0}".format(i)))

      u_mappings = (self.u_cat_indices, self.u_mulhot_indices, 
        self.u_mulhot_segids, self.u_mulhot_lengths)

      # u_biases_cat, u_biases_mulhot = self.bias_parameter(user_attributes, 
      #   'user')
      embedded_user, user_b = self.EmbeddingLayer2(self.user_input, 
        user_embs_cat, user_embs_mulhot, b_cat=None, b_mulhot=None, 
        mappings=u_mappings, mb=batch_size, attributes=user_attributes, 
        prefix='user', concatenation=True, input_att_map=self.input_att_map)
      embedded_user = tf.nn.dropout(embedded_user, self.keep_prob)

      user_model_size = (sum(user_attributes._embedding_size_list_cat) + 
          sum(user_attributes._embedding_size_list_mulhot))

    '''
    item embeddings
      variables -
        embedding parameters (each type)
        attributes mapping 1 (for train: logit ind to embed ind)
          mini-batch
          full (output layer)
        attributes mapping 2 (for test: item ind to embed ind)
    '''
    item_embs_cat, item_embs_mulhot = self.embedded(item_attributes, 
      prefix='item', transpose=False)
    i_biases_cat, i_biases_mulhot = self.bias_parameter(item_attributes, 'item')
    projs_cat, projs_mulhot, projs_cat_b, projs_mulhot_b = self.item_proj_layer(
      item_attributes, hidden_size=user_model_size)
    # i_mappings = self.mapped(item_attributes, True, 'item')    
    

    self.i_cat_indices_tr, self.i_mulhot_indices_tr, self.i_mulhot_segids_tr, self.i_mulhot_lengths_tr = [],[], [], []
    self.i_cat_indices_tr2, self.i_mulhot_indices_tr2, self.i_mulhot_segids_tr2, self.i_mulhot_lengths_tr2 = [],[], [], []
    for i in xrange(user_attributes.num_features_cat):
      self.i_cat_indices_tr.append(tf.placeholder(tf.int32, shape = [mb], 
        name = "i_cat_ind_tr{0}".format(i)))
      self.i_cat_indices_tr2.append(tf.placeholder(tf.int32, shape = [mb], 
        name = "i_cat_ind_tr2{0}".format(i)))
    for i in xrange(user_attributes.num_features_mulhot):
      self.i_mulhot_indices_tr.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_ind_tr{0}".format(i)))
      self.i_mulhot_segids_tr.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_seg_tr{0}".format(i)))
      self.i_mulhot_lengths_tr.append(tf.placeholder(tf.float32, shape= [mb, 1], 
        name = "i_mulhot_len_tr{0}".format(i)))

      self.i_mulhot_indices_tr2.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_ind_tr2{0}".format(i)))
      self.i_mulhot_segids_tr2.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_seg_tr2{0}".format(i)))
      self.i_mulhot_lengths_tr2.append(tf.placeholder(tf.float32, shape= [mb, 1], 
        name = "i_mulhot_len_tr2{0}".format(i)))

    '''
    to change i_mappings_tr etc
    '''
    i_mappings_tr = (self.i_cat_indices_tr, self.i_mulhot_indices_tr, 
      self.i_mulhot_segids_tr, self.i_mulhot_lengths_tr)
    i_mappings_tr2= (self.i_cat_indices_tr2, self.i_mulhot_indices_tr2, 
      self.i_mulhot_segids_tr2, self.i_mulhot_lengths_tr2)
  
    # full vocabulary item indices, also just for train
    full_out_layer = self.full_output_layer(item_attributes)
    indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = full_out_layer

    print("construct postive/negative items/scores ")
    pos_embs_item_cat, pos_embs_item_mulhot, pos_item_b = self.EmbeddingLayer2(
      self.item_input, item_embs_cat, item_embs_mulhot, i_biases_cat, 
      i_biases_mulhot, i_mappings_tr, batch_size, item_attributes, 'item', 
      False, self.input_att_map)
    neg_embs_item_cat, neg_embs_item_mulhot, neg_item_b = self.EmbeddingLayer2(
      self.neg_item_input, item_embs_cat, item_embs_mulhot, i_biases_cat, 
      i_biases_mulhot, i_mappings_tr2, batch_size, item_attributes, 'item', 
      False, self.input_att_map)
    
    pos_scores, neg_scores = [], []
    for i in xrange(item_attributes.num_features_cat):
      proj_user = tf.matmul(embedded_user, projs_cat[i]) + projs_cat_b[i] # mb by d_f
      if self.nonlinear == 'relu':
        proj_user = tf.nn.relu(proj_user)
      elif self.nonlinear == 'tanh':
        proj_user = tf.tanh(proj_user)

      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      pos_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        pos_embs_item_cat[i]), 1))
      neg_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        neg_embs_item_cat[i]), 1))
      # pos_embs_item_cat_drop = tf.nn.dropout(pos_embs_item_cat[i], 
      #   self.keep_prob)
      # pos_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
      #   pos_embs_item_cat_drop), 1))

      # neg_embs_item_cat_drop = tf.nn.dropout(neg_embs_item_cat[i], 
      #   self.keep_prob)
      # neg_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
      #   neg_embs_item_cat_drop), 1))

    for i in xrange(item_attributes.num_features_mulhot):
      proj_user = tf.matmul(embedded_user, projs_mulhot[i]) + projs_mulhot_b[i]
      if self.nonlinear == 'relu':
        proj_user = tf.nn.relu(proj_user)
      elif self.nonlinear == 'tanh':
        proj_user = tf.tanh(proj_user)

      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      pos_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        pos_embs_item_mulhot[i]), 1))      
      neg_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        neg_embs_item_mulhot[i]), 1))
      # pos_embs_item_mulhot_drop = tf.nn.dropout(pos_embs_item_mulhot[i], 
      #   self.keep_prob)
      # pos_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
      #   pos_embs_item_mulhot_drop), 1))

      # neg_embs_item_mulhot_drop = tf.nn.dropout(neg_embs_item_mulhot[i], 
      #   self.keep_prob)
      # neg_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
      #   neg_embs_item_mulhot_drop), 1))

    print("neg_score[0] shape")
    print(neg_scores[0].get_shape())
    print("pos_item_b shape")
    print(pos_item_b.get_shape())
    pos_score = tf.reduce_sum(pos_scores, 0) + pos_item_b
    neg_score = tf.reduce_sum(neg_scores, 0) + neg_item_b
    print("neg_score shape")
    print(neg_score.get_shape())

    neg_pos = neg_score - pos_score
    print("neg_pos shape")
    print(neg_pos.get_shape())
    self.pos_score = pos_score
    self.neg_score = neg_score
    
    print("construct inner products between mb users and full item embeddings")  
    # compute inner product between item_hidden and {user_feature_embedding}
    # then lookup to compute logits
    innerps = []

    for i in xrange(item_attributes.num_features_cat):
      proj_user = tf.matmul(embedded_user, projs_cat[i]) + projs_cat_b[i] # mb by d_f
      if self.nonlinear == 'relu':
        proj_user = tf.nn.relu(proj_user)
      elif self.nonlinear == 'tanh':
        proj_user = tf.tanh(proj_user)
      
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)


      innerp = tf.matmul(item_embs_cat[i], tf.transpose(
        proj_user_drop)) + i_biases_cat[i] # Vf by mb
      innerps.append(embedding_ops.embedding_lookup(innerp, indices_cat[i], 
        name='emb_lookup_innerp_{0}'.format(i))) # V by mb

    for i in xrange(item_attributes.num_features_mulhot):
      proj_user = tf.matmul(embedded_user, projs_mulhot[i]) + projs_mulhot_b[i]
      if self.nonlinear == 'relu':
        proj_user = tf.nn.relu(proj_user)
      elif self.nonlinear == 'tanh':
        proj_user = tf.tanh(proj_user)
      
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)

      innerp = tf.add(tf.matmul(item_embs_mulhot[i], 
        tf.transpose(proj_user_drop)), i_biases_mulhot[i]) # Vf by mb
      innerps.append(tf.div(tf.unsorted_segment_sum(embedding_ops.embedding_lookup(
        innerp, indices_mulhot[i]), segids_mulhot[i], self.logit_size),
        lengths_mulhot[i]))
    
    if self.use_user_bias:
      logits = tf.add(tf.transpose(tf.reduce_sum(innerps, 0)), user_b)
    else:
      logits = tf.transpose(tf.reduce_sum(innerps, 0))
    # there is one transpose here.. can we remove??

    self.output = logits
    self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))

    '''
    test and recommend new items
    '''
    if self.logit_size_test is not None:
      (indices_cat2, indices_mulhot2, segids_mulhot2, 
        lengths_mulhot2) = self.full_output_layer(item_attributes, True)
      innerps_test = []
      # print(indices_cat2)
      # print(indices_cat2[0])
      for i in xrange(item_attributes.num_features_cat):
        proj_user = tf.matmul(embedded_user, projs_cat[i]) + projs_cat_b[i] # mb by d_f
        if self.nonlinear == 'relu':
          proj_user = tf.nn.relu(proj_user)
        elif self.nonlinear == 'tanh':
          proj_user = tf.tanh(proj_user)
          
        proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)

        innerp = tf.matmul(item_embs_cat[i], tf.transpose(
          proj_user_drop)) + i_biases_cat[i] # Vf by mb
        innerps_test.append(embedding_ops.embedding_lookup(innerp, 
          indices_cat2[i], name='emb_lookup_innerp_test_{0}'.format(i))) # V by mb

      for i in xrange(item_attributes.num_features_mulhot):
        proj_user = tf.matmul(embedded_user, projs_mulhot[i]) + projs_mulhot_b[i]
        if self.nonlinear == 'relu':
          proj_user = tf.nn.relu(proj_user)
        elif self.nonlinear == 'tanh':
          proj_user = tf.tanh(proj_user)

        proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
        
        innerp = tf.add(tf.matmul(item_embs_mulhot[i], 
          tf.transpose(proj_user_drop)), i_biases_mulhot[i]) # Vf by mb
        innerps_test.append(tf.div(tf.unsorted_segment_sum(
          embedding_ops.embedding_lookup(
          innerp, indices_mulhot2[i]), segids_mulhot2[i], self.logit_size_test),
          lengths_mulhot2[i]))
      
      if self.use_user_bias:
        logits_test = tf.add(tf.transpose(tf.reduce_sum(innerps_test, 0)), user_b)
      else:
        logits_test = tf.transpose(tf.reduce_sum(innerps_test, 0))
      
      _, self.indices_test= tf.nn.top_k(logits_test, 30, sorted=True)

    print("loss function is %s" % self.loss_function)
    if self.loss_function == 'ce':  # softmax cross-entropy loss
      batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, 
        self.item_input)
    elif self.loss_function == 'bpr_hinge':
      batch_loss = tf.maximum(1 + neg_pos, 0)
      self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))
    elif self.loss_function == 'bpr': 
      batch_loss = tf.log(1+tf.exp(neg_pos))
      self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))
    elif self.loss_function == 'warp':

      self.mask = tf.Variable([True] * self.logit_size * mb,  dtype=tf.bool, 
        trainable=False, name='mask')
      zero_logits = tf.constant([[0.0]*self.logit_size]*mb)
      self.pos_indices = tf.placeholder(tf.int32, shape = [None], 
        name = "mask_indices") # for WARP, should include target itself
      self.l_true = tf.placeholder(tf.bool, shape = [None], name='l_true')
      self.l_false = tf.placeholder(tf.bool, shape = [None], name='l_false')
      self.set_mask = tf.scatter_update(self.mask, self.pos_indices, 
        self.l_false)
      self.reset_mask = tf.scatter_update(self.mask, self.pos_indices, 
        self.l_true)
      flat_matrix = tf.reshape(logits, [-1])
      # idx_flattened0 = tf.constant(np.array(tf.range(0, mb) * self.logit_size))
      idx_flattened0 = tf.range(0, mb) * self.logit_size
      idx_flattened = idx_flattened0 + self.item_input
      logits_ = tf.gather(flat_matrix, idx_flattened)
      logits_ = tf.reshape(logits_, [mb, 1])
      logits2 = tf.sub(logits, logits_) + 1

      mask2 = tf.reshape(self.mask, [mb, self.logit_size])
      target = tf.select(mask2, logits2, zero_logits)
      batch_loss = tf.log(1 + tf.reduce_sum(tf.nn.relu(target), 1))
    else:
      print("No such loss function. Exit.")
      exit()
    '''
    batch warp loss
    logits
    logits = logits - logits[target item] + 1 // index
    logits[mask] = 0 // mask?
    logits = max(logits, 0) ??
    loss = tf.log(1+reduce_sum(logits))
    '''

    # mean over mini-batch
    self.loss = tf.reduce_mean(batch_loss)
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    
    opt = tf.train.AdagradOptimizer(self.learning_rate)
    # opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = tf.gradients(self.loss, params)
    
    self.updates = opt.apply_gradients(
      zip(gradients, params), global_step=self.global_step)

    values, self.indices= tf.nn.top_k(self.output, 30, sorted=True)

    self.saver = tf.train.Saver(tf.all_variables())

  def prepare_warp(self, pos_item_set, pos_item_set_witheval):
    self.pos_item_set = pos_item_set
    self.pos_item_set_witheval = pos_item_set_witheval
    return 

  def mapped(self, attributes, mulhot=False, prefix='unknown'):
    feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = [], [], [], []
    mtls = [] # mulhot_max_lengths
    # with vs.variable_scope(type(self).__name__, reuse=True):
    for i in xrange(attributes.num_features_cat):
      init = tf.constant_initializer(value = attributes.features_cat[i])
      feats_cat.append(tf.get_variable(name=prefix + "_map1_{0}".format(i), 
        shape=attributes.features_cat[i].shape, dtype=tf.int32, 
        initializer=init, trainable=False))

    if mulhot:
      for i in xrange(attributes.num_features_mulhot):
        init = tf.constant_initializer(value = attributes.features_mulhot[i])
        feats_mulhot.append(tf.get_variable(
          name=prefix + "_map2_{0}".format(i), 
          shape=attributes.features_mulhot[i].shape, dtype=tf.int32, 
          initializer=init, trainable=False))
        
        init_s = tf.constant_initializer(value = attributes.mulhot_starts[i])
        mulhot_starts.append(tf.get_variable(
          name=prefix + "_map2_starts{0}".format(i), 
          shape=attributes.mulhot_starts[i].shape, dtype=tf.int32, 
          initializer=init_s, trainable=False))

        init_l = tf.constant_initializer(value = attributes.mulhot_lengths[i])
        mulhot_lengths.append(tf.get_variable(
          name=prefix + "_map2_lengs{0}".format(i), 
          shape=attributes.mulhot_lengths[i].shape, dtype=tf.int32, 
          initializer=init_l, trainable=False))
        mtls.append(attributes.mulhot_max_length[i])
    
    return feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths, mtls

  def mapped_item_tr(self, attributes, mulhot=False, prefix='item'):
    feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = [], [], [], []
    mtls = []
    with vs.variable_scope('item_map_tr', reuse=self.reuse_item_tr) as scope:
      for i in xrange(attributes.num_features_cat):
        init = tf.constant_initializer(value = attributes.features_cat_tr[i])
        feats_cat.append(tf.get_variable(name=prefix + "_tr_map1_{0}".format(i), 
          shape=attributes.features_cat_tr[i].shape, dtype=tf.int32, 
          initializer=init, trainable=False))
      
      if mulhot:
        for i in xrange(attributes.num_features_mulhot):
          init = tf.constant_initializer(value = attributes.features_mulhot_tr[i])
          feats_mulhot.append(tf.get_variable(
            name=prefix + "_tr_map2_{0}".format(i), 
            shape=attributes.features_mulhot_tr[i].shape, dtype=tf.int32, 
            initializer=init, trainable=False))
          
          init_s = tf.constant_initializer(value = attributes.mulhot_starts_tr[i])
          mulhot_starts.append(tf.get_variable(
            name=prefix + "_tr_map2_starts{0}".format(i), 
            shape=attributes.mulhot_starts_tr[i].shape, dtype=tf.int32, 
            initializer=init_s, trainable=False))

          init_l = tf.constant_initializer(
            value = attributes.mulhot_lengs_tr[i])
          mulhot_lengths.append(tf.get_variable(
            name=prefix + "_tr_map2_lengs{0}".format(i), 
            shape=attributes.mulhot_lengs_tr[i].shape, dtype=tf.int32, 
            initializer=init_l, trainable=False))
          mtls.append(attributes.mulhot_max_leng_tr[i])
      self.reuse_item_tr = True

    return feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths, mtls

  def embedded(self, attributes, prefix='', transpose=False):
    '''
    variables of full vocabulary for each type of features
    '''
    embs_cat, embs_mulhot = [], []
    for i in xrange(attributes.num_features_cat):
      d = attributes._embedding_size_list_cat[i]
      V = attributes._embedding_classes_list_cat[i]
      if not transpose:
        embedding = tf.get_variable(name=prefix + "embed_cat_{0}".format(i), 
          shape=[V,d], dtype=tf.float32)
      else:
        embedding = tf.get_variable(name=prefix + "embed_cat_{0}".format(i), 
          shape=[d,V], dtype=tf.float32)
      embs_cat.append(embedding)
    for i in xrange(attributes.num_features_mulhot):
      d = attributes._embedding_size_list_mulhot[i]
      V = attributes._embedding_classes_list_mulhot[i]
      if not transpose:
        embedding = tf.get_variable(name=prefix + "embed_mulhot_{0}".format(i), 
          shape=[V,d], dtype=tf.float32)
      else:
        embedding = tf.get_variable(name=prefix + "embed_mulhot_{0}".format(i), 
          shape=[d,V], dtype=tf.float32)
      embs_mulhot.append(embedding)
    return embs_cat, embs_mulhot

  def bias_parameter(self, attributes, prefix):
    biases_cat, biases_mulhot = [], []
    for i in range(attributes.num_features_cat):
      V = attributes._embedding_classes_list_cat[i]
      b = tf.get_variable(prefix + "_bias_cat_{0}".format(i), [V, 1], 
        dtype = tf.float32)
      biases_cat.append(b)
    for i in range(attributes.num_features_mulhot):
      V = attributes._embedding_classes_list_mulhot[i]
      b = tf.get_variable(prefix + "_bias_mulhot_{0}".format(i), [V, 1], 
        dtype = tf.float32)
      biases_mulhot.append(b)
    return biases_cat, biases_mulhot

  def item_proj_layer(self, attributes, hidden_size, scope = None):
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

  def full_output_layer(self, attributes, test=False):
    indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = [],[],[],[]    
    prefix = 'item'
    if not test:
      feats_cat_tr, _, _ , _, _ = self.mapped_item_tr(attributes, False, 
        prefix=prefix)
      indices_cat = feats_cat_tr
      for i in range(attributes.num_features_mulhot):
        indices_mulhot.append(attributes.full_values_tr[i])
        segids_mulhot.append(attributes.full_segids_tr[i])
        lengths_mulhot.append(attributes.full_lengths_tr[i])
    else:
      for i in xrange(attributes.num_features_cat):
        indices_cat.append(tf.constant(attributes.features_cat[i], 
          dtype=tf.int32))
      for i in range(attributes.num_features_mulhot):
        indices_mulhot.append(attributes.full_values[i])
        segids_mulhot.append(attributes.full_segids[i])
        lengths_mulhot.append(attributes.full_lengths[i])    
    
    return indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot

  def EmbeddingLayer(self, input_user, size, scope = None):
    with vs.variable_scope(scope or type(self).__name__):
      with ops.device("/cpu:0"):
        sqrt3 = math.sqrt(3)
        initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)
        embedding_user = vs.get_variable("embedding_user", 
          [self.user_size, size], initializer=initializer, dtype=tf.float32)
        embeded_user = embedding_ops.embedding_lookup(
          embedding_user, array_ops.reshape(input_user, [-1]))
    return embeded_user
    
  def EmbeddingLayer2(self, input_, embs_cat, embs_mulhot, b_cat, b_mulhot, 
    mappings, mb, attributes, prefix='', concatenation=True, 
    input_att_map=False):
    cat_indices, mulhot_indices, mulhot_segids, mulhot_lengths = mappings
    cat_list, mulhot_list = [], []
    bias_cat_list, bias_mulhot_list = [], []

    for i in xrange(attributes.num_features_cat):
      embedded = embedding_ops.embedding_lookup(embs_cat[i], 
        cat_indices[i], name='emb_lookup_item_{0}'.format(i))  # on cpu
      cat_list.append(embedded)
      if b_cat is not None:
        b = embedding_ops.embedding_lookup(b_cat[i], cat_indices[i], 
          name = 'emb_lookup_item_b_{0}'.format(i))
        bias_cat_list.append(b)
    for i in xrange(attributes.num_features_mulhot):
      embedded_flat = embedding_ops.embedding_lookup(embs_mulhot[i], mulhot_indices[i])
      embedded_sum = tf.unsorted_segment_sum(embedded_flat, mulhot_segids[i], mb)
      embedded = tf.div(embedded_sum, mulhot_lengths[i])
      mulhot_list.append(embedded)
      if b_mulhot is not None:
        b_embedded_flat = embedding_ops.embedding_lookup(b_mulhot[i], 
          mulhot_indices[i])
        b_embedded_sum = tf.unsorted_segment_sum(b_embedded_flat, mulhot_segids[i], 
          mb)
        b_embedded = tf.div(b_embedded_sum, mulhot_lengths[i])
        bias_mulhot_list.append(b_embedded)
    
    if b_cat is None and b_mulhot is None:
      bias = None
    else:
      bias = tf.squeeze(tf.reduce_mean(bias_cat_list + bias_mulhot_list, 0))

    if concatenation:
      return tf.concat(1, cat_list + mulhot_list), bias
    else:
      return cat_list, mulhot_list, bias

  def step(self, session, user_input, item_input, neg_item_input=None, 
    forward_only=False, recommend=False, recommend_new = False, loss=None, 
    run_op=None, run_meta=None):
    input_feed = {}
    '''
    input indices of user/item
    '''
    input_feed[self.user_input.name] = user_input
    input_feed[self.item_input.name] = item_input # logits indices
    if neg_item_input is not None:
      input_feed[self.neg_item_input.name] = neg_item_input
    
    '''
    input mappings
    '''
    if self.user_attributes is not None and self.input_att_map:
      ua = self.user_attributes
      for i in xrange(ua.num_features_cat):
        input_feed[self.u_cat_indices[i].name] = ua.features_cat[i][user_input]
      for i in xrange(ua.num_features_mulhot):
        v_i = ua.features_mulhot[i]
        s_i = ua.mulhot_starts[i]
        l_i = ua.mulhot_lengths[i]
        vals = list(itertools.chain.from_iterable(
          [v_i[s_i[u]:s_i[u]+l_i[u]] for u in user_input]))
        Ls = [l_i[u] for u in user_input]
        i1 = list(itertools.chain.from_iterable(
          Ls[i] * [i] for i in range(len(Ls))))

        input_feed[self.u_mulhot_indices[i].name] = vals
        input_feed[self.u_mulhot_segids[i].name] = i1
        input_feed[self.u_mulhot_lengths[i].name] = np.reshape(Ls, (len(Ls), 1))
      
    if self.item_attributes is not None and self.input_att_map:
      ia = self.item_attributes
      for i in xrange(ia.num_features_cat):
        input_feed[self.i_cat_indices_tr[i].name] = ia.features_cat_tr[i][item_input]
        input_feed[self.i_cat_indices_tr2[i].name] = ia.features_cat_tr[i][neg_item_input]
      for i in xrange(ia.num_features_mulhot):
        v_i = ia.features_mulhot_tr[i]
        s_i = ia.mulhot_starts_tr[i]
        l_i = ia.mulhot_lengs_tr[i]
        vals = list(itertools.chain.from_iterable(
          [v_i[s_i[u]:s_i[u]+l_i[u]] for u in item_input]))
        vals2 = list(itertools.chain.from_iterable(
          [v_i[s_i[u]:s_i[u]+l_i[u]] for u in neg_item_input]))
        Ls = [l_i[u] for u in item_input]
        Ls2= [l_i[u] for u in neg_item_input]
        l = len(Ls)
        l2 = len(Ls2)

        i1 = list(itertools.chain.from_iterable(
          Ls[i] * [i] for i in range(len(Ls))))
        i1_2 = list(itertools.chain.from_iterable(
          Ls2[i] * [i] for i in range(len(Ls2))))

        input_feed[self.i_mulhot_indices_tr[i].name] = vals
        input_feed[self.i_mulhot_segids_tr[i].name] = i1
        input_feed[self.i_mulhot_lengths_tr[i].name] = np.reshape(Ls, (l, 1))

        input_feed[self.i_mulhot_indices_tr2[i].name] = vals2
        input_feed[self.i_mulhot_segids_tr2[i].name] = i1_2
        input_feed[self.i_mulhot_lengths_tr2[i].name] = np.reshape(Ls2, (l2, 1))
      
    if forward_only or recommend:
      input_feed[self.keep_prob.name] = 1.0
    else:
      input_feed[self.keep_prob.name] = self.dropout

    if loss == 'warp' and recommend is False:
      input_feed_warp = {}
      V = self.logit_size
      mask_indices = []
      c = 0
      for u in user_input:
        offset = c * V
        if forward_only:
          mask_indices.extend([v + offset for v in self.pos_item_set_witheval[u]])  # v is logit ind
        else:
          mask_indices.extend([v + offset for v in self.pos_item_set[u]])  # v is logit ind
        c += 1
      L = len(mask_indices)
      input_feed_warp[self.pos_indices.name] = mask_indices
      input_feed_warp[self.l_false.name] = [False] * L
      input_feed_warp[self.l_true.name] = [True] * L

    if not recommend:
      if not forward_only:
        # output_feed = [self.updates, self.loss, self.auc]
        output_feed = [self.updates, self.loss, self.auc, self.neg_score, self.pos_score]
      else:
        output_feed = [self.loss, self.auc]
    else:
      # values, indices= tf.nn.top_k(self.output, 30, sorted=True)
      # self.output = indices
      if recommend_new:
        output_feed = [self.indices_test]
      else:
        output_feed = [self.indices]

    if loss == 'warp' and recommend is False:
      session.run(self.set_mask, input_feed_warp)

    if run_op is not None and run_meta is not None:
      outputs = session.run(output_feed, input_feed, options=run_op, run_metadata=run_meta)
    else:
      outputs = session.run(output_feed, input_feed)

    if loss == 'warp' and recommend is False:
      # print('reset mask %d' % L)
      session.run(self.reset_mask, input_feed_warp)

    if not recommend:
      if not forward_only:
        return outputs[1], outputs[2], outputs[3], outputs[4]
      else:
        return outputs[0], outputs[1]
    else:
      return outputs[0]

  def get_batch(self, data, loss = 'ce', hist = None):
    batch_user_input, batch_item_input = [], []
    batch_neg_item_input = []

    # if loss == 'ce':
    #   count = 0
    #   while count < self.batch_size:
    #     u, i = random.choice(data)
    #     # ii = self.item_ind2logit_ind[i]
    #     # assert(ii!=0)
    #     batch_user_input.append(u)
    #     batch_item_input.append(i)
    #     count += 1
    #   # return batch_user_input, batch_item_input
    # elif loss == 'brp':
    count = 0
    while count < self.batch_size:
      u, i = random.choice(data)
      # ii = self.item_ind2logit_ind[i]
      # assert(ii!=0)
      batch_user_input.append(u)
      batch_item_input.append(i)
      
      i2 = random.choice(self.indices_item)
      while i2 in hist[u]:
        i2 = random.choice(self.indices_item)
      batch_neg_item_input.append(i2)
      count += 1
    return batch_user_input, batch_item_input, batch_neg_item_input






