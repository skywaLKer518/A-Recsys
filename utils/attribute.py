
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import random, math

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import array_ops
import itertools

class Attributes(object):
  def __init__(self, num_feature_cat=0, feature_cat=None,
               num_text_feat=0, feature_mulhot=None, mulhot_max_length=None, 
               mulhot_starts=None, mulhot_lengths=None, 
               v_sizes_cat=None, v_sizes_mulhot=None, 
               embedding_size_list_cat=None):
    self.num_features_cat = num_feature_cat
    self.num_features_mulhot = num_text_feat
    self.features_cat = feature_cat
    self.features_mulhot = feature_mulhot
    self.mulhot_max_length = mulhot_max_length
    self.mulhot_starts = mulhot_starts
    self.mulhot_lengths = mulhot_lengths
    self._embedding_classes_list_cat = v_sizes_cat
    self._embedding_classes_list_mulhot = v_sizes_mulhot
    return 
  
  def set_model_size(self, sizes):

    if isinstance(sizes, list):
      assert(len(sizes) == self.num_features_cat)
      self._embedding_size_list_cat = sizes
    elif isinstance(sizes, int):
      self._embedding_size_list_cat = [sizes] * self.num_features_cat
    # else:
    #   self._embedding_size_list_cat = [sizes] * self.num_features_cat
    if isinstance(sizes, list):
      assert(len(embedding_size_list_mulhot) == self.num_features_mulhot)
      self._embedding_size_list_mulhot = sizes
    elif isinstance(sizes, int):
      self._embedding_size_list_mulhot = [sizes] * self.num_features_mulhot
    # else:
    #   self._embedding_size_list_mulhot = [sizes] * self.num_features_mulhot
  
  def add_sparse_mapping(self, full_indices, full_values, sp_shapes, 
    full_indices_tr, full_values_tr, sp_shapes_tr):
    self.full_indices = full_indices
    self.full_values = full_values
    self.sp_shapes = sp_shapes
    self.full_indices_tr = full_indices_tr
    self.full_values_tr = full_values_tr
    self.sp_shapes_tr = sp_shapes_tr
    return

  def add_sparse_mapping2(self, full_segids, full_lengths, full_segids_tr,
    full_lengths_tr):
    self.full_segids = full_segids
    self.full_lengths = full_lengths
    self.full_segids_tr = full_segids_tr
    self.full_lengths_tr = full_lengths_tr
    return    

  def set_train_mapping(self, features_cat_tr, features_mulhot_tr, 
    mulhot_max_leng_tr, mulhot_starts_tr, mulhot_lengs_tr):
    self.features_cat_tr = features_cat_tr
    self.features_mulhot_tr = features_mulhot_tr
    self.mulhot_max_leng_tr = mulhot_max_leng_tr
    self.mulhot_starts_tr = mulhot_starts_tr
    self.mulhot_lengs_tr = mulhot_lengs_tr
    return
    

class EmbeddingAttribute(object):
  def __init__(self, user_attributes, item_attributes, mb, n_sampled, 
    input_steps=0, item_output=False,
    item_ind2logit_ind=None, logit_ind2item_ind=None, indices_item=None):
    self.user_attributes = user_attributes
    self.item_attributes = item_attributes
    self.batch_size = mb
    self.n_sampled = n_sampled
    self.input_steps = input_steps
    self.item_output = item_output # whether to use separate embedding for item output
    self.num_item_features = (item_attributes.num_features_cat +
      item_attributes.num_features_mulhot)
    self.reuse_item_tr = None

    self.item_ind2logit_ind = item_ind2logit_ind
    self.logit_ind2item_ind = logit_ind2item_ind
    if logit_ind2item_ind is not None:
      self.logit_size = len(logit_ind2item_ind)
    if indices_item is not None:
      self.indices_item = indices_item
    else:
      self.indices_item = range(self.logit_size)
    # self.logit_size_test = logit_size_test

    # self.item_target = tf.placeholder(tf.int32, shape = [mb], name = "item")
    '''
    user embeddings:
      variables -
        embedding parameters (each type of attributes)
        attribute mapping (from user ind (continuous) to embed ind)
      tensors -
        mini batch user representation
    '''    
    self.user_embs_cat, self.user_embs_mulhot = self.embedded(user_attributes, 
      prefix='user')
    u_cat_indices, u_mulhot_indices, u_mulhot_segids, u_mulhot_lengths = [],[], [], []
    for i in xrange(user_attributes.num_features_cat):
      u_cat_indices.append(tf.placeholder(tf.int32, shape = [mb], 
        name = "u_cat_ind{0}".format(i)))
    for i in xrange(user_attributes.num_features_mulhot):
      u_mulhot_indices.append(tf.placeholder(tf.int32, shape = [None], 
        name = "u_mulhot_ind{0}".format(i)))
      u_mulhot_segids.append(tf.placeholder(tf.int32, shape = [None], 
        name = "u_mulhot_seg{0}".format(i)))
      u_mulhot_lengths.append(tf.placeholder(tf.float32, shape= [mb, 1], 
        name = "u_mulhot_len{0}".format(i)))
    self.u_mappings = {}
    self.u_mappings['input'] = (u_cat_indices, u_mulhot_indices, 
      u_mulhot_segids, u_mulhot_lengths)
    '''
    item embeddings
      variables -
        embedding parameters (each type)
        attributes mapping 1 (for train: logit ind to embed ind)
          mini-batch
          full (output layer)
        attributes mapping 2 (for test: item ind to embed ind)
    '''
    self.item_embs_cat, self.item_embs_mulhot = self.embedded(item_attributes, 
      prefix='item', transpose=False)
    if item_output:
      self.item_embs2_cat, self.item_embs2_mulhot = self.embedded(
        item_attributes, prefix='item_output', transpose=False)
    self.i_biases_cat, self.i_biases_mulhot = self.bias_parameter(
      item_attributes, 'item')

    print("construct postive/negative items/scores ")    
    # positive/negative sample indices
    i_cat_indices_pos, i_mulhot_indices_pos, i_mulhot_segids_pos, i_mulhot_lengths_pos = [],[], [], []
    i_cat_indices_neg, i_mulhot_indices_neg, i_mulhot_segids_neg, i_mulhot_lengths_neg = [],[], [], []
    for i in xrange(user_attributes.num_features_cat):
      i_cat_indices_pos.append(tf.placeholder(tf.int32, shape = [mb], 
        name = "i_cat_ind_pos{0}".format(i)))
      i_cat_indices_neg.append(tf.placeholder(tf.int32, shape = [mb], 
        name = "i_cat_ind_neg{0}".format(i)))
    for i in xrange(user_attributes.num_features_mulhot):
      i_mulhot_indices_pos.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_ind_pos{0}".format(i)))
      i_mulhot_segids_pos.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_seg_pos{0}".format(i)))
      i_mulhot_lengths_pos.append(tf.placeholder(tf.float32, shape= [mb, 1], 
        name = "i_mulhot_len_pos{0}".format(i)))

      i_mulhot_indices_neg.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_ind_neg{0}".format(i)))
      i_mulhot_segids_neg.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_seg_neg{0}".format(i)))
      i_mulhot_lengths_neg.append(tf.placeholder(tf.float32, shape= [mb, 1], 
        name = "i_mulhot_len_neg{0}".format(i)))

    self.i_mappings = {}
    self.i_mappings['pos'] = (i_cat_indices_pos, i_mulhot_indices_pos, 
      i_mulhot_segids_pos, i_mulhot_lengths_pos)
    self.i_mappings['neg'] = (i_cat_indices_neg, i_mulhot_indices_neg, 
      i_mulhot_segids_neg, i_mulhot_lengths_neg)

    print("construct mini-batch item candicate pool")
    # mini-batch item candidate pool
    # self.item_sampled = tf.placeholder(tf.int32, shape=[None])
    i_cat_indices, i_mulhot_indices, i_mulhot_segids, i_mulhot_lengths = [],[], [], []
    for i in xrange(user_attributes.num_features_cat):
      i_cat_indices.append(tf.placeholder(tf.int32, shape =[self.n_sampled], 
        name = "i_cat_ind{0}".format(i)))
    for i in xrange(user_attributes.num_features_mulhot):
      i_mulhot_indices.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_ind{0}".format(i)))
      i_mulhot_segids.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_seg{0}".format(i)))
      i_mulhot_lengths.append(tf.placeholder(tf.float32, 
        shape= [self.n_sampled, 1], name = "i_mulhot_len{0}".format(i)))
    self.i_mappings['sampled'] = (i_cat_indices, i_mulhot_indices, 
      i_mulhot_segids, i_mulhot_lengths)

    print("construct input item")
    for step in xrange(input_steps):      
      i_in_cat_indices, i_in_mulhot_indices, i_in_mulhot_segids, i_in_mulhot_lengths = [],[], [], []
      for i in xrange(user_attributes.num_features_cat):
        i_in_cat_indices.append(tf.placeholder(tf.int32, 
          shape =[mb], name = "i_in_cat_ind{}_{}".format(i, step)))
      for i in xrange(user_attributes.num_features_mulhot):
        i_in_mulhot_indices.append(tf.placeholder(tf.int32, shape = [None], 
          name = "i_in_mulhot_ind{}_{}".format(i, step)))
        i_in_mulhot_segids.append(tf.placeholder(tf.int32, shape = [None], 
          name = "i_in_mulhot_seg{}_{}".format(i, step)))
        i_in_mulhot_lengths.append(tf.placeholder(tf.float32, 
          shape= [mb, 1], name = "i_in_mulhot_len{}_{}".format(i, step)))
      self.i_mappings['input{}'.format(step)] = (i_in_cat_indices, 
        i_in_mulhot_indices, i_in_mulhot_segids, i_in_mulhot_lengths)
    
    return

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

  def get_batch_user(self, keep_prob):
    u_mappings = self.u_mappings['input']
    embedded_user, user_b = self.EmbeddingLayer(self.user_embs_cat, 
      self.user_embs_mulhot, b_cat=None, b_mulhot=None, mappings=u_mappings, 
      mb=self.batch_size, attributes=self.user_attributes, prefix='user', 
      concatenation=True)
    embedded_user = tf.nn.dropout(embedded_user, keep_prob)
    return embedded_user, user_b

  def get_batch_item(self, name, batch_size, concat=False, keep_prob=1.0):
    assert(name in self.i_mappings)
    assert(keep_prob == 1.0), 'otherwise not implemented'
    i_mappings = self.i_mappings[name]
    if concat:
      return self.EmbeddingLayer(self.item_embs_cat, self.item_embs_mulhot, 
        self.i_biases_cat, self.i_biases_mulhot, i_mappings, batch_size, 
        self.item_attributes, 'item', True)
    else:
      item_cat, item_mulhot, item_b = self.EmbeddingLayer(self.item_embs_cat, self.item_embs_mulhot, 
        self.i_biases_cat, self.i_biases_mulhot, i_mappings, batch_size, 
        self.item_attributes, 'item', False)
      return item_cat + item_mulhot, item_b

  def get_user_model_size(self):
    return (sum(self.user_attributes._embedding_size_list_cat) + 
        sum(self.user_attributes._embedding_size_list_mulhot))
  def get_item_model_size(self):
    return (sum(self.item_attributes._embedding_size_list_cat) + 
        sum(self.item_attributes._embedding_size_list_mulhot))

  def get_user_proj(self, embedded_user, keep_prob, user_model_size, nonlinear):
    projs_cat, projs_mulhot, projs_cat_b, projs_mulhot_b = self.item_proj_layer(
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

  def get_prediction(self, proj_user_drops):
    print("construct inner products between mb users and full item embeddings")  
    # full vocabulary item indices, also just for train
    full_out_layer = self.full_output_layer(self.item_attributes)
    indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = full_out_layer
    # compute inner product between item_hidden and {user_feature_embedding}
    # then lookup to compute logits
    innerps = []
    
    for i in xrange(self.item_attributes.num_features_cat):
      item_emb_cat = self.item_embs2_cat[i] if self.item_output else self.item_embs_cat[i]
      innerp = tf.matmul(item_emb_cat, tf.transpose(
        proj_user_drops[i])) + self.i_biases_cat[i] # Vf by mb
      innerps.append(embedding_ops.embedding_lookup(innerp, indices_cat[i], 
        name='emb_lookup_innerp_{0}'.format(i))) # V by mb
    offset = self.item_attributes.num_features_cat
    for i in xrange(self.item_attributes.num_features_mulhot):
      item_embs_mulhot = self.item_embs2_mulhot[i] if self.item_output else self.item_embs_mulhot[i]
      innerp = tf.add(tf.matmul(item_embs_mulhot,
        tf.transpose(proj_user_drops[i+offset])), self.i_biases_mulhot[i]) 
      innerps.append(tf.div(tf.unsorted_segment_sum(embedding_ops.embedding_lookup(
        innerp, indices_mulhot[i]), segids_mulhot[i], self.logit_size),
        lengths_mulhot[i]))
    logits = tf.transpose(tf.reduce_sum(innerps, 0))
    return logits

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

  def EmbeddingLayer(self, embs_cat, embs_mulhot, b_cat, b_mulhot, 
    mappings, mb, attributes, prefix='', concatenation=True):
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
      embedded_flat = embedding_ops.embedding_lookup(embs_mulhot[i], 
        mulhot_indices[i])
      embedded_sum = tf.unsorted_segment_sum(embedded_flat, mulhot_segids[i], 
        mb)
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

  def compute_loss(self, logits, item_target, loss='ce'):
    assert(loss in ['ce', 'mce', 'warp', 'mw', 'bpr', 'bpr-hinge'])
    if loss in ['ce', 'mce']:
      return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, item_target)
    elif loss in ['warp', 'mw']:
      V = self.logit_size if loss == 'warp' else self.n_sampled
      mb = self.batch_size
      self.mask = tf.Variable([True] * V * mb, dtype=tf.bool, trainable=False)
      zero_logits = tf.constant([[0.0] * V] * mb)
      self.pos_indices = tf.placeholder(tf.int32, shape = [None])
      self.l_true = tf.placeholder(tf.bool, shape = [None], name='l_true')
      self.l_false = tf.placeholder(tf.bool, shape = [None], name='l_false')
      self.set_mask = tf.scatter_update(self.mask, self.pos_indices, 
        self.l_false)
      self.reset_mask = tf.scatter_update(self.mask, self.pos_indices, 
        self.l_true)
      flat_matrix = tf.reshape(logits, [-1])
      idx_flattened0 = tf.range(0, mb) * V
      idx_flattened = idx_flattened0 + item_target
      logits_ = tf.gather(flat_matrix, idx_flattened)
      logits_ = tf.reshape(logits_, [mb, 1])
      logits2 = tf.sub(logits, logits_) + 1
      mask2 = tf.reshape(self.mask, [mb, V])
      target = tf.select(mask2, logits2, zero_logits)
      return tf.log(1 + tf.reduce_sum(tf.nn.relu(target), 1))
    elif loss == 'bpr':
      return tf.log(1 + tf.exp(logits))
    elif loss == 'bpr-hinge':
      return tf.maximum(1 + logits, 0)

  def get_warp_mask(self):
    return self.set_mask, self.reset_mask

  def prepare_warp(self, pos_item_set, pos_item_set_eval):
    self.pos_item_set = pos_item_set
    self.pos_item_set_eval = pos_item_set_eval
    return 

  def _add_input(self, input_feed, opt, input_, name_):
    if opt == 'user':
      att = self.user_attributes
      mappings = self.u_mappings[name_]
    elif opt == 'item':
      att = self.item_attributes
      mappings = self.i_mappings[name_]
    else:
      exit(-1)

    for i in xrange(att.num_features_cat):
      input_feed[mappings[0][i].name] = att.features_cat[i][input_]
        
    for i in xrange(att.num_features_mulhot):
      v_i, s_i, l_i = (att.features_mulhot[i], att.mulhot_starts[i], 
        att.mulhot_lengths[i])
      vals = list(itertools.chain.from_iterable(
        [v_i[s_i[u]:s_i[u]+l_i[u]] for u in input_]))
      Ls = [l_i[u] for u in input_]
      l = len(Ls) 
      i1 = list(itertools.chain.from_iterable(
        Ls[i] * [i] for i in range(len(Ls))))
      input_feed[mappings[1][i].name] = vals
      input_feed[mappings[2][i].name] = i1
      input_feed[mappings[3][i].name] = np.reshape(Ls, (l, 1))

  def add_input(self, input_feed, user_input, item_input, 
        neg_item_input=None, item_sampled = None, item_sampled_id2idx = None, 
        forward_only=False, recommend=False, recommend_new = False, loss=None):
    # users
    if self.user_attributes is not None:
      self._add_input(input_feed, 'user', user_input, 'input')
    # pos neg: when input_steps = 0 
    if self.item_attributes is not None and recommend is False and self.input_steps == 0:
      self._add_input(input_feed, 'item', item_input, 'pos')
      self._add_input(input_feed, 'item', neg_item_input, 'neg')    
    # sampled item: when sampled-loss is used
    if self.item_attributes is not None and recommend is False and self.n_sampled is not None and loss in ['mw', 'mce']:
      self._add_input(input_feed, 'item', item_sampled, 'sampled')
    # input item: for lstm
    if self.item_attributes is not None and self.input_steps > 0:
      for step in xrange(self.input_steps):
        self._add_input(input_feed, 'item', item_input[step], 
          'input{}'.format(step))
    # for warp loss.
    input_feed_warp = {}
    if loss in ['warp', 'mw'] and recommend is False:
      V = self.logit_size if loss == 'warp' else self.n_sampled
      mask_indices, c = [], 0
      s_2idx = self.item_ind2logit_ind if loss == 'warp' else item_sampled_id2idx      
      item_set = self.pos_item_set_eval if forward_only else self.pos_item_set

      if loss == 'warp':
        for u in user_input:
          offset = c * V
          mask_indices.extend([s_2idx[v] + offset for v in item_set[u]])
          c += 1
      else:
        for u in user_input:
          offset = c * V
          mask_indices.extend([s_2idx[v] + offset for v in item_set[u] 
            if v in s_2idx])
          c += 1          
      L = len(mask_indices)
      input_feed_warp[self.pos_indices.name] = mask_indices
      input_feed_warp[self.l_false.name] = [False] * L
      input_feed_warp[self.l_true.name] = [True] * L

    return input_feed_warp


    # '''
    # test and recommend new items
    # '''
    # if self.logit_size_test is not None:
    #   (indices_cat2, indices_mulhot2, segids_mulhot2, 
    #     lengths_mulhot2) = self.full_output_layer(item_attributes, True)
    #   innerps_test = []
    #   for i in xrange(item_attributes.num_features_cat):
    #     innerp = tf.matmul(item_embs_cat[i], tf.transpose(
    #       proj_user_drops_cat[i])) + i_biases_cat[i] # Vf by mb
    #     innerps_test.append(embedding_ops.embedding_lookup(innerp, 
    #       indices_cat2[i], name='emb_lookup_innerp_test_{0}'.format(i))) # V by mb
    #   for i in xrange(item_attributes.num_features_mulhot):
    #     innerp = tf.add(tf.matmul(item_embs_mulhot[i], 
    #       tf.transpose(proj_user_drops_mulhot[i])), i_biases_mulhot[i]) # Vf by mb
    #     innerps_test.append(tf.div(tf.unsorted_segment_sum(
    #       embedding_ops.embedding_lookup(
    #       innerp, indices_mulhot2[i]), segids_mulhot2[i], self.logit_size_test),
    #       lengths_mulhot2[i]))
      
    #   if self.use_user_bias:
    #     logits_test = tf.add(tf.transpose(tf.reduce_sum(innerps_test, 0)), user_b)
    #   else:
    #     logits_test = tf.transpose(tf.reduce_sum(innerps_test, 0))
      
    #   _, self.indices_test= tf.nn.top_k(logits_test, 30, sorted=True)

    # print("loss function is %s" % loss)
    # if loss == 'ce':  # softmax cross-entropy loss
    #   batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, 
    #     self.item_target)
    # elif loss == 'mce':
    #   batch_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
    #     sampled_logits, self.item_target)
    # elif loss == 'bpr_hinge':
    #   batch_loss = tf.maximum(1 + neg_pos, 0)
    #   self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))
    # elif loss == 'bpr': 
    #   batch_loss = tf.log(1+tf.exp(neg_pos))
    #   self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))
    # elif loss == 'warp' or loss == 'mw':
    #   V = self.logit_size if loss == 'warp' else self.n_sampled
    #   logits0 = logits if loss == 'warp' else sampled_logits
    #   self.mask = tf.Variable([True] * V * mb,  dtype=tf.bool, trainable=False, 
    #     name='mask')
    #   zero_logits = tf.constant([[0.0] * V] * mb)      
    #   self.pos_indices = tf.placeholder(tf.int32, shape = [None], 
    #     name = "mask_indices") # for WARP, should include target itself
    #   self.l_true = tf.placeholder(tf.bool, shape = [None], name='l_true')
    #   self.l_false = tf.placeholder(tf.bool, shape = [None], name='l_false')
    #   self.set_mask = tf.scatter_update(self.mask, self.pos_indices, 
    #     self.l_false)
    #   self.reset_mask = tf.scatter_update(self.mask, self.pos_indices, 
    #     self.l_true)
      
    #   flat_matrix = tf.reshape(logits0, [-1])
    #   idx_flattened0 = tf.range(0, mb) * V
    #   idx_flattened = idx_flattened0 + self.item_target
    #   logits_ = tf.gather(flat_matrix, idx_flattened)
    #   logits_ = tf.reshape(logits_, [mb, 1])
    #   logits2 = tf.sub(logits0, logits_) + 1

    #   mask2 = tf.reshape(self.mask, [mb, V])
    #   target = tf.select(mask2, logits2, zero_logits)
    #   batch_loss = tf.log(1 + tf.reduce_sum(tf.nn.relu(target), 1))

    # else:
    #   print("No such loss function. Exit.")
    #   exit()
    # '''
    # batch warp loss
    # logits
    # logits = logits - logits[target item] + 1 // index
    # logits[mask] = 0 // mask?
    # logits = max(logits, 0) ??
    # loss = tf.log(1+reduce_sum(logits))
    # '''
