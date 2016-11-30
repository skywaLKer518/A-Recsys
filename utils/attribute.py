
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
from tensorflow.python.ops.embedding_ops import embedding_lookup as lookup
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

    # user embeddings
    self.user_embs_cat, self.user_embs_mulhot = self.embedded(user_attributes, 
      prefix='user')
    #item embeddings
    self.item_embs_cat, self.item_embs_mulhot = self.embedded(item_attributes, 
      prefix='item', transpose=False)
    self.i_biases_cat, self.i_biases_mulhot = self.embedded_bias(
      item_attributes, 'item')
    if item_output:
      self.item_embs2_cat, self.item_embs2_mulhot = self.embedded(
        item_attributes, prefix='item_output', transpose=False)
      self.i_biases2_cat, self.i_biases2_mulhot = self.embedded_bias(
        item_attributes, 'item_output')
    # input users
    self.u_mappings = {}
    self.u_mappings['input'] = self._placeholders('user', 'input', mb)

    # item -- positive/negative sample indices
    print("construct postive/negative items/scores ")        
    self.i_mappings = {}
    self.i_mappings['pos'] = self._placeholders('item', 'pos', mb)
    self.i_mappings['neg'] = self._placeholders('item', 'neg', mb)

    # mini-batch item candidate pool
    print("construct mini-batch item candicate pool")
    self.i_mappings['sampled'] = self._placeholders('item', 'sampled', self.n_sampled)

    # input items (for lstm etc)
    print("construct input item")
    for step in xrange(input_steps):
      name_ = 'input{}'.format(step)
      self.i_mappings[name_] = self._placeholders('item', name_, mb)

    # item for prediction
    ia = item_attributes
    print("construct full prediction layer")
    indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = [],[],[],[]
    for i in xrange(ia.num_features_cat):
      indices_cat.append(tf.constant(ia.features_cat_tr[i]))
    for i in xrange(ia.num_features_mulhot):
      indices_mulhot.append(tf.constant(ia.full_values_tr[i]))
      segids_mulhot.append(tf.constant(ia.full_segids_tr[i]))
      lengths_mulhot.append(tf.constant(ia.full_lengths_tr[i]))
    self.i_mappings['prediction'] = (indices_cat, indices_mulhot, segids_mulhot,
      lengths_mulhot)

    return

  def _placeholders(self, opt, name, size):
    cat_indices, mulhot_indices, mulhot_segids, mulhot_lengths = [],[], [], []
    att = self.user_attributes if opt =='user' else self.item_attributes
    for i in xrange(att.num_features_cat):
      cat_indices.append(tf.placeholder(tf.int32, shape = [size], 
        name = "{}_{}_cat_ind_{}".format(opt, name, i)))
    for i in xrange(att.num_features_mulhot):
      mulhot_indices.append(tf.placeholder(tf.int32, shape = [None], 
        name = "{}_{}_mulhot_ind_{}".format(opt, name, i)))
      mulhot_segids.append(tf.placeholder(tf.int32, shape = [None], 
        name = "{}_{}_mulhot_seg_{}".format(opt, name, i)))
      mulhot_lengths.append(tf.placeholder(tf.float32, shape= [size, 1], 
        name = "{}_{}_mulhot_len_{}".format(opt, name, i)))
    return (cat_indices, mulhot_indices, mulhot_segids, mulhot_lengths)

  def get_prediction(self, latent, full='prediction'):
    print("construct inner products between mb users and full item embeddings")  
    full_out_layer = self.i_mappings[full]
    # full_out_layer = self.full_output_layer(self.item_attributes)
    indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = full_out_layer
    # compute inner product between item_hidden and {user_feature_embedding}
    # then lookup to compute logits
    innerps = []
    for i in xrange(self.item_attributes.num_features_cat):
      item_emb_cat = self.item_embs2_cat[i] if self.item_output else self.item_embs_cat[i]
      i_biases_cat = self.i_biases2_cat[i] if self.item_output else self.i_biases_cat[i]
      u = latent[i] if isinstance(latent, list) else latent
      innerp = tf.matmul(item_emb_cat, tf.transpose(u)) + i_biases_cat # Vf by mb
      innerps.append(lookup(innerp, indices_cat[i])) # V by mb
    offset = self.item_attributes.num_features_cat
    for i in xrange(self.item_attributes.num_features_mulhot):
      item_embs_mulhot = self.item_embs2_mulhot[i] if self.item_output else self.item_embs_mulhot[i]
      item_biases_mulhot = self.i_biases2_mulhot[i] if self.item_output else self.i_biases_mulhot[i]
      u = latent[i+offset] if isinstance(latent, list) else latent
      innerp = tf.add(tf.matmul(item_embs_mulhot, tf.transpose(u)), 
        item_biases_mulhot) 
      innerps.append(tf.div(tf.unsorted_segment_sum(lookup(innerp, 
        indices_mulhot[i]), segids_mulhot[i], self.logit_size), lengths_mulhot[i]))

    logits = tf.transpose(tf.reduce_sum(innerps, 0))
    return logits

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

  def embedded_bias(self, attributes, prefix):
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

  def get_batch_user(self, keep_prob, concat=True):
    u_mappings = self.u_mappings['input']
    if concat:
      embedded_user, user_b = self._get_embedded(self.user_embs_cat, 
        self.user_embs_mulhot, b_cat=None, b_mulhot=None, mappings=u_mappings, 
        mb=self.batch_size, attributes=self.user_attributes, prefix='user', 
        concatenation=concat)
    else:
      user_cat, user_mulhot, user_b = self._get_embedded(
        self.user_embs_cat, self.user_embs_mulhot, b_cat=None, b_mulhot=None, 
        mappings=u_mappings, mb=self.batch_size, 
        attributes=self.user_attributes, prefix='user', concatenation=concat)
      embedded_user =  tf.reduce_mean(user_cat + user_mulhot, 0)
    embedded_user = tf.nn.dropout(embedded_user, keep_prob)
    return embedded_user, user_b  

  def get_batch_item(self, name, batch_size, concat=False, keep_prob=1.0):
    assert(name in self.i_mappings)
    assert(keep_prob == 1.0), 'otherwise not implemented'
    i_mappings = self.i_mappings[name]
    if concat:
      return self._get_embedded(self.item_embs_cat, self.item_embs_mulhot, 
        self.i_biases_cat, self.i_biases_mulhot, i_mappings, batch_size, 
        self.item_attributes, 'item', True)
    else:
      item_cat, item_mulhot, item_b = self._get_embedded(self.item_embs_cat, self.item_embs_mulhot, 
        self.i_biases_cat, self.i_biases_mulhot, i_mappings, batch_size, 
        self.item_attributes, 'item', False)
      return item_cat + item_mulhot, item_b

  def get_user_model_size(self):
    return (sum(self.user_attributes._embedding_size_list_cat) + 
        sum(self.user_attributes._embedding_size_list_mulhot))
  def get_item_model_size(self):
    return (sum(self.item_attributes._embedding_size_list_cat) + 
        sum(self.item_attributes._embedding_size_list_mulhot))

  def _get_embedded(self, embs_cat, embs_mulhot, b_cat, b_mulhot, 
    mappings, mb, attributes, prefix='', concatenation=True):
    cat_indices, mulhot_indices, mulhot_segids, mulhot_lengths = mappings
    cat_list, mulhot_list = [], []
    bias_cat_list, bias_mulhot_list = [], []

    for i in xrange(attributes.num_features_cat):
      embedded = lookup(embs_cat[i], cat_indices[i], 
        name='emb_lookup_item_{0}'.format(i))  # on cpu
      cat_list.append(embedded)
      if b_cat is not None:
        b = lookup(b_cat[i], cat_indices[i], 
          name = 'emb_lookup_item_b_{0}'.format(i))
        bias_cat_list.append(b)
    for i in xrange(attributes.num_features_mulhot):
      embedded_flat = lookup(embs_mulhot[i], mulhot_indices[i])
      embedded_sum = tf.unsorted_segment_sum(embedded_flat, mulhot_segids[i], 
        mb)
      embedded = tf.div(embedded_sum, mulhot_lengths[i])
      mulhot_list.append(embedded)
      if b_mulhot is not None:
        b_embedded_flat = lookup(b_mulhot[i], mulhot_indices[i])
        b_embedded_sum = tf.unsorted_segment_sum(b_embedded_flat, 
          mulhot_segids[i], mb)
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



  # def full_output_layer(self, attributes, test=False):
  #   indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = [],[],[],[]    
  #   prefix = 'item'
  #   if not test:
  #     # indices_cat, _, _ , _, _ = self.mapped_item_tr(attributes, False, 
  #     #   prefix=prefix)
  #     # indices_cat = []
  #     for i in xrange(attributes.num_features_cat):
  #       indices_cat.append(tf.constant(attributes.features_cat_tr[i]))
  #     for i in range(attributes.num_features_mulhot):
  #       indices_mulhot.append(attributes.full_values_tr[i])
  #       segids_mulhot.append(attributes.full_segids_tr[i])
  #       lengths_mulhot.append(attributes.full_lengths_tr[i])
  #   else:
  #     for i in xrange(attributes.num_features_cat):
  #       indices_cat.append(tf.constant(attributes.features_cat[i], 
  #         dtype=tf.int32))
  #     for i in range(attributes.num_features_mulhot):
  #       indices_mulhot.append(attributes.full_values[i])
  #       segids_mulhot.append(attributes.full_segids[i])
  #       lengths_mulhot.append(attributes.full_lengths[i])    
  #   return indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot
  
  # def mapped_item_tr(self, attributes, mulhot=False, prefix='item'):
  #   feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = [], [], [], []
  #   mtls = []
  #   with vs.variable_scope('item_map_tr', reuse=self.reuse_item_tr) as scope:
  #     for i in xrange(attributes.num_features_cat):
  #       init = tf.constant_initializer(value = attributes.features_cat_tr[i])
  #       feats_cat.append(tf.get_variable(name=prefix + "_tr_map1_{0}".format(i), 
  #         shape=attributes.features_cat_tr[i].shape, dtype=tf.int32, 
  #         initializer=init, trainable=False))
      
  #     if mulhot:
  #       for i in xrange(attributes.num_features_mulhot):
  #         init = tf.constant_initializer(value = attributes.features_mulhot_tr[i])
  #         feats_mulhot.append(tf.get_variable(
  #           name=prefix + "_tr_map2_{0}".format(i), 
  #           shape=attributes.features_mulhot_tr[i].shape, dtype=tf.int32, 
  #           initializer=init, trainable=False))
          
  #         init_s = tf.constant_initializer(value = attributes.mulhot_starts_tr[i])
  #         mulhot_starts.append(tf.get_variable(
  #           name=prefix + "_tr_map2_starts{0}".format(i), 
  #           shape=attributes.mulhot_starts_tr[i].shape, dtype=tf.int32, 
  #           initializer=init_s, trainable=False))

  #         init_l = tf.constant_initializer(
  #           value = attributes.mulhot_lengs_tr[i])
  #         mulhot_lengths.append(tf.get_variable(
  #           name=prefix + "_tr_map2_lengs{0}".format(i), 
  #           shape=attributes.mulhot_lengs_tr[i].shape, dtype=tf.int32, 
  #           initializer=init_l, trainable=False))
  #         mtls.append(attributes.mulhot_max_leng_tr[i])
  #     self.reuse_item_tr = True

  #   return feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths, mtls


    '''
    test and recommend new items
    '''
    # if self.logit_size_test is not None:
    #   (indices_cat2, indices_mulhot2, segids_mulhot2, 
    #     lengths_mulhot2) = self.full_output_layer(item_attributes, True)
    #   innerps_test = []
    #   for i in xrange(item_attributes.num_features_cat):
    #     innerp = tf.matmul(item_embs_cat[i], tf.transpose(
    #       proj_user_drops_cat[i])) + i_biases_cat[i] # Vf by mb
    #     innerps_test.append(lookup(innerp, 
    #       indices_cat2[i], name='emb_lookup_innerp_test_{0}'.format(i))) # V by mb
    #   for i in xrange(item_attributes.num_features_mulhot):
    #     innerp = tf.add(tf.matmul(item_embs_mulhot[i], 
    #       tf.transpose(proj_user_drops_mulhot[i])), i_biases_mulhot[i]) # Vf by mb
    #     innerps_test.append(tf.div(tf.unsorted_segment_sum(
    #       lookup(
    #       innerp, indices_mulhot2[i]), segids_mulhot2[i], self.logit_size_test),
    #       lengths_mulhot2[i]))
      
    #   if self.use_user_bias:
    #     logits_test = tf.add(tf.transpose(tf.reduce_sum(innerps_test, 0)), user_b)
    #   else:
    #     logits_test = tf.transpose(tf.reduce_sum(innerps_test, 0))
      
    #   _, self.indices_test= tf.nn.top_k(logits_test, 30, sorted=True)

