
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
from sparse_map import create_sparse_map
import itertools

class LatentProductModel(object):
  def __init__(self, user_size, item_size, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, user_attributes=None, 
               item_attributes=None, item_ind2logit_ind=None, 
               logit_ind2item_ind=None, loss_function='ce', GPU=None):

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
    
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    self.input_att_map = True
    # if GPU is not None:
    #   device_name = '/gpu:%d' % GPU
    # else:
    #   device_name = '/cpu:0'
    # with tf.device(device_name):
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
      self.u_cat_indices, self.u_mulhot_indices, self.u_mulhot_values, self.u_mtls = [],[], [], []
      for i in xrange(user_attributes.num_features_cat):
        self.u_cat_indices.append(tf.placeholder(tf.int32, shape = [mb], 
          name = "u_cat_ind{0}".format(i)))
      for i in xrange(user_attributes.num_features_mulhot):
        self.u_mulhot_indices.append(tf.placeholder(tf.int64, shape = [None, 2], 
          name = "u_mulhot_ind{0}".format(i)))
        self.u_mulhot_values.append(tf.placeholder(tf.int32, shape = [None], 
          name = "u_mulhot_val{0}".format(i)))
        self.u_mtls.append(user_attributes.mulhot_max_length[i])

      if self.input_att_map:
        u_mappings = (self.u_cat_indices, self.u_mulhot_indices, 
          self.u_mulhot_values, self.u_mtls)
      else:
        u_mappings = self.mapped(user_attributes, True, 'user')

      embedded_user = self.EmbeddingLayer2(self.user_input, user_embs_cat, 
        user_embs_mulhot, u_mappings, batch_size, user_attributes, 
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
    biases_cat, biases_mulhot, projs_cat, projs_mulhot = self.item_proj_layer(
      item_attributes, hidden_size=user_model_size)
    # i_mappings = self.mapped(item_attributes, True, 'item')    
    

    self.i_cat_indices_tr, self.i_mulhot_indices_tr, self.i_mulhot_values_tr, self.i_mtls_tr = [],[], [], []
    self.i_cat_indices_tr2, self.i_mulhot_indices_tr2, self.i_mulhot_values_tr2, self.i_mtls_tr2 = [],[], [], []
    for i in xrange(user_attributes.num_features_cat):
      self.i_cat_indices_tr.append(tf.placeholder(tf.int32, shape = [mb], 
        name = "i_cat_ind_tr{0}".format(i)))
      self.i_cat_indices_tr2.append(tf.placeholder(tf.int32, shape = [mb], 
        name = "i_cat_ind_tr2{0}".format(i)))
    for i in xrange(user_attributes.num_features_mulhot):
      self.i_mulhot_indices_tr.append(tf.placeholder(tf.int64, shape = [None, 2], 
        name = "i_mulhot_ind_tr{0}".format(i)))
      self.i_mulhot_values_tr.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_val_tr{0}".format(i)))
      self.i_mtls_tr.append(user_attributes.mulhot_max_length[i])

      self.i_mulhot_indices_tr2.append(tf.placeholder(tf.int64, shape = [None, 2], 
        name = "i_mulhot_ind_tr2{0}".format(i)))
      self.i_mulhot_values_tr2.append(tf.placeholder(tf.int32, shape = [None], 
        name = "i_mulhot_val_tr2{0}".format(i)))
      self.i_mtls_tr2.append(user_attributes.mulhot_max_length[i])

    if self.input_att_map:
      i_mappings_tr = (self.i_cat_indices_tr, self.i_mulhot_indices_tr, 
        self.i_mulhot_values_tr, self.i_mtls_tr)
      i_mappings_tr2= (self.i_cat_indices_tr2, self.i_mulhot_indices_tr2, 
        self.i_mulhot_values_tr2, self.i_mtls_tr2)
    else:
      i_mappings_tr = self.mapped_item_tr(item_attributes, True, 'item')
      i_mappings_tr2= i_mappings_tr
    # full vocabulary item indices, also just for train
    indices_cat, indices_mulhot = self.full_output_layer(item_attributes)

    print("construct postive/negative items/scores ")
    pos_embs_item_cat, pos_embs_item_mulhot = self.EmbeddingLayer2(
      self.item_input, item_embs_cat, item_embs_mulhot, i_mappings_tr,
      batch_size, item_attributes, 'item', False, self.input_att_map)
    neg_embs_item_cat, neg_embs_item_mulhot = self.EmbeddingLayer2(
      self.neg_item_input, item_embs_cat, item_embs_mulhot, i_mappings_tr2, 
      batch_size, item_attributes, 'item', False, self.input_att_map)
    
    pos_scores, neg_scores = [], []
    for i in xrange(item_attributes.num_features_cat):
      proj_user = tf.matmul(embedded_user, projs_cat[i]) # mb by d_f
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      
      pos_embs_item_cat_drop = tf.nn.dropout(pos_embs_item_cat[i], 
        self.keep_prob)
      pos_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        pos_embs_item_cat_drop), 1))

      neg_embs_item_cat_drop = tf.nn.dropout(neg_embs_item_cat[i], 
        self.keep_prob)
      neg_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        neg_embs_item_cat_drop), 1))

    for i in xrange(item_attributes.num_features_mulhot):
      proj_user = tf.matmul(embedded_user, projs_mulhot[i]) 
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      
      pos_embs_item_mulhot_drop = tf.nn.dropout(pos_embs_item_mulhot[i], 
        self.keep_prob)
      pos_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        pos_embs_item_mulhot_drop), 1))

      neg_embs_item_mulhot_drop = tf.nn.dropout(neg_embs_item_mulhot[i], 
        self.keep_prob)
      neg_scores.append(tf.reduce_sum(tf.mul(proj_user_drop, 
        neg_embs_item_mulhot_drop), 1))

    print("neg_score[0] shape")
    print(neg_scores[0].get_shape())
    pos_score = tf.reduce_sum(pos_scores, 0)
    neg_score = tf.reduce_sum(neg_scores, 0)
    neg_pos = neg_score - pos_score
    print("neg_pos shape")
    print(neg_pos.get_shape())
    self.pos_score = pos_score
    self.neg_score = neg_score
    
    print("construct inner products between mb users and full item embeddings")  
    # compute inner product between item_hidden and {user_feature_embedding}
    # then lookup to compute logits
    innerps = []
    # for i in xrange(item_attributes.num_features_cat):
    #   proj_user = tf.matmul(embedded_user, projs_cat[i]) # mb by d_f
    #   proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)

    #   innerp = tf.matmul(
    #     proj_user_drop, tf.transpose(item_embs_cat[i])) + biases_cat[i] 
    #   # mb by V_f
    #   innerps.append(tf.transpose(embedding_ops.embedding_lookup(
    #     tf.transpose(innerp), indices_cat[i], 
    #     name='emb_lookup_innerp_{0}'.format(i)))) # mb by V

    # for i in xrange(item_attributes.num_features_mulhot):
    #   proj_user = tf.matmul(embedded_user, projs_mulhot[i])
    #   proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
    #   innerp = tf.matmul(
    #     proj_user_drop, tf.transpose(item_embs_mulhot[i])) + biases_mulhot[i]
    #   innerps.append(tf.transpose(embedding_ops.embedding_lookup_sparse(
    #     tf.transpose(innerp), indices_mulhot[i], sp_weights=None, 
    #     name='emb_lookup_sp_innerp_{0}'.format(i))))
    
    # logits = tf.reduce_sum(innerps, 0) # mb by |logits|



    for i in xrange(item_attributes.num_features_cat):
      proj_user = tf.matmul(embedded_user, projs_cat[i]) # mb by d_f
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      innerp = tf.matmul(item_embs_cat[i], tf.transpose(
        proj_user_drop)) + biases_cat[i] # Vf by mb
      innerps.append(embedding_ops.embedding_lookup(innerp, indices_cat[i], 
        name='emb_lookup_innerp_{0}'.format(i))) # V by mb

    for i in xrange(item_attributes.num_features_mulhot):
      proj_user = tf.matmul(embedded_user, projs_mulhot[i])
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      innerp = tf.add(tf.matmul(item_embs_mulhot[i], 
        tf.transpose(proj_user_drop)), biases_mulhot[i]) # Vf by mb
      innerps.append(embedding_ops.embedding_lookup_sparse(innerp, 
        indices_mulhot[i], sp_weights=None, 
        name='emb_lookup_sp_innerp_{0}'.format(i)))

    logits = tf.transpose(tf.reduce_sum(innerps, 0)) 
    # there is one transpose here.. can we remove??

    self.output = logits

    if self.loss_function == 'ce':
      # softmax cross-entropy loss
      print("loss function is %s" % self.loss_function)
      self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, 
        self.item_input)
      self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))
    elif self.loss_function == 'bpr':
    # # BRP hinge loss
    # print("loss function is bpr with hinge loss")
    # self.loss = tf.maximum(1 - pos_score + neg_score, 0)
    # BRP 
      print("loss function is %s" % self.loss_function)
      self.loss = tf.log(1+tf.exp(neg_pos))
      self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))
    elif self.loss_function == 'warp':
      print("loss function is %s" % self.loss_function)
      self.mask = tf.Variable([True] * self.logit_size * mb,  dtype=tf.bool, trainable=False, 
        name='mask')
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
      self.loss = tf.log(1 + tf.reduce_sum(tf.nn.relu(target), 1))
      self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(neg_pos))
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
    self.loss = tf.reduce_mean(self.loss)
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

  def item_proj_layer(self, attributes, hidden_size, scope = None):
    biases_cat, biases_mulhot = [], []
    projs_cat, projs_mulhot = [], []
    # with vs.variable_scope(scope or type(self).__name__):
    #   with ops.device("/cpu:0"):
    # create projections and biases
    for i in range(attributes.num_features_cat):
      size = attributes._embedding_size_list_cat[i]
      w = tf.get_variable("out_proj_cat_{0}".format(i), [hidden_size, size], 
        dtype=tf.float32)
      projs_cat.append(w)
      V = attributes._embedding_classes_list_cat[i]
      # print("V = %d" % V)
      b = tf.get_variable("out_bias_cat_{0}".format(i), [V, 1], 
        dtype=tf.float32)
      biases_cat.append(b)

    for i in range(attributes.num_features_mulhot):
      size = attributes._embedding_size_list_mulhot[i]
      w = tf.get_variable("out_proj_mulhot_{0}".format(i), 
        [hidden_size, size], dtype=tf.float32)
      projs_mulhot.append(w)
      V = attributes._embedding_classes_list_mulhot[i]
      b = tf.get_variable("out_bias_mulhot_{0}".format(i), [V, 1], 
        dtype=tf.float32)
      biases_mulhot.append(b)
    return biases_cat, biases_mulhot, projs_cat, projs_mulhot

  def full_output_layer(self, attributes):
    indices_cat, indices_mulhot = [], []
    prefix = 'item'
    feats_cat_tr, _, _ , _, _ = self.mapped_item_tr(attributes, False, 
      prefix=prefix)
    indices_cat = feats_cat_tr

    # TODO: only train is constructed now
    # TODO: does these indictes need to be variables
    for i in range(attributes.num_features_mulhot):
      full_indices = tf.to_int64(attributes.full_indices_tr[i])
      full_values = attributes.full_values_tr[i]
      sp_shape = tf.to_int64(attributes.sp_shapes_tr[i])
      st = tf.SparseTensor(full_indices, full_values, sp_shape)
      indices_mulhot.append(st)

    return indices_cat, indices_mulhot

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
    
  def EmbeddingLayer2(self, input_, embs_cat, embs_mulhot, mappings, mb, 
    attributes, prefix='', concatenation=True, input_att_map=False):
    if input_att_map:
      cat_indices, mulhot_indices, mulhot_values, mtls = mappings
    else:
      feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths, mtls = mappings
    
    cat_list, mulhot_list = [], []
    x = array_ops.reshape(input_, [-1])

    # for i in xrange(attributes.num_features_cat):
    #   if input_att_map:
    #     idx = cat_indices[i]
    #   else:
    #     idx = embedding_ops.embedding_lookup(feats_cat[i], x, 
    #       name='emb_lookup_item_map_{0}'.format(i))
    #   embedded = embedding_ops.embedding_lookup(embs_cat[i], 
    #     array_ops.reshape(idx, [-1]), name='emb_lookup_item_{0}'.format(i))
    #   cat_list.append(embedded)          
    for i in xrange(attributes.num_features_cat):
      if input_att_map:
        idx = cat_indices[i]
      else:
        # idx = array_ops.reshape(embedding_ops.embedding_lookup(feats_cat[i], x, 
        #   name='emb_lookup_item_map_{0}'.format(i)), [-1])
        idx = embedding_ops.embedding_lookup(feats_cat[i], x, 
          name='emb_lookup_item_map_{0}'.format(i))            
      embedded = embedding_ops.embedding_lookup(embs_cat[i], 
        idx, name='emb_lookup_item_{0}'.format(i))  # on cpu
      cat_list.append(embedded)          
    for i in xrange(attributes.num_features_mulhot):
      if input_att_map:
        sp_shape = tf.to_int64([self.batch_size, mtls[i]])
        st = tf.SparseTensor(mulhot_indices[i], mulhot_values[i], sp_shape)
      else:
        start = embedding_ops.embedding_lookup(mulhot_starts[i], x, 
        name='emb_lookup_start_{0}'.format(i)) # mb by 1
        leng = embedding_ops.embedding_lookup(mulhot_lengths[i], x, 
        name='emb_lookup_leng_{0}'.format(i)) # mb by 1
        st = create_sparse_map(feats_mulhot[i], start, leng, mb, mtls[i], i, 
          prefix)
      embedded = embedding_ops.embedding_lookup_sparse(embs_mulhot[i], 
        st, sp_weights=None, name='emb_lookup_sp_item_mulhot_{0}'.format(i))
      mulhot_list.append(embedded)
    
    if concatenation:
      return tf.concat(1, cat_list + mulhot_list)
    else:
      return cat_list, mulhot_list

  def step(self, session, user_input, item_input, neg_item_input=None, 
    forward_only=False, recommend=False, loss=None):
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
        i2 = list(itertools.chain.from_iterable(
          range(Ls[i]) for i in range(len(Ls))))
        input_feed[self.u_mulhot_values[i].name] = vals        
        input_feed[self.u_mulhot_indices[i].name] = zip(i1, i2)
        
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

        i1 = list(itertools.chain.from_iterable(
          Ls[i] * [i] for i in range(len(Ls))))
        i2 = list(itertools.chain.from_iterable(
          range(Ls[i]) for i in range(len(Ls))))

        input_feed[self.i_mulhot_indices_tr[i].name] = zip(i1, i2)
        input_feed[self.i_mulhot_values_tr[i].name] = vals

        i1_2 = list(itertools.chain.from_iterable(
          Ls2[i] * [i] for i in range(len(Ls2))))
        i2_2 = list(itertools.chain.from_iterable(
          range(Ls2[i]) for i in range(len(Ls2))))
        input_feed[self.i_mulhot_indices_tr2[i].name] = zip(i1_2, i2_2)
        input_feed[self.i_mulhot_values_tr2[i].name] = vals2


    if forward_only or recommend:
      input_feed[self.keep_prob.name] = 1.0
    else:
      input_feed[self.keep_prob.name] = 0.5

    if loss == 'warp' and recommend is not False:
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
      output_feed = [self.indices]

    if loss == 'warp' and recommend is not False:
      session.run(self.set_mask, input_feed_warp)

    outputs = session.run(output_feed, input_feed)

    if loss == 'warp' and recommend is not False:
      # print('reset mask %d' % L)
      session.run(self.reset_mask, input_feed_warp)

    if not recommend:
      if not forward_only:
        return outputs[1], outputs[2], outputs[3], outputs[4]
      else:
        return outputs[0], outputs[1]
    else:
      return outputs[0]

  def step2(self, session, user_input, item_input, neg_item_input=None, 
    forward_only=False, recommend=False, loss=None, run_op=None, run_meta=None):
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
    if self.user_attributes is not None:
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
        i2 = list(itertools.chain.from_iterable(
          range(Ls[i]) for i in range(len(Ls))))
        input_feed[self.u_mulhot_values[i].name] = vals        
        input_feed[self.u_mulhot_indices[i].name] = zip(i1, i2)
        
    if self.item_attributes is not None:
      ia = self.item_attributes
      for i in xrange(ia.num_features_cat):
        input_feed[self.i_cat_indices_tr[i].name] = ia.features_cat_tr[i][item_input]
      for i in xrange(ia.num_features_mulhot):
        v_i = ia.features_mulhot_tr[i]
        s_i = ia.mulhot_starts_tr[i]
        l_i = ia.mulhot_lengs_tr[i]
        vals = list(itertools.chain.from_iterable(
          [v_i[s_i[u]:s_i[u]+l_i[u]] for u in item_input]))
        Ls = [l_i[u] for u in item_input]
        i1 = list(itertools.chain.from_iterable(
          Ls[i] * [i] for i in range(len(Ls))))
        i2 = list(itertools.chain.from_iterable(
          range(Ls[i]) for i in range(len(Ls))))

        input_feed[self.i_mulhot_indices_tr[i].name] = zip(i1, i2)
        input_feed[self.i_mulhot_values_tr[i].name] = vals
  

    if forward_only or recommend:
      input_feed[self.keep_prob.name] = 1.0
    else:
      input_feed[self.keep_prob.name] = 0.5

    if loss == 'warp':
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

    if not recommend:
      if not forward_only:
        # output_feed = [self.updates, self.loss, self.auc]
        output_feed = [self.updates, self.loss, self.auc, self.neg_score, self.pos_score]
      else:
        output_feed = [self.loss, self.auc]
    else:
      # values, indices= tf.nn.top_k(self.output, 30, sorted=True)
      
      output_feed = [self.indices]

    if loss == 'warp':
      # print('set mask, %d' % L )
      session.run(self.set_mask(L), input_feed_warp)

    outputs = session.run(output_feed, input_feed, options=run_op, run_metadata=run_meta)

    if loss == 'warp':
      # print('reset mask %d' % L)
      session.run(self.reset_mask(L), input_feed_warp)

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









    # with ops.device("/cpu:0"):
    #   size_item = item_attributes._embedding_size_list_cat[0]
    #   embedded_item = tf.get_variable("proj_w", [size_item, self.item_size])

    #   embedded_item_t = tf.transpose(embedded_item)
    #   item_b = tf.get_variable("proj_b", [self.item_size]) 
    #   w = tf.get_variable('transform', [user_model_size, size_item])
    #   print("sizes: %d/%d" % (user_model_size, size_item))
    #   print("embed sizes: %d" % (self.item_size))
    #   print(projs_cat[0].get_shape())
        
    

    # ''' actually the loss is good to go (for no feature cases '''
    # # Sampled softmax only makes sense if we sample less than vocabulary size.
    # if num_samples > 0 and num_samples < self.item_size:
    #   def sampled_loss(inputs, labels):
    #     labels = tf.reshape(labels, [-1, 1])
    #     return tf.nn.sampled_softmax_loss(embedded_item_t, item_b, inputs, 
    #       labels, num_samples, self.item_size)
    #   loss_function = sampled_loss
    

    # proj_user = tf.matmul(embedded_user, w) #+ biases_cat[0]

    # self.loss = loss_function(proj_user, self.item_input)
    # self.loss = tf.reduce_mean(self.loss)
    # # self.output = tf.matmul(embedded_user, embedded_item) + item_b
    # self.output = None

    


    # proj_user = tf.matmul(embedded_user, projs_cat[0]) 
    # innerps = [tf.matmul(proj_user, embedded_item) + item_b]


    # innerps = [tf.matmul(proj_user, embs_cat[0]) + biases_cat[0]]
    # # innerps = [tf.matmul(proj_user, embedded_item) + biases_cat[0]]


'''
mul index 
'''

      # print('shape of logits')
      # print(logits.get_shape())
      # row_indices = tf.constant(np.arange(self.batch_size))
      # # row_indices = self.item_input
      # # col_indices = self.item_input
      # col_indices = tf.constant(np.arange(self.batch_size))
      # col_indices2 = self.neg_item_input
      # flat_ind1 = ravel_multi_index(logits, [row_indices, col_indices])
      # flat_ind2 = ravel_multi_index(logits, [row_indices, col_indices2])
      # flat_matrix = tf.reshape(logits, [-1])
      # logit1 = tf.gather(flat_matrix, flat_ind1)
      # logit2 = tf.gather(flat_matrix, flat_ind2)
      # self.auc = 0.5 - 0.5 * tf.reduce_mean(tf.sign(logit2 - logit1))
