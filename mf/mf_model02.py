
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


def create_sparse_map(Vals, starts, lens, mb, mtl):
    '''
    Create sparse tensor given an array and 2 index vectors.
    input:
        Vals: 1-D tensor (mb, ), dense,
        starts: 1-D tensor (mb, ) starting position
        lens: 1-D tensor (mb, ) size
        mb: minibatch size, scalar
        mtl: max text length, a scalar
    return:
        st: sparse tensor, [mb by  mtl]
    '''
    
    sp_shape = [mb, mtl]
    value_list, idx_list = [], []

    for i in xrange(mb):
      l = tf.reshape(tf.slice(lens, [i], [1]), [])
      s = tf.slice(starts, [i], [1])
      val = tf.slice(Vals, s, [l])
      value_list.append(val)
      col1 = tf.fill([l, 1], i)
      col2 = tf.reshape(tf.range(0, l), [l,1])
      idx_list.append(tf.concat(1, [col1, col2])) # l1 * 2        

    values = tf.concat(0, value_list)
    indices = tf.concat(0, idx_list)

    indices = tf.to_int64(indices)
    sp_shape = tf.to_int64(sp_shape)
    st = tf.SparseTensor(indices, values, sp_shape)
    return st

class LatentProductModel(object):
  def __init__(self, user_size, item_size, size,
               num_layers, max_gradient_norm, batch_size, learning_rate,
               learning_rate_decay_factor, user_attributes=None, 
               item_attributes=None, item_ind2logit_ind=None, num_samples=512):

    self.user_size = user_size
    self.item_size = item_size
    self.item_ind2logit_ind = item_ind2logit_ind

    if user_attributes is not None:
      user_attributes.set_model_size(size)
    if item_attributes is not None:
      item_attributes.set_model_size(size)

    self.batch_size = batch_size
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    self.user_input = tf.placeholder(tf.int32, shape = [None], name = "user")
    self.item_input = tf.placeholder(tf.int32, shape = [None], name = "item")
    self.neg_item_input = tf.placeholder(tf.int32, shape = [None], 
      name = "neg_item")
    self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # user embedding
    if user_attributes is None:
      embedded_user = self.EmbeddingLayer(self.user_input, size)
      user_model_size = size
    else:
      user_embs_cat, user_embs_mulhot = self.embedded(user_attributes, 
        prefix='user')
      embedded_user = self.EmbeddingLayer2(self.user_input, user_embs_cat, 
        user_embs_mulhot, batch_size, size, user_attributes)
      user_model_size = (sum(user_attributes._embedding_size_list_cat) + 
          sum(user_attributes._embedding_size_list_mulhot))

    # item embedding
    item_embs_cat, item_embs_mulhot = self.embedded(item_attributes, 
      prefix='item', transpose=True)
    biases_cat, biases_mulhot, projs_cat, projs_mulhot = self.item_proj_layer(
      item_attributes)

    # full vocabulary item indices
    indices_cat, indices_mulhot = self.full_output_layer(item_attributes)

    print("construct postive/negative items/scores ")
    pos_embs_item_cat, pos_embs_item_mulhot = self.EmbeddingLayer2(
      self.item_input, item_embs_cat, item_embs_mulhot, batch_size, 
      item_attributes, False)
    neg_embs_item_cat, neg_embs_item_mulhot = self.EmbeddingLayer2(
      self.neg_item_input, item_embs_cat, item_embs_mulhot, batch_size, 
      item_attributes, False)
    
    pos_scores, neg_scores = [], []
    for i in xrange(item_attributes.num_features_cat):
      proj_user = tf.matmul(embedded_user, projs_cat[i]) # mb by d_f
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      
      pos_embs_item_cat_drop = tf.nn.dropout(pos_embs_item_cat[i], 
        self.keep_prob)
      pos_scores.append(tf.mul(proj_user_drop, pos_embs_item_cat_drop))

      neg_embs_item_cat_drop = tf.nn.dropout(neg_embs_item_cat[i], 
        self.keep_prob)
      neg_scores.append(tf.mul(proj_user_drop, neg_embs_item_cat_drop))

    for i in xrange(item_attributes.num_features_mulhot):
      proj_user = tf.matmul(embedded_user, projs_mulhot[i]) 
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      
      pos_embs_item_mulhot_drop = tf.nn.dropout(pos_embs_item_mulhot[i], 
        self.keep_prob)
      pos_scores.append(tf.mul(proj_user_drop, pos_embs_item_mulhot_drop))

      neg_embs_item_mulhot_drop = tf.nn.dropout(neg_embs_item_mulhot[i], 
        self.keep_prob)
      neg_scores.append(tf.mul(proj_user_drop, neg_embs_item_mulhot_drop))

    pos_score = tf.reduce_sum(pos_scores, 0)
    neg_score = tf.reduce_sum(neg_scores, 0)

    print("construct inner products")  
    # compute inner product between item_hidden and {user_feature_embedding}
    # then lookup to compute logits
    innerps = []
    for i in xrange(item_attributes.num_features_cat):
      proj_user = tf.matmul(embedded_user, projs_cat[i]) # mb by d_f
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)

      innerp = tf.matmul(proj_user_drop, item_embs_cat[i]) + biases_cat[i] # mb by V_f
      innerps.append(tf.transpose(embedding_ops.embedding_lookup(
        tf.transpose(innerp), indices_cat[i]))) # mb by V

    for i in xrange(item_attributes.num_features_mulhot):
      proj_user = tf.matmul(embedded_user, projs_mulhot[i])
      proj_user_drop = tf.nn.dropout(proj_user, self.keep_prob)
      innerp = tf.matmul(proj_user_drop, item_embs_mulhot[i]) + biases_mulhot[i]
      innerps.append(tf.transpose(embedding_ops.embedding_lookup_sparse(
        tf.transpose(innerp), indices_mulhot[i], sp_weights=None)))

    logits = tf.reduce_sum(innerps, 0)      

    self.output = logits
    # softmax cross-entropy loss
    self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, 
      self.item_input)
    # # BRP hinge loss
    # self.loss = tf.maximum(1 - pos_score + neg_score, 0)
    # # BRP hinge loss
    # self.loss = - tf.log(1+tf.exp(neg_score - pos_score))

    # mean over mini-batch
    self.loss = tf.reduce_mean(self.loss)
    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    
    opt = tf.train.AdagradOptimizer(self.learning_rate)
    gradients = tf.gradients(self.loss, params)
    
    self.updates = opt.apply_gradients(
      zip(gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables())

  def mapped(self, attributes, mulhot=False):
    feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = [], [], [], []

    for i in xrange(attributes.num_features_cat):
      init = tf.constant_initializer(value = attributes.features_cat[i])
      feats_cat.append(tf.get_variable(name=prefix + "map1_{0}".format(i), 
        shape=attributes.features_cat[i].shape, dtype=tf.int32, 
        initializer=init, trainable=False))

    if mulhot:
      for i in xrange(attributes.num_features_mulhot):
        init = tf.constant_initializer(value = attributes.features_mulhot[i])
        feats_mulhot.append(tf.get_variable(name=prefix + "map2_{0}".format(i), 
          shape=attributes.features_mulhot[i].shape, dtype=tf.int32, 
          initializer=init, trainable=False))
        
        init_s = tf.constant_initializer(value = attributes.mulhot_starts[i])
        mulhot_starts.append(tf.get_variable(
          name=prefix + "map2_starts{0}".format(i), 
          shape=attributes.mulhot_starts[i].shape, dtype=tf.int32, 
          initializer=init_s, trainable=False))

        init_l = tf.constant_initializer(value = attributes.mulhot_lengths[i])
        mulhot_lengths.append(tf.get_variable(
          name=prefix + "map2_lengs{0}".format(i), 
          shape=attributes.mulhot_lengths[i].shape, dtype=tf.int32, 
          initializer=init_l, trainable=False))
    
    return feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths

  def mapped_item_tr(self, attributes, prefix):
    feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = [], [], [], []

    for i in xrange(attributes.num_features_cat):
      # print("i = %d" % i)
      # print(np.unique(attributes.features_cat_tr[i]))
      init = tf.constant_initializer(value = attributes.features_cat_tr[i])
      feats_cat.append(tf.get_variable(name=prefix + "_tr_map1_{0}".format(i), 
        shape=attributes.features_cat_tr[i].shape, dtype=tf.int32, 
        initializer=init, trainable=False))
    
    return feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths

  def embedded(self, attributes, prefix='', transpose=False):
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
      b = tf.get_variable("out_bias_cat_{0}".format(i), [V], 
        dtype=tf.float32)
      biases_cat.append(b)

    for i in range(attributes.num_features_mulhot):
      size = attributes._embedding_size_list_mulhot[i]
      w = tf.get_variable("out_proj_mulhot_{0}".format(i), 
        [hidden_size, size], dtype=tf.float32)
      projs_mulhot.append(w)
      V = attributes._embedding_classes_list_mulhot[i]
      b = tf.get_variable("out_bias_mulhot_{0}".format(i), [V], 
        dtype=tf.float32)
      biases_mulhot.append(b)
    return biases_cat, biases_mulhot, projs_cat, projs_mulhot

  def full_output_layer(self, attributes):
    indices_cat, indices_mulhot = [], []
    # prefix = 'item'
    feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = self.mapped(
    attributes, False) # NOT trainable
    feats_cat_tr, _, _ , _ = self.mapped_item_tr(attributes, prefix=prefix)
    indices_cat = feats_cat_tr

    # TODO: only train is constructed now
    
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

  # def self.EmbeddingLayer3(self, item_input, embs_cat, embs_mulhot, 
  #   batch_size, size, attributes):
  #   '''
  #   similar as EmbeddingLayer2, no concatenation
  #   '''
  #   feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = self.mapped(
  #     attributes, mulhot=True) # all NOT trainable

  #   # lookup tables
  #   cat_list, mulhot_list = [], []

  #   x = array_ops.reshape(item_input, [-1])
    
  #   for i in xrange(attributes.num_features_cat):
  #     idx = embedding_ops.embedding_lookup(feats_cat[i], x)
  #     embedded = embedding_ops.embedding_lookup(embs_cat[i], 
  #       array_ops.reshape(idx, [-1]))
  #     cat_list.append(embedded)          
    
  #   for i in xrange(attributes.num_features_mulhot):
  #     mtl = attributes.mulhot_max_length[i]
  #     start = embedding_ops.embedding_lookup(mulhot_starts[i], x) # mb by 1
  #     leng = embedding_ops.embedding_lookup(mulhot_lengths[i], x) # mb by 1
  #     st = create_sparse_map(feats_mulhot[i], start, leng, mb, mtl)
  #     embedded = embedding_ops.embedding_lookup_sparse(embs_mulhot[i], 
  #       st, sp_weights=None)
  #     mulhot_list.append(embedded)
  #   return cat_list, mulhot_list
    
  def EmbeddingLayer2(self, input_, embs_cat, embs_mulhot, mb, attributes, 
    concatenation=True):
    feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = self.mapped(
      attributes, mulhot=True) # all NOT trainable

    # lookup tables
    cat_list, mulhot_list = [], []
    x = array_ops.reshape(input_, [-1])
    
    for i in xrange(attributes.num_features_cat):
      idx = embedding_ops.embedding_lookup(feats_cat[i], x)
      embedded = embedding_ops.embedding_lookup(embs_cat[i], 
        array_ops.reshape(idx, [-1]))
      cat_list.append(embedded)          
    
    for i in xrange(attributes.num_features_mulhot):
      mtl = attributes.mulhot_max_length[i]
      start = embedding_ops.embedding_lookup(mulhot_starts[i], x) # mb by 1
      leng = embedding_ops.embedding_lookup(mulhot_lengths[i], x) # mb by 1
      st = create_sparse_map(feats_mulhot[i], start, leng, mb, mtl)
      embedded = embedding_ops.embedding_lookup_sparse(embs_mulhot[i], 
        st, sp_weights=None)
      mulhot_list.append(embedded)
    if concatenation:
      return tf.concat(1, cat_list + mulhot_list)
    else:
      return cat_list, mulhot_list


  def step(self, session, user_input, item_input, neg_item_input=None, 
    forward_only=False, recommend=False):
    input_feed = {}
    input_feed[self.user_input.name] = user_input
    input_feed[self.item_input.name] = item_input
    input_feed[self.neg_item_input.name] = neg_item_input

    if forward_only or recommend:
      input_feed[self.keep_prob.name] = 1.0
    else:
      input_feed[self.keep_prob.name] = 0.5
    if not recommend:
      if not forward_only:
        output_feed = [self.updates, self.loss]
      else:
        output_feed = [self.loss]
    else:
      values, indices= tf.nn.top_k(self.output, 30, sorted=True)
      self.output = indices
      output_feed = [self.output]
    outputs = session.run(output_feed, input_feed)
    if not recommend:
      if not forward_only:
        return outputs[1]
      else:
        return outputs[0]
    else:
      return outputs[0]

  def get_batch(self, data):
    batch_user_input, batch_item_input = [], []

    count = 0
    while count < self.batch_size:
      u, i = random.choice(data)
      ii = self.item_ind2logit_ind[i]
      # assert(ii!=0)

      batch_user_input.append(u)
      batch_item_input.append(ii)
      count += 1


    return batch_user_input, batch_item_input








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
