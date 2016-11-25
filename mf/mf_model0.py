
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
               item_attributes=None, num_samples=512):

    self.user_size = user_size
    self.item_size = item_size

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

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    loss_function = None
    
    # item embedding

    # indices_cat, indices_mulhot = create_indices()
    # embedded_item = ?? # no embedded item


    with ops.device("/cpu:0"):
      size_item = size
      embedded_item = tf.get_variable("proj_w", [size_item, self.item_size])
#       else:
#         size_item = (sum(item_attributes._embedding_size_list_cat) + 
#           sum(item_attributes._embedding_size_list_mulhot))
#         embedded_item = self.EmbeddingLayer2(self.item_input)
      embedded_item_t = tf.transpose(embedded_item)
      item_b = tf.get_variable("proj_b", [self.item_size]) # item bias
      output_projection = (embedded_item, item_b)

    ''' actually the loss is good to go (for no feature cases '''
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.item_size:
      def sampled_loss(inputs, labels):
        labels = tf.reshape(labels, [-1, 1])
        return tf.nn.sampled_softmax_loss(embedded_item_t, item_b, inputs, 
          labels, num_samples, self.item_size)
      loss_function = sampled_loss

    # user embedding
    if user_attributes is None:
      embedded_user = self.EmbeddingLayer(self.user_input, size)
      user_model_size = size
    else:
      embedded_user = self.EmbeddingLayer2(self.user_input, batch_size, size, 
        user_attributes)
      user_model_size = (sum(user_attributes._embedding_size_list_cat) + 
          sum(user_attributes._embedding_size_list_mulhot))
      print(user_model_size, size)
      w = tf.get_variable("transform", [user_model_size, size])
      embedded_user = tf.matmul(embedded_user, w)

    self.loss = loss_function(embedded_user, self.item_input)
    self.loss = tf.reduce_mean(self.loss)
    self.output = tf.matmul(embedded_user, embedded_item) + item_b

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    
    opt = tf.train.GradientDescentOptimizer(self.learning_rate)
    gradients = tf.gradients(self.loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, 
      max_gradient_norm)
    
    self.gradient_norms = norm
    self.updates = opt.apply_gradients(
      zip(clipped_gradients, params), global_step=self.global_step)

    self.saver = tf.train.Saver(tf.all_variables())

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

  def EmbeddingLayer2(self, input_user, mb, size, attributes, scope = None):
    with vs.variable_scope(scope or type(self).__name__):
      with ops.device("/cpu:0"):
        sqrt3 = math.sqrt(3)
        initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

        feats_cat, feats_mulhot, mulhot_starts, mulhot_lengths = [], [], [], []
        for i in xrange(attributes.num_features_cat):
          init = tf.constant_initializer(value = attributes.features_cat[i])
          feats_cat.append(tf.get_variable(name="map1_{0}".format(i), 
            shape=attributes.features_cat[i].shape, dtype=tf.int32, 
            initializer=init, trainable=False))

        for i in xrange(attributes.num_features_mulhot):
          init = tf.constant_initializer(value = attributes.features_mulhot[i])
          feats_mulhot.append(tf.get_variable(name="map2_{0}".format(i), 
            shape=attributes.features_mulhot[i].shape, dtype=tf.int32, 
            initializer=init, trainable=False))
          
          init_s = tf.constant_initializer(value = attributes.mulhot_starts[i])
          mulhot_starts.append(tf.get_variable(name="map2_starts{0}".format(i), 
            shape=attributes.mulhot_starts[i].shape, dtype=tf.int32, 
            initializer=init_s, trainable=False))

          init_l = tf.constant_initializer(value = attributes.mulhot_lengths[i])
          mulhot_lengths.append(tf.get_variable(name="map2_lengs{0}".format(i), 
            shape=attributes.mulhot_lengths[i].shape, dtype=tf.int32, 
            initializer=init_l, trainable=False))

        embedding_list1, embedding_list2 = [], []
        for i in xrange(attributes.num_features_cat):
          d = attributes._embedding_size_list_cat[i]
          V = attributes._embedding_classes_list_cat[i]
          embedding = tf.get_variable(name="embed_cat_{0}".format(i), shape=[V,d], dtype=tf.float32)
          embedding_list1.append(embedding)
        for i in xrange(attributes.num_features_mulhot):
          d = attributes._embedding_size_list_mulhot[i]
          V = attributes._embedding_classes_list_mulhot[i]
          embedding = tf.get_variable(name="embed_mulhot_{0}".format(i), shape=[V,d], dtype=tf.float32)
          embedding_list2.append(embedding)

        # lookup tables
        embedded_list = []
        x = array_ops.reshape(input_user, [-1])
        
        for i in xrange(attributes.num_features_cat):
          idx = embedding_ops.embedding_lookup(feats_cat[i], x)
          embedded = embedding_ops.embedding_lookup(embedding_list1[i], 
            array_ops.reshape(idx, [-1]))
          embedded_list.append(embedded)          
        
        for i in xrange(attributes.num_features_mulhot):
          mtl = attributes.mulhot_max_length[i]
          start = embedding_ops.embedding_lookup(mulhot_starts[i], x) # mb by 1
          leng = embedding_ops.embedding_lookup(mulhot_lengths[i], x) # mb by 1
          st = create_sparse_map(feats_mulhot[i], start, leng, mb, mtl)
          embedded = embedding_ops.embedding_lookup_sparse(embedding_list2[i], 
            st, sp_weights=None)
          embedded_list.append(embedded)
        embedded = tf.concat(1, embedded_list)

    return embedded

  def step(self, session, user_input, item_input, forward_only=False, 
    recommend=False):
    input_feed = {}
    input_feed[self.user_input.name] = user_input
    input_feed[self.item_input.name] = item_input
    if not recommend:
      if not forward_only:
        output_feed = [self.updates, self.loss]
      else:
        output_feed = [self.loss]
    else:
      values, indices= tf.nn.top_k(self.output, 60, sorted=True)
      self.output = indices
      output_feed = [self.loss, self.output]
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

    for _ in xrange(self.batch_size):
      u, i = random.choice(data)
      batch_user_input.append(u)
      batch_item_input.append(i)

    return batch_user_input, batch_item_input








