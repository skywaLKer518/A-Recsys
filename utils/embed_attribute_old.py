
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    self.mask = {}
    self.zero_logits = {}
    self.pos_indices = {}
    self.l_true = {}
    self.l_false = {}

    # user embeddings
    self.user_embs_cat, self.user_embs_mulhot = self._embedded(user_attributes, 
      prefix='user')
    #item embeddings
    self.item_embs_cat, self.item_embs_mulhot = self._embedded(item_attributes, 
      prefix='item', transpose=False)
    self.i_biases_cat, self.i_biases_mulhot = self._embedded_bias(
      item_attributes, 'item')
    if item_output:
      self.item_embs2_cat, self.item_embs2_mulhot = self.embedded(
        item_attributes, prefix='item_output', transpose=False)
      self.i_biases2_cat, self.i_biases2_mulhot = self._embedded_bias(
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
    ''' TODO: 
    ** change to variables, sampled every now and then 
    ** ??use get_prediction to feed forward/backward
    '''
    print("construct mini-batch item candicate pool")
    if self.n_sampled is not None:
      self.i_mappings['sampled_pass'] = self._placeholders('item', 'sampled', 
        self.n_sampled)

    # input items (for lstm etc)
    print("construct input item")
    for step in xrange(input_steps):
      name_ = 'input{}'.format(step)
      self.i_mappings[name_] = self._placeholders('item', name_, mb)

    # item for prediction
    ''' full version'''
    ia = item_attributes
    print("construct full prediction layer")
    indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = [],[],[],[]
    for i in xrange(ia.num_features_cat):
      indices_cat.append(tf.constant(ia.full_cat_tr[i]))
    for i in xrange(ia.num_features_mulhot):
      indices_mulhot.append(tf.constant(ia.full_values_tr[i]))
      segids_mulhot.append(tf.constant(ia.full_segids_tr[i]))
      lengths_mulhot.append(tf.constant(ia.full_lengths_tr[i]))
    self.i_mappings['full'] = (indices_cat, indices_mulhot, segids_mulhot,
      lengths_mulhot)
    ''' sampled version '''
    print("sampled prediction layer")
    if self.n_sampled is not None:
      self.i_mappings['sampled'] = self._var_indices(self.n_sampled)
      self.update_sampled = self._pass_sampled_items()
    return

  def _var_indices(self, size, name='sampled', opt='item'):
    cat_indices, mulhot_indices, mulhot_segids, mulhot_lengths = [],[], [], []
    att = self.item_attributes
    init_int32 = tf.constant(0)
    for i in xrange(att.num_features_cat):
      cat_indices.append(tf.get_variable(dtype = tf.int32,
        name = "var{}_{}_cat_ind_{}".format(opt, name, i), trainable=False, 
        initializer=tf.zeros([size],dtype=tf.int32)))
    for i in xrange(att.num_features_mulhot):
      l1 = len(att.full_values_tr[i])
      mulhot_indices.append(tf.get_variable(dtype = tf.int32, trainable=False,
        initializer=tf.zeros([l1],dtype=tf.int32), 
        name = "var{}_{}_mulhot_ind_{}".format(opt, name, i)))
      l2 = len(att.full_segids_tr[i])
      assert(l1==l2), 'length of indices/segids should be the same %d/%d'%(l1,l2)
      mulhot_segids.append(tf.get_variable(dtype = tf.int32, trainable=False,
        initializer=tf.zeros([l2],dtype=tf.int32), 
        name = "var{}_{}_mulhot_seg_{}".format(opt, name, i)))
      mulhot_lengths.append(tf.get_variable(dtype =tf.float32, shape= [size, 1], 
        name = "var{}_{}_mulhot_len_{}".format(opt, name, i), trainable=False))
    return (cat_indices, mulhot_indices, mulhot_segids, mulhot_lengths)

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

  def get_prediction(self, latent, pool='full'):
    # compute inner product between item_hidden and {user_feature_embedding}
    # then lookup to compute logits    
    full_out_layer = self.i_mappings[pool]
    indices_cat, indices_mulhot, segids_mulhot, lengths_mulhot = full_out_layer
    innerps = []
    for i in xrange(self.item_attributes.num_features_cat):
      item_emb_cat = self.item_embs2_cat[i] if self.item_output else self.item_embs_cat[i]
      i_biases_cat = self.i_biases2_cat[i] if self.item_output else self.i_biases_cat[i]
      u = latent[i] if isinstance(latent, list) else latent
      inds = indices_cat[i]
      innerp = tf.matmul(item_emb_cat, tf.transpose(u)) + i_biases_cat # Vf by mb
      innerps.append(lookup(innerp, inds)) # V by mb
    offset = self.item_attributes.num_features_cat
    for i in xrange(self.item_attributes.num_features_mulhot):
      item_embs_mulhot = self.item_embs2_mulhot[i] if self.item_output else self.item_embs_mulhot[i]
      item_biases_mulhot = self.i_biases2_mulhot[i] if self.item_output else self.i_biases_mulhot[i]
      u = latent[i+offset] if isinstance(latent, list) else latent
      lengs = lengths_mulhot[i]
      if pool == 'full':
        inds = indices_mulhot[i]
        segids = segids_mulhot[i]
        V = self.logit_size
      else:
        inds = tf.slice(indices_mulhot[i], [0], [self.sampled_mulhot_l[i]])
        segids = tf.slice(segids_mulhot[i], [0], [self.sampled_mulhot_l[i]])
        V = self.n_sampled
      innerp = tf.add(tf.matmul(item_embs_mulhot, tf.transpose(u)), 
        item_biases_mulhot) 
      innerps.append(tf.div(tf.unsorted_segment_sum(lookup(innerp, 
        inds), segids, V), lengs))

    logits = tf.transpose(tf.reduce_mean(innerps, 0))
    return logits

  def _embedded(self, attributes, prefix='', transpose=False):
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

  def _embedded_bias(self, attributes, prefix):
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
    if loss == 'ce':
      return tf.nn.sparse_softmax_cross_entropy_with_logits(logits, item_target)
    elif loss == 'warp':
      return self._compute_warp_loss(logits, item_target)
    elif loss == 'mw':
      return self._compute_mw_loss(logits, item_target)  
    elif loss == 'bpr':
      return tf.log(1 + tf.exp(logits))
    elif loss == 'bpr-hinge':
      return tf.maximum(1 + logits, 0)
    else:
      print('Error: not implemented other loss!!')
      exit(1)

  def _compute_warp_loss(self, logits, item_target):
    loss = 'warp'
    if loss not in self.mask:
      self._prepare_warp_vars(loss)
    V = self.logit_size
    mb = self.batch_size
    flat_matrix = tf.reshape(logits, [-1])
    idx_flattened = self.idx_flattened0 + item_target
    logits_ = tf.gather(flat_matrix, idx_flattened)
    logits_ = tf.reshape(logits_, [mb, 1])
    logits2 = tf.sub(logits, logits_) + 1
    mask2 = tf.reshape(self.mask[loss], [mb, V])
    target = tf.select(mask2, logits2, self.zero_logits[loss])
    return tf.log(1 + tf.reduce_sum(tf.nn.relu(target), 1))

  def _compute_mw_loss(self, logits, item_target):
    if 'mw' not in self.mask:
      self._prepare_warp_vars('mw')
    V = self.n_sampled
    mb = self.batch_size
    logits2 = tf.sub(logits, tf.reshape(item_target, [mb, 1])) + 1
    mask2 = tf.reshape(self.mask['mw'], [mb, V])
    target = tf.select(mask2, logits2, self.zero_logits['mw'])
    return tf.log(1 + tf.reduce_sum(tf.nn.relu(target), 1)) # scale or not??

  def _prepare_warp_vars(self, loss= 'warp'):
    V = self.logit_size if loss == 'warp' else self.n_sampled
    mb = self.batch_size
    self.idx_flattened0 = tf.range(0, mb) * V
    self.mask[loss] = tf.Variable([True] * V * mb, dtype=tf.bool, 
      trainable=False)
    self.zero_logits[loss] = tf.constant([[0.0] * V] * mb)
    self.pos_indices[loss] = tf.placeholder(tf.int32, shape = [None])
    self.l_true[loss] = tf.placeholder(tf.bool, shape = [None], name='l_true')
    self.l_false[loss] = tf.placeholder(tf.bool, shape = [None], name='l_false')
    
  def _pass_sampled_items(self):
    self.updated_indices = []
    self.sampled_mulhot_l = []
    self.sampled_mulhot_l_pass = []
    res = []
    var_s = self.i_mappings['sampled']

    att = self.item_attributes
    for i in xrange(att.num_features_cat):
      vals = self.i_mappings['sampled_pass'][0][i]
      res.append(tf.assign(var_s[0][i], vals))
    for i in xrange(att.num_features_mulhot):
      indices = tf.placeholder(tf.int32, shape=[None])
      self.updated_indices.append(indices)

      vals = self.i_mappings['sampled_pass'][1][i]
      res.append(tf.scatter_update(var_s[1][i], indices, vals))
      segs = self.i_mappings['sampled_pass'][2][i]
      res.append(tf.scatter_update(var_s[2][i], indices, segs))
      lengs = self.i_mappings['sampled_pass'][3][i]
      res.append(tf.assign(var_s[3][i], lengs))
      
      l = tf.get_variable(name='sampled_l_mulhot_{}'.format(i), dtype=tf.int32, 
        initializer=tf.constant(0), trainable=False)      
      self.sampled_mulhot_l.append(l)
      l_pass = tf.placeholder(tf.int32, shape=[])
      self.sampled_mulhot_l_pass.append(l_pass)
      res.append(tf.assign(l, l_pass))
    return res

  def get_warp_mask(self):
    self.set_mask, self.reset_mask = {}, {}
    for loss in ['mw', 'warp']:
      if loss not in self.mask:
        continue
      self.set_mask[loss] = tf.scatter_update(self.mask[loss], 
        self.pos_indices[loss], self.l_false[loss])
      self.reset_mask[loss] = tf.scatter_update(self.mask[loss], 
        self.pos_indices[loss], self.l_true[loss])
    return self.set_mask, self.reset_mask

  def prepare_warp(self, pos_item_set, pos_item_set_eval):
    self.pos_item_set = pos_item_set
    self.pos_item_set_eval = pos_item_set_eval
    return 

  def target_mapping(self, item_target, loss = 'ce'):
    ''' TODO: does not work for sampled loss '''
    m = self.item_ind2logit_ind
    target = []
    for items in item_target:
      target.append([m[v] for v in items])
    return target

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
    
    l_mulhot = []    
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
      l_mulhot.append(len(vals))
    return l_mulhot

  def add_input(self, input_feed, user_input, item_input, 
        neg_item_input=None, item_sampled = None, item_sampled_id2idx = None, 
        forward_only=False, recommend=False, loss=None):
    
    # users
    if self.user_attributes is not None:
      self._add_input(input_feed, 'user', user_input, 'input')
    # pos neg: when input_steps = 0 
    if self.item_attributes is not None and recommend is False and self.input_steps == 0:
      self._add_input(input_feed, 'item', item_input, 'pos')
      # self._add_input(input_feed, 'item', neg_item_input, 'neg')    

    # input item: for lstm
    if self.item_attributes is not None and self.input_steps > 0:
      for step in range(len(item_input)):
        self._add_input(input_feed, 'item', item_input[step], 
          'input{}'.format(step))

    # sampled item: when sampled-loss is used
    input_feed_sampled = {}
    update_sampled = []
    if self.item_attributes is not None and recommend is False and item_sampled is not None and loss in ['mw', 'mce']:      
      l_mulhot = self._add_input(input_feed_sampled, 'item', item_sampled, 
        'sampled_pass')
      for i in range(self.item_attributes.num_features_mulhot):
        input_feed_sampled[self.updated_indices[i].name] = range(l_mulhot[i])
        input_feed_sampled[self.sampled_mulhot_l_pass[i].name] = l_mulhot[i]
      update_sampled = self.update_sampled

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
      input_feed_warp[self.pos_indices[loss].name] = mask_indices
      input_feed_warp[self.l_false[loss].name] = [False] * L
      input_feed_warp[self.l_true[loss].name] = [True] * L

    return update_sampled, input_feed_sampled, input_feed_warp



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

