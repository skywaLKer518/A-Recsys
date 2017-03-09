
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
from linear_seq import LinearSeq
sys.path.insert(0, '../attributes')
import embed_attribute


class Model(LinearSeq):
  def __init__(self, user_size, item_size, size,
               batch_size, learning_rate,
               learning_rate_decay_factor, 
               user_attributes=None, 
               item_attributes=None, 
               item_ind2logit_ind=None, 
               logit_ind2item_ind=None, 
               n_input_items=0, 
               loss_function='ce',
               logit_size_test=None, 
               dropout=1.0, 
               top_N_items=100,
               use_sep_item=True,
               n_sampled=None, 
               output_feat=1,
               indices_item=None, 
               dtype=tf.float32):

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

    self.loss_function = loss_function
    self.n_input_items = n_input_items
    self.n_sampled = n_sampled
    self.batch_size = batch_size
    
    self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    
    self.att_emb = None
    self.dtype=dtype

    mb = self.batch_size
    ''' this is mapped item target '''
    self.item_target = tf.placeholder(tf.int32, shape = [mb], name = "item")
    self.item_id_target = tf.placeholder(tf.int32, shape = [mb], name = "item_id")

    self.dropout = dropout
    self.keep_prob = tf.constant(dropout, dtype=dtype)
    # tf.placeholder(tf.float32, name='keep_prob')

    n_input = max(n_input_items, 1)
    m = embed_attribute.EmbeddingAttribute(user_attributes, item_attributes, mb, 
      self.n_sampled, n_input, use_sep_item, item_ind2logit_ind, logit_ind2item_ind)
    self.att_emb = m

    embedded_user, _ = m.get_batch_user(1.0, False)
    embedded_items = []
    for i in range(n_input):
      embedded_item, _ = m.get_batch_item('input{}'.format(i), batch_size)
      embedded_item = tf.reduce_mean(embedded_item, 0)
      embedded_items.append(embedded_item)

    print("non-sampled prediction")
    input_embed = tf.reduce_mean([embedded_user, embedded_items[0]], 0)
    input_embed = tf.nn.dropout(input_embed, self.keep_prob)
    logits = m.get_prediction(input_embed, output_feat=output_feat)

    if self.n_input_items == 0:
      input_embed_test= embedded_user
    else:
      # including two cases: 1, n items. 2, end_line item
      # input_embed_test = [embedded_user] + embedded_items
      # input_embed_test = tf.reduce_mean(input_embed_test, 0)

      input_embed_test = [embedded_user] + [tf.reduce_mean(embedded_items, 0)]
      input_embed_test = tf.reduce_mean(input_embed_test, 0)      
    logits_test = m.get_prediction(input_embed_test, output_feat=output_feat)

    # mini batch version
    print("sampled prediction")
    if self.n_sampled is not None:
      sampled_logits = m.get_prediction(input_embed, 'sampled', output_feat=output_feat)
      # embedded_item, item_b = m.get_sampled_item(self.n_sampled)
      # sampled_logits = tf.matmul(embedded_user, tf.transpose(embedded_item)) + item_b
      target_score = m.get_target_score(input_embed, self.item_id_target)

    loss = self.loss_function
    if loss in ['warp', 'ce', 'bbpr']:
      batch_loss = m.compute_loss(logits, self.item_target, loss)
      batch_loss_test = m.compute_loss(logits_test, self.item_target, loss)
    elif loss in ['mw']:
      batch_loss = m.compute_loss(sampled_logits, target_score, loss)
      batch_loss_eval = m.compute_loss(logits, self.item_target, 'warp')
    else:
      print("not implemented!")
      exit(-1)
    if loss in ['warp', 'mw', 'bbpr']:
      self.set_mask, self.reset_mask = m.get_warp_mask()

    self.loss = tf.reduce_mean(batch_loss)
    # self.loss_eval = tf.reduce_mean(batch_loss_eval) if loss == 'mw' else self.loss
    self.loss_test = tf.reduce_mean(batch_loss_test)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    opt = tf.train.AdagradOptimizer(self.learning_rate)
    # opt = tf.train.AdamOptimizer(self.learning_rate)
    gradients = tf.gradients(self.loss, params)
    self.updates = opt.apply_gradients(
      zip(gradients, params), global_step=self.global_step)

    self.output = logits_test
    values, self.indices= tf.nn.top_k(self.output, self.top_N_items, sorted=True)
    self.saver = tf.train.Saver(tf.all_variables())
