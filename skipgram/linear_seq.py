
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import sys
sys.path.insert(0, '../attributes')
import embed_attribute

class LinearSeq(object):
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
               use_sep_item=True,
               n_sampled=None, 
               output_feat=1,
               indices_item=None, 
               dtype=tf.float32):

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

  def prepare_warp(self, pos_item_set, pos_item_set_eval):
    self.att_emb.prepare_warp(pos_item_set, pos_item_set_eval)
    return 

  def step(self, session, user_input, item_input=None, item_output=None, 
    item_sampled = None, item_sampled_id2idx = None,
    forward_only=False, recommend=False, recommend_new = False, loss=None, 
    run_op=None, run_meta=None):
    input_feed = {}
        
    if recommend == False:
      targets = self.att_emb.target_mapping([item_output])
      input_feed[self.item_target.name] = targets[0]
      if loss in ['mw']:
        input_feed[self.item_id_target.name] = item_output

    if self.att_emb is not None:
      (update_sampled, input_feed_sampled, 
        input_feed_warp) = self.att_emb.add_input(input_feed, user_input, 
        item_input, neg_item_input=None, item_sampled = item_sampled, 
        item_sampled_id2idx = item_sampled_id2idx, 
        forward_only=forward_only, recommend=recommend, loss=loss)

    if not recommend:
      if not forward_only:
        output_feed = [self.updates, self.loss]
      else:
        output_feed = [self.loss_test]
    else:
      if recommend_new:
        output_feed = [self.indices_test]
      else:
        output_feed = [self.indices]

    if item_sampled is not None and loss in ['mw', 'mce']:
      session.run(update_sampled, input_feed_sampled)

    if (loss in ['warp', 'bbpr', 'mw']) and recommend is False:
      session.run(self.set_mask[loss], input_feed_warp)

    if run_op is not None and run_meta is not None:
      outputs = session.run(output_feed, input_feed, options=run_op, run_metadata=run_meta)
    else:
      outputs = session.run(output_feed, input_feed)

    if (loss in ['warp', 'bbpr', 'mw']) and recommend is False:
      session.run(self.reset_mask[loss], input_feed_warp)

    if not recommend:
      if not forward_only:
        return outputs[1]
      else:
        return outputs[0]
    else:
      return outputs[0]

