from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope

import data_iterator
import env

class SeqModel(object):
    
    def __init__(self,
                 buckets,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 learning_rate_decay_factor,
                 embeddingAttribute,
                 withAdagrad = True,
                 num_samples=512,
                 forward_only=False,
                 dropoutRate = 1.0,
                 START_ID = 0,
                 loss = "ce",
                 devices = "",
                 run_options = None,
                 run_metadata = None,
                 use_concat = True,
                 dtype=tf.float32):
        """Create the model.
        
        Args:
        buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
        size: number of units in each layer of the model.
        num_layers: number of layers in the model.
        max_gradient_norm: gradients will be clipped to maximally this norm.
        batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.

        learning_rate: learning rate to start with.
        learning_rate_decay_factor: decay learning rate by this much when needed.

        num_samples: number of samples for sampled softmax.
        forward_only: if set, we do not construct the backward pass in the model.
        dtype: the data type to use to store internal variables.
        """
        self.embeddingAttribute = embeddingAttribute
        self.buckets = buckets
        self.START_ID = START_ID
        self.PAD_ID = START_ID
        self.USER_PAD_ID = 0
        self.batch_size = batch_size
        self.loss = loss
        self.devices = devices
        self.run_options = run_options
        self.run_metadata = run_metadata

        with tf.device(devices[0]):
            self.dropoutRate = tf.Variable(
                float(dropoutRate), trainable=False, dtype=dtype)        
            self.dropoutAssign_op = self.dropoutRate.assign(dropoutRate)
            self.dropout10_op = self.dropoutRate.assign(1.0)
            self.learning_rate = tf.Variable(
                float(learning_rate), trainable=False, dtype=dtype)
            self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)
            self.global_step = tf.Variable(0, trainable=False)

        with tf.device(devices[1]):
            single_cell = tf.nn.rnn_cell.LSTMCell(size, state_is_tuple=True)
            single_cell = rnn_cell.DropoutWrapper(single_cell,input_keep_prob = self.dropoutRate)
            if num_layers > 1:
                single_cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * num_layers, state_is_tuple=True)
            single_cell = rnn_cell.DropoutWrapper(single_cell, output_keep_prob = self.dropoutRate)
        
        
        # Feeds for inputs.
        with tf.device(devices[2]):
            self.targets = []
            self.target_ids = []
            self.target_weights = []

            # target: 1  2  3  4 
            # inputs: go 1  2  3
            # weights:1  1  1  1

            for i in xrange(buckets[-1]):
                self.targets.append(tf.placeholder(tf.int32, 
                    shape=[self.batch_size], name = "target{}".format(i)))
                self.target_ids.append(tf.placeholder(tf.int32, 
                    shape=[self.batch_size], name = "target_id{}".format(i)))
                self.target_weights.append(tf.placeholder(dtype, 
                    shape = [self.batch_size], name="target_weight{}".format(i)))

        with tf.device(devices[0]):

            self.inputs = []

            if use_concat:
                user_embed, _ = self.embeddingAttribute.get_batch_user(1.0,concat = True, no_id = True)
                user_embed_size = self.embeddingAttribute.get_user_model_size(
                    no_id = True, concat = True)
                item_embed_size = self.embeddingAttribute.get_item_model_size(
                    concat=True)
                w_input_user = tf.get_variable("w_input_user",[user_embed_size, size], dtype = dtype)
                w_input_item = tf.get_variable("w_input_item",[item_embed_size, size], dtype = dtype)
                user_embed_transform = tf.matmul(user_embed, w_input_user)

                for i in xrange(buckets[-1]):
                    name = "input{}".format(i)
                    item_embed, _ = self.embeddingAttribute.get_batch_item(name,
                        self.batch_size, concat = True)
                    item_embed_transform = tf.matmul(item_embed, w_input_item)
                    input_embed = user_embed_transform + item_embed_transform
                    self.inputs.append(input_embed)
            else:
                user_embed, _ = self.embeddingAttribute.get_batch_user(1.0,concat = False, no_id = True)
                
                for i in xrange(buckets[-1]):
                    name = "input{}".format(i)
                    item_embed, _ = self.embeddingAttribute.get_batch_item(name,
                        self.batch_size, concat = False)
                    item_embed = tf.reduce_mean(item_embed, 0)
                    input_embed = tf.reduce_mean([user_embed, item_embed], 0)
                    self.inputs.append(input_embed)
        
        self.outputs, self.losses, self.outputs_full, self.losses_full, self.topk_values, self.topk_indexes = self.model_with_buckets(self.inputs,self.targets, self.target_weights, self.buckets, single_cell,self.embeddingAttribute, dtype, devices = devices)

        # for warp
        if self.loss in ["warp", "mw"]:
            self.set_mask, self.reset_mask = self.embeddingAttribute.get_warp_mask(
                device = self.devices[2])

        #with tf.device(devices[0]):
        # train
        with tf.device(devices[0]):
            params = tf.trainable_variables()
            if not forward_only:
                self.gradient_norms = []
                self.updates = []
                self.gradient_norms = []
                self.updates = []
                if withAdagrad:
                    opt = tf.train.AdagradOptimizer(self.learning_rate)
                else:
                    opt = tf.train.GradientDescentOptimizer(self.learning_rate)

                for b in xrange(len(buckets)):
                    gradients = tf.gradients(self.losses[b], params, colocate_gradients_with_ops=True)
                    clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
                    self.gradient_norms.append(norm)
                    self.updates.append(opt.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step))

        self.saver = tf.train.Saver(tf.all_variables())

    def step(self,session, user_input, item_inputs, targets, target_weights, 
        bucket_id, item_sampled=None, item_sampled_id2idx = None, forward_only = False, recommend = False):

        length = self.buckets[bucket_id]

        targets_mapped = self.embeddingAttribute.target_mapping(targets)
        input_feed = {}
        for l in xrange(length):
            input_feed[self.targets[l].name] = targets_mapped[l]
            input_feed[self.target_weights[l].name] = target_weights[l]
            if self.loss in ['mw', 'ce']:
                input_feed[self.target_ids[l].name] = targets[l]

        #print(input_feed)
        (update_sampled, input_feed_sampled, input_feed_warp) = self.embeddingAttribute.add_input(input_feed, user_input, item_inputs, forward_only = forward_only, recommend = recommend, loss = self.loss, item_sampled_id2idx=item_sampled_id2idx)
        if self.loss in ["warp", "mw"]:
            session.run(self.set_mask[self.loss], input_feed_warp)
        
        if item_sampled is not None and self.loss in ['mw', 'mce']:
            session.run(update_sampled, input_feed_sampled)

        # output_feed
        if forward_only:
            output_feed = [self.losses_full[bucket_id]]
        else:
            output_feed = [self.losses[bucket_id]]
            output_feed += [self.updates[bucket_id], self.gradient_norms[bucket_id]]

        if self.loss in ["warp", "mw"]:
            session.run(self.set_mask[self.loss], input_feed_warp)

        outputs = session.run(output_feed, input_feed, options = self.run_options, run_metadata = self.run_metadata)

        if self.loss in ["warp", "mw"]:
            session.run(self.reset_mask[self.loss], input_feed_warp)

        return outputs[0]
    
    def step_recommend(self,session, user_input, item_inputs, positions, bucket_id):
        length = self.buckets[bucket_id]

        input_feed = {}

        (update_sampled, input_feed_sampled, input_feed_warp) = self.embeddingAttribute.add_input(input_feed, user_input, item_inputs, forward_only = True, recommend = True, loss = self.loss)

        # output_feed
        output_feed = {}
        for pos in positions:
            if not pos in output_feed:
                output_feed[pos] = (self.topk_values[bucket_id][pos], self.topk_indexes[bucket_id][pos])

        outputs = session.run(output_feed, input_feed, options = self.run_options, run_metadata = self.run_metadata)
        
        # results = [(uid, [value], [index])]
        results = []
        for i, pos in enumerate(positions):
            uid = user_input[i]
            values = outputs[pos][0][i,:]
            indexes = outputs[pos][1][i,:]
            results.append((uid,values,indexes))

        return results


    def get_batch(self, data_set, bucket_id, start_id = None):
        length = self.buckets[bucket_id]

        users, item_inputs,item_outputs, weights = [], [], [], []

        for i in xrange(self.batch_size):
            if start_id == None:
                user, item_seq = random.choice(data_set[bucket_id])
            else:
                if start_id + i < len(data_set[bucket_id]):
                    user, item_seq = data_set[bucket_id][start_id + i]
                else:
                    user = self.USER_PAD_ID
                    item_seq = []                    
            
            pad_seq = [self.PAD_ID] * (length - len(item_seq))
            if len(item_seq) == 0:
                item_input_seq = [self.START_ID] + pad_seq[1:]
            else:
                item_input_seq = [self.START_ID] + item_seq[:-1] + pad_seq
            item_output_seq = item_seq + pad_seq
            target_weight = [1.0] * len(item_seq) + [0.0] * len(pad_seq)

            users.append(user)
            item_inputs.append(item_input_seq)
            item_outputs.append(item_output_seq)
            weights.append(target_weight)
            
        # Now we create batch-major vectors from the data selected above.
        def batch_major(l):
            output = []
            for i in xrange(len(l[0])):
                temp = []
                for j in xrange(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output
            
        batch_user = users
        batch_item_inputs = batch_major(item_inputs)
        batch_item_outputs = batch_major(item_outputs)
        batch_weights = batch_major(weights)
        
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True


        return batch_user, batch_item_inputs, batch_item_outputs, batch_weights, finished
        

    def get_batch_recommend(self, data_set, bucket_id, start_id = None):
        length = self.buckets[bucket_id]

        users, item_inputs, positions, valids = [], [], [], []

        for i in xrange(self.batch_size):
            if start_id == None:
                user, item_seq = random.choice(data_set[bucket_id])
                valid = 1
                position = len(item_seq) - 1
            else:
                if start_id + i < len(data_set[bucket_id]):
                    user, item_seq = data_set[bucket_id][start_id + i]
                    valid = 1
                    position = len(item_seq) - 1
                else:
                    user = self.USER_PAD_ID
                    item_seq = []                    
                    valid = 0
                    position = 0
            
            pad_seq = [self.PAD_ID] * (length - len(item_seq))
            item_input_seq = item_seq + pad_seq
            valids.append(valid)
            users.append(user)
            positions.append(position)
            item_inputs.append(item_input_seq)
            
        # Now we create batch-major vectors from the data selected above.
        def batch_major(l):
            output = []
            for i in xrange(len(l[0])):
                temp = []
                for j in xrange(self.batch_size):
                    temp.append(l[j][i])
                output.append(temp)
            return output
            
        batch_item_inputs = batch_major(item_inputs)
        
        finished = False
        if start_id != None and start_id + self.batch_size >= len(data_set[bucket_id]):
            finished = True

        return users, batch_item_inputs, positions, valids, finished

        
    def model_with_buckets(self, inputs, targets, weights,
                           buckets, cell, embeddingAttribute, dtype,
                           per_example_loss=False, name=None, devices = None):

        all_inputs = inputs + targets + weights
        losses = []
        losses_full = []
        outputs = []
        outputs_full = []
        topk_values = []
        topk_indexes = []
        softmax_loss_function = lambda x,y: self.embeddingAttribute.compute_loss(x ,y, loss=self.loss, device = devices[2])

        with tf.device(devices[1]):
            init_state = cell.zero_state(self.batch_size, dtype)
        
        with ops.op_scope(all_inputs, name, "model_with_buckets"):
            for j, bucket in enumerate(buckets):
                with variable_scope.variable_scope(variable_scope.get_variable_scope(),reuse=True if j > 0 else None):
                    
                    with tf.device(devices[1]):
                        bucket_outputs, _ = rnn.rnn(cell,inputs[:bucket],initial_state = init_state)
                    with tf.device(devices[2]):

                        bucket_outputs_full = [self.embeddingAttribute.get_prediction(x, device=devices[2]) for x in bucket_outputs]
                        
                        if self.loss in ['warp', 'ce']:
                            t = targets
                            bucket_outputs = [self.embeddingAttribute.get_prediction(x, device=devices[2]) for x in bucket_outputs]
                        elif self.loss in ['mw']:
                            # bucket_outputs0 = [self.embeddingAttribute.get_prediction(x, pool='sampled', device=devices[2]) for x in bucket_outputs]
                            t, bucket_outputs0 = [], []

                            for i in xrange(len(bucket_outputs)):
                                x = bucket_outputs[i]
                                ids = self.target_ids[i]
                                bucket_outputs0.append(self.embeddingAttribute.get_prediction(x, pool='sampled', device=devices[2]))
                                t.append(self.embeddingAttribute.get_target_score(x, ids, device=devices[2]))
                            bucket_outputs = bucket_outputs0

                        outputs.append(bucket_outputs)
                        outputs_full.append(bucket_outputs_full)

                        if per_example_loss:
                            losses.append(sequence_loss_by_example(
                                    outputs[-1], t[:bucket], weights[:bucket],
                                    softmax_loss_function=softmax_loss_function))
                            losses_full.append(sequence_loss_by_example(
                                    outputs_full[-1], t[:bucket], weights[:bucket],
                                    softmax_loss_function=softmax_loss_function))
                        else:
                            losses.append(sequence_loss(
                                    outputs[-1], t[:bucket], weights[:bucket],
                                    softmax_loss_function=softmax_loss_function))
                            losses_full.append(sequence_loss(
                                    outputs_full[-1], t[:bucket], weights[:bucket],softmax_loss_function=softmax_loss_function))
                        topk_value, topk_index = [], []
                        for full_logits in outputs_full[-1]:
                            value, index = tf.nn.top_k(full_logits, 30, sorted = True)
                            topk_value.append(value)
                            topk_index.append(index)
                        topk_values.append(topk_value)
                        topk_indexes.append(topk_index)
                  
        return outputs, losses, outputs_full, losses_full, topk_values, topk_indexes


def sequence_loss_by_example(logits, targets, weights,
                             average_across_timesteps=True,
                             softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, default: "sequence_loss_by_example".

  Returns:
    1D batch-sized float Tensor: The log-perplexity for each sequence.

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """
  if len(targets) != len(logits) or len(weights) != len(logits):
    raise ValueError("Lengths of logits, weights, and targets must be the same "
                     "%d, %d, %d." % (len(logits), len(weights), len(targets)))
  with ops.op_scope(logits + targets + weights,name, "sequence_loss_by_example"):
    log_perp_list = []
    for logit, target, weight in zip(logits, targets, weights):
      if softmax_loss_function is None:
        # TODO(irving,ebrevdo): This reshape is needed because
        # sequence_loss_by_example is called with scalars sometimes, which
        # violates our general scalar strictness policy.
        target = array_ops.reshape(target, [-1])
        crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
            logit, target)
      else:
        crossent = softmax_loss_function(logit, target)
      log_perp_list.append(crossent * weight)

    log_perps = math_ops.add_n(log_perp_list)
    if average_across_timesteps:
      total_size = math_ops.add_n(weights)
      total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
      log_perps /= total_size
  return log_perps


def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=False,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

  Args:
    logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
    targets: List of 1D batch-sized int32 Tensors of the same length as logits.
    weights: List of 1D batch-sized float-Tensors of the same length as logits.
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".

  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).

  Raises:
    ValueError: If len(logits) is different from len(targets) or len(weights).
  """

  with ops.op_scope(logits + targets + weights, name, "sequence_loss"):
    cost = math_ops.reduce_sum(sequence_loss_by_example(
        logits, targets, weights,
        average_across_timesteps=average_across_timesteps,
        softmax_loss_function=softmax_loss_function))
    if average_across_batch:
        total_size = tf.reduce_sum(tf.sign(weights[0]))
        return cost / math_ops.cast(total_size, cost.dtype)
    else:
      return cost
