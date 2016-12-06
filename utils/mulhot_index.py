
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def batch_slice(target, begin, size, l):
  b = tf.unpack(begin)
  s = tf.unpack(size)
  res = []
  for i in range(l):
    res.append(tf.slice(target, [b[i]], [s[i]]))
  return tf.concat(0, res)

def batch_segids(size, l):
  s = tf.unpack(size)
  res = []
  for i in range(l):
    ok = tf.tile([i], [s[i]])
    res.append(ok)
  return tf.concat(0, res)

def batch_slice_segids(target, begin, size, l):
  b = tf.unpack(begin)
  s = tf.unpack(size)
  res = []
  res2 = []
  for i in range(l):
    res.append(tf.slice(target, [b[i]], [s[i]]))
    res2.append(tf.tile([i], [s[i]]))
  return tf.concat(0, res), tf.concat(0, res2)

def batch_slice2(target, b, s, l):
  res = []
  for i in range(l):
    res.append(tf.slice(target, [b[i]], [s[i]]))
  return tf.concat(0, res)    

def batch_segids2(s, l):
  res = []
  for i in range(l):
    ok = tf.tile([i], [s[i]])
    res.append(ok)
  return tf.concat(0, res)    

