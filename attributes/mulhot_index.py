
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def concat_versions(axis, value):
  if tf.__version__.startswith('0'):
    return tf.concat(axis, value)
  else:
    return tf.concat(value, axis)

def batch_slice(target, begin, size, l):
  b = tf.unstack(begin)
  s = tf.unstack(size)
  res = []
  for i in range(l):
    res.append(tf.slice(target, [b[i]], [s[i]]))
  return concat_versions(0, res)

def batch_segids(size, l):
  s = tf.unstack(size)
  res = []
  for i in range(l):
    ok = tf.tile([i], [s[i]])
    res.append(ok)
  return concat_versions(0, res)

def batch_slice_segids(target, begin, size, l):
  b = tf.unstack(begin)
  s = tf.unstack(size)
  res = []
  res2 = []
  for i in range(l):
    res.append(tf.slice(target, [b[i]], [s[i]]))
    res2.append(tf.tile([i], [s[i]]))
  return concat_versions(0, res), concat_versions(0, res2)

def batch_slice20(target, b, s, l):
  res1, res2 = [], []
  h = int(l/2)
  assert(l/2 == h)
  for i in range(h):
    res1.append(tf.slice(target, [b[i]], [s[i]]))
    res2.append(tf.slice(target, [b[i+h]], [s[i+h]]))
  return concat_versions(0, res1+res2)    

def batch_slice2(target, b, s, l):
  res = []
  for i in range(l):
    res.append(tf.slice(target, [b[i]], [s[i]]))
  return concat_versions(0, res)    

def batch_segids20(s, l):
  res1, res2 = [], []
  h = int(l/2)
  for i in range(h):
    res1.append(tf.tile([i], [s[i]]))
    res2.append(tf.tile([i+h], [s[i+h]]))
  return concat_versions(0, res1 + res2)    

def batch_segids2(s, l):
  res = []
  for i in range(l):
    ok = tf.tile([i], [s[i]])
    res.append(ok)
  return concat_versions(0, res)    

