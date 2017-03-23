import numpy as np
from os import listdir, mkdir, path, rename
from os.path import isfile, join
from tensorflow.python.platform import gfile


_UNK = "_UNK"
_START = "_START"
# _PHANTOM = "_PHANTOM"

UNK_ID = 0
START_ID = 1
# REST_ID = 1

_START_VOCAB = [_UNK, _START ] #, _PHANTOM]


def pickle_save(m, filename):
  import cPickle as pickle
  pickle.dump(m, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def create_dictionary(data_dir, inds, features, feature_types, feature_names, 
  max_vocabulary_size=50000, logits_size_tr = 50000, threshold = 2, 
  prefix='user'):
  filename = 'vocab0_%d' % max_vocabulary_size
  if isfile(join(data_dir, filename)):
    print("vocabulary exists!")
    return
  vocab_counts = {}
  num_uf = len(feature_names)
  assert(len(feature_types) == num_uf), 'length of feature_types should be the same length of feature_names {} vs {}'.format(len(feature_types), num_uf)
  for ind in range(num_uf):
    name = feature_names[ind]
    vocab_counts[name] = {}

  for u in inds: # u index
    uf = features[u, :]
    for ii in range(num_uf):
      name = feature_names[ii]
      if feature_types[ii] == 0:   
        vocab_counts[name][uf[ii]] = vocab_counts[name][uf[ii]] + 1 if uf[ii] in vocab_counts[name] else 1
      elif feature_types[ii] == 1:
        if not isinstance(uf[ii], list):
          if not isinstance(uf[ii], str):
            uf[ii] = str(uf[ii])
          uf[ii] = uf[ii].split(',')
        for t in uf[ii]:
          vocab_counts[name][t] = vocab_counts[name][t] + 1 if t in vocab_counts[name] else 1

  minimum_occurance = []
  for i in range(num_uf):
    name = feature_names[i]
    if feature_types[i] > 1:
      continue
    vocab_list = _START_VOCAB + sorted(vocab_counts[name], 
      key=vocab_counts[name].get, reverse=True)
    if prefix == 'item' and i == 0:
      max_size = logits_size_tr + len(_START_VOCAB)
    elif prefix == 'user' and i == 0:
      max_size = len(features)
    else:
      max_size = max_vocabulary_size

    # max_size += len(_START_VOCAB) 

    # if len(vocab_list) > max_size:
    #   vocab_list= vocab_list[:max_size]
    with gfile.GFile(join(data_dir, ("%s_vocab%d_%d"% (prefix, i,
      max_size))), mode="wb") as vocab_file:

      if prefix == 'user' and i == 0:
        vocab_list2 = [v for v in vocab_list if v in _START_VOCAB or
         vocab_counts[name][v] >= threshold]
      else:
        vocab_list2 = [v for v in vocab_list if v in _START_VOCAB or
         vocab_counts[name][v] >= threshold]
      if len(vocab_list2) > max_size:
        print("vocabulary {}_{} longer than max_vocabulary_size {}. Truncate the tail".format(prefix, len(vocab_list2), max_size))
        vocab_list2= vocab_list2[:max_size]
      for w in vocab_list2:
        vocab_file.write(str(w) + b"\n")
      minimum_occurance.append(vocab_counts[name][vocab_list2[-1]])
  with gfile.GFile(join(data_dir, "%s_minimum_occurance_%d" %(prefix, 
    max_size)), mode="wb") as sum_file:
    sum_file.write('\n'.join([str(v) for v in minimum_occurance]))

  return

def create_dictionary_mix(data_dir, inds, features, feature_types, 
  feature_names, max_vocabulary_size=50000, logits_size_tr = 50000, 
  threshold = 2, prefix='user'):
  filename = 'vocab0_%d' % max_vocabulary_size
  if isfile(join(data_dir, filename)):
    print("vocabulary exists!")
    return
  vocab_counts = {}
  num_uf = len(feature_names)
  assert(len(feature_types) == num_uf), 'length of feature_types should be the same length of feature_names {} vs {}'.format(len(feature_types), num_uf)

  vocab_uid = {}
  vocab = {}
  for u in inds: # u index
    uf = features[u, 0]

    if not isinstance(uf, list):
      uf = uf.split(',')
    for t in uf:
      if t.startswith('uid'):
        vocab_uid[t] = vocab_uid[t] + 1 if t in vocab_uid else 1
      else:
        vocab[t] = vocab[t] + 1 if t in vocab else 1

  minimum_occurance = []
  
  vocab_list = _START_VOCAB + vocab_uid.keys() + sorted(vocab, 
    key=vocab.get, reverse=True)
  
  max_size = max_vocabulary_size

  with gfile.GFile(join(data_dir, ("%s_vocab%d_%d"% (prefix, 0,
    max_size))), mode="wb") as vocab_file:

    vocab_list2 = [v for v in vocab_list if v in _START_VOCAB or (v in vocab and 
      vocab[v] >= threshold) or (v in vocab_uid and vocab_uid[v] >= threshold)]
    if len(vocab_list2) > max_size:
      print("vocabulary {}_{} longer than max_vocabulary_size {}. Truncate the tail".format(prefix, len(vocab_list2), max_size))
      vocab_list2 = vocab_list2[:max_size]

    for w in vocab_list2:
      vocab_file.write(str(w) + b"\n")
    min_occurance = vocab[vocab_list2[-1]] if vocab_list2[-1] in vocab else vocab_uid[vocab_list2[-1]]
    minimum_occurance.append(min_occurance)
  with gfile.GFile(join(data_dir, "%s_minimum_occurance_%d" %(prefix, 
    max_size)), mode="wb") as sum_file:
    sum_file.write('\n'.join([str(v) for v in minimum_occurance]))

  return

def tokenize_attribute_map(data_dir, features, feature_types, max_vocabulary_size, 
  logits_size_tr=50000, prefix='user'):
  """
  read feature maps and tokenize with loaded vocabulary
  output required format for Attributes
  """
  features_cat, features_mulhot = [], []
  v_sizes_cat, v_sizes_mulhot = [], []
  mulhot_max_leng, mulhot_starts, mulhot_lengs = [], [], []
  # logit_ind2item_ind = {}
  for i in range(len(feature_types)):
    ut = feature_types[i]
    if feature_types[i] > 1: 
      continue

    path = "%s_vocab%d_" %(prefix, i)
    vocabulary_paths = [f for f in listdir(data_dir) if f.startswith(path)]
    assert(len(vocabulary_paths) == 1)
    vocabulary_path = join(data_dir, vocabulary_paths[0])

    vocab, _ = initialize_vocabulary(vocabulary_path)

    N = len(features)
    users2 = np.copy(features)
    uf = features[:, i]
    if ut == 0:
      v_sizes_cat.append(len(vocab))
      for n in range(N):
        uf[n] = vocab.get(str(uf[n]), UNK_ID)
      uf = np.append(uf, START_ID)
      features_cat.append(uf)
    else:
      mtl = 0
      idx = 0
      starts, lengs, vals = [idx], [], []
      v_sizes_mulhot.append(len(vocab))
      for n in range(N):
        elem = uf[n]
        if not isinstance(elem, list):
          if not isinstance(elem, str):
            elem = str(elem)
          elem = elem.split(',')
        val = [vocab.get(str(v), UNK_ID) for v in elem]
        val_ = [v for v in val if v != UNK_ID]
        if len(val_) == 0:
          val_ = [UNK_ID]

        vals.extend(val_)
        l_mulhot = len(val_)
        mtl = max(mtl, l_mulhot)
        idx += l_mulhot
        starts.append(idx)
        lengs.append(l_mulhot)

      vals.append(START_ID)
      idx += 1
      starts.append(idx)
      lengs.append(1)

      mulhot_max_leng.append(mtl)
      mulhot_starts.append(np.array(starts))
      mulhot_lengs.append(np.array(lengs))
      features_mulhot.append(np.array(vals))

  num_features_cat = sum(v == 0 for v in feature_types)
  num_features_mulhot= sum(v == 1 for v in feature_types)
  assert(num_features_cat + num_features_mulhot <= len(feature_types))
  return (num_features_cat, features_cat, num_features_mulhot, features_mulhot,
    mulhot_max_leng, mulhot_starts, mulhot_lengs, v_sizes_cat, 
    v_sizes_mulhot)

def filter_cat(num_features_cat, features_cat, logit_ind2item_ind):
  ''' 
  create mapping from logits index [0, logits_size) to features
  '''
  features_cat_tr = []
  size = len(logit_ind2item_ind)
  for i in xrange(num_features_cat):
    feat_cat = features_cat[i]
    feat_cat_tr = []
    for j in xrange(size):
      item_index = logit_ind2item_ind[j]
      feat_cat_tr.append(feat_cat[item_index])
    features_cat_tr.append(np.array(feat_cat_tr))

  return features_cat_tr
  

def filter_mulhot(data_dir, items, feature_types, max_vocabulary_size, 
  logit_ind2item_ind, prefix='item'):
  full_values,  full_values_tr= [], []
  full_segids, full_lengths = [], []
  full_segids_tr, full_lengths_tr = [], []

  L = len(logit_ind2item_ind)
  N = len(items)
  for i in range(len(feature_types)):
    full_index, full_index_tr = [], [] 
    lengs, lengs_tr = [], []
    ut = feature_types[i]
    if feature_types[i] == 1:

      path = "%s_vocab%d_" %(prefix, i)
      vocabulary_paths = [f for f in listdir(data_dir) if f.startswith(path)]
      assert(len(vocabulary_paths)==1), 'more than one dictionaries found! delete unnecessary ones to fix this.'
      vocabulary_path = join(data_dir, vocabulary_paths[0])
      
      vocab, _ = initialize_vocabulary(vocabulary_path)
      
      uf = items[:, i]
      mtl, idx, vals = 0, 0, []
      segids = []

      for n in xrange(N):
        elem = uf[n]
        if not isinstance(elem, list):
          if not isinstance(elem, str):
            elem = str(elem)
          elem = elem.split(',')

        val = [vocab.get(v, UNK_ID) for v in elem]
        val_ = [v for v in val if v != UNK_ID]
        if len(val_) == 0:
          val_ = [UNK_ID]
        vals.extend(val_)
        l_mulhot = len(val_)
        segids.extend([n] * l_mulhot)
        lengs.append([l_mulhot * 1.0])       
      
      full_values.append(vals)
      full_segids.append(segids)
      full_lengths.append(lengs)

      idx2, vals2 = 0, []
      segids_tr = []
      for n in xrange(L):
        i_ind = logit_ind2item_ind[n]
        elem = uf[i_ind]
        if not isinstance(elem, list):
          if not isinstance(elem, str):
            elem = str(elem)
          elem = elem.split(',')

        val = [vocab.get(v, UNK_ID) for v in elem]
        val_ = [v for v in val if v != UNK_ID]
        if len(val_) == 0:
          val_ = [UNK_ID]
        vals2.extend(val_)
        l_mulhot = len(val_)
        lengs_tr.append([l_mulhot * 1.0])
        segids_tr.extend([n] * l_mulhot)
      
      full_values_tr.append(vals2)
      full_segids_tr.append(segids_tr)
      full_lengths_tr.append(lengs_tr)

  return (full_values, full_values_tr, full_segids, full_lengths, 
    full_segids_tr, full_lengths_tr)

