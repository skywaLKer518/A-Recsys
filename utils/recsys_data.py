from load_data import *
import sys
sys.path.insert(0, '../utils')
import attribute

from os import listdir, mkdir, path, rename
from os.path import isfile, join
import cPickle as pickle

from tensorflow.python.platform import gfile


_UNK = "_UNK"
# _PHANTOM = "_PHANTOM"

UNK_ID = 0
# REST_ID = 1

_START_VOCAB = [_UNK ] #, _PHANTOM]

def pickle_save(m, filename):
  pickle.dump(m, open(filename, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

def process_users_nan(users):
  import math
  print 'processing feature'
  for i in range(users.shape[0]):
    # jobrole
    if isinstance(users[i][1], str):
      users[i][1] = users[i][1].split(',')
    # career level
    if users[i][2] == 0 or math.isnan(users[i][2]):
      users[i][2] = 0  # changed from 3 to 0 May 4th
    # dicipline id
    if math.isnan(users[i][3]):
      users[i][3] = 'unknown'
    # industry id
    if math.isnan(users[i][4]):
      users[i][4] = 'unknown'
    # country
    if not isinstance(users[i][5], str):
      users[i][5] = 'unknown'
    # region
    if math.isnan(users[i][6]) or users[i][6] == 0.0:
      users[i][6] = 'unknown'
    # n_entry
    if math.isnan(users[i][7]):
      users[i][7] = 'unknown'
    # n years
    if math.isnan(users[i][8]):
      users[i][8] = 'unknown'
    # years in current
    if math.isnan(users[i][9]):
      users[i][9] = 'unknown'
    # education degree
    if math.isnan(users[i][10]) or users[i][10] == 0.0:
      users[i][10] = 'unknown'
    # edu fields
    if isinstance(users[i][11], str):
      users[i][11] = users[i][11].split(',')
    if isinstance(users[i][11], float):
      users[i][11] = ['-1']            
  return users

def process_items_nan(items):
  import math
  print 'processing feature'
  for i in range(items.shape[0]):
      # jobrole
    if isinstance(items[i][1], str):
      items[i][1] = items[i][1].split(',')
    if isinstance(items[i][1], float):
      items[i][1] = ['-1']

    # career level
    if items[i][2] == 0 or math.isnan(items[i][2]):
      items[i][2] = 0 # changed from 3 to 0 May 9th
    # dicipline id
    if math.isnan(items[i][3]):
      items[i][3] = 'unknown'
    # industry id
    if math.isnan(items[i][4]):
      items[i][4] = 'unknown'
    # country
    if not isinstance(items[i][5], str):
      items[i][5] = 'unknown'
    # region
    if math.isnan(items[i][6]) or items[i][6] == 0.0:
      items[i][6] = 'unknown'
    # latitude
    if math.isnan(items[i][7]):# (505)
      items[i][7] = 'unknown'
    # longitude
    if math.isnan(items[i][8]):# (906)
      items[i][8] = 'unknown'
    # employment
    if math.isnan(items[i][9]):
      items[i][9] = 'unknown'
    # tags
    if isinstance(items[i][10], str):
      items[i][10] = items[i][10].split(',')
    if isinstance(items[i][10], float):
      items[i][10] = ['-1']

    if math.isnan(items[i][11]):
      items[i][11] = -1
    else:
      items[i][11] = min(int((items[i][11] - 1432245600.0) / 15555600.0 * 30), 29)
    # from 2015, 5, 21, 15, 0 to 2015, 11, 17, 15, 0

  return items

def to_index(interact, user_index_all, item_index_all):
  l = len(interact)
  interact_tr = np.zeros((l, 4), dtype=int)
  interact_va = np.zeros((l, 4), dtype=int)
  ind1, ind2 = 0,0
  for i in range(l):
    uid, iid, itype, week, _ = interact[i, :]
    if iid not in item_index_all:
      continue
    if itype == 4:
      continue
    if week <= 44:
      interact_tr[ind1, :] = (user_index_all[uid], item_index_all[iid], itype, week)
      ind1 += 1
    elif week == 45:
      interact_va[ind2, :] = (user_index_all[uid], item_index_all[iid], itype, week)
      ind2 += 1
    else:
      exit(-1)
  interact_tr = interact_tr[:ind1, :]
  interact_va = interact_va[:ind2, :]
  print("train and valid sizes %d/%d" %(ind1, ind2))
  return interact_tr, interact_va

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

def create_dictionary(data_dir, data_tr, features, feature_types, feature_names, 
  max_vocabulary_size=50000, logits_size_tr = 50000, threshold = 2, 
  prefix='user'):
  filename = 'vocab0_%d' % max_vocabulary_size
  if isfile(join(data_dir, filename)):
    print("vocabulary exists!")
    return
  vocab_counts = {}
  num_uf = len(feature_names)
  assert(len(feature_types) == num_uf)
  for ind in range(num_uf):
    name = feature_names[ind]
    vocab_counts[name] = {}

  for u1, u2 in data_tr: # i, u both index
    if prefix=='user':
      u = u1
    else:
      u = u2
    uf = features[u, :]
    for ii in range(num_uf):
      name = feature_names[ii]
      if feature_types[ii] == 0:   
        vocab_counts[name][uf[ii]] = vocab_counts[name][uf[ii]] + 1 if uf[ii] in vocab_counts[name] else 1
      elif feature_types[ii] == 1:
        if not isinstance(uf[ii], list):
          print ii, prefix
          print uf[ii]
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
      max_size = logits_size_tr
    elif prefix == 'user' and i == 0:
      max_size = len(features)
    else:
      max_size = max_vocabulary_size
    if len(vocab_list) > max_size:
      vocab_list= vocab_list[:max_size]
    with gfile.GFile(join(data_dir, ("%s_vocab%d_%d"% (prefix, i,
      max_size))), mode="wb") as vocab_file:
      vocab_list2 = [v for v in vocab_list if v in _START_VOCAB or
       vocab_counts[name][v] >= threshold]
      for w in vocab_list2:
        vocab_file.write(str(w) + b"\n")
      minimum_occurance.append(vocab_counts[name][vocab_list2[-1]])
  with gfile.GFile(join(data_dir, "%s_minimum_occurance_%d" %(prefix, 
    max_size)), mode="wb") as sum_file:
    sum_file.write('\n'.join([str(v) for v in minimum_occurance]))

  return

def tokenize_attribute_map(data_dir, users, feature_types, max_vocabulary_size, 
  logits_size_tr=50000, prefix='user'):
  """
  read feature maps and tokenize with loaded vocabulary
  output required format for Attributes
  """
  features_cat, features_mulhot = [], []
  v_sizes_cat, v_sizes_mulhot = [], []
  mulhot_max_leng, mulhot_starts, mulhot_lengs = [], [], []
  logit_ind2item_ind = {}
  for i in range(len(feature_types)):
    ut = feature_types[i]
    if feature_types[i] > 1: 
      continue
    if prefix == 'user' and i == 0:
      max_size = len(users)
    elif prefix == 'item' and i == 0:
      max_size = logits_size_tr
    else:
      max_size = max_vocabulary_size

    vocabulary_path = join(data_dir, "%s_vocab%d_%d" %(prefix, i, 
      max_size))
    vocab, _ = initialize_vocabulary(vocabulary_path)

    N = len(users)
    users2 = np.copy(users)
    uf = users[:, i]
    if ut == 0:
      v_sizes_cat.append(len(vocab))
      for n in range(N):
        uf[n] = vocab.get(str(uf[n]), UNK_ID)
      features_cat.append(uf)
    else:
      mtl = 0
      idx = 0
      starts, lengs, vals = [idx], [], []
      v_sizes_mulhot.append(len(vocab))
      for n in range(N):
        val = [vocab.get(str(v), UNK_ID) for v in uf[n]]
        val_ = [v for v in val if v != UNK_ID]
        if len(val_) == 0:
          val_ = [UNK_ID]

        vals.extend(val_)
        l_mulhot = len(val_)
        mtl = max(mtl, l_mulhot)
        idx += l_mulhot
        starts.append(idx)
        lengs.append(l_mulhot)

      mulhot_max_leng.append(mtl)
      mulhot_starts.append(np.array(starts))
      mulhot_lengs.append(np.array(lengs))
      features_mulhot.append(np.array(vals))

    if i == 0 and prefix == 'item':
      uf = users2[:, i]
      logit_ind2item_ind[0] = 0
      for n in xrange(N):
        log_ind = vocab.get(str(uf[n]), UNK_ID)
        if log_ind != 0:
          logit_ind2item_ind[log_ind] = n

      assert(len(list(logit_ind2item_ind.keys()))<=logits_size_tr)
      assert(len(list(logit_ind2item_ind.values()))<=logits_size_tr)

  num_features_cat = sum(v == 0 for v in feature_types)
  num_features_mulhot= sum(v == 1 for v in feature_types)
  assert(num_features_cat + num_features_mulhot <= len(feature_types))
  return (num_features_cat, features_cat, num_features_mulhot, features_mulhot,
    mulhot_max_leng, mulhot_starts, mulhot_lengs, v_sizes_cat, 
    v_sizes_mulhot, logit_ind2item_ind)

def tokenize_attribute_sp_indices(data_dir, items, feature_types, 
  max_vocabulary_size, logit_ind2item_ind, prefix='item'):
  full_indices, full_values, sp_shapes = [], [], []
  full_indices_tr, full_values_tr, sp_shapes_tr = [], [], []

  full_segids, full_lengths = [], []
  full_segids_tr, full_lengths_tr = [], []


  L = len(logit_ind2item_ind)
  N = len(items)
  for i in range(len(feature_types)):
    full_index, full_index_tr = [], [] 
    lengs, lengs_tr = [], []
    ut = feature_types[i]
    if feature_types[i] != 1: 
      continue
    if prefix == 'item' and i == 0:
      max_size = logits_size_tr
    else:
      max_size = max_vocabulary_size

    vocabulary_path = join(data_dir, "%s_vocab%d_%d" %(prefix, i, 
      max_size))
    vocab, _ = initialize_vocabulary(vocabulary_path)
    
    uf = items[:, i]
    mtl, idx, vals = 0, 0, []
    segids = []

    for n in xrange(N):
      val = [vocab.get(v, UNK_ID) for v in uf[n]]
      val_ = [v for v in val if v != UNK_ID]
      if len(val_) == 0:
        val_ = [UNK_ID]
      vals.extend(val_)
      l_mulhot = len(val_)
      segids.extend([n] * l_mulhot)
      lengs.append([l_mulhot * 1.0])
      mtl = max(mtl, l_mulhot)
      idx += l_mulhot
      for j in xrange(l_mulhot):
        full_index.append([n, j])

    full_indices.append(full_index)
    full_values.append(vals)
    sp_shapes.append([N, mtl])
    full_segids.append(segids)
    full_lengths.append(lengs)

    mtl2, idx2, vals2 = 0, 0, []
    segids_tr = []
    for n in xrange(L):
      i_ind = logit_ind2item_ind[n]
      val = [vocab.get(v, UNK_ID) for v in uf[i_ind]]
      val_ = [v for v in val if v != UNK_ID]
      if len(val_) == 0:
        val_ = [UNK_ID]
      vals2.extend(val_)
      l_mulhot = len(val_)
      lengs_tr.append([l_mulhot * 1.0])
      segids_tr.extend([n] * l_mulhot)
      mtl2 = max(mtl2, l_mulhot)
      idx2 += l_mulhot
      for j in xrange(l_mulhot):
        full_index_tr.append([n, j])

    full_indices_tr.append(full_index_tr)
    full_values_tr.append(vals2)
    sp_shapes_tr.append([L, mtl2])
    full_segids_tr.append(segids_tr)
    full_lengths_tr.append(lengs_tr)

  return (full_indices, full_values, sp_shapes, full_indices_tr, full_values_tr, 
    sp_shapes_tr, full_segids, full_lengths, full_segids_tr, full_lengths_tr)

def get_item_train_mappings(num_features_cat, features_cat, num_features_mulhot, 
  features_mulhot, mulhot_max_leng, mulhot_starts, mulhot_lengs, 
  v_sizes_cat, v_sizes_mulhot, logit_ind2item_ind):
  ''' 
  create mapping from logits index [0, logits_size) to features
  '''
  (features_cat_tr, features_mulhot_tr, mulhot_max_leng_tr, 
    mulhot_starts_tr, mulhot_lengs_tr) = [], [], [], [], []

  size = len(logit_ind2item_ind)
  for i in xrange(num_features_cat):
    feat_cat = features_cat[i]
    feat_cat_tr = [0]
    for j in xrange(1, size):
      item_index = logit_ind2item_ind[j]
      feat_cat_tr.append(feat_cat[item_index])
    features_cat_tr.append(np.array(feat_cat_tr))

  for i in xrange(num_features_mulhot):
    feat_mulhot = list(features_mulhot[i])
    mulhot_start = mulhot_starts[i]
    mulhot_leng = mulhot_lengs[i]

    feat_mulhot_tr = [0]
    starts_tr, lengths_tr = [0, 1], [1]
    mtl = 1
    idx = 1
    for j in xrange(1, size):
      item_index = logit_ind2item_ind[j]
      st, l = mulhot_start[item_index], mulhot_leng[item_index]
      feat_mulhot_tr.extend(feat_mulhot[st:st+l])
      idx += l
      starts_tr.append(idx)
      lengths_tr.append(l)

      mtl = max(mtl, l)
    features_mulhot_tr.append(np.array(feat_mulhot_tr))
    mulhot_starts_tr.append(np.array(starts_tr))
    mulhot_lengs_tr.append(np.array(lengths_tr))
    mulhot_max_leng_tr.append(mtl)

  return (features_cat_tr, features_mulhot_tr, mulhot_max_leng_tr, 
    mulhot_starts_tr, mulhot_lengs_tr)

def data_read(data_dir, _submit=0, ta=1, max_vocabulary_size=50000, 
  max_vocabulary_size2=50000, logits_size_tr=50000):

  data_filename = join(data_dir, 'recsys_file')
  if isfile(data_filename):
    print("recsys exists, loading")
    (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
      logit_ind2item_ind) = pickle.load(open(data_filename, 'rb'))
    return (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
    logit_ind2item_ind)
  if ta == 1:
    users, user_feature_names, user_index_orig = load_user_target_csv()
    items, item_feature_names, item_index_orig = load_item_active_csv()

  else:
    users, user_feature_names, user_index_orig = load_user_csv()
    items, item_feature_names, item_index_orig = load_item_csv()

  N = len(user_index_orig)
  M = len(item_index_orig)

  interact, _ = load_interactions(1, ta)
  interact_tr, interact_va = to_index(interact, user_index_orig, 
    item_index_orig)

  data_va = None
  if _submit == 1:    
    interact_tr = np.append(interact_tr, interact_va, 0)
    data_tr = zip(list(interact_tr[:, 0]), list(interact_tr[:, 1]))
  else:
    data_tr = zip(list(interact_tr[:, 0]), list(interact_tr[:, 1]))
    data_va = zip(list(interact_va[:, 0]), list(interact_va[:, 1]))

  # clean data
  filename = 'processed_user' + '_ta_' + str(ta)
  if isfile(join(data_dir, filename)):
    users = pickle.load(open(join(data_dir, filename), 'rb'))
  else:
    users = process_users_nan(users)
    if not path.isdir(data_dir):
      mkdir(data_dir)
    pickle_save(users, join(data_dir, filename))

  filename = 'processed_item' + '_ta_' + str(ta)
  if isfile(join(data_dir, filename)):
    items = pickle.load(open(join(data_dir, filename), 'rb'))
  else:
    items = process_items_nan(items)
    if not path.isdir(data_dir):
      mkdir(data_dir)
    pickle_save(items, join(data_dir, filename))

  # create_dictionary
  user_feature_types = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
  create_dictionary(data_dir, data_tr, users, user_feature_types, 
    user_feature_names, max_vocabulary_size, logits_size_tr, prefix='user')

  # create user feature map
  (num_features_cat, features_cat, num_features_mulhot, features_mulhot,
    mulhot_max_leng, mulhot_starts, mulhot_lengs, v_sizes_cat, 
    v_sizes_mulhot, _) = tokenize_attribute_map(data_dir, users, user_feature_types, 
    max_vocabulary_size, logits_size_tr, prefix='user')

  u_attributes = attribute.Attributes(num_features_cat, features_cat, 
    num_features_mulhot, features_mulhot, mulhot_max_leng, mulhot_starts, 
    mulhot_lengs, v_sizes_cat, v_sizes_mulhot)

  # create_dictionary
  item_feature_types = [0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 1, 2, 3]
  create_dictionary(data_dir, data_tr, items, item_feature_types, 
    item_feature_names, max_vocabulary_size2, logits_size_tr, prefix='item')

  # create item feature map
  (num_features_cat2, features_cat2, num_features_mulhot2, features_mulhot2,
    mulhot_max_leng2, mulhot_starts2, mulhot_lengs2, v_sizes_cat2, 
    v_sizes_mulhot2, logit_ind2item_ind) = tokenize_attribute_map(data_dir, 
    items, item_feature_types, max_vocabulary_size2, logits_size_tr, 
    prefix='item')
  
  item_ind2logit_ind = features_cat2[0]

  (features_cat2_tr, features_mulhot2_tr, mulhot_max_leng2_tr, 
    mulhot_starts2_tr, mulhot_lengs2_tr) = get_item_train_mappings(
    num_features_cat2, features_cat2, num_features_mulhot2, features_mulhot2,
    mulhot_max_leng2, mulhot_starts2, mulhot_lengs2, v_sizes_cat2, 
    v_sizes_mulhot2, logit_ind2item_ind)

  (full_indices, full_values, sp_shapes, full_indices_tr, full_values_tr, 
    sp_shapes_tr, full_segids, full_lengths, full_segids_tr, 
    full_lengths_tr) = tokenize_attribute_sp_indices(data_dir, items, 
    item_feature_types, max_vocabulary_size2, logit_ind2item_ind, prefix='item')

  # (full_values_tr, full_segids_tr, full_lengths_tr) = token
  i_attributes = attribute.Attributes(num_features_cat2, features_cat2, 
    num_features_mulhot2, features_mulhot2, mulhot_max_leng2, mulhot_starts2, 
    mulhot_lengs2, v_sizes_cat2, v_sizes_mulhot2)

  i_attributes.set_train_mapping(features_cat2_tr, features_mulhot2_tr, 
    mulhot_max_leng2_tr, mulhot_starts2_tr, mulhot_lengs2_tr)

  i_attributes.add_sparse_mapping(full_indices, full_values, sp_shapes, 
    full_indices_tr, full_values_tr, sp_shapes_tr)
  
  i_attributes.add_sparse_mapping2(full_segids, full_lengths, full_segids_tr,
    full_lengths_tr)

  print("saving data format to data directory")
  pickle_save((data_tr, data_va, u_attributes, i_attributes, 
    item_ind2logit_ind, logit_ind2item_ind), data_filename)
  return (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
    logit_ind2item_ind)

