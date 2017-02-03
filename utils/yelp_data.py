import sys
sys.path.insert(0, '../attributes')
import numpy as np


def process_items(items):
  from load_yelp_data import load_count_view_mapping
  m = load_count_view_mapping()
  import math
  print('processing item features')
 
  # TOKENS = ['[', ']', ' ', '(', ')']

  for i in range(items.shape[0]):
    uid = items[i][0]
    if isinstance(items[i][1], str):
      items[i][1] = items[i][1].split(',')
    else:
      print('1 not str!')
      exit()
    
    if isinstance(items[i][5], int):
      items[i][5] = m[uid]
    else:
      print('5 not int!')
      exit()

    if isinstance(items[i][6], str):
      items[i][6] = items[i][6].split(',')
    elif isinstance(items[i][6],float):
      assert math.isnan(items[i][6])
      items[i][6] = ['-1']
    else:
      print(type(items[i][6]))
      print(items[i][6])
      print('6 not str!')
      exit()

    if isinstance(items[i][7], str):
      items[i][7] = items[i][7].split(',')
    elif isinstance(items[i][7],float):
      assert math.isnan(items[i][7])
      items[i][7] = ['-1']
    else:
      print(type(items[i][7]))
      print(items[i][7])
      print('7 not str!')
      exit()
    
  return items

def process_users(users):
  # add one fake feature dimension (for lstm rec model)
  print('processing user features')
  users[:, 1] = 0
  return users

def interact_split(interact, user_index=None, item_index=None, orig=False, debug=0):
  from load_yelp_data import load_meta
  t20, t10 = load_meta()

  l = len(interact)
  if debug == 1:
    l = l/20
  interact_tr = np.zeros((l, 4), dtype=object)
  interact_va = np.zeros((l, 4), dtype=object)
  interact_te = np.zeros((l, 4), dtype=object)
  ind1, ind2, ind3 = 0,0,0

  ints = {}
  for i in range(l):
    uid, iid, irating, t = interact[i, :]
    if uid not in ints:
      ints[uid] = []
    ints[uid].append((iid, t, irating))
  final_users = set([])
  final_items = set([])
  for u, v in ints.items():
    assert len(v) >= 10, (len(v), v)

    final_users.add(u)
    v = sorted(v, key=lambda tup: tup[1])
    l0 = len(v)
    for j in range(l0):
      t = v[j][1]
      if orig:
        val = (u, v[j][0], v[j][2], v[j][1])
      else:
        val = (user_index[u], item_index[v[j][0]], v[j][2], v[j][1])
      if t < t20:
        interact_tr[ind1, :] = val
        ind1 += 1
      elif t < t10:
        interact_va[ind2, :] = val
        ind2 += 1
      else:
        interact_te[ind3, :] = val
        ind3 += 1
      final_items.add(val[1])


  interact_tr = interact_tr[:ind1, :]
  interact_va = interact_va[:ind2, :]
  interact_te = interact_te[:ind3, :]
  print("train, valid, and test sizes %d/%d/%d" %(ind1, ind2, ind3))
  print("# of users: {}; # of items: {}".format(len(final_users), len(final_items)))
  return interact_tr, interact_va, interact_te

def data_read(data_dir, _submit=0, ta=1, max_vocabulary_size=50000, 
  max_vocabulary_size2=50000, logits_size_tr=20000, thresh=3, sample=1.0, old=False):
  from os import path,mkdir
  from os.path import join, isfile
  import cPickle as pickle

  data_filename = join(data_dir, 'ml_file')
  if isfile(data_filename):
    print("ml_data exists, loading")
    (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
      logit_ind2item_ind) = pickle.load(open(data_filename, 'rb'))
    return (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
    logit_ind2item_ind)

  from preprocess import create_dictionary, tokenize_attribute_map, filter_cat, filter_mulhot, pickle_save
  import attribute
  from load_yelp_data import load_user, load_item, load_interactions

  if not path.isdir(data_dir):
    mkdir(data_dir)
    
  users, user_feature_names, user_index = load_user()
  items, item_feature_names, item_index = load_item()
  items = process_items(items)
  users = process_users(users)

  user_feature_names = ['id', 'fake']
  N = len(users)
  M = len(items)

  interact, names = load_interactions()
  print("loading interactions completed.")

  interact_tr, interact_va, interact_te = interact_split(interact, user_index, item_index, debug=ta)

  # data_tr, data_va, data_te
  data_va, data_te = None, None
  if _submit == 1:    
    interact_tr = np.append(interact_tr, interact_va, 0)
    data_tr = zip(list(interact_tr[:, 0]), list(interact_tr[:, 1]), 
      list(interact_tr[:, 3]))
    data_va = zip(list(interact_te[:, 0]), list(interact_te[:, 1]), 
      list(interact_te[:, 3]))
  else:
    data_tr = zip(list(interact_tr[:, 0]), list(interact_tr[:, 1]), 
      list(interact_tr[:, 3]))
    data_va = zip(list(interact_va[:, 0]), list(interact_va[:, 1]), 
      list(interact_va[:, 3]))
    data_te = zip(list(interact_te[:, 0]), list(interact_te[:, 1]), 
      list(interact_te[:, 3]))

  # create_dictionary

  user_feature_types = [0, 0]
  u_inds = [p[0] for p in data_tr]
  create_dictionary(data_dir, u_inds, users, user_feature_types, 
    user_feature_names, max_vocabulary_size, logits_size_tr, prefix='user', threshold=thresh)

  # create user feature map
  (num_features_cat, features_cat, num_features_mulhot, features_mulhot,
    mulhot_max_leng, mulhot_starts, mulhot_lengs, v_sizes_cat, 
    v_sizes_mulhot) = tokenize_attribute_map(data_dir, users, user_feature_types, 
    max_vocabulary_size, logits_size_tr, prefix='user')

  u_attributes = attribute.Attributes(num_features_cat, features_cat, 
    num_features_mulhot, features_mulhot, mulhot_max_leng, mulhot_starts, 
    mulhot_lengs, v_sizes_cat, v_sizes_mulhot)


  # create_dictionary
  item_feature_types = [0, 1, 0, 0, 0, 0, 1, 1]

  i_inds = [p[1] for p in data_tr]
  create_dictionary(data_dir, i_inds, items, item_feature_types, 
    item_feature_names, max_vocabulary_size2, logits_size_tr, prefix='item', threshold=thresh)

  # create item feature map
  items_cp = np.copy(items)
  (num_features_cat2, features_cat2, num_features_mulhot2, features_mulhot2,
    mulhot_max_leng2, mulhot_starts2, mulhot_lengs2, v_sizes_cat2, 
    v_sizes_mulhot2) = tokenize_attribute_map(data_dir, 
    items_cp, item_feature_types, max_vocabulary_size2, logits_size_tr, 
    prefix='item')

  item_ind2logit_ind = {}
  item2fea0 = features_cat2[0]
  ind = 0
  for i in range(len(items)):
    fea0 = item2fea0[i]
    if fea0 != 0:
      item_ind2logit_ind[i] = ind
      ind += 1
  assert(ind == logits_size_tr), 'Item_vocab_size %d too large! need to be no greater than %d' % (logits_size_tr, ind)
  
  logit_ind2item_ind = {}
  for k, v in item_ind2logit_ind.items():
    logit_ind2item_ind[v] = k

  i_attributes = attribute.Attributes(num_features_cat2, features_cat2, 
    num_features_mulhot2, features_mulhot2, mulhot_max_leng2, mulhot_starts2, 
    mulhot_lengs2, v_sizes_cat2, v_sizes_mulhot2)

  # set target prediction indices
  features_cat2_tr = filter_cat(num_features_cat2, features_cat2, 
    logit_ind2item_ind)

  (full_values, full_values_tr, full_segids, full_lengths, full_segids_tr, 
    full_lengths_tr) = filter_mulhot(data_dir, items, 
    item_feature_types, max_vocabulary_size2, logit_ind2item_ind, prefix='item')
  

  i_attributes.set_target_prediction(features_cat2_tr, full_values_tr, 
    full_segids_tr, full_lengths_tr)

  print("saving data format to data directory")
  pickle_save((data_tr, data_va, u_attributes, i_attributes, 
    item_ind2logit_ind, logit_ind2item_ind), data_filename)
  return (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
    logit_ind2item_ind)


