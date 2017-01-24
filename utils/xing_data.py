import numpy as np
import sys
sys.path.insert(0, '../attributes')


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
  interact_tr = np.zeros((l, 5), dtype=int)
  interact_va = np.zeros((l, 5), dtype=int)
  ind1, ind2 = 0,0
  for i in range(l):
    uid, iid, itype, week, t = interact[i, :]
    if uid not in user_index_all:
      continue
    if iid not in item_index_all:
      continue
    if itype == 4:
      continue
    if week <= 44:
      interact_tr[ind1, :] = (user_index_all[uid], item_index_all[iid], itype, week, t)
      ind1 += 1
    elif week == 45:
      interact_va[ind2, :] = (user_index_all[uid], item_index_all[iid], itype, week, t)
      ind2 += 1
    else:
      exit(-1)
  interact_tr = interact_tr[:ind1, :]
  interact_va = interact_va[:ind2, :]
  print("train and valid sizes %d/%d" %(ind1, ind2))
  return interact_tr, interact_va


def data_read(data_dir, _submit=0, ta=1, max_vocabulary_size=50000, 
  max_vocabulary_size2=50000, logits_size_tr=50000, sample=1.0, old=False):
  from os import mkdir, path
  from os.path import isfile, join
  import cPickle as pickle
  if sample == 1.0:
    data_filename = join(data_dir, 'recsys_file')
  else:
    data_filename = join(data_dir, 'recsys_file' + str(sample))
  if isfile(data_filename):
    print("recsys exists, loading")
    (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
      logit_ind2item_ind) = pickle.load(open(data_filename, 'rb'))
    return (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
    logit_ind2item_ind)

  import attribute
  from load_xing_data import load_user_target_csv, load_item_active_csv, load_user_csv, load_item_csv, load_interactions
  from preprocess import create_dictionary, tokenize_attribute_map, filter_cat, filter_mulhot, pickle_save
  
  if ta == 1:
    users, user_feature_names, user_index_orig = load_user_target_csv()
    items, item_feature_names, item_index_orig = load_item_active_csv()
  else:
    users, user_feature_names, user_index = load_user_target_csv()
    items, item_feature_names, item_index = load_item_active_csv()

    users_all, _, user_index_all = load_user_csv()

    if sample < 1.0:
      # sample only portion of users
      import random
      target_user_ids = set([int(x) for x in list(users[:, 0])])
      users_all2 = np.copy(users_all)
      c = 0
      for i in range(len(users_all)):
        uid = int(users_all[i, 0])
        if uid in target_user_ids or random.random() < sample:
          users_all2[c, :] = users_all[i, :]
          c += 1
      users_all2 = users_all2[:c, :]
      print('total user number is {}'.format(c))
      from pandatools import build_index
      users_all = users_all2
      user_index_all = build_index(users_all)


    items_all, _, item_index_all = load_item_csv()
    user_index_orig = user_index_all
    item_index_orig = item_index
    users = users_all

  N = len(user_index_orig)
  M = len(item_index_orig)

  interact, _ = load_interactions(1, ta)
  interact_tr, interact_va = to_index(interact, user_index_orig, 
    item_index_orig)

  data_va = None
  if _submit == 1:    
    interact_tr = np.append(interact_tr, interact_va, 0)
    data_tr = zip(list(interact_tr[:, 0]), list(interact_tr[:, 1]), 
      list(interact_tr[:, 4]))
  else:
    data_tr = zip(list(interact_tr[:, 0]), list(interact_tr[:, 1]), 
      list(interact_tr[:, 4]))
    data_va = zip(list(interact_va[:, 0]), list(interact_va[:, 1]), 
      list(interact_va[:, 4]))

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
  u_inds = [p[0] for p in data_tr]
  create_dictionary(data_dir, u_inds, users, user_feature_types, 
    user_feature_names, max_vocabulary_size, logits_size_tr, prefix='user')

  # create user feature map
  (num_features_cat, features_cat, num_features_mulhot, features_mulhot,
    mulhot_max_leng, mulhot_starts, mulhot_lengs, v_sizes_cat, 
    v_sizes_mulhot) = tokenize_attribute_map(data_dir, users, user_feature_types, 
    max_vocabulary_size, logits_size_tr, prefix='user')

  u_attributes = attribute.Attributes(num_features_cat, features_cat, 
    num_features_mulhot, features_mulhot, mulhot_max_leng, mulhot_starts, 
    mulhot_lengs, v_sizes_cat, v_sizes_mulhot)


  # create_dictionary
  item_feature_types = [0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 1, 2, 3]
  i_inds = [p[1] for p in data_tr]
  create_dictionary(data_dir, i_inds, items, item_feature_types, 
    item_feature_names, max_vocabulary_size2, logits_size_tr, prefix='item')

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
  assert(ind == logits_size_tr), ' %d vs. %d' %(ind, logits_size_tr)
  
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

