from load_ml_data import *
from os.path import join, isfile
import cPickle as pickle
from preprocess import *
import attribute


def interact_split(interact, user_index, item_index):
  l = len(interact)
  l = l/20
  interact_tr = np.zeros((l, 4), dtype=int)
  interact_va = np.zeros((l, 4), dtype=int)
  interact_te = np.zeros((l, 4), dtype=int)
  ind1, ind2, ind3 = 0,0,0

  
  ints = {}
  for i in range(l):
    uid, iid, irating, t = interact[i, :]
    if irating < 4:
      continue
    if uid not in ints:
      ints[uid] = []
    ints[uid].append((iid, t, irating))
  for u, v in ints.items():
    if len(v) < 10:
      continue
    v = sorted(v, key=lambda tup: tup[1])
    l0 = len(v)
    for j in range(l0):
      val = (user_index[u], item_index[v[j][0]], v[j][2], v[j][1])
      if j < 3 * l0 / 5:
        interact_tr[ind1, :] = val
        ind1 += 1
      elif j < 4 * l0 / 5:
        interact_va[ind2, :] = val
        ind2 += 1
      else:
        interact_te[ind3, :] = val
        ind3 += 1

  interact_tr = interact_tr[:ind1, :]
  interact_va = interact_va[:ind2, :]
  interact_te = interact_te[:ind3, :]
  print("train, valid, and test sizes %d/%d/%d" %(ind1, ind2, ind3))
  return interact_tr, interact_va, interact_te

def data_read(data_dir, _submit=0, ta=1, max_vocabulary_size=50000, 
  max_vocabulary_size2=50000, logits_size_tr=50000):
  data_filename = join(data_dir, 'recsys_file')
  if isfile(data_filename):
    print("recsys exists, loading")
    (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
      logit_ind2item_ind) = pickle.load(open(data_filename, 'rb'))
    return (data_tr, data_va, u_attributes, i_attributes, item_ind2logit_ind, 
    logit_ind2item_ind)

  if not path.isdir(data_dir):
    mkdir(data_dir)
    
  users, user_feature_names, user_index = load_user()
  items, item_feature_names, item_index = load_movie0()

  N = len(users)
  M = len(items)

  interact, names = load_interactions()
  print("loading interactions completed.")

  interact_tr, interact_va, interact_te = interact_split(interact, user_index, item_index)

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
  ## TODO
  user_feature_types = [0]
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
  ## TODO
  item_feature_types = [0]
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



# T20110314 = 1300157602
# T20130322 = 1363970802
# T20150330 = 1427784003

# def interact_split(interact, user_index, item_index):
#   l = len(interact)
#   interact_tr = np.zeros((l, 4), dtype=int)
#   interact_va = np.zeros((l, 4), dtype=int)
#   interact_te = np.zeros((l, 4), dtype=int)
#   ind1, ind2, ind3 = 0,0,0
#   c1, c2 = 0,0
#   for i in range(l/10):
#     if i % 1000000 == 0:
#       print("finished %d" % i)
#     uid, iid, irating, t = interact[i, :]
#     if iid not in item_index:
#       c1 += 1
#       continue
#     if uid not in user_index:
#       c2 += 1
#       continue
#     if irating < 4:
#       continue
#     if t <= T20110314:
#       interact_tr[ind1, :] = (user_index[uid], item_index[iid], irating, t)
#       ind1 += 1
#     elif t <= T20130322:
#       interact_va[ind2, :] = (user_index[uid], item_index[iid], irating, t)
#       ind2 += 1
#     elif t <= T20150330:
#       interact_te[ind3, :] = (user_index[uid], item_index[iid], irating, t)
#       ind3 += 1
#     else:
#       exit(-1)
#   interact_tr = interact_tr[:ind1, :]
#   interact_va = interact_va[:ind2, :]
#   interact_te = interact_te[:ind3, :]
#   print("train, valid, and test sizes %d/%d/%d" %(ind1, ind2, ind3))
#   print("oov users/items %d/%d" %(c2, c1))
#   return interact_tr, interact_va, interact_te
