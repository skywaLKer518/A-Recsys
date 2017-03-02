from preprocess import create_dictionary, create_dictionary_mix, tokenize_attribute_map, filter_cat, filter_mulhot, pickle_save
import numpy as np
import attribute


class Comb_Attributes(object):
  def __init__(self):
    return 

  def get_attributes(self, users, items, data_tr, user_features, item_features):
    # create_dictionary
    user_feature_names, user_feature_types = user_features
    item_feature_names, item_feature_types = item_features
    
    u_inds = [p[0] for p in data_tr]
    self.create_dictionary(self.data_dir, u_inds, users, user_feature_types, 
      user_feature_names, self.max_vocabulary_size, self.logits_size_tr, 
      prefix='user', threshold=self.threshold)

    # create user feature map
    (num_features_cat, features_cat, num_features_mulhot, features_mulhot,
      mulhot_max_leng, mulhot_starts, mulhot_lengs, v_sizes_cat, 
      v_sizes_mulhot) = tokenize_attribute_map(self.data_dir, users, 
      user_feature_types, self.max_vocabulary_size, self.logits_size_tr, 
      prefix='user')

    u_attributes = attribute.Attributes(num_features_cat, features_cat, 
      num_features_mulhot, features_mulhot, mulhot_max_leng, mulhot_starts, 
      mulhot_lengs, v_sizes_cat, v_sizes_mulhot)

    # create_dictionary
    i_inds_tr = [p[1] for p in data_tr]
    self.create_dictionary(self.data_dir, i_inds_tr, items, item_feature_types, 
      item_feature_names, self.max_vocabulary_size, self.logits_size_tr, 
      prefix='item', threshold=self.threshold)

    # create item feature map
    items_cp = np.copy(items)
    (num_features_cat2, features_cat2, num_features_mulhot2, features_mulhot2,
      mulhot_max_leng2, mulhot_starts2, mulhot_lengs2, v_sizes_cat2, 
      v_sizes_mulhot2) = tokenize_attribute_map(self.data_dir, 
      items_cp, item_feature_types, self.max_vocabulary_size, self.logits_size_tr, 
      prefix='item')

    '''
    create an (item-index <--> classification output) mapping
    there are more than one valid mapping as long as 1 to 1 
    '''
    item2fea0 = features_cat2[0] if len(features_cat2) > 0 else None
    item_ind2logit_ind, logit_ind2item_ind = self.index_mapping(item2fea0, 
      i_inds_tr, len(items))

    i_attributes = attribute.Attributes(num_features_cat2, features_cat2, 
      num_features_mulhot2, features_mulhot2, mulhot_max_leng2, mulhot_starts2, 
      mulhot_lengs2, v_sizes_cat2, v_sizes_mulhot2)

    # set target prediction indices
    features_cat2_tr = filter_cat(num_features_cat2, features_cat2, 
      logit_ind2item_ind)

    (full_values, full_values_tr, full_segids, full_lengths, full_segids_tr, 
      full_lengths_tr) = filter_mulhot(self.data_dir, items, 
      item_feature_types, self.max_vocabulary_size, logit_ind2item_ind, 
      prefix='item')

    i_attributes.set_target_prediction(features_cat2_tr, full_values_tr, 
      full_segids_tr, full_lengths_tr)

    return u_attributes, i_attributes, item_ind2logit_ind, logit_ind2item_ind


class MIX(Comb_Attributes):

  def __init__(self, data_dir, max_vocabulary_size=500000, logits_size_tr=50000, 
    threshold=2):
    self.data_dir = data_dir
    self.max_vocabulary_size = max_vocabulary_size
    self.logits_size_tr = logits_size_tr
    self.threshold = threshold
    self.create_dictionary = create_dictionary_mix
    return

  def index_mapping(self, item2fea0, i_inds, M=None):
    item_ind2logit_ind = {}
    logit_ind2item_ind = {}

    item_ind_count = {}
    for i_ind in i_inds:
      item_ind_count[i_ind] = item_ind_count[i_ind] + 1 if i_ind in item_ind_count else 1
    ind_list = sorted(item_ind_count, key=item_ind_count.get, reverse=True)
    assert(self.logits_size_tr <= len(ind_list)), 'Item_vocab_size should be smaller than # of appeared items'
    ind_list = ind_list[:self.logits_size_tr]

    for index, elem in enumerate(ind_list):
      item_ind2logit_ind[elem] = index
      logit_ind2item_ind[index] = elem

    return item_ind2logit_ind, logit_ind2item_ind

  def mix_attr(self, users, items, user_features, item_features):
    user_feature_names, user_feature_types = user_features
    item_feature_names, item_feature_types = item_features
    user_feature_names[0] = 'uid'

    # user
    n = len(users)
    users2  = np.zeros((n, 1), dtype=object)
    for i in range(n):
      v = []
      user = users[i, :]
      for j in range(len(user_feature_types)):
        t = user_feature_types[j]
        n = user_feature_names[j]
        if t == 0:
          v.append(n + str(user[j]))
        elif t == 1:
          v.extend([n + s for s in user[j].split(',')])
        else:
          continue
      users2[i, 0] = ','.join(v)

    # item
    n = len(items)
    items2  = np.zeros((n, 1), dtype=object)
    for i in range(n):
      v = []
      item = items[i, :]
      for j in range(len(item_feature_types)):
        t = item_feature_types[j]
        n = item_feature_names[j]
        if t == 0:
          v.append(n + str(item[j]))
        elif t == 1:
          v.extend([n + s for s in item[j].split(',')])
        else:
          continue
      items2[i, 0] = ','.join(v)

    # modify attribute names and types
    if len(user_feature_types) == 1 and user_feature_types[0] == 0:
      user_features = ([['mix'], [0]])
    else:
      user_features = ([['mix'], [1]])
    if len(item_feature_types) == 1 and item_feature_types[0] == 0:
      item_features = ([['mix'], [0]])
    else:
      item_features = ([['mix'], [1]])
    return users2, items2, user_features, item_features


class HET(Comb_Attributes):

  def __init__(self, data_dir, max_vocabulary_size=50000, logits_size_tr=50000, 
    threshold=2):
    self.data_dir = data_dir
    self.max_vocabulary_size = max_vocabulary_size
    self.logits_size_tr = logits_size_tr
    self.threshold = threshold
    self.create_dictionary = create_dictionary
    return

  def index_mapping(self, item2fea0, i_inds, M):
    item_ind2logit_ind = {}
    logit_ind2item_ind = {}
    ind = 0
    for i in range(M):
      fea0 = item2fea0[i]
      if fea0 != 0:
        item_ind2logit_ind[i] = ind
        ind += 1
    assert(ind == self.logits_size_tr), 'Item_vocab_size %d too large! need to be no greater than %d\nFix: --item_vocab_size [smaller item_vocab_size]\n' % (self.logits_size_tr, ind)
    
    logit_ind2item_ind = {}
    for k, v in item_ind2logit_ind.items():
      logit_ind2item_ind[v] = k
    return item_ind2logit_ind, logit_ind2item_ind

