from preprocess import create_dictionary, tokenize_attribute_map, filter_cat, filter_mulhot, pickle_save
import numpy as np
import attribute


class Comb_Attributes(object):
  def __init__(self, data_dir, max_vocabulary_size=50000, logits_size_tr=50000, 
    threshold=2):
    self.data_dir = data_dir
    self.self.max_vocabulary_size = self.max_vocabulary_size
    self.logits_size_tr = logits_size_tr
    self.threshold = threshold
    return

  def get_attributes(self, users, items, data_tr, user_features, item_features):

    # create_dictionary
    user_feature_names, user_feature_types = user_features
    item_feature_names, item_feature_types = item_features
    
    # user_feature_types = [0, 0, 0, 0]   # TODO # userU
    u_inds = [p[0] for p in data_tr]
    create_dictionary(self.data_dir, u_inds, users, user_feature_types, 
      user_feature_names, self.max_vocabulary_size, self.logits_size_tr, 
      prefix='user', threshold=self.threshold)

    # create user feature map
    (num_features_cat, features_cat, num_features_mulhot, features_mulhot,
      mulhot_max_leng, mulhot_starts, mulhot_lengs, v_sizes_cat, 
      v_sizes_mulhot) = tokenize_attribute_map(self.data_dir, users, user_feature_types, 
      self.max_vocabulary_size, self.logits_size_tr, prefix='user')

    u_attributes = attribute.Attributes(num_features_cat, features_cat, 
      num_features_mulhot, features_mulhot, mulhot_max_leng, mulhot_starts, 
      mulhot_lengs, v_sizes_cat, v_sizes_mulhot)

    # create_dictionary
    # item_feature_types = [0, 1, 1]

    i_inds = [p[1] for p in data_tr]
    create_dictionary(self.data_dir, i_inds, items, item_feature_types, 
      item_feature_names, self.max_vocabulary_size, self.logits_size_tr, 
      prefix='item', threshold=self.threshold)

    # create item feature map
    items_cp = np.copy(items)
    (num_features_cat2, features_cat2, num_features_mulhot2, features_mulhot2,
      mulhot_max_leng2, mulhot_starts2, mulhot_lengs2, v_sizes_cat2, 
      v_sizes_mulhot2) = tokenize_attribute_map(self.data_dir, 
      items_cp, item_feature_types, self.max_vocabulary_size, self.logits_size_tr, 
      prefix='item')

    item_ind2logit_ind = {}
    item2fea0 = features_cat2[0]
    ind = 0
    for i in range(len(items)):
      fea0 = item2fea0[i]
      if fea0 != 0:
        item_ind2logit_ind[i] = ind
        ind += 1
    assert(ind == self.logits_size_tr), 'Item_vocab_size %d too large! need to be no greater than %d\nFix --item_vocab_size [item_vocab_size]\n' % (self.logits_size_tr, ind)
    
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
      full_lengths_tr) = filter_mulhot(self.data_dir, items, 
      item_feature_types, self.max_vocabulary_size, logit_ind2item_ind, 
      prefix='item')

    i_attributes.set_target_prediction(features_cat2_tr, full_values_tr, 
      full_segids_tr, full_lengths_tr)

    return u_attributes, i_attributes, item_ind2logit_ind, logit_ind2item_ind


class HET(Comb_Attributes):
  def __init__(self, data_dir, max_vocabulary_size=50000, logits_size_tr=50000, 
    threshold=2):
    self.data_dir = data_dir
    self.max_vocabulary_size = max_vocabulary_size
    self.logits_size_tr = logits_size_tr
    self.threshold = threshold  
    return

class MIX(Comb_Attributes):
  def __init__(self):
    return
