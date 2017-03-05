
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class Attributes(object):
  def __init__(self, num_feature_cat=0, feature_cat=None,
               num_text_feat=0, feature_mulhot=None, mulhot_max_length=None, 
               mulhot_starts=None, mulhot_lengths=None, 
               v_sizes_cat=None, v_sizes_mulhot=None, 
               embedding_size_list_cat=None):
    self.num_features_cat = num_feature_cat
    self.num_features_mulhot = num_text_feat
    self.features_cat = feature_cat
    self.features_mulhot = feature_mulhot
    # self.mulhot_max_length = mulhot_max_length
    self.mulhot_starts = mulhot_starts
    self.mulhot_lengths = mulhot_lengths
    self._embedding_classes_list_cat = v_sizes_cat
    self._embedding_classes_list_mulhot = v_sizes_mulhot
    return 
  
  def set_model_size(self, sizes, opt=0):
    if isinstance(sizes, list):
      if opt == 0:
        assert(len(sizes) == self.num_features_cat)
        self._embedding_size_list_cat = sizes
      else:
        assert(len(sizes) == self.num_features_mulhot)
        self._embedding_size_list_mulhot = sizes
    elif isinstance(sizes, int):
      self._embedding_size_list_cat = [sizes] * self.num_features_cat
      self._embedding_size_list_mulhot = [sizes] * self.num_features_mulhot
    else:
      print('error: sizes need to be list or int')
      exit(0)
    return
  
  def set_target_prediction(self, features_cat_tr, full_values_tr, 
    full_segids_tr, full_lengths_tr):
    # TODO: move these indices outside this class
    self.full_cat_tr = features_cat_tr
    self.full_values_tr = full_values_tr
    self.full_segids_tr = full_segids_tr
    self.full_lengths_tr = full_lengths_tr
    return

  # def get_item_last_index(self):
  #   return len(self.features_cat[0]) - 1

  def overview(self, out=None):
    def p(val):
      if out:
        out(val)
      else:
        print(val)
    p('# of categorical attributes: {}'.format(self.num_features_cat))
    p('# of multi-hot   attributes: {}'.format(self.num_features_mulhot))
    p('====attributes values===')
    if self.num_features_cat > 0:
      p('\tinput categorical:')
      p('\t{}'.format(self.features_cat))
      if hasattr(self, 'full_cat_tr'):
        p('\toutput categorical:')
        p('\t{}'.format(self.full_cat_tr))
    if self.num_features_mulhot > 0:
      p('\tinput multi-hot:')
      p('\t values: {}'.format(self.features_mulhot))
      p('\t starts:{}'.format(self.mulhot_starts))
      p('\t length:{}'.format(self.mulhot_lengths))  
      if hasattr(self, 'full_values_tr'):
        p('\toutput multi-hot:')
        p('\t values:{}'.format(self.full_values_tr))
        p('\t starts:{}'.format(self.full_segids_tr))
        p('\t length:{}'.format(self.full_lengths_tr))
    p('\n')
