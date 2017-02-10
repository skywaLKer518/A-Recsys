from evaluate import Evaluation
from submit import load_submit, combine_sub
from load_xing_data import load_user_target_csv, load_item_active_csv, load_user_csv
from os.path import join
import cPickle as pickle
from pandatools import build_index

class Evaluate(Evaluation):
    def __init__(self, logit_ind2item_ind, res_filename='../submissions/res_T', 
        hist_filename = '../submissions/historical_train', ta=1, old=None, 
        data_dir=None, test=False):
        self.logit_ind2item_ind = logit_ind2item_ind
        res_filename = res_filename + '_test.csv' if test else res_filename + '.csv'
        self.T = load_submit(res_filename)
        hist_filename = hist_filename + '_test.csv' if test else hist_filename + '.csv'
        self.hist = load_submit(hist_filename)
        self.Iatt, _, self.Iid2ind = load_item_active_csv()
        self.data_dir = data_dir

        if data_dir is not None:
            filename = 'processed_user' + '_ta_' + str(ta)
            self.Uatt = pickle.load(open(join(data_dir, filename), 'rb'))
            self.Uid2ind = build_index(self.Uatt)
        else:
            if ta == 1:
                self.Uatt, _, self.Uid2ind = load_user_target_csv()
            else:
                self.Uatt, _, self.Uid2ind = load_user_csv()

        self.Uids = self.get_uids()
        self.Uinds = [self.Uid2ind[v] for v in self.Uids]
        self.old = old
        self.combine_sub = combine_sub
        return

    def set_uinds(self, uinds):
        self.Uinds = uinds