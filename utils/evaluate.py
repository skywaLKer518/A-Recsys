from eval_metrics import metrics
from submit import load_submit, combine_sub
from load_data import load_users, load_items

class Evaluation(object):

    def __init__(self, raw_data_dir, test=False):

        res_filename = 'res_T_test.csv' if test else 'res_T.csv'
        self.T = load_submit(res_filename, submit_dir=raw_data_dir)
        hist_filename = 'historical_train_test.csv' if test else 'historical_train.csv'
        self.hist = load_submit(hist_filename, submit_dir=raw_data_dir)
        
        self.Iatt, _, self.Iid2ind = load_items(raw_data_dir)
        self.Uatt, _, self.Uid2ind = load_users(raw_data_dir)

        self.Uids = self.get_uids()
        self.Uinds = [self.Uid2ind[v] for v in self.Uids]
        self.combine_sub = combine_sub
        return

    def get_user_n(self):
        return len(self.Uinds)

    def get_uids(self):
        return list(self.T.keys())

    def get_uinds(self):
        return self.Uinds

    def eval_on(self, rec):
        
        self.res = rec

        tmp_filename = 'rec'
        for k, v in rec.items():
            rec[k] = v.split(',')

        r_ex = self.combine_sub(self.hist, rec, 1, users = self.Uatt)

        result = metrics(rec, self.T)
        l = result.values()
        self.s_self = [item for sublist in l for item in sublist]
        l = metrics(r_ex, self.T).values()
        self.s_ex = [item for sublist in l for item in sublist]
        return

    def get_scores(self):
        return self.s_self, self.s_ex


