from eval_metrics import metrics

class Evaluation(object):
    def __init__(self, logit_ind2item_ind, res_filename='../submissions/ml_res_T', 
        hist_filename = '../submissions/ml_historical_train', ta=1, old=False, data_dir=None, test=False):
        return

    def gen_rec(self, rec, recommend_new=False):
        R = {}
        N = self.get_user_n()
        if not recommend_new:
            for i in xrange(N):
                uid = self.Uatt[self.Uinds[i], 0]
                R[uid] = [self.Iatt[self.logit_ind2item_ind[logid], 0] for logid in list(rec[i, :])]
        else:       
            for i in xrange(N):
                uid = self.Uatt[self.Uinds[i], 0]
                R[uid] = [self.Iatt[logid, 0] for logid in list(rec[i, :])]
        for k, v in R.items():
            R[k] = ','.join(str(xx) for xx in v)
        return R

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
        self.s1 = scores(rec, self.T)

        r_combine = self.combine_sub(self.hist, rec, old=self.old, users = self.Uatt)
        self.s2 = scores(r_combine, self.T)

        r_ex = self.combine_sub(self.hist, rec, 1, old=self.old, users = self.Uatt)
        self.s3 = scores(r_ex, self.T)

        result = metrics(rec, self.T)
        print(result)
        l = result.values()
        self.s4 = [item for sublist in l for item in sublist]
        l = metrics(r_ex, self.T).values()
        self.s5 = [item for sublist in l for item in sublist]
        return

    def get_scores(self):
        return self.s1, self.s2, self.s3, self.s4, self.s5

def scores(X, T, opt = 0):
    '''
    X, T: dict   user--> list of recommended items
    '''
    pk = precisionAtK
    score = 0.0
    P2, P4, P6, P20, P30, P100, R30, R100, US = 0,0,0,0,0,0,0,0,0
    R1000 = 0
    l1, l2 = len(X), len(T)
    print 'length of X/T: %i, %i' % (len(X), len(T))
    if opt == 1:
        user_scores = {}
    
    for _, u in enumerate(T):
        t = T[u]
        if u not in X:
            continue
        r = X[u]
        p2,p4,p6, = pk(r, t, 2), pk(r, t, 4), pk(r, t, 6)
        p20,p30,p100 = pk(r, t, 20), pk(r, t, 30),pk(r, t, 100)
        r30, r100 = recall(r[:30], t), recall(r[:100], t)
        r1000 = recall(r, t)
        u_s = userSuccess(r, t)
        s = (20.0 * (p2 + p4 + r30 + u_s) + 10.0 * (p6 + p20))
        assert( s >= 0)
        score += s
        if opt == 1:
            user_scores[u] = s
        P2 += p2
        P4 += p4
        P6 += p6
        P20 += p20
        P30 += p30
        P100 += p100
        R30 += r30
        R100 += r100
        R1000+= r1000
        US += u_s
    n_users = float(len(T))
    P2 = P2 / n_users
    P4 = P4 / n_users
    P6 = P6 / n_users
    P20 = P20 / n_users
    P30 = P30 / n_users
    R30 = R30 / n_users

    res = (score, P2, P4, R30, US, P6, P20, P30, P100,  R100, R1000)
    print_md('res ', res)
    if opt == 1:
        return user_scores
    return score, P2, P4, R30, US, P6, P20, P30, P100,  R100

def print_md(name, res):
    l = len(res)
    print '| %s | |' % name,
    for i in range(l):
        print ' %.2f |' % res[i],
    print '\n'
    return

def precisionAtK(r, t, k):
    # TODO: dealing with the case len(r) < k -- necessary? how
    topK = r[0:k]
    correct = set(topK).intersection(t)
    return 1.0 * len(correct) / k

def precisionAtK_new(r, t, k):
    topK = r[0:k]
    correct = set(topK).intersection(t)
    return 1.0 * len(correct) / min(k, len(r))

def recall(r, t):
    if len(t) > 0:
        return 1.0 * len(set(r).intersection(t)) / len(t)
        # return 1.0 * len(set(r[0:min(30, len(r))]).intersection(t)) / len(t)
    else:
        return 0.0

def userSuccess(r, t):
    if len(set(r[0:min(30, len(r))]).intersection(t)) > 0:
        return 1.0
    else:
        return 0.0

def print_scores(name_):
    r = load_submit(name_)
    # T = load_submit('../submissions/ml_res_T.csv')
    score, P2, P4, R30, US, P6, P20, P30, P100,  R100 = scores(r, self.T)
    print 'final score is %.3f' % score

    return
