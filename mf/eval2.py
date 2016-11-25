from submit import *

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
    res = (score, P2, P4, R30, US, P6, P20, P30, P100,  R100, R1000)
    print_md('res ', res)
    if opt == 1:
        return user_scores
    return score, P2, P4, R30, US, P6, P20, P30, P100,  R100

def print_md(name, res):
    l = len(res)
    print '| %s | |' % name,
    for i in range(l):
        print ' %.0f |' % res[i],
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
    T = load_submit('../submissions/res_T.csv')
    score, P2, P4, R30, US, P6, P20, P30, P100,  R100 = scores(r, T)
    print 'final score is %.3f' % score

    return
