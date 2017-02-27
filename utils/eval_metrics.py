import numpy as np

def metrics(X, T, Ns=[2,5,10,20,30], metrics=['prec', 'recall', 'map', 'ndcg']):
  n_users = float(len(T))
  N_pos = len(Ns)
  funcs = {'prec':PRECISION, 'recall':RECALL, 'map':MAP, 'ndcg':NDCG}
  res = {}
  for m in metrics:
    re = []
    for n in Ns:
      re.append(0.0)
    res[m] = re

  for u, t in T.items():
    t = set(t)
    if u not in X:
      continue
    pred = X[u]
    correct = [int(r in t) for r in pred] # correct or not at each position
    cumsum_x = np.cumsum(correct)
    for m in metrics:
      f = funcs[m]
      s = f(correct, t, Ns, cumsum_x)
      for i in range(N_pos):
        res[m][i] += s[i]
  for m in metrics:
    for i in range(N_pos):
      res[m][i] = res[m][i] / n_users
  return res

def PRECISION(X, T, Ns=[1,2,5,10,20,30], cumsum_x=None):
    ''' return PRECISION@N '''
    # assert(Ns[-1] <= len(X))
    l = len(cumsum_x)
    if l == 0:
      return [0.0 for n in Ns]
    
    return [cumsum_x[min(n-1, l-1)] * 1.0 / min(n,l)  for n in Ns]

def RECALL(X, T, Ns=[1,2,5,10,20,30], cumsum_x=None):
    ''' return RECALL@N '''
    # assert (len(T) > 0)
    n_t = len(T)
    l = len(cumsum_x)
    if l == 0:
      return [0.0 for n in Ns]

    return [cumsum_x[min(n-1, l-1)] * 1.0 / n_t for n in Ns]


def MAP(X, T, Ns=[1,2,5,10,20,30], cumsum_x=None):
    ''' return MAP@N N = 2, 5, 10, 20, 30 '''
    l = len(X)
    if l == 0:
      return [0.0 for n in Ns]
    n_t = len(T)
    ap_i = [X[i] * 1.0 * cumsum_x[i] / (i+1) for i in range(l)]
    ap = np.cumsum(ap_i)
    return [ap[min(n-1, l-1)] / min(min(n_t, n),l) for n in Ns]

N_max = 30
discounts = [1.0 / np.log2(2+n) for n in range(N_max)]
def NDCG(X, T, Ns=[1,2,5,10,20,30], cumsum_x=None):
    ''' return NDCG@N N = 2, 5, 10, 20, 30 '''
    l = len(X)
    if l == 0:
      return [0.0 for n in Ns]
    n_max = l if l < N_max else N_max
    disc = discounts[:n_max]
    DCGs = [x_d[0] * x_d[1] for x_d in zip(X, disc)]
    IDCGs = discounts[:len(T)] + [0.0] * (n_max - len(T))
    cumsum_DCG = np.cumsum(DCGs)
    cumsum_IDCG = np.cumsum(IDCGs)
    return [cumsum_DCG[min(n-1, l-1)] / cumsum_IDCG[min(n-1, l-1)] for n in Ns]


'''deprecated'''

def eval_P5(X, T, K=5):
  score = 0
  for uid in T:
    if uid not in X:
      continue
    t = set(T[uid])  
    pred = set(X[uid][:K])
    score += len(pred.intersection(t)) * 1.0 / len(pred)
  return score * 1.0 / len(T)

def eval_R20(X, T):
  '''
  actually it is the success rate of 20
  '''
  POS = 20
  success = 0
  for uid in T:
    if uid not in X:
      continue
    suc = 1 if len(set(X[uid][:POS]).intersection(T[uid])) > 0 else 0
    success += suc
  return 1.0 * success / (len(T))

