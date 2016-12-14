from load_ml_data import load_movie, load_user, load_interactions
from ml_data import interact_split
from ml_submit import *


interact, names = load_interactions()

interact_tr, interact_va, interact_te = interact_split(interact, orig=True, debug=1)

data_tr = zip(list(interact_tr[:, 0]), list(interact_tr[:, 1]), list(interact_tr[:, 3]))
data_va = zip(list(interact_va[:, 0]), list(interact_va[:, 1]), list(interact_va[:, 3]))
data_te = zip(list(interact_te[:, 0]), list(interact_te[:, 1]), list(interact_te[:, 3]))

seq_tr, seq_va, seq_te = {}, {}, {}

for u, i , t in data_tr:
  if u not in seq_tr:
    seq_tr[u] = []
  seq_tr[u].append((i, t))

for u, i , t in data_va:
  if u not in seq_va:
    seq_va[u] = []
  seq_va[u].append(i)

for u, i , t in data_te:
  if u not in seq_te:
    seq_te[u] = []
  seq_te[u].append(i)

for u, v in seq_tr.items():
  l = sorted(v, key = lambda x:x[1], reverse=True)
  seq_tr[u] = ','.join([str(p[0]) for p in l])

for u, v in seq_va.items():
  seq_va[u] = ','.join(str(p) for p in seq_va[u])

for u, v in seq_te.items():
  seq_te[u] = ','.join(str(p) for p in seq_te[u])


format_submit(seq_tr, 'ml_historical_train.csv')
format_submit(seq_va, 'ml_res_T.csv')
format_submit(seq_te, 'ml_res_T_test.csv')