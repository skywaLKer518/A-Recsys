from load_ml100k_data import load_movie, load_user, load_interactions
from ml100k_data import interact_split, process_items
from ml100k_submit import *
from scipy.sparse import lil_matrix, csr_matrix

def gen_label_data(debug = 0):
  interact, names = load_interactions()

  interact_tr, interact_va, interact_te = interact_split(interact, orig=True, debug=debug)

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


  format_submit(seq_tr, 'ml100k_historical_train.csv')
  format_submit(seq_va, 'ml100k_res_T.csv')
  format_submit(seq_te, 'ml100k_res_T_test.csv')

def extract_user_features(users, scale=1):
  feature_tokens = {}
  feat2ind = {}
  subsets1 = [0]
  user_f = {0:0}
  V = len(users)
  print('V = {}'.format(V))
  for j in range(len(user_f)):
    feature_tokens[j] = {}
    feat2ind[j] = {}
  user_unknown = 0
  n, ind = users.shape[0], 0
  A = lil_matrix((n,V))

  for i in range(n):
    j = 0
    lv = users[i][j]
    t = user_f[j]
    v = lv
    if v in feature_tokens[t]:
      A[i, feat2ind[t][v]] = 1
      feature_tokens[t][v] += 1
    else:
      feature_tokens[t][v] = 1
      feat2ind[t][v] = ind
      A[i, ind] = 1
      ind += 1
  print('ind = {}'.format(ind))
  return csr_matrix(A)

def extract_item_features(users, scale=1):
    feature_tokens = {}
    feat2ind = {}
    subsets1 = [0, 1]
    user_f = {0: 0, 1:1}

    V = len(users) + 2773
    print('V = {}'.format(V))
    for j in range(len(user_f)):
        feature_tokens[j] = {}
        feat2ind[j] = {}

    user_unknown = 0
    n, ind = users.shape[0], 0
    A = lil_matrix((n, V))
    for i in range(n):
        for j in subsets1:
            lv = users[i][j]
            t = user_f[j]

            if isinstance(lv, list):
                if scale == 1:
                    ll = 1.0 * len(lv)
                else:
                    ll = 1.0
                for v in lv:
                    if v in feature_tokens[t]:
                        A[i, feat2ind[t][v]] = 1.0 / ll
                        feature_tokens[t][v] += 1
                    else:
                        feature_tokens[t][v] = 1
                        feat2ind[t][v] = ind
                        A[i, ind] = 1.0 / ll 
                        ind += 1
            else:
                v = lv
                if v in feature_tokens[t]:
                    A[i, feat2ind[t][v]] = 1
                    feature_tokens[t][v] += 1
                else:
                    feature_tokens[t][v] = 1
                    feat2ind[t][v] = ind
                    A[i, ind] = 1
                    ind += 1

    print 'ind = ', ind
    return csr_matrix(A)


def gen_features_lightfm():
  users, user_feature_names, user_index = load_user()
  items, item_feature_names, item_index = load_movie()
  items = process_items(items)
  itemf = extract_item_features(items)
  uf = extract_user_features(users)
  import numpy
  numpy.save('../dataset/ml-100k/ml_features', (uf, itemf))


gen_label_data(0)
