import numpy as np
import pandas as pd
import time
from scipy.sparse import coo_matrix
import socket


NAME = socket.gethostname()
if NAME == 'Kuans-MacBook-Pro.local':
    DIR = '/Users/Kuan/Project/vertex_nomination/data/recsys16/'
elif NAME.startswith("hpc"):
    DIR = '/home/nlg-05/xingshi/kuan/recsys16/data/recsys16/'
else:
    DIR = '/nfs/isicvlnas01/users/liukuan/vertex_nomination/data/recsys16/'

def load_csv(filename, types = 1):
    data = pd.read_csv(filename, delimiter='\t')
    if types == 0:
        return data
    columns = list(data.columns)
    values = data.values
    return values, columns

def write_csv(x, filename, header, columns=None):
    x.to_csv(path_or_buf=filename, sep='\t',
        index=False, header = header, columns = columns)
    return

def build_index(values, opt = 1):
    count, index = 0, {}
    for v in values:
        if opt == 1:
            index[v[0]] = count
        elif opt == 0:
            index[v] = count
        count += 1
    return index

def load_user_csv():
    filename = DIR + 'users.csv'
    users, user_feature_names = load_csv(filename)
    user_index = build_index(users)
    return users, user_feature_names, user_index

def load_user_target_csv():
    filename = DIR + 'users_target_feat.csv'
    items, item_feature_names = load_csv(filename)
    item_index = build_index(items)
    return items, item_feature_names, item_index

def load_item_csv():
    filename = DIR + 'items.csv'
    items, item_feature_names = load_csv(filename)
    item_index = build_index(items)
    return items, item_feature_names, item_index

def load_item_active_csv():
    filename = DIR + 'items_active.csv'
    items, item_feature_names = load_csv(filename)
    item_index = build_index(items)
    return items, item_feature_names, item_index

def load_interactions(submit=1, ta = 1):
    if submit == 1 and ta == 1:
        filename = DIR + 'interactions_ta2.csv'
    elif submit == 1 and ta == 0:
        filename = DIR + 'interactions2.csv' # 'interactions_week.csv'
    elif submit == 0 and ta == 1:
        filename = DIR + 'interactions_ta_train2.csv'
    elif submit == 0 and ta == 0:
        filename = DIR + 'interactions_train2.csv' ## todo
    elif submit == 2 and ta == 1:
        filename = DIR + 'interactions_ta_valid2.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_impressions(submit=1, ta = 1):
    if submit == 1 and ta == 1:
        filename = DIR + 'impressions_ta2.csv'
    elif submit == 1 and ta == 0:
        filename = DIR + 'impressions_c2.csv'
    elif submit == 0 and ta == 1:
        filename = DIR + 'impressions_ta_train2.csv'
    else:
        filename = DIR + 'impressions_train2.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_target_users():
    # only the name list... no profile info
    filename = DIR + 'target_users.csv'
    values, columns = load_csv(filename)
    return values, columns

def save_impression_time():
    '''
    ignore 'items' in impressions.csv
    to speed up when we do no need 'items'
    '''
    impr, _ = load_impressions_csv()
    columns_filter = [0, 1, 2]
    header = ['user_id', 'year', 'week']
    filename = DIR + 'impressions_time.csv'
    x = pd.DataFrame(impr)
    write_csv(x, filename, header, columns_filter)
    return

def save_user_target():
    users, header, _ = load_user_csv()
    tu, _ = load_target_users()
    tu = set(tu.reshape(tu.shape[0],))
    ind = [i for i in range(users.shape[0]) if users[i][0] in tu]
    users = users[ind, :]
    x = pd.DataFrame(users)
    filename = DIR + 'users_target_feat.csv'
    write_csv(x, filename, header)
    return

def save_item_active():
    items, header, _ = load_item_csv()
    items_active = items[np.where(items[:,12]>0)]
    x = pd.DataFrame(items_active)
    filename = DIR + 'items_active.csv'
    write_csv(x, filename, header)
    return

def matrix2tsv(M, filename):
    x = pd.DataFrame(M)
    write_csv(x, filename, None)
    return

def dict2tsv(D, filename):
    l, ind = 0, 0
    for k in D:
        l += len(D[k])
    x = np.zeros((l, 3), dtype=object)

    for k in D:
        for p in D[k]:
            x[ind, :] = (k, p, D[k][p])
            ind += 1
    assert(ind == l)
    x = pd.DataFrame(x)
    write_csv(x, filename, None)
    return

def sparse2tsv(M, filename):
    '''
    from sparse matrix 2 tsv file
    '''
    M = M.todok()
    pos = np.array(M.keys())
    val = np.array(M.values()).reshape(pos.shape[0],1)
    x = np.append(pos, val, 1)
    x = x.astype(int)
    x = pd.DataFrame(x)
    write_csv(x, filename, None)
    return

def sparse2tsv2(A, filename, matlab):
    a, b = A.nonzero()
    d = [i for sublist in A.data for i in sublist]
    l = len(d)
    A2 = np.zeros((l, 3), dtype=int)
    for i in range(l):
        A2[i, 0] = a[i]+matlab
        A2[i, 1] = b[i]+matlab
        A2[i, 2] = d[i]
    matrix2tsv(A2, filename)

def tsv2dict(filename):
    x = pd.read_csv(filename, delimiter='\t', header=None).values
    D = {}
    assert(x.shape[1] == 3)
    for i in range(len(x)):
        uid, iid, sim = x[i,:]
        uid, iid = int(uid), int(iid)
        if uid not in D:
            D[uid] = {}
        D[uid][iid] = sim
    return D

def tsv2matrix(filename, opt=1):
    x = pd.read_csv(filename, delimiter='\t', header=None).values
    if opt == 0:
        return x
    d = int(x[:,1].max() + 1)
    n = int(x[:,0].max() + 1)
    p = coo_matrix( (x[:,2], (x[:,0], x[:,1])), shape=(n,d))
    p = p.todense()
    return p

def load_interactions_csv():
    filename = DIR + 'interactions.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_interactions_week():
    filename = DIR + 'interactions_week.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_interactions_ta():
    filename = DIR + 'interactions_ta2.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_impressions_csv():  # 60s to load
    filename = DIR + 'impressions.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_impressions_compact():
    filename = DIR + 'impressions_c2.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_impressions_target_active(opt = 0):
    filename = DIR + 'impressions_ta2.csv'
    if opt == 1: # train
        filename = DIR + 'impressions_ta_train2.csv'
    values, columns = load_csv(filename)
    return values, columns

def load_impressions_time():
    filename = DIR + 'impressions_time.csv'
    values, columns = load_csv(filename)
    return values, columns


if __name__ == '__main__':
    print time.time()
    #
    # users, user_feature_names, user_index = load_user_csv()
    # users, user_feature_names = load_csv(DIR + 'impressions.csv')
    print time.time()
    # print len(users)
    # print user_feature_names
    # print user_index[4]
    # print user_index[7]
    # print user_index[9]


    # print user_index[19]
    # print user_index[116]
    # print user_index[140]

    # print user_index.has_key(2)
    # load_user()
