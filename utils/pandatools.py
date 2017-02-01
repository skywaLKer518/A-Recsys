import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix

def load_csv(filename, sep = '\t', types = 1, header=0):
    data = pd.read_csv(filename, delimiter=sep, header=header)
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
