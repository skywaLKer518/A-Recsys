import pandas as pd
from load_xing_data import write_csv, load_csv, load_user_target_csv, load_item_active_csv
from os.path import isfile
from scipy.sparse import lil_matrix, csr_matrix
# import socket
# NAME = socket.gethostname()
# print NAME
# if NAME == 'Kuans-MacBook-Pro.local' or NAME == 'Kuans-MBP':
#     SUBMIT_DIR = '/Users/Kuan/Project/vertex_nomination/recsys16/src/'
# elif NAME.startswith("hpc"):
#     SUBMIT_DIR = '/home/nlg-05/xingshi/kuan/recsys/submissions/'    
# else:
#     SUBMIT_DIR = '/nfs/isicvlnas01/users/liukuan/vertex_nomination/recsys16/src/'
    
SUBMIT_DIR = '../submissions/'
NUM_TARGET_USERS = 150000
NUM_ACTIVE_ITEMS = 327003


def load_hist_interact_ta():
    filename = SUBMIT_DIR + '../submissions/submitted/res_HISTORY_0324_2.csv'
    his, _ = load_csv(filename)
    _, _, user_index = load_user_target_csv()
    _, _, item_index = load_item_active_csv()

    A = lil_matrix((NUM_TARGET_USERS, NUM_ACTIVE_ITEMS))
    for i in range(len(his)):
        l = his[i, 1].split(',')
        for p in l:
            A[user_index[his[i, 0]], item_index[int(p)]] = 4
    return A

def load_hist_impress_ta():
    filename = SUBMIT_DIR + '../submissions/submitted/res_IMPRESSION.csv'
    his, _ = load_csv(filename)
    _, _, user_index = load_user_target_csv()
    _, _, item_index = load_item_active_csv()

    A = lil_matrix((NUM_TARGET_USERS, NUM_ACTIVE_ITEMS))
    for i in range(len(his)):
        l = his[i, 1].split(',')
        for p in l:
            A[user_index[his[i, 0]], item_index[int(p)]] = 3
    return A

def combine_sub(r1, r2, opt = 0):
    users, _, _ = load_user_target_csv()
    rec = {}
    for i in range(len(users)):
        uid = users[i, 0]
        if uid not in r1 and uid not in r2:
            continue
        i_set = set()
        rec[uid] = []
        if uid in r1:
            for iid in r1[uid]:
                if iid not in i_set:
                    i_set.add(iid)
                    if opt == 0:
                        rec[uid].append(iid)
        if uid in r2:
            for iid in r2[uid]:
                if iid not in i_set:
                    i_set.add(iid)
                    rec[uid].append(iid)
    return rec

if __name__ == '__main__':
    print 'test'

