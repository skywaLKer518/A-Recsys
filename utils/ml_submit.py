import pandas as pd
from load_ml_data import write_csv, load_csv, load_movie, load_user
from os.path import isfile
# import socket
# NAME = socket.gethostname()
# print NAME
# if NAME == 'Kuans-MacBook-Pro.local' or NAME == 'Kuans-MBP':
#     SUBMIT_DIR = '/Users/Kuan/Project/recsys/submissions/'
# elif NAME.startswith("hpc"):
#     SUBMIT_DIR = '/home/nlg-05/xingshi/kuan/recsys/submissions/'    
# else:
#     SUBMIT_DIR = '/nfs/isicvlnas01/users/liukuan/recsys/submissions/'

SUBMIT_DIR = '../submissions/'


def format_submit(X, sub_id):
    '''
    save recommendation result to submission file
    input:
        X : dict. Ex: X[1400] = 1232,1123,5325
        sub_id: submission id
    '''
    header = ['user_id', 'items']
    for pos, key in enumerate(X):
        l = X[key]
        if isinstance(l, list):
            X[key] = ','.join(str(xx) for xx in l)
        else:
            print 'not a list. No need to convert.'
            break
    x = pd.DataFrame(X.items())
    write_csv(x, SUBMIT_DIR+sub_id, header)
    return

def load_submit(sub_id):

    # filename = SUBMIT_DIR+'res_'+str(sub_id)+'.csv'
    # if not isfile(filename):
    #     filename = SUBMIT_DIR2+'res_'+str(sub_id)+'.csv'
    filename = SUBMIT_DIR + sub_id
    data = load_csv(filename, types=0)
    x = data.set_index('user_id').to_dict()['items']
    for _, key in enumerate(x):
        l = x[key]
        if isinstance(l, str):
            x[key] = l.split(',')
        elif isinstance(l, int):
            x[key] = [str(l)]
        else:
            # print 'not a str, break.'
            x[key] = []
            # break
    return x

# def load_hist_interact_ta():
#     filename = SUBMIT_DIR + '../submissions/submitted/res_HISTORY_0324_2.csv'
#     his, _ = load_csv(filename)
#     _, _, user_index = load_user_target_csv()
#     _, _, item_index = load_item_active_csv()

#     A = lil_matrix((NUM_TARGET_USERS, NUM_ACTIVE_ITEMS))
#     for i in range(len(his)):
#         l = his[i, 1].split(',')
#         for p in l:
#             A[user_index[his[i, 0]], item_index[int(p)]] = 4
#     return A

# def load_hist_impress_ta():
#     filename = SUBMIT_DIR + '../submissions/submitted/res_IMPRESSION.csv'
#     his, _ = load_csv(filename)
#     _, _, user_index = load_user_target_csv()
#     _, _, item_index = load_item_active_csv()

#     A = lil_matrix((NUM_TARGET_USERS, NUM_ACTIVE_ITEMS))
#     for i in range(len(his)):
#         l = his[i, 1].split(',')
#         for p in l:
#             A[user_index[his[i, 0]], item_index[int(p)]] = 3
#     return A

def combine_sub(r1, r2, opt = 0, old=False):
    users, user_feature_names, user_index = load_user()
    items, item_header, item_index = load_movie(20, old=old)
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

