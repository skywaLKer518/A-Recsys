from pandatools import write_csv, load_csv, pd
from os.path import join


def load_submit(sub_id):
    SUBMIT_DIR = '../submissions/'
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
            x[key] = []
    return x

def format_submit(X, sub_id):
    '''
    save recommendation result to submission file
    input:
        X : dict. Ex: X[1400] = 1232,1123,5325
        sub_id: submission id
    '''
    SUBMIT_DIR = '../submissions/'

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


def combine_sub(r1, r2, opt = 0, old=False, users=None):
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