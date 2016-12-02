from pandatools import pd, np, load_csv, write_csv, build_index
import socket


NAME = socket.gethostname()
if NAME == 'Kuans-MacBook-Pro.local':
    DIR = '/Users/Kuan/Project/vertex_nomination/data/recsys16/'
elif NAME.startswith("hpc"):
    DIR = '/home/nlg-05/xingshi/kuan/recsys16/data/recsys16/'
else:
    DIR = '/nfs/isicvlnas01/users/liukuan/vertex_nomination/data/recsys16/'


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

