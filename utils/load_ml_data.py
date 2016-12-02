from pandatools import pd, np, load_csv, write_csv, build_index
from os.path import join
import socket


NAME = socket.gethostname()
if NAME == 'Kuans-MacBook-Pro.local':
    DIR = '/Users/Kuan/Project/recsys/dataset/ml-20m/'
elif NAME.startswith("hpc"):
    DIR = ''
else:
    DIR = ''

def load_interactions():
    '''
    timestamps range
    789652004.0,  
    1995-01-09 03:46:44

    1427784002.0,
    2015-03-30 23:40:02
    '''
    filename = join(DIR, 'ratings.csv')
    values, columns = load_csv(filename, ',')
    return values, columns

def load_user():
    N = 138493 # 1 to 138493
    users = np.reshape(1 + np.array(range(N)), (N,1))
    user_index = build_index(users)
    return users, ['id'], user_index

def load_movie0():
    M = 131262
    items = np.reshape(1 + np.array(range(M)) , (M, 1))
    item_index = build_index(items)
    return items, ['id'], item_index



