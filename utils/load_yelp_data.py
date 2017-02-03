from pandatools import load_csv, build_index
from os.path import join
import cPickle as pickle

DIR = '../raw_data/yelp/'

def load_interactions():
    filename = join(DIR, 'reviews_filtered.csv')
    values, columns = load_csv(filename, '\t')
    return values, columns

def load_user():
    filename = join(DIR, 'users_filtered.csv')
    values, columns = load_csv(filename, '\t')
    user_index = build_index(values)
    return values, columns, user_index

def load_item():
    filename = 'items_filtered.csv'
    filename = join(DIR, filename)
    movies, columns = load_csv(filename, '\t')
    index = build_index(movies)
    return movies, columns, index

def load_meta():
    filename = join(DIR, 'filtered_meta.csv')
    meta = load_csv(filename, '\t', types = 0)
    t20 = meta['last 20 percent'][0]
    t10 = meta['last 10 percent'][0]
    return t20, t10

def load_count_view_mapping():
    filename = join(DIR, 'business_review_count.pkl')
    with open(filename, 'rb') as f:
        m = pickle.load(f)
    return m
