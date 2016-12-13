from pandatools import pd, np, load_csv, write_csv, build_index
from os.path import join
import socket


NAME = socket.gethostname()
if NAME == 'Kuans-MacBook-Pro.local':
    DIR = '/Users/Kuan/Project/recsys/dataset/ml-20m/'
elif NAME.startswith("hpc"):
    DIR = '/home/nlg-05/xingshi/lstm/tensorflow/recsys/data/data_ml_orig'
else:
    DIR = '/nfs/isicvlnas01/users/liukuan/recsys/dataset/ml-20m/'

def load_interactions():
    '''
    timestamps range
    789652004.0,  
    1995-01-09 03:46:44

    1427784002.0,
    2015-03-30 23:40:02
    '''
    filename = join(DIR, 'ratings_filtered.csv')
    values, columns = load_csv(filename, '\t')
    return values, columns

def filter_interactions():
    filename = join(DIR, 'ratings.csv')
    values, columns = load_csv(filename, ',')
    values2 = np.array(np.copy(values), dtype=int)
    c = 0
    for i in range(len(values)):
        r = float(values[i, 2])
        if r >= 4.0:
            values2[c, :] = values[i, :]
            c += 1
    values2 = values2[:c, :]
    write_csv(pd.DataFrame(values2), join(DIR, 'ratings_filtered.csv'), columns)

def load_user():
    filename = join(DIR, 'users.csv')
    values, columns = load_csv(filename, '\t')
    user_index = build_index(values)
    return values, columns, user_index

def load_movie(thresh=0.8):

    filename = 'movie_attributes' + '_' + str(thresh) + '.csv'
    filename = join(DIR, filename)

    movies, columns = load_csv(filename, '\t')
    index = build_index(movies)
    return movies, columns, index

def create_movie(thresh=0.8):
    filename0 = join(DIR, 'movies.csv')
    movies, _ = load_csv(filename0, ',')
    movie_ids = [int(mid) for mid in list(movies[:, 0])]

    filename = join(DIR, 'genome-scores.csv')
    values, columns = load_csv(filename, ',')
    print('create movie dict')
    movie = {}
    tags = set([])
    for i in range(len(values)):
        movie_id, tag_id, s = values[i, :]
        if s >= thresh:
            if movie_id not in movie:
                movie[movie_id] = set([])
            movie[movie_id].add(int(tag_id))
            tags.add(tag_id)

    print('create movie attributes')
    n_movies = len(movie_ids)
    values = np.zeros((n_movies, 2), dtype=object)
    l = []
    c = 0
    for mid in movie_ids:
        values[c, 0] = mid
        if mid in movie:
            t = list(movie[mid])
            values[c, 1] = t
            l.append(len(t))
        else:
            values[c, 1]= []
            l.append(0)
        c += 1
    print('c = {}'.format(c))
    assert(c == n_movies)

    print('tag lengths, max: {}, min: {}, mean: {}'.format(max(l), min(l), sum(l) * 1.0 / len(l)))
    print('total tags {}'.format(len(tags)))
    filename = 'movie_attributes' + '_' + str(thresh) + '.csv'
    filename = join(DIR + filename)
    print('saving to file {}'.format(filename))

    write_csv(pd.DataFrame(values), filename, ['id', 'tags'])

def process_user():
    filename = join(DIR, 'ratings.csv')
    values, columns = load_csv(filename, ',')
    users = np.unique(values[:, 0])
    users = [int(u) for u in list(users)]
    N = len(users)
    print('how many users? {}'.format(N))
    users = np.reshape(users, (N, 1))
    filename = join(DIR, 'users.csv')
    user_index = build_index(users)
    write_csv(pd.DataFrame(users), filename, ['id'])



# create_movie()
# x, index = load_movie()
# print x[:5, :]

# process_user()
# x, y, z = load_user()
# print(x, y)

# filter_interactions()
