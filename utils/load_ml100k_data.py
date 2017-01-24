from pandatools import pd, np, load_csv, write_csv, build_index
from os.path import join

DIR = '../raw_data/ml-100k/'

def load_interactions():
    filename = join(DIR, 'u.data')
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
    filename = join(DIR, 'u.user')
    values, columns = load_csv(filename, '|')
    user_index = build_index(values)
    return values, columns, user_index

def load_movie():

    filename = 'items.csv'
    filename = join(DIR, filename)

    movies, columns = load_csv(filename, '\t')
    index = build_index(movies)
    return movies, columns, index

def create_movie():
    filename0 = join(DIR, 'u.item')
    movies, _ = load_csv(filename0, '|')

    values = np.zeros((len(movies), 2), dtype=object)
    for i in range(len(movies)):
        values[i, 0] = int(movies[i, 0])
        g = []
        for j in range(5, 24):
            if int(movies[i, j]) > 0:
                g.append(j)
        if len(g) == 0:
            g = [-1]
        values[i, 1] = ','.join([str(x) for x in g])
    filename = join(DIR, 'items.csv')

    write_csv(pd.DataFrame(values), filename, ['id', 'genres'])


def create_users():
    f = join(DIR, 'u.user')
    users, _ =  load_csv(f, '|')
    print(users)
    print(type(users))
    print(users[0, :])


def create_movie2(thresh=20):
    '''
    at most 20 tags per movie
    '''
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
        movie_id = int(movie_id)
        if movie_id not in movie:
            movie[movie_id] = []
        movie[movie_id].append((int(tag_id), float(s)))
        tags.add(int(tag_id))

    for mid in movie:
        seq = sorted(movie[mid], key = lambda x:x[1], reverse=True)
        seq = seq[:thresh]
        movie[mid] = [x[0] for x in seq]



    print('create movie attributes')
    n_movies = len(movie_ids)
    values = np.zeros((n_movies, 3), dtype=object)
    l = []
    for i in range(len(movies)):
        mid = int(movies[i, 0])
        values[i, 0] = mid
        values[i, 1] = movies[i, 2]
        if mid in movie:
            t = movie[mid]
            values[i, 2] = t
            l.append(len(t))
        else:
            values[i, 2] = []
            l.append(0)
    
    
    print('tag lengths, max: {}, min: {}, mean: {}'.format(max(l), min(l), sum(l) * 1.0 / len(l)))
    print('total tags {}'.format(len(tags)))
    filename = 'movie_attributes' + '_max_' + str(thresh) + '.csv'
    filename = join(DIR + filename)
    print('saving to file {}'.format(filename))

    write_csv(pd.DataFrame(values), filename, ['id', 'genres', 'tags'])


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


# create_movie2()
# create_movie()
# create_users()
# x, index = load_movie()
# print x[:5, :]

# process_user()
# x, y, z = load_user()
# print(x, y)

# filter_interactions()
