from pandatools import pd, np, load_csv, write_csv, build_index
from os.path import join, isfile

DIR = '../raw_data/ml-1m/'

def load_interactions(thresh=4.0):
    filename = join(DIR, 'ratings_' + str(thresh) + '.csv')
    if not isfile(filename):
        print("Rating file not exist. Creating new one.")
        filter_interactions(thresh)
    values, columns = load_csv(filename, '\t')
    return values, columns

def load_user():
    filename = join(DIR, 'users.csv')
    if not isfile(filename):
        print("User file not exist. Creating new one.")
        create_users()
    values, columns = load_csv(filename, '\t')
    user_index = build_index(values)
    return values, columns, user_index

def load_movie():
    filename = 'items.csv'
    filename = join(DIR, filename)
    if not isfile(filename):
        print("Item file not exist. Creating new one.")
        create_movie()

    movies, columns = load_csv(filename, '\t')
    index = build_index(movies)
    return movies, columns, index


''' 
raw data processing
'''

def create_movie():
    filename0 = join(DIR, 'movies.dat')
    movies, _ = load_csv(filename0, '::', header=None)

    values = np.zeros((len(movies), 3), dtype=object)
    for i in range(len(movies)):
        values[i, 0] = int(movies[i, 0])
        values[i, 1] = str(movies[i, 2]).replace('|', ',')
        values[i, 2] = ','.join([str(x) for x in str(movies[i, 1]).split(' ') if x !=''])
    filename = join(DIR, 'items.csv')
    print("saving data to {}".format(filename))
    write_csv(pd.DataFrame(values), filename, ['id', 'genres', 'title'])


def create_users():
    f = join(DIR, 'users.dat')
    users, _ =  load_csv(f, '::', header=None)
    users = users[:, 0:4]
    # print(users)
    print(type(users))
    print(users[0, :])
    filename = join(DIR, 'users.csv')
    write_csv(pd.DataFrame(users), filename, ['id', 'gender', 'age', 'job'])


def filter_interactions(thresh=4.0):
    filename = join(DIR, 'ratings.dat')
    values, columns = load_csv(filename, '::', header=None)

    values2 = np.array(np.copy(values), dtype=int)
    c = 0
    for i in range(len(values)):
        r = float(values[i, 2])
        if r >= thresh:
            values2[c, :] = values[i, :]
            c += 1
    values2 = values2[:c, :]
    filename = join(DIR, 'ratings_' + str(thresh) + '.csv')
    write_csv(pd.DataFrame(values2), filename, ["user", "movie", "rating", "time"])

# create_movie()