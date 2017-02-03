from pandatools import load_csv, np, write_csv, pd
from os.path import join, isfile
import unicodedata
from datetime import datetime
import time

DIR_RAW = '../raw_data/yelp_dataset_challenge_round9/'
DIR = '../raw_data/yelp/'

def unicode_norm(val):
    return unicodedata.normalize('NFKD', val).encode('ascii', 'ignore')

def clean_data(val, t):
    if t == 0:
        # print(val)
        return unicode_norm(val)
        # return str(val)
    elif t == 1:
        # a list of unicode
        if val == None:
            return ''
        return ','.join([str(x) for x in val])
    elif t == 2:
        # name, in unicode

        return ','.join(unicode_norm(val).split(' '))
    elif t == 3:
        # postcode
        return str(val)
    elif t == 4 or t == 5:
        return val
    elif t == 6:
        # time
        tval = datetime.strptime(str(val), '%Y-%m-%d')
        return time.mktime(tval.timetuple()) # float
    else:
        print("type: {}, val: {}".format(type(val), val))


def parse_data(source1, headers, header_types, output_path):
    import json
    import csv
    from tqdm import tqdm
    # setup an array for writing each row in the csv file
    rows = []
    # extract fields from business json data set #
    # setup an array for storing each json entry
    business_data = []

    # open the business source file
    with open(source1) as f:
        # for each line in the json file
        for line in f:
            # store the line in the array for manipulation
            business_data.append(json.loads(line))
    # close the reader
    f.close()

    print('processing data in the {}...'.format(source1))
    # for every entry in the business data array
    for entry in tqdm(range(0, len(business_data))):
        row = []
        for h in headers:
            val = business_data[entry][h]

            # if entry < 5:
            #     print('entry {} type {} value {}'.format(entry, type(val), val))
            row.append(clean_data(val, header_types[h]))
        # if entry < 5:
        #     print('\n')
        # else:
        #     exit()

        rows.append(row)

    # write to csv file
    with open(output_path, 'w') as out:
        writer = csv.writer(out, delimiter='\t')
        # write the csv headers
        writer.writerow(headers)
        # for each entry in the row array
        print('writing contents to csv...')
        for entry in tqdm(range(0, len(rows))):
            try:
                # write the row to the csv
                writer.writerow(rows[entry])
            # if there is an error, continue to the next row
            except UnicodeEncodeError:
                continue
    out.close()


def filter_data(vals, valid):
    vals2 = np.copy(vals)
    ind = 0
    for i in range(len(vals)):
        v = vals[i, 0]
        if v in valid:
            vals2[ind, :] = vals[i, :]
            ind += 1
    vals2 = vals2[:ind, :]
    return vals2
    


if not isfile(join(DIR, 'items.csv')):
    # setup an array for business headers
    headers = ['business_id', 'name', 'city', 'state', 'stars', 'review_count', 'categories', 'attributes'] 
    header_types = {'business_id':0, 'name':2, 'city':0, 'state':0, 'stars':4, 'review_count':5, 'categories':1, 'attributes':1}
    # # setup an array for headers we are not using strictly
    # business_header_removals = ['neighborhood', 'is_open', 'address', 'latitude', 'longitude', 'hours', type']

    parse_data(join(DIR_RAW, 'yelp_academic_dataset_business.json'), headers, 
        header_types, join(DIR, 'items.csv'))

if not isfile(join(DIR, 'users.csv')):
    headers = ['user_id', 'name']
    header_types = {'user_id':0, 'name':2}
    parse_data(join(DIR_RAW, 'yelp_academic_dataset_user.json'), headers, 
        header_types, join(DIR, 'users.csv'))


if not isfile(join(DIR, 'reviews.csv')):
    headers = ['user_id', 'business_id', 'stars', 'date']
    header_types = {'user_id':0, 'business_id':0, 'stars':4, 'date':6}
    parse_data(join(DIR_RAW, 'yelp_academic_dataset_review.json'), headers, 
        header_types, join(DIR, 'reviews.csv'))

if not isfile(join(DIR, 'reviews_filtered.csv')):
    vals, header = load_csv(join(DIR, 'reviews.csv'))
    counts = {}
    for i in range(len(vals)):
        uid = vals[i, 0]
        if uid not in counts:
            counts[uid] = 1
        else:
            counts[uid] +=1

    targets = set([])
    for uid, c in counts.items():
        if c >= 10:
            targets.add(uid)
    print('there are totally {} users'.format(len(targets)))

    vals2 = filter_data(vals, targets)
    write_csv(pd.DataFrame(vals2), join(DIR, 'reviews_filtered.csv'), header)

meta_file = join(DIR, 'filtered_meta.csv')
if not isfile(meta_file):
    import csv
    print('loading reviews...')
    reviews = load_csv(join(DIR, 'reviews_filtered.csv'), types=0)
    print('...done')
    t_ = reviews['date']
    i_ = reviews['business_id']
    ii = set([])
    for i in range(len(i_)):
        ii.add(i_[i])

    u_ = reviews['user_id']
    u = set([])
    for i in range(len(u_)):
        u.add(u_[i])

    users, header = load_csv(join(DIR, 'users.csv'))
    users2 = filter_data(users, u)
    print(header)
    print(type(users2))
    print(users2.shape)
    write_csv(pd.DataFrame(users2), join(DIR, 'users_filtered.csv'), header)

    items, header = load_csv(join(DIR, 'items.csv'))
    items2 = filter_data(items, ii)
    write_csv(pd.DataFrame(items2), join(DIR, 'items_filtered.csv'), header)

    t = sorted([t_[i] for i in range(len(t_))])
    headers = ['earliest', 'latest', 'median', 'mean','last 20 percent', 'last 10 percent', 'user #', 'business #']
    row = [t[0], t[-1], t[len(t)/2], sum(t) / len(t), t[len(t) * 8 / 10], t[len(t)*9/10], len(u), len(ii)]
    print headers
    print row
    with open(meta_file, 'w') as out:
        writer = csv.writer(out, delimiter='\t')
        writer.writerow(headers)
        writer.writerow(row)


if not isfile(join(DIR, 'business_review_count.pkl')):
    print('generating review_count clustering results')
    from sklearn.cluster import KMeans as kmeans
    import cPickle as pickle
    items = load_csv(join(DIR, 'items_filtered.csv'), types=0)
    r = items['review_count']
    ids = items['business_id']
    vals = [[r[i]] for i in range(len(r))]
    
    m = kmeans(20)
    m.fit(vals)
    y = m.labels_
    id2c = {}
    for i in range(len(r)):
        id2c[ids[i]] = y[i]

    print('writing to file...')
    with open(join(DIR, 'business_review_count.pkl'), 'wb') as f:
        pickle.dump(id2c, f, pickle.HIGHEST_PROTOCOL)
