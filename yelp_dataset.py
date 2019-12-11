import pandas as pd
import pickle
from math import sin, cos, sqrt, atan2, radians
from multiprocessing import Pool
import ijson
import numpy as np


def dist(lat1, lon1, lat2, lon2):
    R = 6373.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def divide_chunks(l, n):
    m = (len(l)+1)/n
    for i in range(0, len(l), m):
        yield l[i:i + m]


def neighbouring_restaurants(center_business, business):
    neighbours = []
    lat = radians(business.at[center_business, 'latitude'])
    lon = radians(business.at[center_business, 'longitude'])
    neighbours = business[business.apply(lambda row: dist(lat, lon,
            radians(row['latitude']), radians(row['longitude'])) < 2, axis=1)]
    return neighbours


def category_neighbours_func(categories, neighbour_categories):
    if not(isinstance(neighbour_categories, str)):
        return False
    categories = set(categories.split(", "))
    neighbour_categories = set(neighbour_categories.split(", "))
    if categories & neighbour_categories:
        return True
    else:
        return False



def compute_relative_features(args):
    center_business = args[0]
    business = args[1]
    neighbours = neighbouring_restaurants(center_business, business)
    relative_stars = business.at[center_business, 'stars'] / neighbours['stars'].mean()
    relative_stars2017 = business.at[center_business, 'stars2017'] / neighbours['stars2017'].mean()
    relative_review = business.at[center_business, 'review_count'] / neighbours['review_count'].mean()
    relative_review2017 = business.at[center_business, 'review_count2017'] / neighbours['review_count2017'].mean()
    relative_tips = business.at[center_business, 'tips'] / neighbours['tips'].mean()
    relative_tips2017 = business.at[center_business, 'tips2017'] / neighbours['tips2017'].mean()
    relative_checkin = business.at[center_business, 'checkin'] / neighbours['checkin'].mean()
    relative_checkin2017 = business.at[center_business, 'checkin2017'] / neighbours['checkin2017'].mean()
    density = neighbours.shape[0]
    category_density = 0
    if isinstance(business.at[center_business, 'categories'], str):
        category_neighbours = neighbours[neighbours.apply(lambda row:
                category_neighbours_func(business.at[center_business, 'categories'],
                row['categories']), axis=1)]
        category_density = category_neighbours.shape[0] / density
    return (relative_stars, relative_stars2017, relative_review, relative_review2017,
            relative_tips, relative_tips2017, relative_checkin, relative_checkin2017,
            density, category_density)


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_business_ids(fname, start, stop):
    businesses = []
    with open(fname, "r") as f:
        for i, line in enumerate(f):
            if i >= start and i < stop:
                dict = eval(line)
                date = dict['date']
                id = dict['business_id']
                if int(date[:4]) == 2017:
                    if not(id in businesses):
                        businesses.append(id)
    return businesses


def get_review_data(fname, start, stop):
    stars2017 = {}
    review_count2017 = {}
    with open(fname, "r") as f:
        line = f.readline()
        while line:
            try:
                dict = eval(line)
                date = dict['date']
                id = dict['business_id']
                star = dict['stars']
                if int(date[:4]) == 2017:
                    if id in stars2017.keys():
                        stars2017[id] += star
                        review_count2017[id] += 1
                    else:
                        stars2017[id] = star
                        review_count2017[id] = 1
            except:
                pass
            line = f.readline()
    return (stars2017, review_count2017)


with open("yelp_dataset/business.json", "rb") as f:
    business = pd.read_json(f, lines=True)
    chains = business['name'].value_counts()
    chains = pd.DataFrame({'name': chains.index, 'chain': chains})
    business = business[['business_id', 'is_open', 'name', 'latitude', 'longitude', 'stars', 'review_count', 'categories']]

num_reviews = file_len("yelp_dataset/review.json")

stars2017, review_count2017 = get_review_data("yelp_dataset/review.json", 0, num_reviews)

for id in stars2017.keys():
    stars2017[id] /= review_count2017[id]
stars_frame = pd.DataFrame([(id, stars) for id, stars in stars2017.items()],
            columns=['business_id', 'stars2017'])
review_count_frame = pd.DataFrame([(id, review_count) for id, review_count in
            review_count2017.items()], columns=['business_id', 'review_count2017'])
business = pd.merge(business, stars_frame, on='business_id')
business = pd.merge(business, review_count_frame, on='business_id')
business = pd.merge(business, chains, on='name')

with open("yelp_dataset/tip.json", "r") as f:
    tip_frame = pd.read_json(f, lines=True)
    tips = tip_frame['business_id'].value_counts()
    tips = pd.DataFrame({'business_id': tips.index, 'tips': tips})
    business = pd.merge(business, tips, on='business_id')
    tip_frame = tip_frame[tip_frame.apply(lambda tip: tip['date'].year == 2017, axis=1)]
    tips2017 = tip_frame['business_id'].value_counts()
    tips2017 = pd.DataFrame({'business_id': tips2017.index, 'tips2017': tips2017})
    business = pd.merge(business, tips2017, on='business_id')

with open("yelp_dataset/checkin.json", "r") as f:
    checkin_frame = pd.read_json(f, lines=True)
    business_ids = business['business_id'].tolist()
    checkin, checkin2017 = ([0 for id in business_ids] for i in range(2))
    for _, row in checkin_frame.iterrows():
        id = row['business_id']
        if id in business_ids:
            i = business_ids.index(id)
            timestamps = row['date'].split(", ")
            checkin[i] = len(timestamps)
            timestamps = [timestamp for timestamp in timestamps if int(timestamp[:4]) == 2017]
            checkin2017[i] = len(timestamps)
    business['checkin'] = checkin
    business['checkin2017'] = checkin2017

age = [float('inf') for i in range(business.shape[0])]
id_list = business['business_id'].tolist()

with open("yelp_dataset/review.json", "r") as f:
    line = f.readline()
    while line:
        try:
            dict = eval(line)
            id = dict['business_id']
            if id in id_list:
                i = id_list.index(id)
                year = int(dict['date'][:4])
                if year < age[i]:
                    age[i] = year
        except:
            pass
        line = f.readline()

with open("yelp_dataset/tip.json", "rb") as f:
    tip = pd.read_json(f, lines=True)
    for _, row in tip.iterrows():
        if row['business_id'] in id_list:
            i = id_list.index(row['business_id'])
            if age[i] > row['date'].year:
                age[i] = row['date'].year


with open("yelp_dataset/checkin.json", "rb") as f:
    checkin = pd.read_json(f, lines=True)
    for _, row in checkin.iterrows():
        if row['business_id'] in id_list:
            i = id_list.index(row['business_id'])
            if age[i] > int(row['date'][:4]):
                age[i] = int(row['date'][:4])

business['age'] = age

business['stars'] = business.apply(lambda row: row['stars'] / row['age'], axis=1)
business['review_count'] = business.apply(lambda row: row['review_count'] / row['age'], axis=1)
business['tips'] = business.apply(lambda row: row['tips'] / row['age'], axis=1)
business['checkin'] = business.apply(lambda row: row['checkin'] / row['age'], axis=1)

map_list = [(i, business) for i in range(business.shape[0])]
pool = Pool()
relative_features = pool.map(compute_relative_features, map_list)

business['stars'] = [sublist[0] for sublist in relative_features]
business['stars2017'] = [sublist[1] for sublist in relative_features]
business['review_count'] = [sublist[2] for sublist in relative_features]
business['review_count2017'] = [sublist[2] for sublist in relative_features]
business['tips'] = [sublist[4] for sublist in relative_features]
business['tips2017'] = [sublist[5] for sublist in relative_features]
business['checkin'] = [sublist[6] for sublist in relative_features]
business['checkin2017'] = [sublist[7] for sublist in relative_features]
business['density'] = [sublist[8] for sublist in relative_features]
business['category_density'] = [sublist[9] for sublist in relative_features]

print(business.iloc[0])

with open("business.csv", "w") as f:
    business.to_csv(f)


#split = int(num_reviews/12)
#split = 10
'''
pool = Pool()
results = list()
for i in range(12):
    results.append(pool.apply_async(get_business_ids, args=("yelp_dataset/review.json", split*i, split*(i+1))))
businesses = list()
for result in results:
    for id in result.get():
        if not(id in businesses):
            businesses.append(id)
with open("businesses.pickle", "wb") as f:
    pickle.dump(businesses, f)
print(len(businesses))
'''
'''
with open("businesses.pickle", "rb") as f:
    business_list = pickle.load(f)

business = business[business.business_id.isin(business_list)]

pool = Pool(1)
results = list()
print(len(business_list))
'''
'''
for i in range(12):
    results.append(pool.apply_async(get_review_data, args={"yelp_dataset/review.json", i*split, (i+1)*split}))
for result in results:
    star_dict, review_count_dict = result.get()
    for id in business_list:
        stars2017[id] += star_dict[id]
        review_count2017[id] += review_count_dict[id]
'''
'''
with open("star2017.pickle", "rb") as f:
    stars2017 = pickle.load(f)
with open("review_count2017.pickle", "rb") as f:
    review_count2017 = pickle.load(f)
for id in stars2017.keys():
    stars2017[id] /= review_count2017[id]
stars_frame = pd.DataFrame([(id, stars) for id, stars in stars2017.items()],
            columns=['business_id', 'stars2017'])
review_count_frame = pd.DataFrame([(id, review_count) for id, review_count in
            review_count2017.items()], columns=['business_id', 'review_count2017'])
business = pd.merge(business, stars_frame, on='business_id')
business = pd.merge(business, review_count_frame, on='business_id')
business = pd.merge(business, chains, on='name')
print(business.shape)
print(business.iloc[0])






with open("business.csv", "w") as f:
    business.to_csv(f)
print(business.iloc[0])
'''

'''
with open("")

with open("yelp_dataset/checkin")
    exit(0)
    X = [dataset['business_id'], dataset['latitude'], dataset['longitude'],
        dataset['review_count'], dataset['stars'], dataset['categories'],
        []]
    y = [dataset['is_open']]

with open("yelp_dataset/X.pickle", "wb") as f:
    pickle.dump(X, f)

with open("yelp_dataset/y.pickle", "wb") as f:
    pickle.dump(y, f)
'''
'''
business = pd.read_csv("business.csv")
print(business['is_open'].value_counts())
id_list = business['business_id'].tolist()
'''

'''


with open("age.pickle", "wb") as f:
    pickle.dump(age, f)
'''
'''
with open("yelp_dataset/tip.json", "rb") as f:
    tip = pd.read_json(f, lines=True)
    for _, row in tip.iterrows():
        if row['business_id'] in id_list:
            i = id_list.index(row['business_id'])
            if age[i] > row['date'].year:
                age[i] = row['date'].year

with open("age_tip.pickle", "wb") as f:
    pickle.dump(age, f)

'''
'''
with open("yelp_dataset/checkin.json", "rb") as f:
    checkin = pd.read_json(f, lines=True)
    for _, row in checkin.iterrows():
        if row['business_id'] in id_list:
            age[id_list.index(row['business_id'])] = int(row['date'][:4])

with open("age_checkin.pickle", "wb") as f:
    pickle.dump(age, f)

'''
'''
with open("age.pickle", "rb") as f:
    age1 = pickle.load(f)

with open("age_tip.pickle", "rb") as f:
    age2 = pickle.load(f)

with open("age_checkin.pickle", "rb") as f:
    age3 = pickle.load(f)

age = [min(a1, min(a2, a3)) for a1, a2, a3 in zip(age1, age2, age3)]
business['age'] = age
print(business.iloc[0])

with open("business.csv", "w") as f:
    business.to_csv(f)
'''
