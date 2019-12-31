import pandas as pd
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians


def count(timestamps, year):
    sum = int()
    try:
        for timestamp in timestamps:
            sum += int(int(timestamp[:4]) == year)
    except:
        pass
    return sum



def dist(lat1, lon1, lat2, lon2):
    R = 6373.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = R * c
    return distance


def find_neighbours(lat, lon, learning_set):
    neighbours = business[business.apply(lambda row: dist(lat, lon,
            radians(row['latitude']), radians(row['longitude'])) < 2, axis=1)]
    return neighbours


def category_neighbours_func(neighbour_categories, categories):
    neighbour_categories = neighbour_categories.strip('][').split(', ')
    for category in neighbour_categories:
        if category in categories:
            return True
    return False


def calculate_features(restaurants, year):
    learning_set = []
    reviews = pd.read_csv("yelp_dataset/reviews_"+str(year)+".csv")
    review_count = reviews['business_id'].value_counts()
    star_sum = reviews.groupby('business_id').sum()['stars']
    tip = pd.read_csv("yelp_dataset/tip_"+str(year)+".csv")
    tip_count = tip['business_id'].value_counts()
    checkin = pd.read_json("yelp_dataset/checkin.json", lines=True)
    checkin = checkin.set_index('business_id')['date']
    for _, row in restaurants.iterrows():
        if row['start_year'] < year and row['end_year'] > year:
            id = row['business_id']
            row['age'] = year - row['start_year']
            row['review_count'] = review_count.get(id, default=0)
            row['stars'] = star_sum.get(id, default=0) / (row['review_count'] + 0.00001)
            row['tip_count'] = tip_count.get(id, default=0)
            checkin_timestamps = checkin.get(id, default="").split(", ")
            row['checkin_count'] = count(checkin_timestamps, year)
            row['years_to_shutdown'] = row['end_year'] - year
            learning_set.append(row)
    learning_set = pd.DataFrame(learning_set)
    chain = learning_set['name'].value_counts().rename('chain')
    learning_set = learning_set.merge(chain, left_on='name', right_index=True)
    return learning_set


def compute_relative_features(learning_set):
    for i, row in learning_set.iterrows():
        lat, lon = radians(row['latitude']), radians(row['longitude'])
        neighbours = learning_set[learning_set.apply(lambda row_: dist(lat, lon,
                    radians(row_['latitude']), radians(row_['longitude'])) < 2, axis=1)]
        for feature in ['stars', 'review_count', 'tip_count', 'checkin_count']:
            learning_set.at[i, feature] = row[feature] / (neighbours[feature].mean() + 0.00001)
        learning_set.at[i, 'density'] = neighbours.shape[0]
        category_list = row['categories'].strip('][').split(', ')
        category_neighbours = neighbours[neighbours.apply(lambda row_:
                    category_neighbours_func(row_['categories'], category_list), axis=1)]
        learning_set.at[i, 'category_density'] = category_neighbours.shape[0] / neighbours.shape[0]
    return learning_set


def generate_learning_set(restaurants, year):
    learning_set = calculate_features(restaurants, year)
    #learning_set = learning_set[learning_set.apply(lambda row: row['checkin_count'] > 0, axis=1)]
    learning_set = compute_relative_features(learning_set)
    return learning_set


if __name__ == '__main__':
    restaurants = pd.read_csv("yelp_dataset/restaurants.csv")
    restaurants = restaurants[restaurants.apply(lambda row: row['review_count'] != None, axis=1)]
    restaurants = restaurants.drop(columns=['Unnamed: 0'])
    columns = list(restaurants.columns)
    columns.extend(['age', 'tip_count', 'checkin_count', 'years_to_shutdown', 'chain',
                'density', 'category_density'])
    learning_set = pd.DataFrame(columns=columns)
    pool = Pool()
    results = list()
    for year in range(2010, 2018):
        results.append(pool.apply_async(generate_learning_set, args=(restaurants, year)))
    pool.close()
    for result in results:
        learning_set = learning_set.append(result.get())
    learning_set.to_csv("yelp_dataset/learning_set_2.csv")
