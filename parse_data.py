import pandas as pd
import numpy as np
from multiprocessing import Pool
from collections import defaultdict
import matplotlib.pyplot as plt


def compute_age(business):
    min_age = defaultdict(lambda: float('inf'))
    max_age = defaultdict(int)

    reviews = pd.read_json("yelp_dataset/review.json", lines=True, chunksize=10000)
    for chunk in reviews:
        for i, row in chunk.iterrows():
            id = row['business_id']
            year = row['date'].year
            if year < min_age[id]:
                min_age[id] = year
            elif year > max_age[id]:
                max_age[id] = year

    tips = pd.read_json("yelp_dataset/tip.json", lines=True)
    for i, row in tips.iterrows():
        id = row['business_id']
        year = row['date'].year
        if year < min_age[id]:
            min_age[id] = year
        elif year > max_age[id]:
            max_age[id] = year

    checkin = pd.read_json("yelp_dataset/checkin.json", lines=True)
    for i, row in checkin.iterrows():
        id = row['business_id']
        timestamps = row['date'].split(", ")
        min_year, max_year = int(timestamps[0][:4]), int(timestamps[-1][:4])
        if min_year < min_age[id]:
            min_age[id] = min_year
        if max_year > max_age[id]:
            max_age[id] = max_year

    start_year = pd.DataFrame.from_records(list(min_age.items()), columns=['business_id',
                'start_year'])
    end_year = pd.DataFrame.from_records(list(max_age.items()), columns=['business_id',
                'end_year'])

    start_year.to_csv("start_year.csv")
    end_year.to_csv("end_year.csv")

    start_year = start_year.drop(columns=['Unnamed: 0'])
    end_year = end_year.drop(columns=['Unnamed: 0'])

    business = pd.merge(business, start_year, on='business_id')
    business = pd.merge(business, end_year, on='business_id')

    return business


def compute_features(restaurants, year):
    id_list = restaurant['business_id'].tolist()
    age, stars, review, tip, checkin = {}, {}, {}, {}, {}
    for _, row in restaurants.iterrows():
        id = row['business_id']
        if row['start_year'] < year and row['end_year'] > year:
            age[id] = year - row['start_year'] + 1
            review_frame = pd.read_csv("reviews_"+str(year)+".csv")



'''
tip = pd.read_json("yelp_dataset/tip.json", lines=True)
min_age = float('inf')
for _, row in tip.iterrows():
    if row['date'].year < min_age:
        min_age = row['date'].year
print(min_age)
exit(0)
'''
'''
restaurants = pd.read_csv("yelp_dataset/restaurants.csv")
restaurants = restaurants[restaurants.apply(lambda row: row['start_year'] < row['end_year'] - 1, axis=1)]
age = restaurants['end_year'].to_numpy() - restaurants['start_year'].to_numpy()
review_count = restaurants['review_count'].to_numpy() / age

plt.hist(review_count, bins=100)
plt.show()
exit(0)
'''
business = pd.read_csv("yelp_dataset/business.csv")
business = business.dropna()
business = business[business.apply(lambda row: row['end_year'] >= 2010, axis=1)]
restaurants = business[business.apply(lambda row: 'Restaurants' in row['categories'], axis=1)]
for i, row in restaurants.iterrows():
    row['categories'] = row['categories'].split(", ")
    row['categories'].remove("Restaurants")
    if "Food" in row['categories']:
        row['categories'].remove("Food")
    restaurants.at[i, 'categories'] = row['categories']
print(restaurants.iloc[0])
restaurants.to_csv("yelp_dataset/restaurants.csv")
exit(0)

for year in range(2011, 2018):
    tip = pd.read_json("yelp_dataset/tip.json", lines=True)
    df = tip[tip.apply(lambda row: row['date'].year == year, axis=1)]
    df.to_csv("yelp_dataset/tip_"+str(year)+".csv")
    print(df.head())
exit(0)


min_age = float('inf')
for _, row in business.iterrows():
    if row['start_year'] < min_age:
        min_age = row['start_year']
print(min_age)
exit(0)

open_businesses, closed_businesses = ([0 for i in range(16)] for j in range(2))
for _, row in business.iterrows():
    age = row['end_year'] - row['start_year']
    if row['is_open']:
        open_businesses[age] += 1
    else:
        closed_businesses[age] += 1

open_restaurants, closed_restaurants = ([0 for i in range(16)] for j in range(2))
for _, row in restaurants.iterrows():
    age = row['end_year'] - row['start_year']
    if row['is_open']:
        open_restaurants[age] += 1
    else:
        closed_restaurants[age] += 1


plt.plot(range(1,17), open_businesses, label="open businesses")
plt.plot(range(1,17), closed_businesses, label="closed businesses")

plt.plot(range(1,17), open_restaurants, label="open restaurants")
plt.plot(range(1,17), closed_restaurants, label="closed restaurants")

plt.legend()
plt.show()
