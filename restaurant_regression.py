import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

'''
learning_set = pd.read_csv('yelp_dataset/learning_set_2.csv')

learning_set['chain'] = learning_set['chain'].apply(np.log)

id_list = learning_set[learning_set.apply(lambda row: row['is_open'] == 1
            and row['years_to_shutdown'] == 1 and row['age'] >= 4, axis=1)]['business_id'].tolist()
open_restaurant = learning_set[learning_set.apply(lambda row: row['years_to_shutdown'] == 4
            and row['business_id'] in id_list, axis=1)]
shutdown_in_three = learning_set[learning_set.apply(lambda row: not(row['is_open'])
            and row['years_to_shutdown'] == 3, axis=1)]
id_list = shutdown_in_three['business_id'].tolist()
shutdown_in_two = learning_set[learning_set.apply(lambda row: not(row['is_open'])
            and row['years_to_shutdown'] == 2 and not(row['business_id'] in id_list), axis=1)]
id_list.extend(shutdown_in_two['business_id'].tolist())
shutdown_in_one = learning_set[learning_set.apply(lambda row: not(row['is_open'])
            and row['years_to_shutdown'] == 1 and not(row['business_id'] in id_list), axis=1)]

n = shutdown_in_two.shape[0]
restaurant = pd.concat([shutdown_in_one.sample(n), shutdown_in_two,
            shutdown_in_three.sample(n), open_restaurant.sample(n)]).reset_index()
print(restaurant['years_to_shutdown'].value_counts())

for i, row in restaurant.iterrows():
    if i % 500 == 0:
        print(i)
    data = learning_set.loc[(learning_set['years_to_shutdown'] >= row['years_to_shutdown'])
                & (learning_set['business_id'] == row['business_id'])]
    for feature in ['chain', 'density', 'category_density', 'review_count', 'stars',
                'tip_count', 'checkin_count']:
        restaurant.at[i, 'mean_'+feature] = data[feature].mean()

restaurant.to_csv('multi_class_data.csv')
'''

restaurant = pd.read_csv('multi_class_data.csv')

feature_list = ['age', 'mean_review_count', 'review_count', 'mean_chain', 'chain', 'mean_stars',
            'stars', 'mean_tip_count', 'tip_count', 'mean_checkin_count', 'checkin_count',
            'mean_density', 'density', 'mean_category_density', 'category_density']

X = restaurant[feature_list]
y = restaurant['years_to_shutdown']

model = LogisticRegression(solver='lbfgs', max_iter=500, multi_class='multinomial')
scaler = StandardScaler()
pca = PCA()
X = scaler.fit_transform(X)
X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train)

predictions = model.predict_proba(X_test)

for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    confusion_matrix = np.zeros((4,4))
    difference = list()
    for i, prediction in zip(y_test, predictions):
        sum = 0
        for j, probability in enumerate(prediction):
            sum += probability
            if sum > threshold:
                confusion_matrix[i-1][j] += 1
                difference.append(np.abs(i-j-1))
    print(threshold, "\n", confusion_matrix, "\n", np.sum(np.diag(confusion_matrix)), np.sum(difference))
