from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

learning_set = pd.read_csv("yelp_dataset/learning_set_2.csv")
learning_set['chain'] = learning_set['chain'].apply(np.log)

closed_restaurants = learning_set[learning_set.apply(lambda row: not(row['is_open'])
        and row['years_to_shutdown'] == 1, axis=1)]

open_restaurants = learning_set[learning_set.apply(lambda row: row['is_open'] == 1
        and row['years_to_shutdown'] == 1, axis=1)]
#open_restaurants = open_restaurants.sample(closed_restaurants.shape[0])

restaurant = closed_restaurants.append(open_restaurants)

for i, row in restaurant.iterrows():
    data = learning_set.loc[learning_set['business_id'] == row['business_id']]
    for feature in ['chain', 'density', 'category_density', 'review_count', 'stars',
                'tip_count', 'checkin_count']:
        restaurant.at[i, 'mean_'+feature] = data[feature].mean()

restaurant.to_csv('binary_data_2.csv')

#restaurant = pd.read_csv('binary_data.csv')

print(restaurant.shape)

scaler = StandardScaler()
pca = PCA()

print(restaurant['is_open'].value_counts())

feature_list = ['age', 'mean_review_count', 'review_count', 'chain', 'mean_stars',
            'stars', 'mean_tip_count', 'tip_count', 'mean_checkin_count', 'checkin_count',
            'mean_density', 'density', 'category_density']

#feature_list = ['age', 'chain', 'category_density']

X = restaurant[feature_list]
y = restaurant['is_open']

for feature in feature_list:
    print(feature, pearsonr(X[feature], y)[0])

X = scaler.fit_transform(X)
X = pca.fit_transform(X)

for model in [LogisticRegression(solver='lbfgs'), SVC(gamma='scale', C=5), MLPClassifier(max_iter=400)]:
    results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
            'f1'), cv=10, n_jobs=-1)
    print(np.mean(results['test_accuracy']), ", ", np.std(results['test_accuracy']),
            ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
            ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
            np.mean(results['test_recall']), ",", np.std(results['test_recall']))
