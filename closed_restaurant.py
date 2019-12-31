from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import cross_validate
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

learning_set = pd.read_csv("yelp_dataset/learning_set_2.csv")

closed_restaurant = learning_set[learning_set.apply(lambda row: not(row['is_open'])
            and row['years_to_shutdown'] == 1, axis=1)]
open_restaurant = learning_set[learning_set.apply(lambda row: row['is_open'] == 1, axis=1)]
open_restaurant = open_restaurant.sample(closed_restaurant.shape[0])
restaurant = closed_restaurant.append(open_restaurant)

X = restaurant[['age', 'review_count', 'chain', 'stars', 'tip_count', 'checkin_count',
            'density', 'category_density']]
y = restaurant['is_open']

X.to_csv('X_2.csv')
y.to_csv('y_2.csv')

for C in range(5,6):
    model = SVC(gamma='scale', C=C)

    #scaler = StandardScaler()
    #X = scaler.fit_transform(X)

    results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
            'f1'), cv=10, n_jobs=-1)

    print(C, np.mean(results['test_accuracy']), ", ", np.std(results['test_accuracy']),
            ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
            ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
            np.mean(results['test_recall']), ",", np.std(results['test_recall']))
