import pandas as pd
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import numpy as np


data = pd.read_csv("business.csv")
data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
#print(data['is_open'].value_counts())
#print(data.iloc[0])
X = data[['stars', 'review_count', 'stars2017', 'review_count2017', 'chain', 'tips',
        'tips2017', 'checkin', 'checkin2017', 'age', 'density', 'category_density']].to_numpy()
y = data['is_open'].to_numpy()
indices = list(np.where(y == 0)[0])
for i in range(3384):
    while True:
        j = int(np.random.rand()*y.size)
        if not(j in indices):
            if y[j] == 1:
                indices.append(j)
                break
X = X.take(indices, axis=0)
y = y.take(indices)

np.savetxt('X.csv', X, delimiter=',')
np.savetxt('y.csv', y, delimiter=',')

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

#print(X.shape, y.shape)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
#print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

model = Perceptron()
results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
print(np.mean(results['test_accuracy']), ",", np.std(results['test_accuracy']),
        ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
        ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
        np.mean(results['test_recall']), ",", np.std(results['test_recall']))

model = LogisticRegression(solver='lbfgs')
results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
print(np.mean(results['test_accuracy']), ",", np.std(results['test_accuracy']),
        ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
        ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
        np.mean(results['test_recall']), ",", np.std(results['test_recall']))

model = SVC(kernel='linear', C=8)
results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
print(np.mean(results['test_accuracy']), ",", np.std(results['test_accuracy']),
        ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
        ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
        np.mean(results['test_recall']), ",", np.std(results['test_recall']))

model = SVC(kernel='poly', C=8, gamma='scale')
results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
print(np.mean(results['test_accuracy']), ",", np.std(results['test_accuracy']),
        ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
        ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
        np.mean(results['test_recall']), ",", np.std(results['test_recall']))

model = SVC(C=8, gamma='scale')
results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
print(np.mean(results['test_accuracy']), ",", np.std(results['test_accuracy']),
        ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
        ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
        np.mean(results['test_recall']), ",", np.std(results['test_recall']))

model = DecisionTreeClassifier(min_samples_split=10)
results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
print(np.mean(results['test_accuracy']), ",", np.std(results['test_accuracy']),
        ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
        ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
        np.mean(results['test_recall']), ",", np.std(results['test_recall']))

model.fit(X, y)
feat_importances = pd.Series(model.feature_importances_, index=['stars', 'review_count',
        'stars2017', 'review_count2017', 'chain', 'tips', 'tips2017', 'checkin',
        'checkin2017', 'age', 'density', 'category_density'])

model = MLPClassifier(max_iter=1000)
results = cross_validate(model, X, y, scoring=('accuracy', 'precision', 'recall',
        'f1'), cv=10, n_jobs=-1)
print(np.mean(results['test_accuracy']), ",", np.std(results['test_accuracy']),
        ",", np.mean(results['test_precision']), ",", np.std(results['test_precision']),
        ",", np.mean(results['test_f1']), ",", np.std(results['test_f1']), ",",
        np.mean(results['test_recall']), ",", np.std(results['test_recall']))

feat_importances.plot(kind='barh')
plt.show()
