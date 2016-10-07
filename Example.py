import autosklearn.classification
import sklearn.datasets
import numpy as np

digits = sklearn.datasets.load_digits()
X = digits.data
y = digits.target

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
X_train = X[:1000]
y_train = y[:1000]
X_test = X[1000:]
y_test = y[1000:]
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
print(automl.score(X_test, y_test))
