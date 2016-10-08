import os
import autosklearn.classification
from mnist import MNIST
from tests.loader import root_path
import sklearn.datasets
import numpy as np

DATA_PATH = os.path.join(root_path, 'data notMNIST')
mndata = MNIST(DATA_PATH)
mndata.load_training()
mndata.load_testing()

digits = sklearn.datasets.fetch_lfw_people()
X = digits.data
y = digits.target

indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X = X[indices]
y = y[indices]
X_train = X[:100]
y_train = y[:100]
X_test = X[100:]
y_test = y[100:]
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
print(automl.score(X_test, y_test))