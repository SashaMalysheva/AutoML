import os
import autosklearn.classification
from mnist import MNIST
from tests.loader import root_path

DATA_PATH = os.path.join(root_path, 'data notMNIST')
mndata = MNIST(DATA_PATH)
mndata.load_training()
mndata.load_testing()

automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(mndata.train_labels, mndata.train_images)
print(automl.score(mndata.test_images, mndata.test_labels))