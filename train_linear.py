import pandas as pd
from utils import *
from dataset import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import torch
import sys
import numpy as np

features = pd.read_csv('./features/train_test_feature_map.csv')
features_normalized = perform_feature_normalization(features).values[:, 1:]
labels = data['is_duplicate'].to_numpy()
labels = (labels - 0.5) * 2.0
features_train, features_test, labels_train, labels_test = train_test_split(features_normalized, labels, test_size=0.33)



model = LinearRegression()

print(features_train.shape)
print(features_normalized.shape)
print(labels_train.shape)
print(labels.shape)

if sys.argv[1] == 'test':
    classifier = model.fit(features_train, labels_train)
    # train_score = classifier.score(features_train, labels_train)
    # test_score = classifier.score(features_test, labels_test)

    test_predictions = (classifier.predict(features_test) >= 0).astype(np.int64)

    print('Test Score', np.mean((test_predictions == labels_test).astype(np.float32)))

    pickle.dump(classifier, open('models/linear_test.pickle', 'wb'))

if sys.argv[1] == 'train':
    classifier = model.fit(features_normalized, labels)
    score = classifier.score(features_normalized, labels)

    print('Score', score)

    pickle.dump(classifier, open('models/linear_full.pickle', 'wb'))