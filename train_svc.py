import pandas as pd
from utils import *
from dataset import data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import torch
import sys
from sklearn import svm


features = pd.read_csv('./features/train_test_feature_map.csv')
features_normalized = perform_feature_normalization(features).values[:, 1:]
labels = data['is_duplicate'].to_numpy()
features_train, features_test, labels_train, labels_test = train_test_split(features_normalized, labels, test_size=0.33)



model = svm.SVC(max_iter=1000, C=0.01, kernel='linear', gamma='auto', verbose=True)

print(features_train.shape)
print(features_normalized.shape)
print(labels_train.shape)
print(labels.shape)

if sys.argv[1] == 'test':
    classifier = model.fit(features_train, labels_train)
    train_score = classifier.score(features_train, labels_train)
    test_score = classifier.score(features_test, labels_test)

    print('Train Score', train_score)
    print('Test Score', test_score)

    pickle.dump(classifier, open('models/svc_test.pickle', 'wb'))

if sys.argv[1] == 'train':
    classifier = model.fit(features_normalized, labels)
    score = classifier.score(features_normalized, labels)

    print('Score', score)

    pickle.dump(classifier, open('models/svc_full.pickle', 'wb'))