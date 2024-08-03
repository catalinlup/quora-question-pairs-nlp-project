import pandas as pd
from utils import *
from dataset import data
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np

import torch
import sys

features = pd.read_csv('./features/train_test_feature_map.csv')
features_normalized = perform_feature_normalization(features).values[:, 1:]
embeddings = load_feature('train_test_emb_concat')
print(embeddings.shape)
features_with_embeddings = np.concatenate([features_normalized, embeddings], axis=1)
print(features_with_embeddings[0])
labels = data['is_duplicate'].to_numpy()
features_train, features_test, labels_train, labels_test = train_test_split(features_with_embeddings, labels, test_size=0.33)



model = XGBClassifier()

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

    pickle.dump(classifier, open('models/xgb_test.pickle', 'wb'))

if sys.argv[1] == 'train':
    classifier = model.fit(features_train, labels_train)
    score = classifier.score(features_with_embeddings, labels)

    print('Score', score)

    pickle.dump(classifier, open('models/xgb.pickle', 'wb'))