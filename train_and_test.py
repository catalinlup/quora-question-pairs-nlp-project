import pandas as pd
from utils import *
from dataset import data
from sklearn.model_selection import train_test_split
from model import QuoraNet, train_net, test_net
import torch

device = 'cpu'
features = pd.read_csv('./features/train_test_feature_map.csv')
features_normalized = perform_feature_normalization(features).values[:, 1:]
labels = data['is_duplicate'].to_numpy()
features_train, features_test, labels_train, labels_test = train_test_split(features_normalized, labels, test_size=0.33)

features_train = torch.from_numpy(features_train).to(device).type(torch.FloatTensor)
features_test = torch.from_numpy(features_test).to(device).type(torch.FloatTensor)
labels_train = torch.from_numpy(labels_train).to(device).type(torch.FloatTensor)
labels_test = torch.from_numpy(labels_test).to(device).type(torch.FloatTensor)


print(features_train)

net = QuoraNet()
# train_net(net, features_train, labels_train, 15, lr=0.1)

# evaluate on test set
# print(test_net(net, features_test, labels_test))