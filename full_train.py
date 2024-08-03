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

features_normalized = torch.from_numpy(features_normalized).to(device).type(torch.FloatTensor)
labels = torch.from_numpy(labels).to(device).type(torch.FloatTensor)



net = QuoraNet()
train_net(net, features_normalized, labels, 30, lr=0.1)

save_model(net, 'mlp')

# evaluate on test set
# print(test_net(net, features_test, labels_test))