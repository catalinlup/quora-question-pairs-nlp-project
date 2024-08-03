import pickle
from bert_similarity import predict_similarity
import pandas as pd
from utils import *
import torch
import numpy as np
from dataset import validation_data



logistic_full = pickle.load(open('./models/xgb.pickle', 'rb'))

features = pd.read_csv('./features/validation_feature_map.csv')
features_normalized = perform_feature_normalization(features).values[:, 1:]
embeddings = load_feature('validation_emb_concat')
features_with_embeddings = np.concatenate([features_normalized, embeddings], axis=1)

features_tensor = torch.from_numpy(features_with_embeddings).type(torch.FloatTensor)


final_predictions = logistic_full.predict(features_with_embeddings).astype(dtype=np.int64)

ids = validation_data['id'].to_numpy().astype(dtype=np.int64)
predictions = final_predictions.reshape((final_predictions.shape[0], 1))
ids = ids.reshape((ids.shape[0], 1))

result = np.concatenate([ids, predictions], axis=1)


np.savetxt('predictions/xgb.csv', result, delimiter=',', fmt='%d')