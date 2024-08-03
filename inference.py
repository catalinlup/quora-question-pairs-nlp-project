import pickle
from bert_similarity import predict_similarity
import pandas as pd
from utils import *
import torch
import numpy as np
from dataset import validation_data

question1_embeddings = pickle.load(open('./saved/test_questions1_embeddings.pickle', 'rb'))
question2_embeddings = pickle.load(open('./saved/test_questions2_embeddings.pickle', 'rb'))

bert_predictions = predict_similarity(question1_embeddings, question2_embeddings, 0.9)

mlp_net = pickle.load(open('./models/mlp.pickle', 'rb'))

features = pd.read_csv('./features/validation_feature_map.csv')
features_normalized = perform_feature_normalization(features).values[:, 1:]
features_tensor = torch.from_numpy(features_normalized).type(torch.FloatTensor)

mlp_predictions = mlp_net(features_tensor)

print(mlp_predictions.shape)

def compine_predictions(mlp, bert, thr1=0.3, thr2=0.8):
    predictions = []

    for i in range(bert.shape[0]):
        p1 = mlp[i]
        p2 = bert[i].item()

        if p1 < thr1:
            predictions.append(0)
        elif p1 > thr2:
            predictions.append(1)
        else:
            predictions.append(p2)
    
    return np.array(predictions)

final_predictions = compine_predictions(mlp_predictions, bert_predictions)

ids = validation_data['id'].to_numpy().astype(dtype=np.int64)
predictions = final_predictions.reshape((final_predictions.shape[0], 1))
ids = ids.reshape((ids.shape[0], 1))

result = np.concatenate([ids, predictions], axis=1)


np.savetxt('predictions/mlp_bert.csv', result, delimiter=',', fmt='%d')