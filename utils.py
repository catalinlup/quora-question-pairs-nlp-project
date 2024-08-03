from sklearn import preprocessing
import pandas as pd
import pickle


def perform_feature_normalization(dataset):
    min_max_scaler = preprocessing.MinMaxScaler()
    # new_dataset = dataset.copy()

    # for column in new_dataset:
    #     new_dataset[column] = min_max_scaler.fit_transform(new_dataset[column].values.reshape(-1,1))

    
    # return new_dataset

    new_data = min_max_scaler.fit_transform(dataset.values)
    return pd.DataFrame(new_data)




def save_feature(feature, feature_name):
    pickle.dump(feature, open(f'features/{feature_name}.pickle', 'wb'))


def load_feature(feature_name):
    return pickle.load(open(f'features/{feature_name}.pickle', 'rb'))


def save_model(model, model_name):
    pickle.dump(model, open(f'models/{model_name}.pickle', 'wb'))