import pandas as pd
import numpy as np
from sklearn import preprocessing
from utils.ABY import Role, main

client_feature_num = 12


client_feature_num = 6

def parse_data(role):
    data = pd.read_csv("data/cs-training.csv")
    data = np.array(data.iloc[:, :])
    data = data[~np.isnan(data).any(axis=1), :]
    if(role == Role.SERVER):
        features = np.array(data[:, 2:7], dtype=np.float32)

    else:
        features = np.array(data[:, 7:12], dtype=np.float32)

    features = preprocessing.scale(features)
    features = np.insert(features, np.shape(features)[1], 1, axis=1)
    labels = np.array(data[:, 1])

    features = np.mat(features, dtype=np.float32)
    labels = np.mat(labels, dtype=np.float32).transpose()
    # print(labels)

    weights = np.mat(np.zeros((np.shape(features)[1], 1), dtype=np.float32))
    # weights = np.random.rand(np.shape(features)[1], 1)
    # logger.info(f"features.shape: {features.shape}")
    # logger.info(f"labels.shape: {labels.shape}")
    # logger.info(f"weights.shape: {weights.shape}")
    # logger.info(f"init weights:")
    # print(weights)
    return features, labels, weights