import pandas as pd
import numpy as np
from sklearn import preprocessing
from utils.ABY import Role, main

client_feature_num = 12


def parse_data(role):
    global client_feature_num
    data = pd.read_csv("data/breast.csv")
    data = np.array(data.iloc[:, :])
    if(role == Role.SERVER):
        features = np.array(data[:, 2:-12], dtype=np.float32)
    else:
        features = np.array(data[:, -12:-1], dtype=np.float32)
    features = preprocessing.scale(features)
    features = np.insert(features, np.shape(features)[1], 1, axis=1)
    labels = np.array(data[:, 1])
    for i in range(len(labels)):
        if labels[i] == "M":
            labels[i] = 1.0
        else:
            labels[i] = 0.0
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