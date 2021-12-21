import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.algorithm import Role
from utils.common import Config


config = Config(client_feature_num=6, encryted=False, alpha=0.3, epochs=20, threshold=0.6)


def parse_data(role):
    scaler = StandardScaler()
    data = pd.read_csv("data/cs-training.csv")
    data = np.array(data.iloc[:, :])
    data = data[~np.isnan(data).any(axis=1), :]
    if(role == Role.SERVER):
        features = np.array(data[:, 2:7], dtype=np.float32)

    else:
        features = np.array(data[:, 7:12], dtype=np.float32)

    features = scaler.fit_transform(features)
    features = np.insert(features, np.shape(features)[1], 1, axis=1)
    labels = np.array(data[:, 1])

    features = np.mat(features, dtype=np.float32)
    labels = np.mat(labels, dtype=np.float32).transpose()

    return features, labels
