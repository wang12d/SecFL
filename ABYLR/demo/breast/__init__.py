import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.algorithm import Role
from utils.common import Config


config = Config(server_feature_num=20, encryted=False, alpha=1e-4, epochs=1)


def parse_data(role):
    scaler = StandardScaler()
    if (role == Role.GUEST):
        data = pd.read_csv("data/breast_hetero_guest.csv")
        data = np.array(data.iloc[:, :])
        features = np.array(data[:, 2:], dtype=np.float32)
        features = scaler.fit_transform(features)
        features = np.insert(features, np.shape(features)[1], 1, axis=1)
        features = np.mat(features, dtype=np.float32)
        labels = np.mat(np.array(data[:, 1]), dtype=np.float32).transpose()
        return features, labels
    else:
        data = pd.read_csv("data/breast_hetero_host.csv")
        data = np.array(data.iloc[:, :])
        features = np.array(data[:, 1:], dtype=np.float32)
        features = scaler.fit_transform(features)
        features = np.insert(features, np.shape(features)[1], 1, axis=1)
        features = np.mat(features, dtype=np.float32)

        return features
