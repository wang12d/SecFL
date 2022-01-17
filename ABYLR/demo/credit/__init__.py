import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils.algorithm import Role
from utils.common import Config


config = Config(server_feature_num=5, encryted=False, alpha=0.1, epochs=20, threshold=0.6)


def parse_data(role):
    scaler = StandardScaler()
    if (role == Role.GUEST):
        data = pd.read_csv("data/give_credit_hetero_guest.csv")
        data = np.around(np.array(data.iloc[:, :]), 5)
        features = np.array(data[:, 2:], dtype=np.float32)
        features = scaler.fit_transform(features)
        features = np.insert(features, np.shape(features)[1], 1, axis=1)
        features = np.mat(features, dtype=np.float32)
        labels = np.mat(np.array(data[:, 1]), dtype=np.float32).transpose()
        return features, labels
    else:
        data = pd.read_csv("data/give_credit_hetero_host.csv")
        data = np.around(np.array(data.iloc[:, :]), 5)
        features = np.array(data[:, 1:], dtype=np.float32)
        features = scaler.fit_transform(features)
        features = np.insert(features, np.shape(features)[1], 1, axis=1)
        features = np.mat(features, dtype=np.float32)

        return features
