from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics


def parse_data():

    data = pd.read_csv("data/cs-training.csv")
    data = np.array(data.iloc[:, :])
    data = data[~np.isnan(data).any(axis=1), :]

    features = np.array(data[:, 2:12], dtype=np.float32)
    labels = np.array(data[:, 1])
    #features = preprocessing.scale(features)
    return features, labels


X, y = parse_data()
lr = linear_model.LogisticRegression(solver='liblinear')
lr.fit(X, y)

print(f"{100*metrics.accuracy_score(lr.predict(X), y):.6f}")
