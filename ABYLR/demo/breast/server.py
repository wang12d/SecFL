from utils.algorithm import Role, train
from sklearn.model_selection import train_test_split
from . import parse_data, config


role = Role.SERVER
raw_features, raw_labels = parse_data(role)
X_train, X_test, y_train, y_test = train_test_split(raw_features, raw_labels, test_size=0.2, random_state=0)
train(role, config, X_train, X_test, y_train, y_test)
