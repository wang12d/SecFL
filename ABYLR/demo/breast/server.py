from utils.algorithm import Role, train
from . import parse_data, config


role = Role.SERVER
features, labels, weights = parse_data(role)
train(role, config, features, labels, weights)
