from utils.algorithm import Role, main
from utils.common import Config
from . import parse_data, client_feature_num

role = Role.CLIENT
features, labels, weights = parse_data(role)
main(role, Config(), features, labels, weights, client_feature_num)
