from . import main, Role
# from demo.credit import parse_data, client_feature_num
from demo.breast import parse_data, client_feature_num

role = Role.CLIENT
features, labels, weights = parse_data(role)
main(role, features, labels, weights, client_feature_num)

