from . import parse_data, main, Role,client_feature_num


role = Role.SERVER
features, labels, weights = parse_data(role)
main(role, features, labels, weights, client_feature_num)