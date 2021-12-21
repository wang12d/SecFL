class Role:
    SERVER = 0
    CLIENT = 1
    ALL = 2


class Config:
    def __init__(self, client_feature_num, alpha=0.1,encryted=False, epochs=20, batch_size=100):
        self.alpha = alpha
        self.epochs = epochs
        self.encryted = encryted
        self.client_feature_num = client_feature_num
        self.batch_size = batch_size
        self.batch_list = []
