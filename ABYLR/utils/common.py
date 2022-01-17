class Role:
    HOST = 0
    GUEST = 1
    ALL = 2


class Config:
    def __init__(self, server_feature_num, alpha=0.1, encryted=False, epochs=20, batch_size=100, threshold=0.5):
        self.alpha = alpha
        self.epochs = epochs
        self.encryted = encryted
        self.server_feature_num = server_feature_num + 1
        self.batch_size = batch_size
        self.batch_list = []
        self.threshold = threshold
