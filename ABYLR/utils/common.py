class Role:
    SERVER = 0
    CLIENT = 1
    ALL = 2


class Config:
    def __init__(self, alpha=0.1, epochs=20, batch_size=100):
        self.alpha = alpha
        self.epochs = epochs
        self.encryted = False
        self.batch_size = batch_size
        self.batch_list = []