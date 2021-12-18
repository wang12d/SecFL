from utils.comm import recvby, sendby
import numpy as np


class Role:
    SERVER = 0
    CLIENT = 1
    ALL = 2


def communicate(role, data):
    other_data = str()
    if(role == Role.SERVER):
        other_data = recvby(role)
        sendby(role, data)
    else:
        sendby(role, data)
        other_data = recvby(role)

    return other_data


def fuck(role):
    for i in range(10000):
        a = np.random.rand(1000000).tolist()

        b = communicate(role, str(a))
        #b = eval(b)
        print(f"{i}:{len(b)}:{len(str(a))}")
