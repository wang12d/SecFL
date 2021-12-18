import ctypes
import numpy as np
from utils.logging import logger
from utils.timing import timecal

alpha = 0.1
epochs = 20
batch_size = 100
batch_list = []


class Role:
    SERVER = 0
    CLIENT = 1
    ALL = 2


secureLR = ctypes.CDLL("ABY/build/lib/libsecureLR.so")
secureLR.loss_mu_computation.restype = ctypes.c_double
secureLR.loss_first_computation.restype = ctypes.c_double


def ABY_loss_compute(role, x_theta, theta, mu):

    # calculate the second
    loss = 1/4 * np.dot(theta.T, mu)
    loss = secureLR.loss_first_computation(role, (ctypes.c_double)(loss))

    # calculate the first
    number = x_theta.shape[0]
    x_theta = np.array(x_theta).squeeze()
    x_theta = (ctypes.c_double * number)(*x_theta)
    x_theta_sum = (ctypes.c_double * number)(*[])

    secureLR.loss_xtheta_sum_computation(role, number, x_theta,  x_theta_sum)
    x_theta_sum = np.around(x_theta_sum, decimals=6)

    loss = loss + 1/8 * np.dot(x_theta_sum, x_theta_sum)

    # divide h
    return loss / number


def ABY_grad_compute(role, x_theta, x, Y):
    x_theta, Y = np.array(x_theta).squeeze(), np.array(Y).squeeze()
    assert(x_theta.shape[0] == Y.shape[0])
    number = x_theta.shape[0]

    # calculate w_a,w_b
    if(role == Role.SERVER):
        input = 1/4 * x_theta - 1/2 * Y
    else:
        input = 1/4 * x_theta
    input = (input / number).tolist()

    # W <- w_a + w_b
    input = (ctypes.c_double * number)(*input)
    w = (ctypes.c_double * number)(*[])

    secureLR.grad_communicate(role, number, input,  w)

    # X \times w
    w = np.array(w, dtype=np.float32).reshape(number, 1)
    return np.dot(x.transpose(), w)


def mu_cache(role, X, Y, client_feature_num):
    number = X.shape[0]

    if(role == Role.SERVER):
        mu = 0.69314718056 - np.dot(Y.transpose(), X) / number
        Y = (ctypes.c_double * number)(*Y)
        for i in range(client_feature_num):
            secureLR.loss_mu_computation(role, number, Y)
    else:
        assert(X.shape[1] == client_feature_num)
        mu = []
        for i in range(client_feature_num):
            col = np.array(X[:, i]).squeeze().tolist()
            col = (ctypes.c_double * number)(*col)
            ret = secureLR.loss_mu_computation(role, number, col)
            mu.append(0.69314718056 - ret / number)
        mu = np.array(mu, dtype=np.float32)
    return np.mat(mu).transpose()


def batch_train(role, features, labels, weights, grad):

    for batch in batch_list:
        batch_features = features[batch[0]:batch[1], :]
        batch_labels = labels[batch[0]:batch[1], :]
        x_theta = np.dot(batch_features, weights)
        grad = ABY_grad_compute(role, x_theta, batch_features, batch_labels)
        weights = weights - alpha * grad
        weights = weights.reshape(weights.shape[0], 1)

    return weights


def grad_descent(role, features, labels, weights, client_feature_num):

    loss_array = []
    num = int(features.shape[0] / batch_size)
    for i in range(num):
        batch_list.append([i*batch_size, (i+1)*batch_size-1])
    batch_list.append([num*batch_size, features.shape[0]])

    mu = timecal(mu_cache)(role, features, labels, client_feature_num)
    print(mu)

    for i in range(epochs):
        logger.debug(f"Epoch: {i}")

        x_theta = np.dot(features, weights)
        grad = timecal(ABY_grad_compute)(role, x_theta, features, labels)

        weights = timecal(batch_train)(role, features, labels, weights, grad)

        x_theta = np.dot(features, weights)
        loss = timecal(ABY_loss_compute)(role, x_theta, weights,  mu)

        print(f"weights :\n {weights}")
        print(f"grad :\n {grad}")
        loss_array.append(loss)
        test(weights, features, labels)
        print(f"loss : {loss}")

    print("loss is: ", loss)
    # print("weights shape: ", weights.shape)
    print("weights is: ", weights)

    return weights, loss_array


def test(weights, features, labels):
    def sigmoid(x):
        # return 1 / (1 + np.exp(-x))
        x_ravel = x.ravel()  # 将numpy数组展平
        length = len(x_ravel)
        y = []
        for index in range(length):
            if x_ravel[index] >= 0:
                y.append(1.0 / (1 + np.exp(-x_ravel[index])))
            else:
                y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
        return np.array(y).reshape(x.shape)

    pred = sigmoid(np.array(np.dot(features, weights)))

    # print(pred)
    # print(features.shape)
    # print(weights.shape)
    cnt = 0
    total = len(labels)
    for i, j in zip(pred, labels):
        if (i < 0.5 and j == 0.0) or (i >= 0.5 and j == 1.0):
            cnt = cnt + 1

    # np.set_printoptions(threshold=np.inf)
    # print(np.column_stack((pred, labels)))

    print(f"Accuracy: {(cnt/total)*100:.3f}%")
    # assert(False)
    pass


def main(role, features, labels, weights, client_feature_num):
    # print(f"data/{'client' if role == Role.CLIENT else 'server' }.csv")
    logger.info(f"{'Server' if role == Role.SERVER else 'Client' } start")

    # print(features)
    # print(labels)
    # print(weights)
    test(weights, features, labels)
    weights, loss_array = grad_descent(role, features, labels, weights, client_feature_num)
    # r = np.mat(r).transpose()
    pass
