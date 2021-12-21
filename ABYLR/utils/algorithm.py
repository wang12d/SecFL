import ctypes
import numpy as np
from utils.logging import logger
from utils.timing import timecal
from utils.baseline import communicate
from utils.common import Role
from sklearn.metrics import precision_score, accuracy_score, recall_score

secureLR = ctypes.CDLL("ABY/build/lib/libsecureLR.so")
secureLR.loss_mu_computation.restype = ctypes.c_double
secureLR.loss_first_computation.restype = ctypes.c_double


def loss_mu_computation(role, number, x_theta):
    ret = 0
    x_theta = np.array(x_theta.T).squeeze().tolist()

    other_x_theta = communicate(role, str(x_theta))
    other_x_theta = np.mat(eval(other_x_theta))
    ret += np.dot(np.mat(x_theta), other_x_theta.T)
    return ret


def ABY_loss_compute(role, encryted, x_theta, theta, mu):

    # calculate the second
    loss = -1/2 * np.dot(mu, theta)

    if (encryted):
        loss = secureLR.loss_first_computation(role, (ctypes.c_double)(loss))
    else:
        other_loss = eval(communicate(role, loss))
        loss = loss + other_loss

    # calculate the first
    number = x_theta.shape[0]
    x_theta = np.array(x_theta).squeeze()

    if (encryted):
        x_theta = (ctypes.c_double * number)(*x_theta)
        x_theta_sum = (ctypes.c_double * number)(*[])
        secureLR.loss_xtheta_sum_computation(role, number, x_theta,  x_theta_sum)
    else:
        other_x_theta = np.array(eval(communicate(role, x_theta.tolist())))
        x_theta_sum = x_theta + other_x_theta

    x_theta_sum = np.around(x_theta_sum, decimals=6)

    loss = loss + 1/8 * np.dot(x_theta_sum, x_theta_sum)

    loss = np.array(loss).squeeze()
    # divide h and plus log(2)
    return np.log(2) + loss / number


def ABY_grad_compute(role, encryted, x_theta, x, Y):
    x_theta, Y = np.array(x_theta).squeeze(), np.array(Y).squeeze()
    assert(x_theta.shape[0] == Y.shape[0])
    number = x_theta.shape[0]

    # calculate w_a,w_b
    if(role == Role.SERVER):
        wpart = 1/4 * x_theta - 1/2 * Y
    else:
        wpart = 1/4 * x_theta
    wpart = wpart.tolist()

    # W <- w_a + w_b
    if (encryted):
        wpart = (ctypes.c_double * number)(*wpart)
        w = (ctypes.c_double * number)(*[])

        secureLR.grad_communicate(role, number, wpart,  w)
        w = np.array(w, dtype=np.float32).reshape(number, 1)
    else:
        other_wpart = np.array(eval(communicate(role, wpart)))
        w = np.mat(wpart+other_wpart).transpose()
    # X.transpose() \times w
    return np.dot(x.transpose(), w) / number


def mu_cache(role, config, X, Y):
    number = X.shape[0]

    if(role == Role.SERVER):
        mu = np.dot(Y.transpose(), X)
        if(config.encryted):
            Y = (ctypes.c_double * number)(*Y)
            for i in range(config.client_feature_num):
                secureLR.loss_mu_computation(role, number, Y)
        else:
            for i in range(config.client_feature_num):
                loss_mu_computation(role, number, Y)
    else:
        assert(X.shape[1] == config.client_feature_num)
        mu = []
        if(config.encryted):
            for i in range(config.client_feature_num):
                col = np.array(X[:, i]).squeeze().tolist()
                col = (ctypes.c_double * number)(*col)
                ret = secureLR.loss_mu_computation(role, number, col)
                mu.append(ret)
        else:
            for i in range(config.client_feature_num):
                col = np.array(X[:, i]).squeeze()
                ret = loss_mu_computation(role, number, col)
                mu.append(ret)
        mu = np.array(mu, dtype=np.float32)
    return np.mat(mu)


def batch_train(role, config, features, labels, weights):

    for batch in config.batch_list:
        batch_features = features[batch[0]:batch[1], :]
        batch_labels = labels[batch[0]:batch[1], :]
        x_theta = np.dot(batch_features, weights)
        grad = ABY_grad_compute(role, config.encryted, x_theta, batch_features, batch_labels)
        weights = weights - config.alpha * grad
        weights = weights.reshape(weights.shape[0], 1)

    return weights


def grad_descent(role, config, X_train, X_test, y_train, y_test, weights):

    loss_array = []
    num = int(X_train.shape[0] / config.batch_size)
    for i in range(num):
        config.batch_list.append([i*config.batch_size, (i+1)*config.batch_size-1])
    config.batch_list.append([num*config.batch_size, X_train.shape[0]])

    mu = timecal(mu_cache)(role, config, X_train, y_train)

    for i in range(config.epochs):
        logger.debug(f"Epoch: {i}")

        x_theta = np.dot(X_train, weights)

        weights = timecal(batch_train)(role, config, X_train, y_train, weights)

        x_theta = np.dot(X_train, weights)
        loss = timecal(ABY_loss_compute)(role, config.encryted, x_theta, weights,  mu)

        print(f"weights :\n {weights}")
        loss_array.append(loss)
        test(role, config, weights, X_test, y_test)
        print(f"loss : {loss}")

    # print("weights shape: ", weights.shape)
    print("weights is: ", weights)

    return weights, loss_array


def test(role, config, weights, features, labels):
    def sigmoid(x):
        x_ravel = x.ravel()
        length = len(x_ravel)
        y = []
        for index in range(length):
            if x_ravel[index] >= 0:
                y.append(1.0 / (1 + np.exp(-x_ravel[index])))
            else:
                y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
        return np.array(y).reshape(x.shape)

    x_theta = np.array(np.dot(features, weights)).T.squeeze().tolist()
    other_x_theta = eval(communicate(role, str(x_theta)))
    x_theta = np.array(x_theta) + np.array(other_x_theta)
    labels = np.array(labels.transpose()).squeeze().tolist()

    pred = sigmoid(x_theta).tolist()

    for i in range(len(pred)):
        if (pred[i] < config.threshold):
            pred[i] = 0.0
        else:
            pred[i] = 1.0

    logger.info(f"Accuracy Score: {accuracy_score(labels,pred) *100:.3f}%")
    logger.info(f"Precision Score: {precision_score(labels,pred) *100:.3f}%")
    logger.info(f"Recall Score: {recall_score(labels,pred) *100:.3f}%")


def train(role, config, X_train, X_test, y_train, y_test):
    logger.info(f"{'Server' if role == Role.SERVER else 'Client' } start")

    weights = np.mat(np.zeros((np.shape(X_train)[1], 1), dtype=np.float32))
    weights, _ = grad_descent(role, config, X_train, X_test, y_train, y_test, weights)
