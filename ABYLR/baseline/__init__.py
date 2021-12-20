import numpy as np
from utils.comm import recvby, sendby
from utils.logging import logger
from utils.timing import timecal
from utils.ABY import Role


alpha = 1e-4
epochs = 200
batch_size = 100
batch_list = []


def communicate(role, data):
    data = str(data)
    if(role == Role.SERVER):
        other_data = recvby(role)
        sendby(role, data)
    else:
        sendby(role, data)
        other_data = recvby(role)
    return other_data


def loss_first_computation(role, loss):
    other_loss = eval(communicate(role, loss))
    return loss + other_loss


def loss_xtheta_sum_computation(role, number, x_theta):
    other_x_theta = np.array(eval(communicate(role, str(x_theta.tolist()))))
    return x_theta + other_x_theta


def loss_mu_computation(role, number, x_theta):
    ret = 0
    x_theta = np.array(x_theta.T).squeeze().tolist()

    other_x_theta = communicate(role, str(x_theta))
    other_x_theta = np.mat(eval(other_x_theta))
    ret += np.dot(np.mat(x_theta), other_x_theta.T)
    return ret


def grad_communicate(role, number, wpart):
    return wpart + np.array(eval(communicate(role, wpart)))


def model_aggregation(weights):

    print(weights.shape)
    return weights


############################################################################################

def loss_compute(role, x_theta, theta, mu):

    # calculate the second
    loss = -1/2 * np.dot(mu, theta)

    loss = loss_first_computation(role, loss)

    # calculate the first
    number = x_theta.shape[0]
    x_theta = np.array(x_theta).squeeze()

    x_theta_sum = loss_xtheta_sum_computation(role, number, x_theta)
    x_theta_sum = np.around(x_theta_sum, decimals=6)

    loss = loss + 1/8 * np.dot(x_theta_sum, x_theta_sum)

    # divide h and plus log(2)
    return np.log(2) + loss / number


def grad_compute(role, x_theta, x, Y):
    x_theta, Y = np.array(x_theta).squeeze(), np.array(Y).squeeze()
    #assert(x_theta.shape[0] == Y.shape[0])
    number = x_theta.shape[0]

    # calculate w_a,w_b
    if(role == Role.SERVER):
        wpart = 1/4 * x_theta - 1/2 * Y
    else:
        wpart = 1/4 * x_theta
    wpart = wpart .tolist()

    # W <- w_a + w_b

    w = np.mat(grad_communicate(role, number, wpart)).transpose()

    # X^T \times w
    return np.dot(x.transpose(), w) / number


def mu_cache(role, X, Y, client_feature_num):
    number = X.shape[0]

    if(role == Role.SERVER):
        mu = np.dot(Y.transpose(), X)

        for i in range(client_feature_num):
            loss_mu_computation(role, number, Y)
    else:
        assert(X.shape[1] == client_feature_num)
        mu = []
        for i in range(client_feature_num):
            col = np.array(X[:, i]).squeeze()

            ret = loss_mu_computation(role, number, col)
            mu.append(ret)
        mu = np.array(mu)
    return np.mat(mu)


def batch_train(role, features, labels, weights):

    for batch in batch_list:
        batch_features = features[batch[0]:batch[1], :]
        batch_labels = labels[batch[0]:batch[1], :]

        x_theta = np.dot(batch_features, weights)

        grad = grad_compute(role, x_theta, batch_features, batch_labels)
        weights = weights - alpha * grad
        weights = weights.reshape(weights.shape[0], 1)

    return weights


def grad_descent(role, features, labels, weights, client_feature_num):

    loss_array = []
    num = int(features.shape[0] / batch_size)
    for i in range(num):
        batch_list.append([i*batch_size, (i+1)*batch_size])
    batch_list.append([num*batch_size, features.shape[0]])

    mu = timecal(mu_cache)(role, features, labels, client_feature_num)
    print(mu)

    for i in range(epochs):
        logger.debug(f"Epoch: {i}")

        x_theta = np.dot(features, weights)

        weights = timecal(batch_train)(role, features, labels, weights)

        x_theta = np.dot(features, weights)
        loss = timecal(loss_compute)(role, x_theta, weights,  mu)

        print(f"weights :\n {weights}")
        # print(f"grad :\n {grad}")
        loss_array.append(loss)
        test(role, weights, features, labels)
        print(f"loss : {loss}")

    print("loss is: ", loss)
    # print("weights shape: ", weights.shape)
    print("weights is: ", weights)

    return weights, loss_array


def test(role, weights, features, labels):
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

    x_theta = np.array(np.dot(features, weights)).T.squeeze().tolist()
    other_x_theta = eval(communicate(role, str(x_theta)))
    x_theta = np.array(x_theta) + np.array(other_x_theta)

    pred = sigmoid(x_theta)

    cnt = 0
    total = len(labels)
    for i, j in zip(pred, labels):
        if (i < 0.5 and j == 0.0) or (i >= 0.5 and j == 1.0):
            cnt = cnt + 1

    # np.set_printoptions(threshold=np.inf)
    # print(np.column_stack((pred, labels)))

    logger.info(f"Accuracy: {(cnt/total)*100:.3f}%")
    # assert(False)
    pass


def main(role, features, labels, weights, client_feature_num):
    logger.info(f"{'Server' if role == Role.SERVER else 'Client' } start")

    # print(features)
    # print(labels)
    # print(weights)
    # test(weights, features, labels)
    weights, loss_array = grad_descent(role, features, labels, weights, client_feature_num)
    # r = np.mat(r).transpose()
    pass


def fuck(role):
    for i in range(10000):
        if(role == Role.CLIENT):
            print(communicate(role, str(i)))
        else:
            print(communicate(role, str(i)))
