import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tqdm import tqdm

X, Y = fetch_california_housing(return_X_y=True)
test_size = 0.2
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

def get_batch(batch_size, X, Y):
    assert X.shape[0] % batch_size == 0
    batch_num = X.shape[0] // batch_size
    X_new = X.reshape(batch_num, batch_size, -1)
    Y_new = Y.reshape(batch_num, batch_size)
    for i in range(batch_num):
        yield X_new[i, :, :], Y_new[i, :]

def mse(X, Y, W):
    # @矩阵乘法
    return np.mean(np.square(X@W-Y)) * 0.5

def diff_mse(X, Y, W):
    return X.T @ (X@W-Y) / X.shape[0]

# lr一定要足够小，不然直接梯度爆炸
lr = 1e-8
batch_size = 64
epoch_num = 50000


def train():
    W = np.random.random(size=(X.shape[1]))
    loop = tqdm(range(epoch_num))
    for epoch in loop:
        for batch_x, batch_y in get_batch(batch_size, x_train, y_train):
            W_grad = diff_mse(batch_x, batch_y, W)
            W = W - lr * W_grad

        if epoch % 10 == 0:
            test_loss = mse(x_test, y_test, W)
            print("Epoch: {} test_loss: {}".format(epoch, test_loss))



train()