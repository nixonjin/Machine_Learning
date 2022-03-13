#作业题目：https://www.zybuluo.com/wujiaju/note/1619984
from sklearn.datasets import load_svmlight_file
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    "'sigmoid函数'"
    s = 1.0 / (1.0 + np.exp(-x))
    return s


def load_data():
    "'数据预处理'"
    #你可以在 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a 下载数据
    filepath1 = r"D:\ACoder\AllMyLab\MLLab\Lab2\data\a9a_train.txt"  #convert to your path
    train_data = load_svmlight_file(filepath1)
    X_train = train_data[0].toarray()
    y_train = train_data[1]
    filepath2 = r"D:\ACoder\AllMyLab\MLLab\Lab2\data\a9a_train.txt"   #convert to you path
    test_data = load_svmlight_file(filepath2)
    X_test = test_data[0].toarray()
    y_test = test_data[1]

    # 去不整齐的数据
    if X_train.shape[1] > X_test.shape[1]:
        max_colomn = X_train.shape[1] - 1
        X_train = np.delete(X_train, max_colomn, axis=1)
    elif X_train.shape[1] < X_test.shape[1]:
        max_colomn = X_test.shape[1] - 1
        X_test = np.delete(X_test, max_colomn, axis=1)

    # 将label从(-1,1)->(0,1)
    for i in range(len(y_train)):
        if y_train[i] == -1:
            y_train[i] = 0
    for j in range(len(y_test)):
        if y_test[j] == -1:
            y_test[j] = 0

    # 向Xtain Xtest中粘贴一列1以模拟b
    rows_train = X_train.shape[0]
    rows_test = X_test.shape[0]
    colomn = X_train.shape[1]
    X_train = np.insert(X_train, colomn - 1, np.ones(rows_train), axis=1)
    X_test = np.insert(X_test, colomn - 1, np.ones(rows_test), axis=1)
    return X_train, X_test, y_train, y_test


def batch_generator(X, y, batch_size):
    nsamples = len(X)
    batch_num = int(nsamples / batch_size)
    indexes = np.random.permutation(nsamples)
    for i in range(batch_num):
        yield (X[indexes[i*batch_size:(i+1)*batch_size]],
               y[indexes[i*batch_size:(i+1)*batch_size]])


def cross_entropy_loss(y_true, y_pred):
    loss = 0
    y_true = np.squeeze(y_true)
    y_pred = np.squeeze(y_pred)
    for k in range(len(y_pred)):
        loss += - y_true[k] * np.log(y_pred[k] + 1e-5) - (
            1 - y_true[k]) * np.log(1 - y_pred[k] + 1e-5)
    return loss/len(y_pred)


def train(alpha, beta1, beta2, epoches, epsilon, batch_size):
    # 初始化参数
    X_train, X_test, y_train, y_test = load_data()
    colomn = X_train.shape[1]
    print(colomn)
    W = np.random.uniform(low=1.0, high=10.0, size=(1, colomn))
    W_pre = np.zeros([1, colomn])
    # 训练参数
    losses_val = []
    m = np.zeros([1, colomn])
    v = np.zeros([1, colomn])
    t = 1
    while t <= epoches:
        for x_batch, y_batch in batch_generator(X_train, y_train, batch_size):
            y = sigmoid(np.dot(W, x_batch.transpose()))
            deviation = y - y_batch.reshape(y.shape)
            gradient = 1/len(x_batch)*np.dot(deviation, x_batch)

            # Adam优化方法
            m = beta1 * m + (1 - beta1) * gradient
            derivative_square = np.array([i*i for i in \
                                          np.squeeze(gradient)]).reshape(gradient.shape)
            v = beta2 * v + (1 - beta2) * derivative_square
            m_hat = m / (1 - pow(beta1, t))
            v_hat = v / (1 - pow(beta2, t))
            temp = m_hat / (v_hat**0.5 + epsilon)
            W = W - alpha * temp

        # 判断是否收敛
        if np.linalg.norm(W - W_pre) < 1e-20:
            print("中断")
            break
        else:
            W_pre = W

        # 在验证集上进行验证
        y_pred = sigmoid(np.dot(W, X_test.T))
        losses_val.append(cross_entropy_loss(y_test, y_pred))
        print("第{}次训练，在验证机上的损失函数值为：{}".format(t, losses_val[t-1]))
        t += 1
    return losses_val


def plot():
    "'画图'"
    losses_val = train(0.005, 0.9, 0.999, 200, 1e-6, 500)
    plt.figure()
    plt.title("logistic_model")
    plt.xlabel("training_number")
    plt.ylabel("losses_val")
    plt.plot(np.arange(len(losses_val)), losses_val, color='orange')
    plt.show()


plot()
