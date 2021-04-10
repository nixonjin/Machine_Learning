from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
import numpy as np
import random
import time
def load_data(filename):
    data = load_svmlight_file(filename)
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.25)
    X_train = X_train.toarray()
    X_test = X_test.toarray()
    return X_train, X_test, y_train, y_test

def train():
    # 读取实验数据，切分数据集
    # you can dowload the dataset from https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#housing
    filename = 'D:\ACoder\AllMyLab\MLLab\Lab1\housing_scale.txt'
    X_train, X_test, y_train, y_test = load_data(filename)

    # 向Xtrain Xtest中粘贴一列1,代替bias
    b = np.ones(379)
    X_train = np.insert(X_train, 13, b, axis=1)
    X_test = np.insert(X_test, 13, b[0:127], axis=1)

    # 初始化参数,训练参数
    weight = np.random.uniform(low=0.0, high=5.0, size=(1, 14))
    learning_rate = 0.1
    loss_test = []
    loss_train = []
    epoches = 15
    batch_size = 200
    start = time.time()
    for epoch in range(epoches):
        for x_batch, y_batch in batch_generator(X_train, y_train, batch_size):
            y_hat = np.dot(weight, x_batch.T)
            deviation = y_hat - y_batch.reshape(y_hat.shape)
            gradient = 1/len(x_batch)* np.dot(deviation, x_batch)
            weight = weight - learning_rate*gradient
        loss1 = mse(y_train, np.dot(weight, X_train.T))
        loss_train.append(loss1)
        loss2 = mse(y_test, np.dot(weight, X_test.T))
        loss_test.append(loss2)
        print('epoch:{},loss:{:.4f}'.format(epoch+1,loss1))
    print('weight:',weight)
    print('time interval:{:.2f}'.format(time.time() - start))
    return loss_train, loss_test

def batch_generator(x, y, batch_size):
    nsamples = len(x)
    batch_num = int(nsamples / batch_size)
    indexes = np.random.permutation(nsamples)
    for i in range(batch_num):
        yield (x[indexes[i*batch_size:(i+1)*batch_size]], y[
            indexes[i*batch_size:(i+1)*batch_size]])

def mse(y,y_hat):
    m = len(y)
    mean_square = np.sum((y - y_hat.reshape(y.shape))**2) / (2 * m)
    return mean_square

# 画图
def plot(loss_train,loss_test):
    x_axis = []
    for i in range(len(loss_train)):
        x_axis.append(i)
    plt.figure()
    plt.title("Linear Regression(SGD)")
    plt.xlabel("training_num")
    t = int(len(loss_train) * 0.95)
    plt.annotate(round(loss_train[t], 2),
         xy=(t, loss_train[t]), xycoords='data',
         xytext=(+10, +30), textcoords='offset points', fontsize=12, color='blue',
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.plot(x_axis, loss_train, label="loss_train", color='orange')
    plt.plot(x_axis, loss_test, label="loss_test", color='blue')
    plt.legend(loc='upper right')
    plt.show()

if __name__ == "__main__":
    loss_train,loss_test = train()
    plot(loss_train,loss_test)
