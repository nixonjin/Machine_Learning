#%%
import time
import mxnet as mx
import numpy as np 
import matplotlib.pyplot as plt 
from mxnet import gluon, init, autograd
from sklearn.datasets import load_svmlight_file
#%%
def load_data():
    "'数据预处理'"
    #你可以在 https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#a9a 下载数据
    #需要修改成你的文件路径
    filepath1 = r"D:\ACoder\AllMyLab\MLLab\Lab2\data\a9a_train.txt"
    train_data = load_svmlight_file(filepath1)
    X_train = train_data[0].toarray()
    y_train = train_data[1]
    filepath2 = r"D:\ACoder\AllMyLab\MLLab\Lab2\data\a9a_test.txt"
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

    X_train = mx.nd.array(X_train)
    X_test = mx.nd.array(X_test)
    y_train = mx.nd.array(y_train)
    y_test = mx.nd.array(y_test)
    return X_train, X_test, y_train, y_test

def plot(loss_train, loss_test):
    plt.figure()
    plt.title("Logistic Regression(SGD)")
    plt.xlabel("training_num")
    t = int(len(loss_train) * 0.95)
    plt.annotate(round(loss_train[t], 2),
                 xy=(t, loss_train[t]), xycoords='data',
                 xytext=(+10, +30), textcoords='offset points', fontsize=12, color='blue',
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.plot(np.arange(len(loss_train)), loss_train, label="loss_train", color='orange')
    plt.plot(np.arange(len(loss_train)), loss_test, label="loss_test", color='blue')
    plt.legend(loc='upper right')
    plt.show()

#%%
X_train, X_test, y_train, y_test = load_data()

#%%
net = gluon.nn.Sequential()
net.add(gluon.nn.Dense(1))
net.add(gluon.nn.Activation('sigmoid'))
net.initialize(init.Normal())
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = gluon.Trainer(net.collect_params(), 'sgd',{'learning_rate':0.05})

epoches = 15
batch_size = 100
dataset = gluon.data.ArrayDataset(X_train, y_train)
data_iterator = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

start = time.time()
loss_train = []
loss_test = []
#%%
for epoch in range(epoches):
    print("[INFO] epoch %s is running..."%epoch)
    for batch_x, batch_y in data_iterator:
        with autograd.record():
            ls = loss_fn(net(batch_x), batch_y)
        ls.backward()
        # step：更新参数 需在backward()后，record()外
        trainer.step(batch_size)
    l_test = loss_fn(net(X_test), y_test).mean().asscalar()
    l_train = loss_fn(net(X_train), y_train).mean().asscalar()
    loss_train.append(l_train)
    loss_test.append(l_test)

print('weight:{}'.format(net[0].weight.data()))
print('weight:{}'.format(net[0].bias.data()))
print('time cost:{:.2f}'.format(time.time()-start))
plot(loss_train, loss_test)

