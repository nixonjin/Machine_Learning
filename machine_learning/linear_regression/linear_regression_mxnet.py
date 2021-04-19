from sklearn.model_selection import train_test_split
from sklearn.datasets import load_svmlight_file
from mxnet.gluon import nn, Trainer
from mxnet.gluon import loss
from mxnet.gluon import data as gdata
from mxnet import init, autograd
import time
def load_data(filename):
    data = load_svmlight_file(filename)
    X_train, X_test, y_train, y_test = train_test_split(data[0], data[1], test_size=0.25)
    X_train = mx.nd.array(X_train.toarray())
    X_test = mx.nd.array(X_test.toarray())
    y_train = mx.nd.array(y_train)
    y_test = mx.nd.array(y_test)
    return X_train, X_test, y_train, y_test

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

# convert filename to your file path
# you can download the dataset from 
# https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression.html#housing
filename = 'D:\ACoder\AllMyLab\MLLab\Lab1\housing_scale.txt'
X_train, X_test, y_train, y_test = load_data(filename)
#定义模型
net = nn.Sequential()
net.add(nn.Dense(1))

# 初始化参数
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
loss_fn = loss.L2Loss()

# 定义训练器
trainer = Trainer(net.collect_params(), 'sgd', {'learning_rate':0.05})

epoches = 15
batch_size = 100
dataset = gdata.ArrayDataset(X_train,y_train)
data_iterator = gdata.DataLoader(dataset, batch_size, shuffle=True)

# 训练
start = time.time()
loss_train = []
loss_test = []
for epoch in range(epoches):
    for batch_x, batch_y in data_iterator:
        with autograd.record():
            l = loss_fn(net(batch_x), batch_y)
        l.backward()
        # step：更新参数 需在bachward()后，record()外
        trainer.step(batch_size)
    l_test = loss_fn(net(X_test), y_test).mean().asscalar()
    l_train = loss_fn(net(X_train),y_train).mean().asscalar()
    loss_train.append(l_train)
    loss_test.append(l_test)
    print('epoch:{}, loss_train:{}, loss_test:{}'.format(
        epoch+1, l_train, l_test))

print('weight:{}'.format(net[0].wright.data()))
print('weight:{}'.format(net[0].bias.data()))
print('time cost:{:.2f}'.format(time.time()-start))
plot(loss_train,loss_test)
