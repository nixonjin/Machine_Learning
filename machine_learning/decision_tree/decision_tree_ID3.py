# encoding: utf-8
# author: Jackson Kim
# time: 2020/5/3

#%%
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from collections import Counter
import copy
from tree import Tree


def cal_entropy(x):
    x_value_list = set(x)
    entropy = 0.0
    for x_value in x_value_list:
        prob = float(x[x == x_value].shape[0]) / x.shape[0]
        entropy -= prob * np.log2(prob)
    return entropy


def cal_entropy_slow(label):
    label_num = len(label)
    label_count = {}
    entropy = 0.0
    for each in label:
        label_count[each] = label_count.get(each, 0)+1
    for key in label_count.keys():
        prob = float(label_count[key]) / label_num
        entropy -= prob * np.log2(prob)
    return entropy


# 计算条件熵H(label/attr)
def cal_conditional_entropy(attr, label):
    attr_values = set(np.squeeze(attr))
    entropy = 0.0
    for value in attr_values:
        sub_label = label[attr == value]
        sub_entropy = cal_entropy(sub_label)
        entropy += (float(len(sub_label))/len(label))*sub_entropy
    return entropy


def create_tree(data, label, features):
    # 1.如果训练集中所有数据均属于同一类，即可返回
    label_set = set(label)
    if len(label_set) == 1:
        category = label_set.pop()
        subtree = Tree('LEAF', category=category)
        print("[INFO]: create a leaf node, category is %s"%(category))
        return subtree

    # 2.如果特征集为空
    label_set = set(label)
    max_class = label[0]
    max_num = 0
    for each in label_set:
        class_num = label[label==each].shape[0]
        if  class_num > max_num:
            max_class = each
            max_num = class_num
    if len(features) == 0:
        subtree = Tree('LEAF', category=max_class)
        print("[INFO]: create a leaf node, category is %s"%(max_class))
        return subtree

    # 3.找到信息增益最大的特征
    base_entropy = cal_entropy(label)
    best_feature = features[0]
    max_info_gain = 0
    for i in features:
        attr = data[:,i]
        info_gain = base_entropy - cal_conditional_entropy(attr, label)
        if info_gain > max_info_gain:
            best_feature = i
            max_info_gain = info_gain

    # 4.如果信息增益小于阈值，进行预剪枝
    # if max_info_gain < 0.001:
        # return Tree('LEAF', category=max_class)

    # 5.构建内部节点以及递归创造子树
    sub_features = list(filter(lambda x: x != best_feature, features))
    tree = Tree('INTERNAL', feature=best_feature, category=max_class)
    print("[INFO]: create a internal node，feature is %sth"%(best_feature))
    best_feature_values = set(data[:, best_feature])
    for value in best_feature_values:
        indexs = data[:, best_feature] == value
        sub_data = data[indexs]
        sub_label = label[indexs]
        sub_tree = create_tree(sub_data, sub_label, sub_features)
        tree.add_subtree(value, sub_tree)

    return tree

def predict(test_data, tree):
    result = []
    for each in test_data:
        result.append(tree.predict(each))
    return np.array(result)


def process_leafs_parent(val_data,val_label, 
            dataset, root, features, tree, path):
    # 在未剪枝的决策树上进行验证，并得出f1
    old_score = f1_score(val_label, predict(val_data, root),average='micro')
    # 将未剪枝的子树保存起来
    temp_tree = copy.deepcopy(tree)

    # 根据路径找到训练集中能到达该节点的数据
    df = pd.DataFrame(dataset, columns=features+['label'])
    query = str(path[0][0])+"=="+str(path[0][1])
    if len(path) >= 2:
        for each in path[1:]:
            query += " and "+str(each[0])+"=="+str(each[1])
    print(query)
    df = df.query(query)

    # 进行剪枝
    tree.category = Counter(df.iloc[:, -1]).most_common(1)[0][0]
    tree.node_type = "LEAF"
    tree.mydict = {}

    # 若剪枝后分数变低，那么还原子树
    new_score = f1_score(val_label, predict(val_data, root),average='micro')
    if(new_score < old_score):
        tree.category = temp_tree.category
        tree.node_type = temp_tree.node_type
        tree.mydict = temp_tree.mydict


def DFS(tree, path):
    if tree.node_type == "LEAF":
        print("[INFO]:post pruning, arrive at leaf node")
        return
    
    # 没有return，该节点是内部节点，检测该节点的子节点是不是都是叶子节点
    leafs_parent = True
    for sub_tree in tree.mydict.values():
        if sub_tree.node_type != "LEAF":
            leafs_parent = False
            break
    # 如果该节点的子节点全部是叶子节点
    if leafs_parent and len(path) > 0:
        process_leafs_parent(DFS.val_data,DFS.val_label, 
            DFS.dataset, DFS.root, DFS.features,tree, path)

    # 如果该节点的子节点也是内部节点，继续递归
    else:
        for key, sub_tree in tree.mydict.items():
            path.append((DFS.features[tree.feature], key))
            DFS(sub_tree, path)
            path.pop()

        # 对子节点的递归结束，
        # 再次检测该节点的子节点是不是都是叶子节点
        leafs_parent = True
        for sub_tree in tree.mydict.values():
            if sub_tree.node_type != "LEAF":
                leafs_parent = False
                break
        # 如果该节点的子节点全部是叶子节点
        if leafs_parent and len(path) > 0:
            process_leafs_parent(DFS.val_data,DFS.val_label, 
                DFS.dataset, DFS.root, DFS.features, tree, path)


def post_pruning(val_data, val_label, dataset, tree, features):
    path = []
    DFS.val_data = val_data
    DFS.val_label = val_label 
    DFS.dataset = dataset
    DFS.root = tree
    DFS.features = features
    DFS(tree, path)
    return tree


def find_best_split(attr, label, base_entropy, stride):
    # 进行深复制，不然会更改原来的值
    new_attr = copy.deepcopy(attr)
    new_label = copy.deepcopy(label)
    data = np.c_[new_attr, new_label]
    data = data[np.argsort(data[:,0])]
    if not stride:
        stride = int(len(attr)/100) + 1
    max_info_gain = 0
    max_gain_index = 0
    for i in range(0, data.shape[0], stride):
        entropy1 = cal_entropy(data[:i,1])
        entropy2 = cal_entropy(data[i:,1])
        proportion = float(i)/data.shape[0]
        info_gain =  proportion*entropy1+(1-proportion)*entropy2 - \
            base_entropy
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            max_gain_index = i
    return (data[max_gain_index, 0]+data[max_gain_index-1, 0]) / 2


def binaryzation_features(data, label, columns, stride=None):
    data = np.array(data)
    label = np.array(label)
    base_entropy = cal_entropy(label)
    for column in columns:
        print("[INFO]: doing binaryzation of feature %s ..."%column)
        split_point = find_best_split(data[:,column], label, \
            base_entropy, stride)
        print("[INFO]: best split point of column %s is %s"%(column,split_point))
        data[data[:, column] < split_point, column] = 0
        data[data[:, column] >= split_point, column] = 1
    return data, label

# %%
from sklearn.datasets import load_iris
from tree_plotter.tree_plotter import create_plot
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#%%
data = load_iris() 
features = data['feature_names']
x = data['data']
y = data['target']
#%%
data, label = binaryzation_features(x, y, columns=[0,1,2,3],stride=1)
data_train, data_val, label_train, label_val = train_test_split(data,label,test_size=0.2)
#%%
data_train, data_val, label_train, label_val = train_test_split(x,y,test_size=0.2)
#%%
mytree = create_tree(data_train, label_train,[0,1,2,3])
#%%
post_pruning(data_val, label_val, np.c_[data_train, label_train], mytree,['a','b','c','d'])
#%%
create_plot(mytree)
#%%
print(classification_report(label_val, predict(data_val, mytree)))
