#%%
import matplotlib.pyplot as plt
#%%
class Tree(object):
    def __init__(self,node_type,category=None,feature=None,\
        feature_idx=None,mydict=None):
        self.node_type = node_type  # 节点类型
        self.category = category    # 叶子节点表示的类，若是内部节点则为空
        self.feature = feature      # 表示当前的树由第feature个特征进行划分，方便可视化
        if not mydict:
            self.mydict = {}
        else:
            self.mydict = mydict    # 字典

    def add_subtree(self, key, subtree):
        self.mydict[key] = subtree

    def predict(self, x):
        if self.node_type == 'LEAF'\
            or x[self.feature] not in self.mydict:
            return self.category
        tree = self.mydict.get(x[self.feature])
        return tree.predict(x)
#%%
decision_node = dict(boxstyle="square", fc=(1.0,1.0,0.8784313725490196))
leaf_node = dict(boxstyle="round4", fc=(0.5647058823529412,0.9333333333333333,0.5647058823529412))
arrow_args = dict(arrowstyle="<-")

def plot_node(node_txt, center_pt, parent_pt, node_type):
    create_plot.ax1.annotate(node_txt, xy=parent_pt, \
        xycoords='axes fraction', xytext=center_pt, 
        textcoords='axes fraction', va="center", ha="center",\
        bbox=node_type, arrowprops=arrow_args)

def get_leafnum(mytree):
    num_leafs = 0
    tree_dict = mytree.mydict
    for key in tree_dict.keys():
        if tree_dict[key].node_type == "INTERNAL":
            num_leafs += get_leafnum(tree_dict[key])
        else: 
            num_leafs += 1
    return num_leafs

def get_treedepth(mytree):
    max_depth = 0
    tree_dict = mytree.mydict
    for key in tree_dict.keys():
        if tree_dict[key].node_type == "INTERNAL":
            depth = 1 + get_treedepth(tree_dict[key])
        else:
            depth = 1
        if depth > max_depth:
            max_depth = depth
    return max_depth

#%%
def retrieve_tree(i):
    dict1 = {0:Tree("LEAF", category='no'), 1: Tree("LEAF", category='yes')}
    tree1 = Tree(node_type="INTERNAL",feature="flippers",mydict=dict1)
    dict2 = {0:Tree("LEAF",category='no'),1:tree1}
    mytree1 = Tree("INTERNAL", feature="no surfacing",mydict=dict2)
    tree3 = Tree("INTERNAL",feature="head",mydict=dict1)
    dict3 = {0:tree3,1:Tree("LEAF",category='no')}
    tree4 = Tree("INTERNAL",feature="flippers", mydict=dict3)
    dict4 = {0:Tree("LEAF",category='no'),1:tree4}
    mytree2 = Tree("INTERNAL", feature="no surfacing",mydict=dict4)
    trees_list = [mytree1, mytree2]
    return trees_list[i]

#%%
def plot_mid_text(cntr_pt, parent_pt, txt):
    xMid = (parent_pt[0]-cntr_pt[0])/2.0 + cntr_pt[0]
    yMid = (parent_pt[1]-cntr_pt[1])/2.0 + cntr_pt[1]
    create_plot.ax1.text(xMid, yMid, txt)

def plot_tree(mytree, parent_pt, node_text):
    num_leafs = get_leafnum(mytree)
    depth = get_treedepth(mytree)
    cntr_pt = (plot_tree.xOff + (1.0 + float(num_leafs))/2.0/\
        plot_tree.totalW, plot_tree.yOff)
    plot_mid_text(cntr_pt, parent_pt, node_text)
    plot_node(mytree.feature, cntr_pt, parent_pt, decision_node)
    tree_dict = mytree.mydict
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in tree_dict.keys():
        if tree_dict[key].node_type == "INTERNAL":
            plot_tree(tree_dict[key], cntr_pt, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            plot_node(tree_dict[key].category, (plot_tree.xOff, plot_tree.yOff),
                cntr_pt, leaf_node)
            plot_mid_text((plot_tree.xOff, plot_tree.yOff),cntr_pt, str(key))
    plot_tree.yOff = plot_tree.yOff + 1.0/plot_tree.totalD

def create_plot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plot_tree.totalW = float(get_leafnum(tree))
    plot_tree.totalD = float(get_treedepth(tree))
    plot_tree.xOff = -0.5/plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(tree,(0.5,1.0),'')
    plt.show()
