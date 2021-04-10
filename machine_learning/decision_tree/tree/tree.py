class Tree(object):
    def __init__(self,node_type,category=None,feature=None,mydict=None):
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
