# encoding: utf-8
# author: Jackson Kim
# date: 2020/5/3
#%%
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
#%%
data = load_iris() 
features = data['feature_names']
x = data['data']
y = data['target']
data_train, data_val, label_train, label_val = train_test_split(x,y,test_size=0.2)
mytree= DecisionTreeClassifier()
mytree.fit(data_train, label_train)
label_pred = mytree.predict(data_val)
print(classification_report(label_val, label_pred))
dot_data = export_graphviz(mytree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("iris")
