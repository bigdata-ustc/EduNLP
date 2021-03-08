# coding: utf-8
# 2021/3/8 @ tongshiwei

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn import tree

clf = tree.DecisionTreeClassifier(random_state=0)
iris = load_iris()

clf = clf.fit(iris.data, iris.target)
anns = tree.plot_tree(clf, filled=True)
plt.show()