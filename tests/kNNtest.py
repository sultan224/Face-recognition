#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/8/18 6:04 PM

# 引入ｋＮＮClassifier
from faceyou.kNN import KNNClassifier
# 导入train_split，用于分割数据集
from faceyou import train_split
from sklearn import datasets

# 导入鸢尾花数据集
iris = datasets.load_iris()
set_x = iris.data
set_y = iris.target

# 分割数据，获取训练数据集和测试数据集
train_x, train_y, test_x, test_y = train_split.split(set_x, set_y)

# 实例化
kc = KNNClassifier(k=6)

# 验证数据集是否合法
kc.fit(train_x, train_y)
# 运行kNN，获取预测结果
predict_y = kc.predict_set(test_x)

# 与测试集中标签比较，统计准确率
accuracy = sum(predict_y == test_y) / len(test_y)
print('kNN训练准确率为:' + str(accuracy))
