#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/8/18 12:00 PM

import numpy as np
from math import sqrt
from collections import Counter


class KNNClassifier:

    # 初始化分类器
    def __init__(self, k):
        # 断言验证k值有效
        assert k >= 1, 'k is not valid'
        self.k = k
        self.x_train = None
        self.y_train = None

    # 验证数据
    def fit(self, x_train, y_train):
        # 验证训练数据size
        assert x_train.shape[0] == y_train.shape[0], \
            'x_train can not fit y_train'
        assert self.k <= x_train.shape[0], \
            'k can not greater than the axis of x_train'
        self.x_train = x_train
        self.y_train = y_train
        return self

    # 开始预测结果
    def predict_set(self, x_predict):
        assert self.x_train is not None and self.y_train is not None, \
            'data for predict must be fitted'
        assert self.x_train.shape[1] == x_predict.shape[1], \
            'the axis of x_predict must be equal to that of x_train'

        # 分别将每组数据进行预测
        y_predict = [self.predict(x) for x in x_predict]
        return np.array(y_predict)

    # 对于单组数据的预测
    def predict(self, x):
        # 计算欧式距离
        distances = [sqrt(np.sum(x_train - x) ** 2) for x_train in self.x_train]

        # 对距离进行排序
        nearest = np.argsort(distances)

        # 获取最近的k个点的距离所对应的y标签
        top_y = [self.y_train[i] for i in nearest[:self.k]]

        # 投票法确定预测值
        vote_result = Counter(top_y)
        return vote_result.most_common(1)[0][0]
