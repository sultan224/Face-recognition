#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/8/18 5:19 PM

import numpy as np


def split(set_x, set_y, ratio=0.2, seed=None):
    assert set_x.shape[0] == set_y.shape[0], \
        'the axis of set_x must be equal to that of set_y'
    assert 0.0 <= ratio <= 1.0, \
        'ratio must be between 0.0 and 1.0'

    if seed:
        np.random.seed(seed)

    # 根据集合数量获取N个取值为[0, N-1]的随机数
    rand_index = np.random.permutation(len(set_x))
    # 计算测试集数量
    num = int(len(set_x) * ratio)

    # 确定索引，此时索引是随机的
    test_index = rand_index[:num]
    train_index = rand_index[num:]

    # 获取训练集和测试集，因为索引的随机性，保证了所分配集合的随机性
    train_x = set_x[train_index]
    train_y = set_y[train_index]
    test_x = set_x[test_index]
    test_y = set_y[test_index]

    return train_x, train_y, test_x, test_y
