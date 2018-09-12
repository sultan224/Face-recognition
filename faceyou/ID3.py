#!/usr/bin/python
# -*- coding: utf-8 -*-
# @Time    : 9/10/18 8:21 PM

from math import log
import operator


def calcShannonEnt(dataSet):
    # 数据条数
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        # 每行数据的最后一个列（类别）
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        # 统计有多少个类以及每个类的数量
        labelCounts[currentLabel] += 1
    shannonEnt = 0
    for key in labelCounts:
        # 计算单个类的熵值
        prob = float(labelCounts[key]) / numEntries
        # 累加每个类的熵值
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 创造示例数据
def createDateSet1():
    dataSet = [['长', '粗', '男'],
               ['短', '粗', '男'],
               ['短', '粗', '男'],
               ['长', '细', '女'],
               ['短', '细', '女'],
               ['短', '粗', '女'],
               ['长', '粗', '女'],
               ['长', '粗', '女']]
    # 两个特征
    labels = ['头发', '声音']
    return dataSet, labels


# 按某个特征分类后的数据
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最优的分类特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    # 原始的熵值
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy = 0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            # 按特征分类后的熵值
            newEntropy += prob * calcShannonEnt(subDataSet)
            # 原始熵与按特征分类后的熵的差值
        infoGain = baseEntropy - newEntropy
        # 若按某特征划分后，熵值减小的最大，则次特征为最优分类特征
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree


if __name__ == '__main__':
    dataSet, labels = createDateSet1()
    print(createTree(dataSet, labels))
