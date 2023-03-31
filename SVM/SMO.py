#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/31 20:05
# @Author : cc
# @File : SMO.py
# @Software: PyCharm

import random
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


def load_data(path):
    """ 数据集预处理
    :param path: 数据集路径
    :return: ndarray 样本 + label
    """
    data = pd.read_csv(path, header=None)
    new_columns = [str(n) for n in range(32)]
    data.columns = new_columns
    data['1'] = data['1'].replace({'M': 1, 'B': 0})  # 恶性：1 良性：0
    y = data['1'].values  # ndarray
    data.drop(["0", "1"], axis=1, inplace=True)
    X = data.values
    return X, y



def linear_kernel(X):
    """
    Generate linear kernel matrix.

    Args:
    X: numpy.ndarray, shape (n_samples, n_features)
       The input data matrix.

    Returns:
    K: numpy.ndarray, shape (n_samples, n_samples)
       The computed linear kernel matrix.
    """
    return np.dot(X, X.T)

def polynomial_kernel(X, degree=3, gamma=1): # 多项式核函数
    return (gamma * np.dot(X, X.T) + 1) ** degree


def rbf_kernel_matrix(X, sigma=1):
    # 计算样本点间的欧几里得距离的平方矩阵
    pairwise_dists = cdist(X, X, 'euclidean') ** 2
    # 计算RBF核矩阵
    K = np.exp(-pairwise_dists / (2 * sigma ** 2))
    return K



def SMO(X, y, C, toler, maxIter):
    """
    :param X: 样本
    :param y: label
    :param C: 正则化项系数
    :param toler: 容忍度
    :param maxIter: 最大迭代次数
    :return:
    """
    m, n = X.shape  # 数据集中样本数和特征数，特征数30
    alphas = np.zeros(m)  # 初始化拉格朗日乘数
    b = 0  # 初始化偏置项
    passes = 0  # 初始化迭代次数
    # 计算核矩阵
    kernelMat = linear_kernel(X)
    # kernelMat = polynomial_kernel(X)  # 计算核矩阵
    # kernelMat = rbf_kernel_matrix(X)  # 计算核矩阵
    # 训练
    while passes < maxIter:  # 迭代次数小于最大迭代次数
        num_changed_alphas = 0
        for i in range(m): # 遍历所有样本
            # 计算样本xi的预测值（根据支持向量展式）：kernelMat[:, i]是所有支持向量与当前样本i的核函数值，即核矩阵的第i列
            fXi = np.dot(alphas * y, kernelMat[:, i]) + b
            # 计算样本xi的预测误差 = 预测值- label
            Ei = fXi - float(y[i])
            # 判断是否满足KKT条件
            if ((y[i] * Ei < -toler) and (alphas[i] < C)) or ((y[i] * Ei > toler) and (alphas[i] > 0)):
                # 随机选择另一个样本xj，更新第二个拉格朗日乘数
                j = random.randint(0, m - 1)
                while j == i: # 不能是xi自身
                    j = random.randint(0, m - 1)
                fXj = np.dot(alphas * y, kernelMat[:, j]) + b # xj的预测值
                Ej = fXj - float(y[j]) # xj预测误差
                alphaIold = alphas[i].copy() # 原始ai
                alphaJold = alphas[j].copy() # 原始aj

                # 计算eta：偏导的一部分
                eta = 2.0 * kernelMat[i, j] - kernelMat[i, i] - kernelMat[j, j]
                if eta >= 0: # 当eta>=0时，无法保证参数更新后的值下降，因此需要跳出本次循环，进行下一次循环。
                    continue
                # 更新第二个拉格朗日乘数
                alphas[j] -= y[j] * (Ei - Ej) / eta



                # 对第二个拉格朗日乘数进行修剪
                    # 计算拉格朗日乘子的范围：方形约束(Bosk constraint)得到
                if y[i] != y[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    continue
                    # 修剪
                alphas[j] = min(H, alphas[j])
                alphas[j] = max(L, alphas[j])
                if abs(alphas[j] - alphaJold) < 1e-5:
                    continue
                # 更新第一个拉格朗日乘数
                alphas[i] += y[i] * y[j] * (alphaJold - alphas[j])
                # 更新偏置项
                b1 = b - Ei - y[i] * (alphas[i] - alphaIold) * kernelMat[i, i] \
                     - y[j] * (alphas[j] - alphaJold) * kernelMat[i, j]
                b2 = b - Ej - y[i] * (alphas[i] - alphaIold) * kernelMat[i, j] \
                     - y[j] * (alphas[j] - alphaJold) * kernelMat[j, j]
                if 0 < alphas[i] and alphas[i] < C:
                    b = b1
                elif 0 < alphas[j] and alphas[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                num_changed_alphas += 1
        if num_changed_alphas == 0: # 本轮训练参数没更新，迭代+1
            passes += 1
        else:
            passes = 0
    return alphas, b





if __name__ == '__main__':
    # 加载数据，样本 + label
    X, y = load_data('SVM/wdbc.data')
    # 划分训练集测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    # SVM模型训练
    C = 1.0  # 软间隔SVM的正则化参数
    toler = 0.001  # SMO算法的容错率
    maxIter = 5  # SMO算法的最大迭代次数
    # SVM算法执行，返回模型参数
    alphas, b = SMO(X_train, y_train, C, toler, maxIter)
    # SVM模型预测
    kernelMat = np.dot(X_train, X_train.T)  # 计算训练集的核矩阵
    y_train_pred = np.sign(np.dot(alphas * y_train, kernelMat) + b)  # 预测训练集
    kernelMat = np.dot(X_test, X_train.T)  # 计算测试集和训练集之间的核矩阵
    y_test_pred = np.sign(np.dot(alphas * y_train, kernelMat.T) + b)  # 预测测试集
    # SVM模型评估
    train_accuracy = np.mean(y_train_pred == y_train)  # 计算训练集精度
    test_accuracy = np.mean(y_test_pred == y_test)  # 计算测试集精度
    print('训练集精度：', train_accuracy)
    print('测试集精度：', test_accuracy)

