#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/31 22:45
# @Author : cc
# @File : sklearn_SVM.py
# @Software: PyCharm
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd

# # 加载数据
# cancer = load_breast_cancer()
# X = cancer.data
# y = cancer.target
#
# # 数据预处理
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# # 分割数据集为训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

data = pd.read_csv('wdbc.data', header=None)
new_columns = [str(n) for n in range(32)]
data.columns = new_columns
data['1'] = data['1'].replace({'M': 1, 'B': 0})  # 恶性：1 良性：0
y = data['1'].values  # ndarray
data.drop(["0", "1"], axis=1, inplace=True) # delete identity id + diagnosis column
X = data.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建SVM模型
svm_model = SVC(kernel='rbf', C=1.0)

# 训练模型
svm_model.fit(X_train, y_train)

# 预测测试集
y_pred = svm_model.predict(X_test)

# 评估模型
accuracy = svm_model.score(X_test, y_test)
print("Accuracy: {:.2f}%".format(accuracy*100))
