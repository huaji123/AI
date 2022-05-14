import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        '''根据训练数据集X获得数据的均值和方差'''
        self.mean_ = np.array([np.mean(X[:, i]) for i in range(X.shape[1])])
        self.scale_ = np.array([np.std(X[:, i]) for i in range(X.shape[1])])
        return self

    def transform(self, X):
        '''将X根据Standardcaler进行均值方差归一化处理'''
        resX = np.empty(shape=X.shape, dtype=float)
        for col in range(X.shape[1]):
            resX[:, col] = (X[:, col] - self.mean_[col]) / (self.scale_[col])
        return resX


iris = datasets.load_iris()  # 下载数据集
X = iris.data  # 获取特征值
y = iris.target  # 获取标签值

X_train, X_test, y_train, y_test = train_test_split(X, y)
StandardScaler = StandardScaler()
StandardScaler.fit(X_train)
X_train = StandardScaler.transform(X_train)
X_test = StandardScaler.transform(X_test)
print("训练集归一化后数据")
print(X_train)
print("测试集归一化后数据")
print(X_test)
