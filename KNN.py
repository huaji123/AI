import numpy as np

class KNN(object):
    # 定义内置函数，方便自己传参，默认值为3
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        # 用于后期的模型计算，输入训练集目标数据以及训练数据x
        self.x = x
        self.y = y

    # 计算距离
    def _square_distance(self, v1, v2):
        # 计算欧式距离
        return np.sum(np.square(v1 - v2))

    def _vote(self, ys):
        ys_unique = np.unique(ys)
        vote_dict = {}
        for y in ys:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1
        sorted_vote_dict = sorted(vote_dict.items(), key=operator.itemgetter(1), reverse=True)
        return sorted_vote_dict[0][0]

    def predict(self, x):
        y_pred = []

        for i in range(len(x)):
            dist_arr = [self._square_distance(x[i], self.x[j]) for j in range(len(self.x))]