# -*- coding: utf-8 -*-
import numpy as np

class RAS(object):
    def __init__(self, A0, X1, u1, v1, epsilon, revisedElements):
        self.A0 = A0
        self.X1 = X1
        self.u1 = u1
        self.v1 = v1
        self.i = np.array([1, 1, 1])
        self.epsilon = epsilon
        self.revisedElements = revisedElements

    def revise(self):
        for idx, val in self.revisedElements.items():
            self.A0[idx] = 0
            self.X1[idx[0], idx[0]] -= val
            self.u1[idx[0]] -= val
            self.v1[idx[1]] -= val

    def run(self):
        Zk = np.dot(self.A0, self.X1)
        uk = np.dot(self.i, np.transpose(Zk))
        vk = np.dot(self.i, Zk)
        k = 1

        R = np.diag(self.u1 * (1 / uk))
        S = np.diag(self.v1 * (1 / vk))

        u1, v1 = self.u1, self.v1

        while sum(abs(u1 - uk)) >= self.epsilon or sum(abs(v1 - vk)) >= self.epsilon:
            Rk = u1 * (1 / uk)
            Zk = np.dot(np.diag(Rk), Zk)

            u1 = np.dot(self.i, np.transpose(Zk))
            vk = np.dot(self.i, Zk)

            Sk = v1 * (1 / vk)
            Zk = np.dot(Zk, np.diag(Sk))

            uk = np.dot(self.i, np.transpose(Zk))
            v1 = np.dot(self.i, Zk)

            if k == 1:
                R = np.diag(Rk)
                S = np.diag(Sk)
            else:
                R = np.dot(np.diag(Rk), R)
                S = np.dot(np.diag(Sk), S)
            k += 1

        Z2n = np.dot(np.dot(np.dot(R, self.A0), self.X1), S)
        A1 = np.dot(np.dot(R, self.A0), S)

        for idx, val in self.revisedElements.items():
            Z2n[idx] = val
            A1[idx] = val / (self.X1[idx[0], idx[0]] + val)

        return k, A1, Z2n, R, S

def test():
    A0 = np.array([[0, 120, 40],
                   [90, 60, 90],
                   [60, 40, 100]])
    X1 = np.diag([400, 500, 600])
    u1 = np.array([180, 330, 220])
    v1 = np.array([200, 240, 290])
    epsilon = 0.01
    revisedElements = {(1, 1): 75}
    ras = RAS(A0, X1, u1, v1, epsilon, revisedElements)
    ras.revise()
    k, A1, Z2n, R, S = ras.run()
    print("1.修正RAS法计算过程循环次数：" + str(k - 1))
    print("2.修正RAS法估计的目标年直接消耗系数矩阵A1：")
    print(A1)
    print("3.修正RAS法估计的目标年消耗矩阵A1*X1：")
    print(Z2n)
    print("4.修正RAS法估计的目标年R矩阵：")
    print(R)
    print("5.修正RAS法估计的目标年S矩阵：")
    print(S)

# if __name__ == "main":
test()