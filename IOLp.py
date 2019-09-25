# -*- coding: utf-8 -*-
# @Time    : 2019/9/23
# @Author  : github.com/xfLee

import numpy as np
from pulp import *

def IOLp(A0, X1, u1, v1):
    w_a_set = set([str(i) + "_" + str(j) + "_a" for i in range(A0.shape[0]) for j in range(A0.shape[1])])
    w_plus_set = set([str(i) + "_" + str(j) + "_plus" for i in range(A0.shape[0]) for j in range(A0.shape[1])])
    w_minus_set = set([str(i) + "_" + str(j) + "_minus" for i in range(A0.shape[0]) for j in range(A0.shape[1])])

    prob = LpProblem('probIOLp', LpMinimize)
    w_a = LpVariable.dicts('x', w_a_set, 0, 1, LpContinuous)
    w_plus = LpVariable.dicts('x', w_plus_set, 0, 1, LpContinuous)
    w_minus = LpVariable.dicts('x', w_minus_set, 0, 1, LpContinuous)
    x = dict(w_a, **dict(w_plus, **w_minus))

    # Sequential
    # Objective function
    prob += lpSum(i for i in w_plus.values()) + lpSum(j for j in w_minus.values())

    # Constraints
    for i in range(A0.shape[0]):
        w_a_id = [id for id in w_a.keys() if int(id.split("_")[0]) == i]
        prob += lpSum(w_a[w_id] * X1[int(w_id.split("_")[1])] for w_id in w_a_id) == u1[i]

    for j in range(A0.shape[1]):
        w_a_id = [id for id in w_a.keys() if int(id.split("_")[1]) == j]
        prob += X1[j] * (lpSum(w_a[w_id] for w_id in w_a_id)) == v1[j]

    for key in list(w_plus.keys()):
        prob += w_plus[key] - w_minus[key.split("_")[0] + "_" + key.split("_")[1] + "_minus"] + w_a[key.split("_")[0] + "_" + key.split("_")[1] + "_a"] - A0[int(key.split("_")[0]), int(key.split("_")[1])] == 0

    # prob += x['1_1_a'] == 0.150
    # prob += x['0_1_a'] == 0.260
    # prob += x['2_0_a'] == 0.173

    prob.writeLP('ALL.lp')

    # Solve
    prob.solve(PULP_CBC_CMD(cuts='on', msg=1, threads=4))
    # prob.solve()
    print(prob.objective.value())
    result =[(id, x[id].varValue) for id in x.keys()]
    print(result)
    A1 = np.zeros((A0.shape))
    A1_value = [(id, x[id].varValue) for id in x.keys() if "a" in id]
    for tp in A1_value:
        A1[(int(tp[0].split("_")[0]), int(tp[0].split("_")[1]))] = tp[1]
    return A1

def test():
    Z0 = np.array([[0, 120, 40],
                   [90, 60, 90],
                   [60, 40, 100]])
    X0 = np.array([300, 400, 500])
    A0 = np.dot(Z0, np.linalg.inv(np.diag(X0)))
    # Y1 = np.array([220, 170, 380])
    X1 = np.array([400, 500, 600])
    u1 = np.array([180, 330, 220])
    v1 = np.array([200, 240, 290])
    A1 = IOLp(A0, X1, u1, v1)
    print("通过线性规划求解更新年直接消耗系数矩阵A1：")
    print(A1)

test()