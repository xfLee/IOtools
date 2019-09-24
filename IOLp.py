# -*- coding: utf-8 -*-
# @Time    : 2019/9/23
# @Author  : github.com/xfLee

import pandas as pd
import numpy as np
from pulp import *

Z0 = np.array([[0, 120, 40],
               [90, 60, 90],
               [60, 40, 100]])
X0 = np.array([300, 400, 500])
A0 = np.dot(Z0, np.linalg.inv(np.diag(X0)))
X1 = np.array([400, 500, 600])
u1 = np.array([180, 330, 220])
v1 = np.array([200, 240, 290])

w_plus_set = set([str(i) + "_" + str(j) + "_plus" for i in range(A0.shape[0]) for j in range(A0.shape[1])])
w_minus_set = set([str(i) + "_" + str(j) + "_minus" for i in range(A0.shape[0]) for j in range(A0.shape[1])])

prob = LpProblem('probIOLp', LpMinimize)
w_plus = LpVariable.dicts('x', w_plus_set, 0, 1, LpContinuous)
w_minus = LpVariable.dicts('x', w_minus_set, -1, 0, LpContinuous)
x = dict(w_plus, **w_minus)

# Sequential
# Objective function
prob += lpSum(i for i in w_plus.values()) + lpSum(j for j in w_minus.values())

# Constraints
for i in range(A0.shape[0]):
    w_plus_id = [id for id in w_plus.keys() if id.split("_")[0] == i ]
    w_minus_id = [id for id in w_minus.keys() if id.split("_")[0] == i]
    prob += lpSum(w_plus[w_id] * X1[int(w_id.split("_")[1])] for w_id in w_plus_id) + lpSum(w_minus[w_id] * X1[int(w_id.split("_")[1])] for w_id in w_minus_id) + lpSum(tp[0] * tp[1] for tp in zip(A0[i, :], X1)) <= u1[i]

for j in range(A0.shape[1]):
    w_plus_id = [id for id in w_plus.keys() if id.split("_")[1] == j]
    w_minus_id = [id for id in w_minus.keys() if id.split("_")[1] == j]
    prob += X1[j] * (lpSum(w_plus[w_id] for w_id in w_plus_id) + lpSum(w_minus[w_id] for w_id in w_minus_id) + lpSum(A0[:, j])) <= v1[j]

for key in list(w_plus.keys()):
    prob += w_plus[key] - w_minus[key.split("_")[0] + "_" + key.split("_")[1] + "_minus"] + A0[int(key.split("_")[0]), int(key.split("_")[1])] >= 0

prob.writeLP('ALL.lp')

# Solve
prob.solve(PULP_CBC_CMD(cuts='on', msg=1, threads=4))
print(prob.objective.value())
result =[(id, x[id].varValue) for id in x.keys()]
print(result)
