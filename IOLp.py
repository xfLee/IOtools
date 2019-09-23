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

w_plus_set = set([str(i) + "#" + str(j) + "#plus" for i in range(A0.shape[0]) for j in range(A0.shape[1])])
w_minus_set = set([str(i) + "#" + str(j) + "#minus" for i in range(A0.shape[0]) for j in range(A0.shape[1])])

prob = LpProblem('probIOLp', LpMinimize)
w_plus = LpVariable.dicts('x', w_plus_set, 0, 1, LpContinuous)
w_minus = LpVariable.dicts('x', w_minus_set, -1, 0, LpContinuous)
x = dict(w_plus, **w_minus)

# Sequential
# Target
prob += lpSum(i for i in w_plus.values()) + lpSum(j for j in w_minus.values())


