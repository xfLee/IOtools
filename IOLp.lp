\* probIOLp *\
Minimize
OBJ: x_0_0_minus + x_0_0_plus + x_0_1_minus + x_0_1_plus + x_0_2_minus
 + x_0_2_plus + x_1_0_minus + x_1_0_plus + x_1_1_minus + x_1_1_plus
 + x_1_2_minus + x_1_2_plus + x_2_0_minus + x_2_0_plus + x_2_1_minus
 + x_2_1_plus + x_2_2_minus + x_2_2_plus
Subject To
_C1: 400 x_0_0_a + 500 x_0_1_a + 600 x_0_2_a = 180
_C10: x_2_0_a - x_2_0_minus + x_2_0_plus = 0.2
_C11: x_2_2_a - x_2_2_minus + x_2_2_plus = 0.2
_C12: x_2_1_a - x_2_1_minus + x_2_1_plus = 0.1
_C13: x_1_2_a - x_1_2_minus + x_1_2_plus = 0.18
_C14: x_1_0_a - x_1_0_minus + x_1_0_plus = 0.3
_C15: x_0_0_a - x_0_0_minus + x_0_0_plus = 0
_C2: 400 x_1_0_a + 500 x_1_1_a + 600 x_1_2_a = 330
_C3: 400 x_2_0_a + 500 x_2_1_a + 600 x_2_2_a = 220
_C4: 400 x_0_0_a + 400 x_1_0_a + 400 x_2_0_a = 200
_C5: 500 x_0_1_a + 500 x_1_1_a + 500 x_2_1_a = 240
_C6: 600 x_0_2_a + 600 x_1_2_a + 600 x_2_2_a = 290
_C7: x_0_1_a - x_0_1_minus + x_0_1_plus = 0.3
_C8: x_1_1_a - x_1_1_minus + x_1_1_plus = 0.15
_C9: x_0_2_a - x_0_2_minus + x_0_2_plus = 0.08
Bounds
x_0_0_a <= 1
x_0_0_minus <= 1
x_0_0_plus <= 1
x_0_1_a <= 1
x_0_1_minus <= 1
x_0_1_plus <= 1
x_0_2_a <= 1
x_0_2_minus <= 1
x_0_2_plus <= 1
x_1_0_a <= 1
x_1_0_minus <= 1
x_1_0_plus <= 1
x_1_1_a <= 1
x_1_1_minus <= 1
x_1_1_plus <= 1
x_1_2_a <= 1
x_1_2_minus <= 1
x_1_2_plus <= 1
x_2_0_a <= 1
x_2_0_minus <= 1
x_2_0_plus <= 1
x_2_1_a <= 1
x_2_1_minus <= 1
x_2_1_plus <= 1
x_2_2_a <= 1
x_2_2_minus <= 1
x_2_2_plus <= 1
End
