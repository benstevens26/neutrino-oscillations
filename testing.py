from main import *
from numpy import cos
from numpy import sin
import matplotlib.pyplot as plt
import numpy as np


def test_function(x):
    """has about 10 local minima in range 0 < x < 7"""
    f = cos(x) + 5*cos(1.6*x) - 2*cos(2*x) + 5*cos(4.5*x) + 7*cos(9*x)
    return f

def dim2_test_function(u):
    """has about 10 local minima in range 0 < x < 7"""
    x = u[0]
    y = u[1]
    f = x**2 + y**2
    return f

y_min = parabolic_minimiser_nd(dim2_test_function, axis=1, x0 = [-1,1], xrange = 10, full_output=False)

print(y_min)
# print(dim2_test_function(u=[0, 0.5]))

#
# xs = [coord[0] for coord in points]
# ys = [coord[1] for coord in points]
#
# x = np.linspace(0, 2*np.pi, 1000)
# y = test_function(x)
#
# plt.figure()
# plt.plot(xs, ys, 'o')
# plt.plot(x, y, linestyle="--")
# plt.vlines([0.3, 0.35, 0.5], ymin=min(y), ymax=max(y), linestyle="--")
# plt.show()

