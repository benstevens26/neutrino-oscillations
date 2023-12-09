from neutrino_oscillation import *
from numpy import cos
from numpy import sin
import matplotlib.pyplot as plt
import numpy as np
import neutrino_oscillation


def test_function(x):
    """has about 10 local minima in range 0 < x < 7"""
    f = cos(x) + 5*cos(1.6*x) - 2*cos(2*x) + 5*cos(4.5*x) + 7*cos(9*x)
    return f


def test_function_2(u):
    """simple test function for 2d minimisation"""
    x = u[0]
    y = u[1]
    f = x**2 + y**2
    return f


def test_function_3(u):
    """simple test function for 3d minimisation"""
    x = u[0]
    y = u[1]
    z = u[2]
    f = x**2 + y**2 + z**2
    return f


def test_function_4(u):
    """simple test function for 3d minimisation"""
    return u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2
