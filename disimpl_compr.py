from _shgo import *
import logging
import sys

class TestFunction(object):
    def __init__(self, bounds, expected_x, expected_fun=None,
                 expected_xl=None, expected_funl=None):
        self.bounds = bounds
        self.expected_x = expected_x
        self.expected_fun = expected_fun
        self.expected_xl = expected_xl
        self.expected_funl = expected_funl

class Test1(TestFunction):
    def f(self, x):
        return x[0]**2 + x[1]**2

    def g(self, x):
       return -(numpy.sum(x, axis=0) - 6.0)

test1_1 = Test1(bounds=[(-1, 6), (-1, 6)],
                expected_x=[0, 0])