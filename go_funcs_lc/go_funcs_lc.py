# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy
from numpy import abs, cos, exp, pi, prod, sin, sqrt, sum
from .go_benchmark import Benchmark


class Horst1(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(0, 3), (0, 2)]
        self.global_optimum = [0.75, 2.0]
        self.fglob = -1.062

        def g1(x):
            return 1 - (-4 * x[0] + 2 * x[1])

        def g2(x):
            return 4 - (x[0] + x[1])

        def g3(x):
            return 1 - (x[0] - 4 * x[1])

        self.g = (g1, g2, g3)

        self.x0 = [1.5, 1.0]

    def fun(self, x):
        self.nfev += 1
        return -x[0]**2 - 4*x[1]**2 + 4*x[0]*x[1] + 2*x[0] + 4*x[1]


class Horst2(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self.fglob = -6.8995
        self.global_optimum = [2.5, 0.75]
        self._bounds = [(0, 2.5), (0, 2.0)]

        def g1(x):
            return 4 - (x[0] + 2 * x[1])

        def g2(x):
            return 1 - (x[0] - 2 * x[1])

        def g3(x):
            return 1 - (-x[0] + x[1])

        self.g = (g1, g2, g3)

        self.x0 = [1.0, 1.0]

    def fun(self, x):
        self.nfev += 1
        return -x[0]**2 - x[1]**(3/2.0)

class Horst3(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 1.5), ] * 2
        self.fglob = -4 / 9.0
        self.global_optimum = [0.0, 0.0]

        def g1(x):
            return 1 - (-2 * x[0] + x[1])

        def g2(x):
            return 3 / 2.0 - (x[0] + x[1])

        def g3(x):
            return 1 - (x[0] + 1 / 10.0 * x[1])

        self.g = (g1, g2, g3)

        self.x0 = [1.0, 1.0]

    def fun(self, x):
        self.nfev += 1
        return -x[0]**2 + 4/3.0 * x[0] + numpy.log(1 + x[1]) - 4/9.0


class Horst4(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 2.0), (0, 2.0), (0, 2.0)]
        self.fglob = -6.0858
        self.global_optimum = [2.0, 0.0, 2.0]

        def g1(x):
            return 6 - (x[0] + x[1] + 2 * x[2])

        def g2(x):
            return 2 - (x[0] + 0.5 * x[1])

        def g3(x):
            return -1 - (-x[1] - 2 * x[2])

        self.g = (g1, g2, g3)

        self.x0 = [0.0, 0.0, 0.0]


    def fun(self, x):
        self.nfev += 1
        return - (numpy.abs(x[0] + 0.5*x[1] + 2/3.0 * x[2]))**(3/2.0)


class Horst5(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 2), (0, 2), (0, 3)]
        self.fglob = -3.722
        self.global_optimum = [1.2, 0.0, 0.8]

        def g1(x):
            return 2 - (x[0] + x[1] + x[2])

        def g2(x):
            return 1 - (x[0] + x[1] - (1 / 4.0) * x[2])

        def g3(x):
            return -1 - (-2 * x[0] - 2 * x[1] + x[2])

        self.g = (g1, g2, g3)

        self.x0 = [1.0, 1.0, 1.5]

    def fun(self, x):
        self.nfev += 1
        return - (numpy.abs(x[0] + 0.5*x[1] + 2/3.0 * x[2]))**(3/2.0) - x[0]**2

class Horst6(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 10), ] * 3

        # Literature values unfeasible (close to value with wrong constraint)
        #self.fglob = -31.5285
        #self.global_optimum = [5.210677, 5.027908, 0.0]

        # shgo feasible value
        self.fglob = -12.003240034450666  # SHGO best
        self.global_optimum = [1.36656132, 2.96722951, 0.0]


        def g1(x):
            return 2.865062 - (0.488509 * x[0] + 0.063565 * x[1] + 0.945686 * x[2])

        def g2(x):
            return -1.491608 - (-0.578592 * x[0] - 0.324014 * x[1] - 0.501754 * x[2])

        def g3(x):
            return 0.519588 - (-0.719203 * x[0] + 0.099562 * x[1] + 0.445225 * x[2])

        def g4(x):
            return 1.584087 - (-0.346896 * x[0] + 0.637939 * x[1] - 0.257623 * x[2])

        def g5(x):
            # Paulavius textbook:
            # return  2.198036 - (-0.202821*x[0] + 0.647361*x[1] + 0.920135*x[2])
            # https://link.springer.com/content/pdf/10.1007/BF00429750.pdf
            return 2.198036 - (+0.202821 * x[0] + 0.647361 * x[1] + 0.920135 * x[2])

        def g6(x):
            return -1.301853 - (-0.983091 * x[0] - 0.886420 * x[1] - 0.802444 * x[2])

        def g7(x):
            return -0.738290 - (-0.305441 * x[0] - 0.180123 * x[1] - 0.515399 * x[2])

        self.g = (g1, g2, g3, g4, g5, g6, g6, g7)

        self.x0 = [1.0, ] * 7


    def fun(self, x):
        self.nfev += 1
        x = numpy.atleast_2d(x).T
        Q = numpy.array([[ 0.992934, -0.640117, 0.337286],
                         [-0.640117, -0.814622, 0.960807],
                         [ 0.337286,  0.960807, 0.500874]])

        q = numpy.array([[-0.992372],
                         [-0.046466],
                         [ 0.891766]])

        xT_Q = numpy.dot(x.T, Q)
        return numpy.dot(xT_Q, x) + numpy.dot(q.T, x)

class Horst7(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 6), (0, 1), (0, 3)]
        #self.fglob = -44.859
        self.fglob = -52.87741699796952  # NOTE LOWER VALUE THAN LITERATURE
        #self.global_optimum = [6.0, 0.0, 2.0]
        self.global_optimum = [ 6.,  0.,  3.] # NOTE LOWER VALUE THAN LITERATURE

        def g1(x):
            return 1 - (-x[0] - x[1] + (1 / 2.0) * x[2])

        def g2(x):
            return 6 - (x[0] + 2 * x[1])

        def g3(x):
            return - 1 + (2 * x[0] + 4 * x[1] + 2 * x[2])

        self.g = (g1, g2, g3)

        self.x0 = [1.0, 1.0, 1.0]


    def fun(self, x):
        self.nfev += 1
        return -(x[0] + (1/2.0) * x[2] - 2)**2 - (numpy.abs(x[0] + 0.5*x[1] + 2/3.0 * x[2]))**(3/2.0)



class Hs021(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(2, 50), (-50, 50)]
        self.fglob = -99.96
        self.global_optimum = [2.0, 0.0]

        def g1(x):
            return 10 * x[0] - x[1] - 10

        self.g = (g1,)

        self.x0 = [-1, -1]

    def fun(self, x):
        self.nfev += 1
        return x[0]**2/100 + x[1]**2 - 100


class Hs024(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(0, 5), ] * 2  # TODO: (0, None) ?
        self.fglob = -1.0
        self.global_optimum = [3.0, numpy.sqrt(3)]

        def g1(x):
            return (x[0] / numpy.sqrt(3) - x[1])

        def g2(x):
            return (x[0] + numpy.sqrt(3) * x[1])

        def g3(x):
            return -x[0] - numpy.sqrt(3) * x[1] + 6

        self.g = (g1, g2, g3)

        self.x0 = [1.0, 0.5]

    def fun(self, x):
        self.nfev += 1
        return (((x[0] - 3)**2 - 9.0) * x[1]**3) / (27.0*numpy.sqrt(3))

class Hs035(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 5), ] * 3  # TODO: (0, None) ?
        self.fglob = 1 / 9.0
        self.global_optimum = [4 / 3.0, 7 / 9.0, 4 / 9.0]

        def g1(x):
            return -(x[0] + x[1] + 2 * x[2] - 3)

        self.g = (g1,)

        self.x0 = [0.5, 0.5, 0.5]

    def fun(self, x):
        self.nfev += 1
        return 9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2+ 2*x[0]*x[1] + 2*x[0]*x[2]


class Hs036(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 20), (0, 11), (0, 42)]
        self.fglob = -3300
        self.global_optimum = [20, 11, 15]

        def g1(x):
            #return (x[0] + 2 * x[1] + 2 * x[2] - 72)
            return -x[0] - 2 * x[1] - 2 * x[2] + 72

        self.g = (g1,)

        self.x0 = [10.0, 10.0, 10.0]

    def fun(self, x):
        self.nfev += 1
        return -x[0]*x[1]*x[2]


class Hs037(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 42), ] * 3
        self.fglob = -3456
        self.global_optimum = [24, 12, 12]

        def g1(x):
            return -(x[0] + 2 * x[1] + 2 * x[2] - 72)

        def g2(x):
            return x[0] + 2 * x[1] + 2 * x[2]

        self.g = (g1, g2)

        self.x0 = [10, 10, 10]

    def fun(self, x):
        self.nfev += 1
        return -x[0]*x[1]*x[2]

class Hs038(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-10, 10), ] * 4
        self.fglob = 0.0
        self.global_optimum = [1, 1, 1, 1]

        def g1(x):
            return None

        self.g = None

        self.x0 = [0.0, 0.0, 0.0, 0.0]

    # Colville function in SciPy suite
    def fun(self, x):
        self.nfev += 1
        return (100.0 * (x[0] - x[1] ** 2) ** 2
                + (1 - x[0]) ** 2 + (1 - x[2]) ** 2
                + 90 * (x[3] - x[2] ** 2) ** 2
                + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)
                + 19.8 * (x[1] - 1) * (x[3] - 1))


class Hs044(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0.0, 5.0), ] * 4  # TODO: (0, None) ?
        self.fglob = -15.0
        self.global_optimum = [0.0, 3.0, 0.0, 4.0]

        def g1(x):
            return -(x[0] + 2 * x[1] - 8.0)

        def g2(x):
            return -(4 * x[0] + x[1] - 12.0)

        def g3(x):
            return -(3 * x[0] + 4 * x[1] - 12.0)

        def g4(x):
            return -(2 * x[2] + x[3] - 8.0)

        def g5(x):
            return -(x[2] + 2 * x[3] - 8.0)

        def g6(x):
            return -(x[2] + x[3] - 5.0)

        self.g = (g1, g2, g3, g4, g5, g6)

        self.x0 = [0.0, 0.0, 0.0, 0.0]

    def fun(self, x):
        self.nfev += 1
        return x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3]


class Hs076(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 5), ] * 4
        self.fglob = -4.681818181
        self.global_optimum = [0.2727273, 2.09090, 0.26e-10, 0.5454545]

        def g1(x):
            return -(x[0] + 2 * x[1] + x[2] + x[3] - 5)

        def g2(x):
            return -(3 * x[0] + x[1] + 2 * x[2] - x[3] - 4)

        def g3(x):
            return (x[1] + 4 * x[2] - 1.5)

        self.g = (g1, g2, g3)

        self.x0 = [0.5, 0.5, 0.5, 0.5]


    def fun(self, x):
        self.nfev += 1
        return x[0]**2 + 0.5*x[1]**2 + x[2]**2 + 0.5*x[3]**2 - x[0]*x[2] + x[2]*x[3]- x[0] - 3*x[1] + x[2] - x[3]


class S224(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 6), ] * 2
        self.fglob = -304.0
        self.global_optimum = [4.0, 4.0]

        def g1(x):
            return x[0] + 3 * x[1]

        def g2(x):
            return 18 - x[0] - 3 * x[1]

        def g3(x):
            return x[0] + x[1]

        def g4(x):
            return 8 - x[0] - x[1]

        self.g = (g1, g2, g3, g4)

        self.x0 = [0.1, 0.1]

    def fun(self, x):
        self.nfev += 1
        return 2*x[0]**2 + x[1]**2 - 48*x[0] - 40*x[1]


class S231(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(-2, 2), ] * 2
        self.fglob = 0.0
        self.global_optimum = [1.0, 1.0]

        def g1(x):
            return x[0]/3 + x[1] + 0.1

        def g2(x):
            return -1*x[0]/3 + x[1] + 0.1

        self.g = (g1,g2)

        self.x0 = [-1.2, 1]

    def fun(self, x):
        self.nfev += 1
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2


class S232(Benchmark):
    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 5), ] * 2  # TODO: (0, None) ?
        self.fglob = -1.0
        self.global_optimum = [3.0, numpy.sqrt(3)]

        def g1(x):
            return (x[0] / numpy.sqrt(3) - x[1])

        def g2(x):
            return x[0] / numpy.sqrt(3) - x[1]

        def g3(x):
            return 6 - x[0] - numpy.sqrt(3) * x[1]

        self.g = (g1, g2, g3)

        self.x0 = [1.0, 0.5]

    def fun(self, x):
        self.nfev += 1
        return -1*(9-(x[0] - 3)**2)*(x[1]**3/(27*numpy.sqrt(3)))


class S250(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 20), (0, 11), (0, 42)]
        self.fglob = -3300
        self.global_optimum = [20, 11, 15]

        def g1(x):
            # return (x[0] + 2 * x[1] + 2 * x[2] - 72)
            return -x[0] - 2 * x[1] - 2 * x[2] + 72

        self.g = (g1,)

        self.x0 = [10.0, 10.0, 10.0]

    def fun(self, x):
        self.nfev += 1
        return -x[0] * x[1] * x[2]


class S251(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 42), ] * 3
        self.fglob = -3456
        self.global_optimum = [24, 12, 12]

        def g1(x):
            return -(x[0] + 2 * x[1] + 2 * x[2] - 72)

        def g2(x):
            return x[0] + 2 * x[1] + 2 * x[2]

        self.g = (g1, g2)

        self.x0 = [10, 10, 10]

    def fun(self, x):
        self.nfev += 1
        return -x[0] * x[1] * x[2]


class Bunnag1(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 3), ] * 3
        # Try to find lower
        self.fglob = 1 / 9.0  # Exact analytical solution
        self.global_optimum = [1.33333327, 0.77777775, 0.44444449]

        def g1(x):
            return -(x[0] + x[1] + 2 * x[2] - 3)

        self.g = (g1,)

    def fun(self, x):
        self.nfev += 1
        return 9-8*x[0]-6*x[1]-4*x[2]+2*x[0]**2+2*x[1]**2+x[2]**2+2*x[0]*x[1]+2*x[0]*x[2]


class Bunnag1(Benchmark):
    def __init__(self, dimensions=3):
        Benchmark.__init__(self, dimensions)

        self._bounds = [(0, 3), (0, 3), (0, 3)]
        # Try to find lower
        self.fglob = 1 / 9.0  # Exact analytical solution
        self.global_optimum = [1.33333327, 0.77777775, 0.44444449]

        def g1(x):
            return -(x[0] + x[1] + 2 * x[2] - 3)

        self.g = (g1,)

    def fun(self, x):
        self.nfev += 1
        return 9-8*x[0]-6*x[1]-4*x[2]+2*x[0]**2+2*x[1]**2+x[2]**2+2*x[0]*x[1]+2*x[0]*x[2]


class Bunnag2(Benchmark):
    def __init__(self, dimensions=4):
        Benchmark.__init__(self, dimensions)
        self._bounds = [(0, 4), (0, 4), (0, 4), (0, 4)]
        # self.fglob=-2.07,
        self.fglob = -6.4052065800118605  # shgo solution
        # self.global_optimum=[4/3.0, 4, 0, 0]
        self.global_optimum = [1., 4., 0., 4.]  # shgo solution

        def g1(x):
            return -(x[0] + 2 * x[2] - 4)

        def g2(x):
            return -(-3 * x[0] + x[3] - 1)

        self.g = (g1, g2)

    def fun(self, x):
        self.nfev += 1
        return x[0]**0.6 + 2*x[1]**0.6 - 2*x[1] + 2*x[2] - x[3]
