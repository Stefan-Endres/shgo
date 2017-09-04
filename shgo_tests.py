"""
Unit tests for topographical global optimization algorithm.

NOTE: For TestTgoFuncs test_f1 and test_f2 adequately test the
      functionality of the algorithm, the rest can be omitted to
      increase speed.
"""
import unittest
import numpy
#from _shgo_sobol import *
from _shgo import *
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class TestFunction(object):
    def __init__(self, bounds, expected_x, expected_fun=None,
                 expected_xl=None, expected_funl=None):
        self.bounds = bounds
        self.expected_x = expected_x
        self.expected_fun = expected_fun
        self.expected_xl = expected_xl
        self.expected_funl = expected_funl

class Test1(TestFunction):
    def f(self, x, r=0, s=0):
        return x[0]**2 + x[1]**2

    def g(self, x, r=0, s=0):
       return -(numpy.sum(x, axis=0) - 6.0)

test1_1 = Test1(bounds=[(-1, 6), (-1, 6)],
                expected_x=[0, 0])
test1_2 = Test1(bounds=[(0, 1), (0, 1)],
                expected_x=[0, 0])
test1_3 = Test1(bounds=[(None, None), (None, None)],
                expected_x=[0, 0])

class Test2(TestFunction):
    """
    Scalar function with several minima to test all minimiser retrievals
    """
    def f(self, x):
        return (x - 30) * numpy.sin(x)

    def g(self, x):
        return 58 - numpy.sum(x, axis=0)

test2_1 = Test2(bounds=[(0, 60)],
                expected_x=[1.53567906],
                expected_fun=[-28.44677132],
                # Important to test that funl return is in the correct order
                expected_xl=numpy.array([[1.53567906],
                                         [55.01782167],
                                         [7.80894889],
                                         [48.74797493],
                                         [14.07445705],
                                         [42.4913859],
                                         [20.31743841],
                                         [36.28607535],
                                         [26.43039605],
                                         [30.76371366]]),

                expected_funl=numpy.array([-28.44677132, -24.99785984,
                                           -22.16855376, -18.72136195,
                                           -15.89423937, -12.45154942,
                                           -9.63133158,  -6.20801301,
                                           -3.43727232,  -0.46353338])
                )

test2_2 = Test2(bounds=[(0, 4.5)],
                expected_x=[1.53567906],
                expected_fun=[-28.44677132],
                expected_xl=numpy.array([[1.53567906]]),
                expected_funl=numpy.array([-28.44677132])
                )


class Test3(TestFunction):
    """
    Hock and Schittkowski 19 problem (HS19). Hoch and Schittkowski (1981)
    http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    Minimize: f = (x_1 - 5)**3 + (x_2 - 20)**3

    Subject to: -(x_1 - 5)**2  - (x_2 - 20)**2 + 100 <= 0,
                -(x_1 - 6)**2  - (x_2 - 20)**2 + 82.81 <= 0,
                13 <= x_1 <= 100,
                0 <= x_2 <= 100.

    Approx. Answer:
        f([14.095, 0.84296]) = -6961.814744487831

    """
    def f(self, x):     # TODO: Add f bounds from original problem
        return (x[0] - 10.0)**3.0 + (x[1] - 20.0)**3.0

    # def f2(x):  #
    #     return (x[0] - 10.0) ** 3.0 + (x[1] - 20.0) ** 3.0

    def g1(x):
        return -(-(x[0] - 5.0)**2.0 - (x[1] - 5.0)**2.0 + 100.0)

    def g2(x):
        #return -(-(x[0] - 6.0)**2.0 - (x[1] - 5.0)**2.0 + 82.81)
        return -(+(x[0] - 6)**2 - (x[1] - 5)**2 - 82.81)

    g = (g1, g2)

test3_1 = Test3(bounds=[(13.0, 100.0), (0.0, 100.0)],
                expected_x=[13.6602540, 0.])
                # Note this is a lower value that is still within the bounds
                # There appears to be a typo in Henderson (2015), but the
                # original solution in the collection of
                # Hock and Shittkowski 1981 is outside the specified bounds.
                #expected_x=[14.095, 0.84296])

class Test8(TestFunction):
    """
    Hock and Schittkowski 29 problem (HS29). Hoch and Schittkowski (1981)
    http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    Minimize: f = - x_1 * x_2 * x_3

    Subject to: - (x_1**2 + 2 * x_2**2 + 4 * x_3**2 - 48.0)<= 0,
                -5 <= x_1 <= 5,
                -4 <= x_2 <= 4,
                -3 <= x_3 <= 3.

    Approx. Answer:
        f([4.0,  -2 * 2**0.5, -2.0]) = -16.0 * 2**0.5

    NOTE: Other minimizers: [4.0,  2 * 2**0.5, 2.0]
                            [-4.0, 2 * 2**0.5, -2.0]
                            [-4.0, -2 * 2**0.5, 2.0]

    """

    def f(self, x):
        return - x[0] * x[1] * x[2]

    def g(self, x):
        return - (x[0]**2 + 2 * x[1]**2 + 4 * x[2]**2 - 48.0)


test8_1 = Test8(bounds=[(-5, 5), (-4, 4), (-3, 3)],
                expected_x=[4.0,  -2 * 2**0.5, -2.0],
                expected_fun=[-16.0 * 2**0.5],  # For all minimizers
                expected_xl = numpy.array([[4.0,  -2 * 2**0.5, -2.0],
                                           [4.0, 2 * 2 ** 0.5, 2.0],
                                           [-4.0, 2 * 2 ** 0.5, -2.0],
                                           [-4.0, -2 * 2 ** 0.5, 2.0]]),

                expected_funl = numpy.array([-16.0 * 2**0.5,]*4)
                )

class Test9(TestFunction):
    """
    Hock and Schittkowski 18 problem (HS18). Hoch and Schittkowski (1981)
    http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    Minimize: f = 0.01 * (x_1)**2 + (x_2)**2

    Subject to: x_1 * x_2 - 25.0 >= 0,
                (x_1)**2 + (x_2)**2 - 25.0 >= 0,
                2 <= x_1 <= 50,
                0 <= x_2 <= 50.

    Approx. Answer:
        f([(250)**0.5 , (2.5)**0.5]) = 5.0


    """
    def f(self, x):
        return 0.01 * (x[0])**2 + (x[1])**2

    def g1(x):
        return x[0] * x[1] - 25.0

    def g2(x):
        return x[0]**2 + x[1]**2 - 25.0

    g = (g1, g2)

test9_1 = Test9(bounds=[(2, 50), (0, 50)],
                expected_x=[250**0.5, 2.5**0.5],
                expected_fun=[5.0]
                )

class Test10(TestFunction):
    """
    Hock and Schittkowski 11 problem (HS11). Hoch and Schittkowski (1981)

    NOTE: Did not find in original reference to HS collection, refer to
          Henderson (2015) problem 7 instead. 02.03.2016
    """

    def f(self, x):
        return ((x[0] - 10)**2 + 5*(x[1] - 12)**2 + x[2]**4
                 + 3*(x[3] - 11)**2 + 10*x[4]**6 + 7*x[5]**2 + x[6]**4
                 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]
                )

    def g1(x):
        return -(2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4] - 127)

    def g2(x):
        return -(7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4] - 282.0)

    def g3(x):
        return -(23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6] - 196)

    def g4(x):
        return -(4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5]
                 - 11*x[6])

    g = (g1, g2, g3, g4)

test10_1 = Test10(bounds=[(-10, 10),]*7,
                  expected_x=[2.330499, 1.951372, -0.4775414,
                              4.365726, -0.6244870, 1.038131, 1.594227],
                   expected_fun=[680.6300573]
                  )

def run_test(test, args=(), g_args=(), test_atol=1e-5,
             n=100, iter=None, sampling_method='sobol'):

    if test == test10_1:
        n = 1000
        if sampling_method =='simplicial':
            n = 1

    #if test == test9_1:
    #    n = 50000

    res = shgo(test.f, test.bounds, args=args, g_cons=test.g,
                g_args=g_args, n=n, iter=iter,
                sampling_method=sampling_method)

    logging.info(res)

    if test.expected_x is not None:
        numpy.testing.assert_allclose(res.x, test.expected_x,
                                      rtol=test_atol,
                                      atol=test_atol)

    # (Optional tests)
    if test.expected_fun is not None:
        numpy.testing.assert_allclose(res.fun,
                                      test.expected_fun,
                                      atol=test_atol)

    if test.expected_xl is not None:
        numpy.testing.assert_allclose(res.xl,
                                      test.expected_xl,
                                      atol=test_atol)

    if test.expected_funl is not None:
        numpy.testing.assert_allclose(res.funl,
                                      test.expected_funl,
                                      atol=test_atol)


# $ python2 -m unittest -v tgo_tests.TestTgoFuncs
class TestShgoSobolTestFunctions(unittest.TestCase):
    """
    Global optimisation tests:
    """
    # Sobol algorithm
    def test_f1_sobol(self):
        """Multivariate test function 1: x[0]**2 + x[1]**2"""
        run_test(test1_1)
        run_test(test1_2)

    def test_f2_sobol(self):
        """Scalar opt test on f(x) = (x - 30) * sin(x)"""
        run_test(test2_1)
        run_test(test2_2)

    def test_f3_sobol(self):
        """Hock and Schittkowski problem 19"""
        run_test(test3_1)

    def test_t8_sobol(self):
        """Hock and Schittkowski problem 29"""
        run_test(test8_1)

    def test_t9_sobol(self):
        """Hock and Schittkowski problem 18 """
        run_test(test9_1)

    def test_t910_sobol(self):
        """ Hock and Schittkowski 11 problem (HS11)"""
        run_test(test10_1)

    #def test_t911(self):
    #    """1D tabletop function"""
    #    run_test(test11_1)

class TestShgoSimplicialTestFunctions(unittest.TestCase):
    # Simplicial algorithm
    def test_f1_1_simplicial(self):
        """Multivariate test function 1: x[0]**2 + x[1]**2"""
        run_test(test1_1,  sampling_method='simplicial')

    def test_f1_2_simplicial(self):
        """Scalar opt test on f(x) = (x - 30) * sin(x)"""
        run_test(test1_2,  sampling_method='simplicial')

    def test_f2_simplicial(self):
        """Scalar opt test on f(x) = (x - 30) * sin(x)"""
        run_test(test2_1, sampling_method='simplicial')
        run_test(test2_2, sampling_method='simplicial')

    def test_f3_simplicial(self):
        """Hock and Schittkowski problem 19"""
        run_test(test3_1, sampling_method='simplicial')

    def test_t8_simplicial(self):
        """Hock and Schittkowski problem 29"""
        run_test(test8_1, sampling_method='simplicial')

    def test_t9_simplicial(self):
        """Hock and Schittkowski problem 18 """
        run_test(test9_1, sampling_method='simplicial')

    def test_t910_sobol(self):
        """ Hock and Schittkowski 11 problem (HS11)"""
        run_test(test10_1, sampling_method='simplicial')

def tgo_suite():
    """
    Gather all the TGO tests from this module in a test suite.
    """
    TestTgo = unittest.TestSuite()
    tgo_suite1 = unittest.makeSuite(TestTgoFuncs)
    #tgo_suite2 = unittest.makeSuite(TestTgoSubFuncs)
    TestTgo.addTest(tgo_suite1)
    return TestTgo


if __name__ == '__main__':
    TestTgo=tgo_suite()
    unittest.TextTestRunner(verbosity=2).run(TestTgo)
