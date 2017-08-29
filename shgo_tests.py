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


class Test4(TestFunction):
    """ Rosenbrock's function  Ans x1 = 1, x2 = 1, f = 0 """
    g = None

    def f(self, x):
        return (1.0 - x[0])**2.0 + 100.0*(x[1] - x[0]**2.0)**2.0


test4_1 = Test4(bounds=[(-3.0, 3.0), (-3.0, 3.0)],
                expected_x=[1, 1])

test4_2 = Test4(bounds=[(None, None), (-numpy.inf, numpy.inf)],
                expected_x=[1, 1])

test_atol = 1e-5


class Test5(TestFunction):
    """
    Himmelblau's function
    https://en.wikipedia.org/wiki/Himmelblau's_function
    """
    g = None

    def f(self, x):
        return (x[0]**2 + x[1] - 11)**2 + (x[0] + x[1]**2 - 7)**2


test5_1 = Test5(bounds=[(-6, 6),
                        (-6, 6)],
                expected_x=None,
                expected_fun=[0.0],  # Important to test that fun
                # return is in the correct order
                expected_xl=numpy.array([[3.0, 2.0],
                                         [-2.805118, 3.1313212],
                                         [-3.779310, -3.283186],
                                         [3.584428, -1.848126]]),

                expected_funl=numpy.array([0.0, 0.0, 0.0, 0.0])
                )


class Test6(TestFunction):
    """
    Eggholder function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    g = None

    def f(self, x):
        return (-(x[1] + 47.0)
                * numpy.sin(numpy.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
                - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))
                )


test6_1 = Test6(bounds=[(-512, 512),
                        (-512, 512)],
                expected_x=[512, 404.2319],
                expected_fun=[-959.6407]
                )

class Test7(TestFunction):
    """
    Ackley function
    https://en.wikipedia.org/wiki/Test_functions_for_optimization
    """
    g = None

    def f(self, x):
        arg1 = -0.2 * numpy.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
        arg2 = 0.5 * (numpy.cos(2. * numpy.pi * x[0])
                      + numpy.cos(2. * numpy.pi * x[1]))
        return -20. * numpy.exp(arg1) - numpy.exp(arg2) + 20. + numpy.e

test7_1 = Test7(bounds=[(-5, 5), (-5, 5)],
                expected_x=[0.,  0.],
                expected_fun=[0.0]
                )

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
                expected_fun=[-16.0 * 2**0.5]  # For all minimizers
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

def run_test(test, args=(), g_args=()):
    ThirdDev = False
    if ThirdDev:
        # minimizer_kwargs2 = {'method': 'TNC'}
        # minimizer_kwargs2 = {'method': 'CG'}
        # minimizer_kwargs2 = {'method': 'BFGS'}
        # minimizer_kwargs2 = {'method': 'L-BFGS-B'}
        # minimizer_kwargs2 = {'method': 'TNC'}
        minimizer_kwargs2 = {'method': 'SLSQP'}

    if test is not test10_1:
        res = shgo(test.f, test.bounds, args=args, g_cons=test.g,
                    g_args=g_args, n=100,
                    sampling_method='sobol')

        ares = shgo(test.f, test.bounds, args=args, g_cons=test.g,
                    g_args=g_args, n=100,
                    sampling_method='simplicial')

        if ThirdDev:
            ares2 = shgo(test.f, test.bounds, args=args, g_cons=test.g,
                         g_args=g_args, n=100, iter=None,
                         minimizer_kwargs=minimizer_kwargs2)

    if test == test5_1:
        # Remove the extra minimizer found in this test
        # (note all minima is at the global 0.0 value)
        res.xl = [res.xl[0], res.xl[1],
                  res.xl[3], res.xl[2]]
        res.funl = res.funl[:4]

    if test == test10_1:
        res = shgo(test.f, test.bounds, args=args, g_cons=test.g,
                    g_args=g_args, n=1000, iter=None, sampling_method='sobol')#, crystal_mode=True)

        if ThirdDev:
            ares2 = shgo(test.f, test.bounds, args=args, g_cons=test.g,
                         g_args=g_args, n=100, iter=None,
                         minimizer_kwargs=minimizer_kwargs2)

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
class TestShgoFuncs(unittest.TestCase):
    """
    Global optimisation tests:
    """
    def test_f1(self):
        """Multivariate test function 1: x[0]**2 + x[1]**2"""
        run_test(test1_1)
        run_test(test1_2)

    def test_f2(self):
        """Scalar opt test on f(x) = (x - 30) * sin(x)"""
        run_test(test2_1)
        run_test(test2_2)

    def test_f3(self):
        """Hock and Schittkowski problem 19"""
        run_test(test3_1)

    def test_t4(self):
        """Rosenbrock function"""
        run_test(test4_1)
        #run_test(test4_2)

    def test_t5(self):
        """Himmelblau's function"""
        run_test(test5_1)

    #def test_t6(self):
    #    """Eggholder function"""
    #    run_test(test6_1)

    def test_t7(self):
        """Ackley function"""
        run_test(test7_1)

    #def test_t8(self):
    #    """Hock and Schittkowski problem 29"""
    #    run_test(test8_1)

    def test_t9(self):
        """Hock and Schittkowski problem 18 """
        run_test(test9_1)

    def test_t910(self):
        """ Hock and Schittkowski 11 problem (HS11)"""
        run_test(test10_1)

    #def test_t911(self):
    #    """1D tabletop function"""
    #    run_test(test11_1)


def shgo_suite():
    """
    Gather all the shgo tests from this module in a test suite.
    """
    TestShgo = unittest.TestSuite()
    tgo_suite1 = unittest.makeSuite(TestShgoFuncs)
    TestShgo.addTest(tgo_suite1)
    return TestShgo


if __name__ == '__main__':
    TestShgo=shgo_suite()
    unittest.TextTestRunner(verbosity=2).run(TestTgo)
