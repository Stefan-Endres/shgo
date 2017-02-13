#from _shgo_sobol import shgo
from _shgo import shgo
from _tgo import tgo
import scipy.optimize
import numpy
import logging
import sys
import time
if 0:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Benchmark(object):
    def __init__(self, N=6):
        self.N = N
        self.nfev = 0


class LennardJones(Benchmark):
    r"""
    LennardJones objective function.

    This class defines the Lennard-Jones global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{LennardJones}}(\mathbf{x}) = \sum_{i=0}^{n-2}\sum_{j>1}^{n-1}
        \frac{1}{r_{ij}^{12}} - \frac{1}{r_{ij}^{6}}


    Where, in this exercise:

    .. math::

        r_{ij} = \sqrt{(x_{3i}-x_{3j})^2 + (x_{3i+1}-x_{3j+1})^2)
        + (x_{3i+2}-x_{3j+2})^2}


    Valid for any dimension, :math:`n = 3*k, k=2 , 3, 4, ..., 20`. :math:`k`
    is the number of atoms in 3-D space constraints: unconstrained type:
    multi-modal with one global minimum; non-separable

    Value-to-reach: :math:`minima[k-2] + 0.0001`. See array of minima below;
    additional minima available at the Cambridge cluster database:

    http://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-4, 4]` for :math:`i = 1 ,..., n`.

    *Global optimum*:

    .. math::

        \text{minima} = [-1.,-3.,-6.,-9.103852,-12.712062,-16.505384,\\
                         -19.821489, -24.113360, -28.422532,-32.765970,\\
                         -37.967600,-44.326801, -47.845157,-52.322627,\\
                         -56.815742,-61.317995, -66.530949, -72.659782,\\
                         -77.1777043]\\


    """

    def __init__(self, dimensions=6):
        # dimensions is in [6:60]
        # max dimensions is going to be 60.
        if dimensions not in range(6, 61):
            raise ValueError("LJ dimensions must be in (6, 60)")

        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-4.0] * self.N, [4.0] * self.N))
        print("len(bounds) = {}".format(len(self._bounds)))

        self.global_optimum = [[]]

        self.minima = [-1.0, -3.0, -6.0, -9.103852, -12.712062,
                       -16.505384, -19.821489, -24.113360, -28.422532,
                       -32.765970, -37.967600, -44.326801, -47.845157,
                       -52.322627, -56.815742, -61.317995, -66.530949,
                       -72.659782, -77.1777043]

        k = int(dimensions / 3)
        self.atoms = k
        print('Number of atoms = {}'.format(self.atoms))
        self.fglob = self.minima[k - 2]
        self.change_dimensionality = True

    def change_dimensions(self, ndim):
        if ndim not in range(6, 61):
            raise ValueError("LJ dimensions must be in (6, 60)")

        Benchmark.change_dimensions(self, ndim)
        self.fglob = self.minima[int(self.N / 3) - 2]

    def fun(self, x, *args):
        self.nfev += 1

        k = int(self.N / 3)
        s = 0.0

        for i in range(k - 1):
            for j in range(i + 1, k):
                a = 3 * i
                b = 3 * j
                xd = x[a] - x[b]
                yd = x[a + 1] - x[b + 1]
                zd = x[a + 2] - x[b + 2]
                ed = xd * xd + yd * yd + zd * zd
                ud = ed * ed * ed
                if ed > 0.0:
                    s += (1.0 / ud - 2.0) / ud

        return s


def unique_minima(xl, tol=1e-5):
    """
    Returns the number of points in `xl` that are unique to the default
    tolerance of numpy.allclose
    """
    import itertools

    uniql = len(xl)
    if uniql == 1:
        uniq = 1
    else:
        xll = len(xl)
        flag = []
        for i in range(xll):
            for k in range(i + 1, xll):
                if numpy.allclose(xl[i], [xl[k]],
                                  rtol=tol,
                                  atol=tol):
                    flag.append(k)

        uniq = uniql - len(numpy.unique(numpy.array(flag)))
    return uniq

#atoms = 10
atoms = 2
atoms = 10#38
atoms = 3
atoms = 4
atoms = 2
atoms = 4
atoms = 4
atoms = 10
N = atoms * 3
LJ = LennardJones(N)
print(LJ.fun([0.1] * N))

options = {'disp': True}
options = {'disp': False}
#res = shgo(LJ.fun, LJ._bounds, options=options, n=40)
#res = shgo(LJ.fun, LJ._bounds, options=options, n=100)
#res = shgo(LJ.fun, LJ._bounds, options=options, n=1000)

#res = shgo(LJ.fun, LJ._bounds, options=options, n=50, crystal_mode=True)


print("="*30)
print("="*30)
print("Optimizing with SHGO...")
print("="*40)


# Symmetry shit
options = {'symmetry': True,
          # 'crystal_iter': 8}
          # 'crystal_iter': 15}
           'disp': True,
           'crystal_iter': 11}

t0 = time.time()
res = shgo(LJ.fun, LJ._bounds, options=options, crystal_mode=True,
               sampling_method='simplicial')
tshgo = time.time() - t0

#shgo(test.f, test.bounds, args=args, g_cons=test.g,
#                    g_args=g_args, n=100, iter=None, crystal_mode=True)
print('=' * 11)
print('Global out:')
print('=' * 11)
print('LJ cluster of {} atoms with dimensionality '
      '= {}:'.format(LJ.atoms, len(LJ._bounds)))
print(res)

# if LJ.fglob == res.fun:
if abs(LJ.fglob - res.fun) <= 1e-4:
    print("Correct global minima found")
    print("LJ.fglob = {}".format(LJ.fglob))
else:
    print("INCORRECT global minima found!")
    print("res.fun= {}".format(res.fun))
    print("LJ.fglob = {}".format(LJ.fglob))

print("="*30)
print("="*30)
print("Optimizing with Basinhopping...")
print("="*40)
x0 = numpy.zeros(N)
x0 = numpy.random.randn(N)
t0 = time.time()
res2 = scipy.optimize.basinhopping(LJ.fun, x0)
tbh = time.time() - t0
print(res2)

print("="*30)
print("="*30)
print("Optimizing with TGO...")
print("="*40)
res3 = tgo(LJ.fun, LJ._bounds, options=options, n=100)
print(res3)
print("="*40)
print("="*40)
print('shgo.res.xl = {}'.format(res.xl))
print('tgo.res.xl = {}'.format(res3.xl))
print('tgo.res.nfev = {}'.format(res3.nfev))
print('shgo.res.nfev = {}'.format(res.nfev))
print('basinhopping.res.nfev = {}'.format(res2.nfev))
print('basinhopping.res.fun = {}'.format(res2.fun))
print('shgo.res.fun = {}'.format(res.fun))
print('tgo.res.fun = {}'.format(res3.fun))
#print('len(basinhopping.res.fun) = {}'.format(len(res2.xl)))
print('len(shgo.res.xl) = {}'.format(len(res.xl)))
print('len(tgo.res.fun) = {}'.format(len(res3.xl)))


tol = 5
res.funl = numpy.around(res.funl, tol)
res3.funl = numpy.around(res3.funl, tol)

#shgo_uniq = numpy.unique(res.funl)
shgo_uniq = unique_minima(res.xl, tol=1e-5)
#tgo_uniq = numpy.unique(res3.funl)
tgo_uniq = unique_minima(res3.xl)

#print('Unique local minima SHGO = {}'.format(len(shgo_uniq)))
print('SHGO symmetry homology group = {}'.format(shgo_uniq))
print('Unique local minima SHGO = {}'.format(shgo_uniq))
#print('Unique local minima TGO = {}'.format(len(tgo_uniq)))
print('Unique local minima TGO = {}'.format(tgo_uniq))
print('====================='*3)
print('SHGO improvment on BH')
print('=====================')
print('nfev: {} %'.format( round(res2.nfev/res.nfev *100, 5)))
print('Simulation time: {} %'.format( round(tbh/tshgo *100, 5)))
 #     ' nfev: {}')