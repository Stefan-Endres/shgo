from _shgo import shgo
import scipy.optimize
import numpy

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


#atoms = 10
atoms = 2
N = atoms * 3
LJ = LennardJones(N)
print(LJ.fun([0.1] * N))

options = {'disp': True}
res = shgo(LJ.fun, LJ._bounds, options=options, n=40)
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
res2 = scipy.optimize.basinhopping(LJ.fun, x0)
print(res2)