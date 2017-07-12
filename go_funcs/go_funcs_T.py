# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

from numpy import (abs, arctan2, array, asarray, cos, exp,
                   floor, identity, linalg, log, log10,
                   matrix, arange, pi, sign, sin, sqrt, sum,
                   tan, tanh, tensordot, atleast_2d)
import  scipy.spatial.distance
from .go_benchmark import Benchmark


class TestTubeHolder(Benchmark):

    r"""
    TestTubeHolder objective function.

    This class defines the TestTubeHolder [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{TestTubeHolder}}(x) = - 4 \left | {e^{\left|{\cos 
        \left(\frac{1}{200} x_{1}^{2} + \frac{1}{200} x_{2}^{2}\right)}
        \right|}\sin\left(x_{1}\right) \cos\left(x_{2}\right)}\right|


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -10.872299901558` for
    :math:`x= [-\pi/2, 0]`

    .. [1] Mishra, S. Global Optimization by Differential Evolution and
    Particle Swarm Methods: Evaluation on Some Benchmark Functions.
    Munich Personal RePEc Archive, 2006, 1005

    TODO Jamil#148 has got incorrect equation, missing an abs around the square
    brackets
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))

        self.global_optimum = [[-pi / 2, 0.0]]
        self.fglob = -10.87229990155800

    def fun(self, x, *args):
        self.nfev += 1

        u = sin(x[0]) * cos(x[1])
        v = (x[0] ** 2 + x[1] ** 2) / 200
        return -4 * abs(u * exp(abs(cos(v))))


class Thurber(Benchmark):

    r"""
    Thurber [1]_ objective function.

    .. [1] http://www.itl.nist.gov/div898/strd/nls/data/thurber.shtml
    """

    def __init__(self, dimensions=7):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip(
            [500., 500., 100., 10., 0.1, 0.1, 0.],
            [2000., 2000., 1000., 150., 2., 1., 0.2]))
        self.global_optimum = [[1.288139680e3, 1.4910792535e3, 5.8323836877e2,
                                75.416644291, 0.96629502864, 0.39797285797,
                                4.9727297349e-2]]
        self.fglob = 5642.7082397
        self.a = asarray([80.574, 84.248, 87.264, 87.195, 89.076, 89.608,
                          89.868, 90.101, 92.405, 95.854, 100.696, 101.06,
                          401.672, 390.724, 567.534, 635.316, 733.054, 759.087,
                          894.206, 990.785, 1090.109, 1080.914, 1122.643,
                          1178.351, 1260.531, 1273.514, 1288.339, 1327.543,
                          1353.863, 1414.509, 1425.208, 1421.384, 1442.962,
                          1464.350, 1468.705, 1447.894, 1457.628])
        self.b = asarray([-3.067, -2.981, -2.921, -2.912, -2.840, -2.797,
                             -2.702, -2.699, -2.633, -2.481, -2.363, -2.322,
                             -1.501, -1.460, -1.274, -1.212, -1.100, -1.046,
                             -0.915, -0.714, -0.566, -0.545, -0.400, -0.309,
                             -0.109, -0.103, 0.010, 0.119, 0.377, 0.790, 0.963,
                             1.006, 1.115, 1.572, 1.841, 2.047, 2.200])

    def fun(self, x, *args):
        self.nfev += 1

        vec = x[0] + x[1] * self.b + x[2] * self.b ** 2 + x[3] * self.b ** 3
        vec /= 1 + x[4] * self.b + x[5] * self.b ** 2 + x[6] * self.b ** 3

        return sum((self.a - vec) ** 2)


class Treccani(Benchmark):

    r"""
    Treccani objective function.

    This class defines the Treccani [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Treccani}}(x) = x_1^4 + 4x_1^3 + 4x_1^2 + x_2^2


    with :math:`x_i \in
    [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [-2, 0]` or
    :math:`x = [0, 0]`.

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.custom_bounds = [(-2, 2), (-2, 2)]

        self.global_optimum = [[-2.0, 0.0]]
        self.fglob = 0

    def fun(self, x, *args):
        self.nfev += 1

        return x[0] ** 4 + 4.0 * x[0] ** 3 + 4.0 * x[0] ** 2 + x[1] ** 2


class Trefethen(Benchmark):

    r"""
    Trefethen objective function.

    This class defines the Trefethen [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Trefethen}}(x) = 0.25 x_{1}^{2} + 0.25 x_{2}^{2}
                                  + e^{\sin\left(50 x_{1}\right)}
                                  - \sin\left(10 x_{1} + 10 x_{2}\right)
                                  + \sin\left(60 e^{x_{2}}\right)
                                  + \sin\left[70 \sin\left(x_{1}\right)\right]
                                  + \sin\left[\sin\left(80 x_{2}\right)\right]


    with :math:`x_i \in [-10, 10]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = -3.3068686474` for
    :math:`x = [-0.02440307923, 0.2106124261]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-10.0] * self.N, [10.0] * self.N))
        self.custom_bounds = [(-5, 5), (-5, 5)]

        self.global_optimum = [[-0.02440307923, 0.2106124261]]
        self.fglob = -3.3068686474

    def fun(self, x, *args):
        self.nfev += 1

        val = 0.25 * x[0] ** 2 + 0.25 * x[1] ** 2
        val += exp(sin(50. * x[0])) - sin(10 * x[0] + 10 * x[1])
        val += sin(60 * exp(x[1]))
        val += sin(70 * sin(x[0]))
        val += sin(sin(80 * x[1]))
        return val


class ThreeHumpCamel(Benchmark):

    r"""
    Three Hump Camel objective function.

    This class defines the Three Hump Camel [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{ThreeHumpCamel}}(x) = 2x_1^2 - 1.05x_1^4 + \frac{x_1^6}{6}
                                       + x_1x_2 + x_2^2


    with :math:`x_i \in [-5, 5]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, 0]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-5.0] * self.N, [5.0] * self.N))
        self.custom_bounds = [(-2, 2), (-1.5, 1.5)]

        self.global_optimum = [[0.0, 0.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1

        return (2.0 * x[0] ** 2.0 - 1.05 * x[0] ** 4.0 + x[0] ** 6 / 6.0
                + x[0] * x[1] + x[1] ** 2.0)

class TIP4P(Benchmark):
    r"""
    TIP4P objective function evaluated for a water molecule cluster.

    This class defines a TIP4P global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{TIP4P}}(\mathbf{x}) = \sum_{n<m} [ \frac {A} { r_{OO}^{12} }
        - \frac {B} { r_{OO}^{6} } + \sum_{i}^{on m} \sum_{j}^{on n} \frac{ k_c q_i q_j }{ r_{ij} }]

    References:
    Chill, S.T., Stevenson, J., Ruehle, V., Shang, C.,  Xiao, P., Farrell, J.D., Wales, D.J., and Henkelman, G. (2014)
    "Benchmarks for Characterization of Minima, Transition States, and Pathways in Atomic, Molecular, and Condensed Matter Systems"
    J. Chem. Theory Comput., 2014, 10 (12), 5476–5482.
    """

    def __init__(self, W, fglob=None, global_optimum=[[]], xyz = True, bounds = [()], name =""):
        """Initialises Class
	
	Parameters
	----------
    W : int
        Number of water molecules
    fglob : float
		Value of the global minimum

	global_optimum : [[float]]
		List of x_lists, containing coordinates at which the
		objective function evaluates to fglob

    xyz : bool
        If true then it implies that global_optimum lists
        are given in terms of xyz coordinates.
        If false then it implies that global_optimum lists
        are given in terms of parametrised coordinates.
	Either fglob or global_optimum must be specified.
 
	bounds : [(float, float)]
        List of bounds for each dimension.
        Defaults to: [ (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0),
            (0.0, numpy.pi), (0.0, 2 * numpy.pi), (0.0, 2 * numpy.pi)] * W

    name : String
        Optional name for benchmark. """

        dimensions = 6 * W

        Benchmark.__init__(self, dimensions)

        if bounds == [()]:
            self._bounds = [ (-10.0, 10.0), (-10.0, 10.0), (-10.0, 10.0),
                            (0.0, numpy.pi), (0.0, 2 * numpy.pi), (0.0, 2 * numpy.pi)] * W
        else:
            self._bounds = bounds

        if global_optimum != [[]]:
            self.global_optimum = global_optimum
            if xyz:
                self.fglob = self.xyz_fun(global_optimum[0])
            else:
                self.fglob = self.fun(global_optimum[0])
            if fglob !=None and abs(self.fglob - fglob) > 1e-5:
                raise ValueError("Global optimum does not correspond to supposed global minimum. {0} != {1}".format(self.fglob, fglob))
        elif fglob != None:
            self.fglob = fglob
        else:
            raise ValueError("Either fglob or global_optimim must be specified")
	
        self.change_dimensionality = False

        if name == "":
            self._name = str(id(self))
        else:
            self._name = name

    def xyz_fun(self, x):
        """Objective function
	
	Parameters
	----------
	x : [float]
    Coordinates at which to evaluate objective function
    The values are semantically grouped into sets of nine, and then
    into groups of three where each group represent the Euclidian
    coordinates of an atom of a water molecule. The first group in
    every set represents the oxygen atom.

	Returns
	-------
	t : float
		Value of the TIP4P potential"""
        A = 6e5
        B = 6.1e2
        k_c = 332.1
        q_O = -1.04
        q_H = 0.52
        
        self.nfev += 1
        k = int(self.N / 6)
        t = 0.0

        molecules = [ [ x[9*mol + index:9*mol + index + 3] for index in range(0,9,3) ] for mol in range(k)]
        for molecule in molecules:
            O, H1, H2 = molecule
            O, H1, H2 = matrix(O), matrix(H1), matrix(H2)

            avg = (H1+H2)/2 - O
            avg = avg / linalg.norm(avg)
            charge_site = avg * 0.15 + O
            charge_site = array(charge_site)[0].tolist()
            molecule.append(charge_site)

        for n in range(k-1):
            for m in range(n+1, k):
                M = molecules[m] 
                N = molecules[n]
                       
                dist = scipy.spatial.distance.cdist(M, N)
                if dist[0][0] != 0:
                    t += A/dist[0][0]**12 - B/dist[0][0]**6
                else:
                    t += 1e9
                for i in range(1,4):
                    for j in range(1,4):
                        d = dist[i][j]
                        if d==0:
                            t += 1e9
                            continue
                        if i==3 and j==3:                   
                            t += (k_c * q_O * q_O)/d
                        elif i==3 or j==3:
                            t += (k_c * q_O * q_H)/d
                        else:
                            t += (k_c * q_H * q_H)/d

        return t

    def fun(self, p):
        """Objective function
	
	Parameters
	----------
	x : [float]
    Parametrised coordinates at which to evaluate objective function
    The values are semantically grouped into sets of six, where each
    group represents that parametrised form of a water molecule.

	Returns
	-------
	t : float
		Value of the TIP4P potential"""

        k = int(self.N / 6)
        molecules_param = [ p[6*mol:6*mol + 6] for mol in range(k)]
        molecules_coord = [ coord for molecule in molecules_param for coord in self.param2coord(molecule)]
        return self.xyz_fun(molecules_coord)
       

    def rotationMatrix(self, theta, u):
        """
    See https://en.wikipedia.org/wiki/Rotation_matrix

    Parameters
    ----------
    theta : float
        An angle in radians
    u : [float, float, float]
        A unit normal vector, that defines the axis of rotation
    
    Returns
    -------
    M : [[float]]
        A 3x3 rotation matrix
    """
        u1, u2, u3 = u
        u_cross = matrix( [  [0   , -u3 , u2  ],
                                   [u3  , 0   , -u1 ],
                                   [-u2 , u1  , 0   ] ] )

        
        M = cos(theta) * identity(3) + sin(theta) * u_cross + (1 - cos(theta) ) * tensordot(u, u, axes=0)
        return M

    def param2coord(self, p):
        """
    This function takes a parametrised coordinate set for a water
    molecule, and returns a set of 3 (x,y,z) coordinates for each atom.
    The parametrisation is as follows:
    p = [ x, y, z, theta, phi, gamma ]
    where x,y,z give the coordinates of the oxygen atom. The three
    angles describe rotations that are applied to the molecule.
    The initial configuration has the oxygen atom at (0,0,0),
    one hydrogen atom at (d*cos(alpha/2), d*sin(alpha/2), 0)
    and the other at (d*cos(alpha/2), -d*sin(alpha/2), 0),
    where d is the distance from the oxygen atom to a hydrogen atom,
    and alpha is the angle between the two hydrogen atoms.

    Parameters
    ----------
    p : [float]
        An array of the form:
        [ x, y, z, theta, phi, gamma ]
        where theta is in [0, pi], phi is in [0, 2*pi], and
        gamma is in [0, 2*pi].
        See above for information on what each parameter means.

    Returns
    -------
    xyz : [float]
        An array of the form:
        [ x1, y1, z1, x2, y2, z2, x3, y3, z3 ]
        Where (x1, y1, z1) is the coordinate of the oxygen atom
        (x2, y2, z2) is the coordinate of the first hydrogen atom and
        (x3, y3, z3) is the coordinate of the second hydrogen atom.
    """

        x, y, z, theta, phi, gamma = p

        d = 0.9572
        alpha = 104.52 / 180 * pi

        H1 = matrix( [ [d * cos(alpha/2),  d * sin(alpha/2), 0] ] )
        H2 = matrix( [ [d * cos(alpha/2), -d * sin(alpha/2), 0] ] )
        O  = array( [x, y, z] )

        rx = self.rotationMatrix(theta, [1,0,0])
        ry = self.rotationMatrix(phi, [0,1,0])
        rz = self.rotationMatrix(gamma, [0,0,1])
        r = rz * ry * rx

        H1 = (r * H1.T).T
        H2 = (r * H2.T).T
        
        H1 += O
        H2 += O
        
        H1 = array(H1)[0]
        H2 = array(H2)[0]

        return [ coord for atom in [O, H1, H2] for coord in atom]

    def coord2param(self, xyz):
        """
    This function takes the (x,y,z) coordinates of the atoms of a water molecule,
    and returns a parametrised coordinate set for the water molecule.
    The parametrisation is as follows:
    p = [ x, y, z, theta, phi, gamma ]
    where x,y,z give the coordinates of the oxygen atom. The three
    angles describe rotations that are applied to the molecule.
    The initial configuration has the oxygen atom at (0,0,0),
    one hydrogen atom at (d*cos(alpha/2), d*sin(alpha/2), 0)
    and the other at (d*cos(alpha/2), -d*sin(alpha/2), 0),
    where d is the distance from the oxygen atom to a hydrogen atom,
    and alpha is the angle between the two hydrogen atoms.

    Parameters
    ----------
    xyz : [float]
        An array of the form:
        [ x1, y1, z1, x2, y2, z2, x3, y3, z3 ]
        Where (x1, y1, z1) is the coordinate of the oxygen atom
        (x2, y2, z2) is the coordinate of the first hydrogen atom and
        (x3, y3, z3) is the coordinate of the second hydrogen atom.

    Returns
    -------
    p : [float]
        An array of the form:
        [ x, y, z, theta, phi, gamma ]
        where theta is in [0, pi], phi is in [0, 2*pi], and
        gamma is in [0, 2*pi].
        See above for information on what each parameter means.
    """

        O = matrix(xyz[:3])
        H1 = matrix(xyz[3:6]) - O
        H2 = matrix(xyz[6:9]) - O

        u = (H1+H2)/2
        u = u / linalg.norm(u) 
        v  = array(u)[0]

        gamma = arctan2( -v[1], v[0] )
        phi = arctan2( -v[2], ( v[1] * sin(gamma) - v[0] * cos(gamma) ) ) + pi

        rz = self.rotationMatrix(gamma, [0,0,1])
        ry = self.rotationMatrix(phi, [0,1,0])
        r = ry * rz

        H1 = (r * H1.T).T
        H1 = array(H1)[0]
        theta = arctan2(-H1[2],H1[1])

        return( [value for s in [array(O)[0], [-theta, -phi, -gamma] ] for value in s]  )

class Trid(Benchmark):

    r"""
    Trid objective function.

    This class defines the Trid [1]_ global optimization problem. This is a
    multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Trid}}(x) = \sum_{i=1}^{n} (x_i - 1)^2
                            - \sum_{i=2}^{n} x_i x_{i-1}


    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-20, 20]` for :math:`i = 1, ..., 6`.

    *Global optimum*: :math:`f(x) = -50` for :math:`x = [6, 10, 12, 12, 10, 6]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO Jamil#150, starting index of second summation term should be 2.
    """

    def __init__(self, dimensions=6):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-20.0] * self.N, [20.0] * self.N))

        self.global_optimum = [[6, 10, 12, 12, 10, 6]]
        self.fglob = -50.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1

        return sum((x - 1.0) ** 2.0) - sum(x[1:] * x[:-1])


class Trigonometric01(Benchmark):

    r"""
    Trigonometric 1 objective function.

    This class defines the Trigonometric 1 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Trigonometric01}}(x) = \sum_{i=1}^{n} \left [n -
                                        \sum_{j=1}^{n} \cos(x_j)
                                        + i \left(1 - cos(x_i)
                                        - sin(x_i) \right ) \right]^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [0, \pi]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x_i = 0` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.

    TODO: equaiton uncertain here.  Is it just supposed to be the cos term
    in the inner sum, or the whole of the second line in Jamil #153.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([0.0] * self.N, [pi] * self.N))

        self.global_optimum = [[0.0 for _ in range(self.N)]]
        self.fglob = 0.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1
        i = atleast_2d(arange(1.0, self.N + 1)).T
        inner = cos(x) + i * (1 - cos(x) - sin(x))
        return sum((self.N - sum(inner, axis=1)) ** 2)


class Trigonometric02(Benchmark):

    r"""
    Trigonometric 2 objective function.

    This class defines the Trigonometric 2 [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Trigonometric2}}(x) = 1 + \sum_{i=1}^{n} 8 \sin^2
                                       \left[7(x_i - 0.9)^2 \right]
                                       + 6 \sin^2 \left[14(x_i - 0.9)^2 \right]
                                       + (x_i - 0.9)^2

    Here, :math:`n` represents the number of dimensions and
    :math:`x_i \in [-500, 500]` for :math:`i = 1, ..., n`.

    *Global optimum*: :math:`f(x) = 1` for :math:`x_i = 0.9` for
    :math:`i = 1, ..., n`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-500.0] * self.N,
                           [500.0] * self.N))
        self.custom_bounds = [(0, 2), (0, 2)]

        self.global_optimum = [[0.9 for _ in range(self.N)]]
        self.fglob = 1.0
        self.change_dimensionality = True

    def fun(self, x, *args):
        self.nfev += 1

        vec = (8 * sin(7 * (x - 0.9) ** 2) ** 2
               + 6 * sin(14 * (x - 0.9) ** 2) ** 2
               + (x - 0.9) ** 2)
        return 1.0 + sum(vec)


class Tripod(Benchmark):

    r"""
    Tripod objective function.

    This class defines the Tripod [1]_ global optimization problem. This
    is a multimodal minimization problem defined as follows:

    .. math::

        f_{\text{Tripod}}(x) = p(x_2) \left[1 + p(x_1) \right] +
                               \lvert x_1 + 50p(x_2) \left[1 - 2p(x_1) \right]
                               \rvert + \lvert x_2 + 50\left[1 - 2p(x_2)\right]
                               \rvert

    with :math:`x_i \in [-100, 100]` for :math:`i = 1, 2`.

    *Global optimum*: :math:`f(x) = 0` for :math:`x = [0, -50]`

    .. [1] Jamil, M. & Yang, X.-S. A Literature Survey of Benchmark Functions
    For Global Optimization Problems Int. Journal of Mathematical Modelling
    and Numerical Optimisation, 2013, 4, 150-194.
    """

    def __init__(self, dimensions=2):
        Benchmark.__init__(self, dimensions)

        self._bounds = list(zip([-100.0] * self.N,
                           [100.0] * self.N))

        self.global_optimum = [[0.0, -50.0]]
        self.fglob = 0.0

    def fun(self, x, *args):
        self.nfev += 1

        p1 = float(x[0] >= 0)
        p2 = float(x[1] >= 0)

        return (p2 * (1.0 + p1) + abs(x[0] + 50.0 * p2 * (1.0 - 2.0 * p1))
                + abs(x[1] + 50.0 * (1.0 - 2.0 * p2)))
