import itertools
import numpy
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class Cell:
    """
    Contains a cell that is symmetric to the initial hypercube triangulation
    """
    def __init__(self, gen, hg):
        self.gen = gen  # generation
        self.hg = hg  # parent homology group
        self.C = []

    def __call__(self):
        return self.C

    def add_vertex(self, V):
        if V not in self.C:
            self.C.append(V)

    def homology_group(self):
        """
        Returns the homology group of the current cell
        """
        pass


class Complex:
    def __init__(self, dim, func, func_args=(), symmetry=False, g_cons=None):
        self.dim = dim
        self.gen = 0
        self.perm_cycle = 0
        # List of cells
        self.C = []


        #self.cell = 0
        # ^ Put every vertex into its own cell which changes
        # every generation

        # We set self.V to be the list of vertices in every generation
        # So self.V[0] is the list of vertices in the initial cube
        # generation
        # self.V[1] = [(0.5, 0.5, 0.5, ...)]
        # self.V[2] is the second cycle permutation split
        # etc.
        #self.V = []
        #self.V.append(VertexCached())
        self.V = VertexCached(func, func_args)

        # Generate n-cube here:
        self.n_cube(dim, symmetry=symmetry)
        #self.C.append()

    def n_cube(self, dim, symmetry=False, printout=True):
        """
        Generate the simplicial triangulation of the n dimensional hypercube
        containing 2**n vertices
        """
        import numpy
        origin = list(numpy.zeros(dim, dtype=int))
        self.origin = origin
        supremum = list(numpy.ones(dim, dtype=int))
        self.suprenum = supremum
        self.C0 = Cell(0, 0)  # Initial cell object
        self.C0.add_vertex(self.V(tuple(origin)))
        self.C0.add_vertex(self.V(tuple(supremum)))

        i1 = dim + 10 # Let all first rows be generation in first perm
        i1 = 0
        i1 = -1
        i_parents = []
        self.perm(i1, i_parents, origin)

        if printout:
            print("Initial hyper cube:")
            for v in self.C0():
                print("Vertex: {}".format(v.x))
                constr = 'Connections: '
                for vc in v.nn:
                    constr += '{} '.format(vc.x)

                print(constr)

    def perm(self, i1, i_parents, xi):
        #TODO: Cut out of for if outside linear constraint cutting planes
        xi_t = tuple(xi)

        # Construct required iterator
        iter_range = [x for x in range(self.dim) if x not in i_parents]

        for i in iter_range:
            i2_parents = i_parents.copy()
            i2_parents.append(i)
        #for i in range(dim_i):
            #print('i = ')
            xi2 = xi.copy()
            #if i is not i1:  #TODO: Not needed
            xi2[i] = 1
            # Make new vertex list a hashable tuple
            xi2_t = tuple(xi2)
            # Append to cell
            self.C0.add_vertex(self.V(tuple(xi2_t)))
            # Connect neighbours and vice versa
            # Parent point
            self.V(xi2_t).connect(self.V(tuple(xi_t)))
            self.V(xi_t).connect(self.V(tuple(xi2_t)))
            # Origin
            self.V(xi2_t).connect(self.V(tuple(self.origin)))
            self.V(tuple(self.origin)).connect(self.V(xi_t))
            # Suprenum
            self.V(xi_t).connect(self.V(tuple(self.suprenum)))
            self.V(tuple(self.suprenum)).connect(self.V(xi_t))

            # Permutate
            #self.perm(dim_i - 1, i1 - 1, xi2)
            #self.perm(dim_i-1, i, xi2)
            self.perm(i, i2_parents, xi2)


class Vertex:
    def __init__(self, x, func=None, func_args=(), nn=None):
        import numpy
        self.x = x
        x_a = numpy.array(x)
        self.f = func(x_a, *func_args)
        if nn is not None:
            self.nn = nn
        else:
            self.nn = []

        self.fval = None

    def connect(self, v):
        if v not in self.nn and v is not self:  # <-- Cool
            self.nn.append(v)

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)

class VertexCached:
    def __init__(self, func, func_args):
    #    from collections import Counter
        self.cache = {}
        self.func = func
        self.func_args = func_args
    #    self.Index = Counter()

    def __call__(self, x):
        if x in self.cache:
            return self.cache[x]
        else:
            xval = Vertex(x, func=self.func, func_args=self.func_args)
            #xval = Vertex(x)
            logging.info("New generated vertex at x = {}".format(x))
            self.cache[x] = xval
            return xval



if __name__ == '__main__':
    def test_func(x):
        import numpy
        return numpy.sum(x ** 2)


    HC = Complex(200, test_func)
    #Cc = Cell(0, 0)
    #print(Cc())

    if 0:
        HC.V((0, 0, 0))
        HC.V((0, 0, 0))
        HC.V((1, 0, 0))
        HC.V((0, 1, 0))
        HC.V((0, 0, 1))
        HC.V((1, 1, 1))
        HC.V((0, 0, 0)).connect(HC.V((1, 0, 0)))
        HC.V((0, 0, 0)).connect(HC.V((0, 1, 0)))
        HC.V((0, 0, 0)).connect(HC.V((0, 0, 1)))
        HC.V((0, 0, 0)).connect(HC.V((1, 1, 1)))
        HC.V((0, 0, 0)).connect(HC.V((1, 1, 0)))
        HC.V((0, 0, 0)).disconnect(HC.V((1, 1, 1)))
        #print(HC.V((0, 0, 0)).nn)
        #print(HC.V((0, 0, 0)).nn[3].x)

        print(HC.V((0, 0, 0)).nn)
        print(HC.V((0, 0, 0)).nn[3].x)
        print(HC.V((0, 0, 0)).f)
        print(HC.V((0, 1, 1)).f)


    def fact(N):
        #print(N)
        if N == 0 or N == 1:
            return 1
        else:
            return fact(N - 1)*N

    #fact(10)

