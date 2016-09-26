import itertools
import numpy
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class Complex:
    def __init__(self, dim, symmetry=False, g_cons=None):
        self.dim = dim
        self.gen = 0
        self.perm_cycle = 0
        #self.cell = 0
        # ^ Put every vertex into its own cell which changes
        # every generation

        origin = list(numpy.zeros(dim, dtype=int))
        supremum = list(numpy.ones(dim, dtype=int))

        # We set self.V to be the list of vertices in every generation
        # So self.V[0] is the list of vertices in the intitial cube
        # generation
        # self.V[1] = [(0.5, 0.5, 0.5, ...)]
        # self.V[2] is the second cycle permutation split
        # etc.
        #self.V = []
        #self.V.append(VertexCached())
        self.V = VertexCached()

    def perm(self):
        for i in range(self.dimx0, i0, Vsize, D=1):
            if i is not i0:
                x0[i] = 1
                V.append(Vertex(x0, [V[0], V[1]]))
                V[0].nn.append(V[i1 + Vsize])
                V[1].nn.append(V[i1 + Vsize])

class Vertex:
    def __init__(self, x, nn=None):
        self.x = x
        if nn is not None:
            self.nn = nn
        else:
            self.nn = []

        self.fval = None

    def add(self, v):
        if v not in self.nn:
            self.nn.append(v)

class VertexCached:
    def __init__(self):
        self.cache = {}

    def __call__(self, x):
        if x in self.cache:
            return self.cache[x]
        else:
            import numpy
            x_a = numpy.array(x)
            value = Vertex(x_a)
            logging.info("New generated vertex at x = {}".format(x))
            self.cache[x] = value
            return value

HC = Complex(3)
HC.V((0, 0, 0))
HC.V((0, 0, 0))
HC.V((1, 0, 0))
HC.V((0, 1, 0))
HC.V((0, 0, 1))
HC.V((1, 1, 1))
HC.V((0, 0, 0)).nn.append(HC.V((1, 0, 0)))
HC.V((0, 0, 0)).nn.append(HC.V((0, 1, 0)))
HC.V((0, 0, 0)).nn.append(HC.V((0, 0, 1)))
HC.V((0, 0, 0)).nn.append(HC.V((1, 1, 1)))
#print(HC.V((0, 0, 0)).nn)
#print(HC.V((0, 0, 0)).nn[3].x)

print(HC.V((0, 0, 0)).nn)
print(HC.V((0, 0, 0)).nn[3].x)


def fact(N):
    #print(N)
    if N == 0 or N == 1:
        return 1
    else:
        return fact(N - 1)*N

fact(10)