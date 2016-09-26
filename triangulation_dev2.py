import itertools
import numpy
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

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

V_cached = VertexCached()

V_cached((0, 0, 0))

V_cached((0, 0, 0))
V_cached((0, 0, 0))
V_cached((1, 0, 0))
V_cached((0, 1, 0))
V_cached((0, 0, 1))
V_cached((1, 1, 1))
V_cached((0, 0, 0)).nn.append(V_cached((1, 0, 0)))
V_cached((0, 0, 0)).nn.append(V_cached((0, 1, 0)))
V_cached((0, 0, 0)).nn.append(V_cached((0, 0, 1)))
V_cached((0, 0, 0)).nn.append(V_cached((1, 1, 1)))
print(V_cached((0, 0, 0)).nn)
print(V_cached((0, 0, 0)).nn[3].x)
