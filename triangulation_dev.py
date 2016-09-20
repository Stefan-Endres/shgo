import itertools
import numpy
class Vertex:
    def __init__(self, x, nn=None):
        self.x = x
        if nn is not None:
            self.nn = nn
        else:
            self.nn = []

        self.fval = None


# p1 = Point([0, 0])
# p2 = Point([0, 1])
# p3 = Point([1, 0])
# p4 = Point([1, 1])
# p1.neighbours = [p2, p3, p4]

# v = [p1, p2, p3, p4]
#V[0]

dim = 3
V = []
# Note that these two vertices are connected to all others
origin = list(numpy.zeros(dim, dtype=int))
supremum = list(numpy.ones(dim, dtype=int))
V.append(Vertex(origin))
V.append(Vertex(supremum, [V[0]]))
V[0].nn.append(V[1])

Vsize = len(V)
for i1 in range(dim):  # Add exception handling
    print('i1 = {}'.format(i1))
    x0 = origin.copy()
    x0[i1] = 1
    V.append(Vertex(x0, [V[0], V[1]]))
    V[0].nn.append(V[i1 + Vsize])
    V[1].nn.append(V[i1 + Vsize])

    ind1 = i1 + Vsize
    Vsize1 = Vsize + dim
    for i2 in range(dim):
        x1 = x0.copy()
        ind2 = ind1
        if i2 is not i1:
            ind2 += 1
            x1[i2] = 1
            V.append(Vertex(x1, [V[0], V[1], V[ind1]]))
            V.append(Vertex(x1, [V[0], V[1], V[ind1]]))
            print(ind1)
            print(len(V))
            #V[0].nn.append(V[i2 + Vsize1])
            V[0].nn.append(V[ind2])
            #V[1].nn.append(V[i2 + Vsize1])
            V[1].nn.append(V[ind2])
            V[ind1].nn.append(V[ind2])

def perm(x0, i0, Vsize, D=1):
    for i in range(dim):
        if i is not i0:
            x0[i] = 1
            V.append(Vertex(x0, [V[0], V[1]]))
            V[0].nn.append(V[i1 + Vsize])
            V[1].nn.append(V[i1 + Vsize])


for v in V:
    print('v.x = {}'.format(v.x))
    for vn in v.nn:
        print('vn.x = {}'.format(vn.x))

