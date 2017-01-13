import itertools
import numpy
import logging
import sys
import copy

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class Cell:
    """
    Contains a cell that is symmetric to the initial hypercube triangulation
    """
    def __init__(self, gen, hg):
        self.gen = gen  # generation
        self.hg = hg  # parent homology group
        self.C = []
        #TODO: self.bounds

    def __call__(self):
        return self.C

    def add_vertex(self, V):
        if V not in self.C:
            self.C.append(V)

    def homology_group_order(self):
        """
        Returns the homology group order of the current cell
        """
        hg_n = 0
        for v in self.C:
            if v.minimiser():
                hg_n += 1

        self.hg_n = hg_n
        return hg_n

    def homology_group_differential(self):
        """
        Returns the difference between the current homology group of the
        cell and it's parent group
        """
        self.hgd = self.hg_n - self.hg

    def polytopial_sperner_lemma(self):
        """
        Returns the number of stationary points theoretically contained in the
        cell based information currently known about the cell
        """
        pass

    def print_out(self):
        """
        Print the current cell to console
        """
        return

class Complex:
    def __init__(self, dim, func, func_args=(), symmetry=False, g_cons=None):
        self.dim = dim
        self.gen = 1
        self.perm_cycle = 0

        self.C = []  # List of cells
        self.V = VertexCached(func, func_args)  # Cache of all vertices

        # Generate n-cube here:
        self.n_cube(dim, symmetry=symmetry)
        self.C.append(self.C0)
        self.hg0 = self.C0.homology_group_order()
        #for v in self.C0():
         #   print(v)

    def n_cube(self, dim, symmetry=False, printout=False):
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

        i_parents = []
        x_parents = []
        x_parents.append(self.origin)
        self.perm(i_parents, x_parents, origin)

        if printout:
            print("Initial hyper cube:")
            for v in self.C0():
                print("Vertex: {}".format(v.x))
                constr = 'Connections: '
                for vc in v.nn:
                    constr += '{} '.format(vc.x)

                print(constr)
                print('Order = {}'.format(v.order))

    def perm(self, i_parents, x_parents, xi):
        #TODO: Cut out of for if outside linear constraint cutting planes
        xi_t = tuple(xi)

        # Construct required iterator
        iter_range = [x for x in range(self.dim) if x not in i_parents]

        for i in iter_range:
            i2_parents = copy.copy(i_parents)#.copy()
            i2_parents.append(i)
            xi2 = copy.copy(xi)#.copy()
            xi2[i] = 1
            # Make new vertex list a hashable tuple
            xi2_t = tuple(xi2)
            # Append to cell
            self.C0.add_vertex(self.V(tuple(xi2_t)))
            # Connect neighbours and vice versa
            # Parent point
            self.V(xi2_t).connect(self.V(tuple(xi_t)))

            # Connect all family of simplices in parent containers
            for x_ip in x_parents:
                self.V(xi2_t).connect(self.V(tuple(x_ip)))

            x_parents2 = copy.copy(x_parents)#.copy()
            x_parents2.append(xi_t)

            # Permutate
            self.perm(i2_parents, x_parents2, xi2)

    def add_centroid(self):
        """Split the central edge between the origin and suprenum of
        a cell and add the new vertex to the complex"""
        self.centroid = list((numpy.array(self.origin) + numpy.array(self.suprenum))/2.0)
        self.C0.add_vertex(self.V(tuple(self.centroid)))

        # Disconnect origin and suprenum
        self.V(tuple(self.origin)).disconnect(self.V(tuple(self.suprenum)))

        # Connect centroid to all other vertices
        for v in HC.C0():
             self.V(tuple(self.centroid)).connect(self.V(tuple(v.x)))
        print(self.centroid)

    def generate_gen(self):
        """Generate all cells in the next generation of subdivisions"""
        self.gen += 1
        self.C_new = []
        for cell in self.C:
            self.C_new_cells = self.generate_sub(cell, self.gen, self.hg0)
            for c_new in self.C_new_cells:
                self.C_new.append(c_new)

        # Set new complex
        self.C = self.C_new
        return

    def generate_sub(self, cell, gen, hg0=0):
        """Generate the subdivision of a specified cell"""
        # Shrink the initial cell
        factor = float(1/(gen))
        print(factor)

        return [None]

    def stretch(self, cell, factor):
        """
        Stretch transformation of all vertices in a cell.
        """
        # TODO: Optimize with numpy arrays and matrix operations
        C1 = Cell(1, 0)
        for v in cell.C:
            print(tuple(numpy.array(v.x) * factor))

        # (loop through all neighbours and stretch


        return

    def translate(self, cell, v_start, v_end):
        """
        translate the cell from a vector starting at v_start pointing at v_end
        """

        return


    def rotation(self, cell):
        # Return all SO(n) group rotations of input cell
        return

class Vertex:
    def __init__(self, x, func=None, func_args=(), nn=None):
        import numpy
        self.x = x
        self.order = sum(x)
        x_a = numpy.array(x)
        # Note Vertex is only initiate once for all x so only
        # evaluated once
        if func is not None:
            self.f = func(x_a, *func_args)

        if nn is not None:
            self.nn = nn
        else:
            self.nn = []

        self.fval = None
        self.check_min = True

    def connect(self, v):
        if v not in self.nn and v is not self:  # <-- Cool
            self.nn.append(v)
            v.nn.append(self)
            if self.f > v.f:
                self.min = False
            #self.check_min = True

    def disconnect(self, v):
        if v in self.nn:
            self.nn.remove(v)
            v.nn.remove(self)
            self.check_min = True

    def minimiser(self):
        if self.check_min:
            # Check if the current vertex is a minimiser
            self.min = False
            for v in self.nn:
                if self.f > v.f:
                    break

                self.min = True

            self.check_min = False
            return (self.min)
        else:
            return(self.min)

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
            logging.info("New generated vertex at x = {}".format(x))
            self.cache[x] = xval
            return xval



if __name__ == '__main__':
    def test_func(x):
        import numpy
        return numpy.sum(x ** 2) + 2.0 * x[0]

    tr = []
    nr = list(range(9))
    HC = Complex(2, test_func)
    for n in range(9):
        import time
        ts = time.time()
        #HC = Complex(4, test_func)

        #tr


    #Complex.stretch(None, HC.C0, 0.5)

    #Complex.generate_gen(HC)

    HC.add_centroid()

    print(HC.C0())

    if 1:
        print(HC.V((0.5, 0.5, 0.5, 0.5)).nn)
        for v in HC.V((0.5, 0.5, 0.5, 0.5)).nn:
            print('v = {}'.format(v.x))


    if 0:
        for v in HC.C0:
            print('v = {}'.format(v))