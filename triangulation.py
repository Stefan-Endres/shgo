import itertools
import numpy
import logging
import sys
import copy

try:
    pass
    #from multiprocessing_on_dill import Pool
except ImportError:
    from multiprocessing import Pool

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

class Complex:
    def __init__(self, dim, func, func_args=(), symmetry=False, g_cons=None):
        self.dim = dim
        self.gen = 0
        self.perm_cycle = 0

        # Every cell is stored in a list of its generation,
        # ex. the initial cell is stored in self.H[0]
        # 1st get new cells are stored in self.H[1] etc.
        # When a cell is subgenerated it is removed from this list

        self.H = []  # Storage structure of cells

        self.V = VertexCached(func, func_args)  # Cache of all vertices

        # Generate n-cube here:
        self.n_cube(dim, symmetry=symmetry, printout=True)
        self.add_centroid()
        self.H.append([])
        self.H[0].append(self.C0)
        self.hg0 = self.C0.homology_group_order()
        #for v in self.C0():
         #   print(v)

    def __call__(self):
        return self.H


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
        self.C0 = Cell(0, 0, self.origin, self.suprenum)  # Initial cell object
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
        self.C0.centroid = self.centroid

        # Disconnect origin and suprenum
        self.V(tuple(self.origin)).disconnect(self.V(tuple(self.suprenum)))

        # Connect centroid to all other vertices
        for v in self.C0():
             self.V(tuple(self.centroid)).connect(self.V(tuple(v.x)))

        self.centroid_added = True
        return

    # Construct incidence array:
    def incidence(self):
        if self.centroid_added:
            self.structure = numpy.zeros([2**self.dim + 1, 2**self.dim + 1], dtype=int)
        else:
            self.structure = numpy.zeros([2**self.dim, 2**self.dim], dtype=int)\


        for v in HC.C0():
            for v2 in v.nn:
                #self.structure[0, 15] = 1
                self.structure[v.I, v2.I] = 1

        return

    # A more sparse incidence generator:
    def graph_map(self):
        """ Make a list of size 2**n + 1 where an entry is a vertex
        incidence, each list element contains a list of indexes
        corresponding to that entries neighbours"""
        self.graph = []
        for i, v in enumerate(HC.C0()):
            self.graph.append([])
            for v2 in v.nn:
                self.graph[i].append(v2.I)

    # Graph structure method:
    # 0. Capture the indices of the initial cell.
    # 1. Generate new origin and suprenum scalars based on current generation
    # 2. Generate a new set of vertices corresponding to a new
    #    "origin" and "suprenum"
    # 3. Connected based on the indices of the previous graph structure
    # 4. Disconnect the edges in the original cell

    def sub_generate_cell(self, C_i, gen):
        """Subgenerate a cell `C_i` of generation `gen` and
        homology group rank `hgr`."""
        origin_new = C_i.centroid
        centroid_index = len(C_i()) - 1

        # If not gen append
        try:
            self.H[gen]
        except IndexError:
            self.H.append([])

        # Generate subcubes using every extreme verex in C_i as a suprenum
        # and the centroid of C_i as the origin
        H_new = []  # list storing all the new cubes split from C_i
        for i, v in enumerate(C_i()):
            #origin = tuple(C_i.centroid)
            if i is not centroid_index:
                suprenum = tuple(v.x)
                H_new.append(
                    self.construct_hypercube(origin_new, suprenum, gen, C_i.hgr))

        # Disconnected all edges of parent cells (except origin to sup)
        for i, connections in enumerate(self.graph):
            # Present vertex V_new[i]; connect to all connections:
            if i == centroid_index:  # Break out of centroid
                break

            for j in connections:
                C_i()[i].disconnect(C_i()[j])

        # Destroy the old cell
        if C_i is not self.C0:  # Garbage collector does this anyway; not needed
            del(C_i)

        #TODO: Recalculate all the homology group ranks of each cell
        return H_new

    def split_generation(self):
        """
        Run sub_generate_cell for every cell in the current complex self.gen
        """
        try:
            for c in self.H[self.gen]:
                self.sub_generate_cell(c, self.gen + 1)
        except IndexError:
            pass

        self.gen += 1


    def construct_hypercube(self, origin, suprenum, gen, hgr, printout=False):
        """
        Build a hypercube with triangulations symmetric to C0.

        Parameters
        ----------
        origin : vec
        suprenum : vec
        gen : generation
        hgr : parent homology group rank
        """

        # Initiate new cell
        C_new = Cell(gen, hgr, origin, suprenum)
        C_new.centroid = list((numpy.array(origin) + numpy.array(suprenum))/2.0)

        centroid_index = len(HC.C0()) - 1
        # Build new indexed vertex list
        V_new = []

        for i, v in enumerate(HC.C0()):
            if i == centroid_index:  # (This should be the last index of HC.C0()
                C_new.add_vertex(self.V(tuple(C_new.centroid)))
                V_new.append(C_new.centroid)
                break

            vec = list(origin)  # set a vec equal to origin tuple
            for j, x_i in enumerate(v.x):
                if x_i:  # (if x_i = 1, else leave x_i = 0 at origin
                    vec[j] = suprenum[j]

            C_new.add_vertex(self.V(tuple(vec)))
            V_new.append(vec)

        # Connect new vertices
        for i, connections in enumerate(self.graph):
            # Present vertex V_new[i]; connect to all connections:
            for j in connections:
                self.V(tuple(V_new[i])).connect(self.V(tuple(V_new[j])))

        #print('V_new = {}'.format(V_new))

        if printout:
            print("A sub hyper cube with:")
            print("origin: {}".format(origin))
            print("suprenum: {}".format(suprenum))
            for v in C_new():
                print("Vertex: {}".format(v.x))
                constr = 'Connections: '
                for vc in v.nn:
                    constr += '{} '.format(vc.x)

                print(constr)
                print('Order = {}'.format(v.order))

        # Append the new cell to the to complex
        self.H[gen].append(C_new)

        return C_new

    # Not completed zone:
    ## Symmetry group topological transformation methods
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

    # Plots
    def plot_complex(self):
        """
             Here C is the LIST of simplexes S in the
             2 or 3 dimensional complex

             To plot a single simplex S in a set C, use ex. [C[0]]
        """
        from matplotlib import pyplot
        if self.dim == 2:
            pyplot.figure()
            for C in self.H:
                for c in C:
                    for v in c():
                        logging.info('v.x = {}'.format(v.x))

                        pyplot.plot([v.x[0]], [v.x[1]], 'o')

                        xlines = []
                        ylines = []
                        for vn in v.nn:
                            logging.info('vn.x = {}'.format(vn.x))

                            xlines.append(vn.x[0])
                            ylines.append(vn.x[1])
                            xlines.append(v.x[0])
                            ylines.append(v.x[1])
                        pyplot.plot(xlines, ylines)

            pyplot.ylim([-1e-2, 1 + 1e-2])
            pyplot.xlim([-1e-2, 1 + 1e-2])

            pyplot.show()

        elif self.dim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')

            for C in self.H:
                for c in C:
                    for v in c():
                        x = []
                        y = []
                        z = []
                        logging.info('v.x = {}'.format(v.x))
                        x.append(v.x[0])
                        y.append(v.x[1])
                        z.append(v.x[2])
                        for vn in v.nn:
                            x.append(vn.x[0])
                            y.append(vn.x[1])
                            z.append(vn.x[2])
                            x.append(v.x[0])
                            y.append(v.x[1])
                            z.append(v.x[2])
                            logging.info('vn.x = {}'.format(vn.x))

                        ax.plot(x, y, z, label='simplex')

            pyplot.show()
        else:
            print("dimension higher than 3 or wrong complex format")
        return


class Cell:
    """
    Contains a cell that is symmetric to the initial hypercube triangulation
    """
    def __init__(self, p_gen, p_hgr, origin, suprenum):
        self.p_gen = p_gen  # parent generation
        self.p_hgr = p_hgr  # parent homology group rank
        self.C = []
        self.origin = origin
        self.suprenum = suprenum
        self.centroid = None  # (Not always used)
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
        for v in self():
            print("Vertex: {}".format(v.x))
            constr = 'Connections: '
            for vc in v.nn:
                constr += '{} '.format(vc.x)

            print(constr)
            print('Order = {}'.format(v.order))


class Vertex:
    def __init__(self, x, func=None, func_args=(), nn=None, I=None):
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

        # Index:
        if I is not None:
            self.I = I

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
    def __init__(self, func, func_args, indexed=True):

        self.cache = {}
        self.func = func
        self.func_args = func_args

        if indexed:
            self.Index = -1

    def __call__(self, x, indexed=True):
        if x in self.cache:
            return self.cache[x]
        else:
            if indexed:
                self.Index += 1
                xval = Vertex(x, func=self.func, func_args=self.func_args, I=self.Index)
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
    HC = Complex(4, test_func)
    if 0:
        for n in range(9):
            import time
            ts = time.time()
            HC = Complex(n, test_func)
            logging.info('Total time at n = {}: {}'.format(n, time.time() - ts))

    #Complex.stretch(None, HC.C0, 0.5)

    #Complex.generate_gen(HC)

    HC.add_centroid()

    if 0:
        HC.incidence()
        print(HC.structure)

    HC.graph_map()
    logging.info('HC.graph = {}'.format(HC.graph))
    #print(HC.C0())
    #print(HC())


    origin = (0.5, 0.5, 0.5, 0.5)
    suprenum = (1.0, 1.0, 1.0, 1.0)
    gen = 1
    HC.H.append([])

   # HC.sub_generate_cell(HC.C0, gen)

    #for i in range(4):
    for i in range(0):
        HC.split_generation()

    #print(HC.V)
    HC.plot_complex()
    #HC.construct_hypercube(origin, suprenum, gen, hgr)

    if 0:
        print(HC.H)
        print(len(HC.H[1]))
        print(HC.H[1][0])
        HC.H[1][0].print_out()
