import logging
import numpy

class Complex():
    """
    A class relating the vertexes of a complex
    through a triangulation of hypersimplices without
    physical storage by exploiting the symmetry of
    the simplicial powerset.
    """

    def __init__(self, dim=2):
        self.dim = dim
        self.V = []
        self.S = []  # Face indexes, edges = 1 face
        self.I = []  # Index sets
        self.i_gen = []
        self.i_current = []
        self.generation_cycle = 1
        for i in range(dim + 1):
            self.S.append([])
            self.I.append([])
            self.i_gen.append(self.index_gen())
            self.i_current.append(0)

            # Intiate first generation of vertices

    def n_cube(self, dim, printout=False, symmetry=False):
        """
        Generate the simplicial triangulation of the n cube
        containing 2**n vertices
        """
        import numpy

        # Symmetry simplex:
        if symmetry:
            C = []
            S = numpy.ones([dim + 1, dim])
            for i in range(dim + 1):
                S[i, :-i] = 0

            S[0, :] = 0
            C.append(S)
            self.C = C
            return C

        # else:
        import itertools

        permgroups = list(itertools.permutations(range(dim)))
        self.permgroups = permgroups
        # Build the D set feasible region to use for symmetery groups later:
        D = [[0, 1], ] * dim  # Domain := hypercube
        D = numpy.array(D)

        C = []
        for tau in permgroups:  # n! simplices
            S = numpy.tile(D[:, 0], (dim + 1, 1))

            for i in range(dim):
                for j in range(dim):
                    S[i + 1] = S[i]
                    # (Needed since looping through i will use these)

                tau[i]
                S[i + 1][tau[i]] = D[tau[i], 1]

            C.append(S)

        if printout:
            self.print_complex(permgroups, C)

        self.C = C

        return C


    def split_generation(self, Ci, V, build_complex_array=False):
        """
        Ci = index simplexes
        V = List of vertexes
        build_complex_arra = Memory expensive arrays containing
            the actual vertex vectors; used to plot the complex

        This method relies on building a hyperplane which connects
        to a new vertex on an edge (the longest edge in dim = {2, 3})
        and every other vertex in the simplex that is not connected to
        the edge being split.

        The hyperplane devides the parent simplex into 2 new simplices.

        The longest edge (in dim = 2?) is tracked by an ordering of the
        vertices in every simplices, the edge between first and last
        vertex is the longest edge to be split in the next iteration.
        """
        # Split the simplices by sampling the longest edge
        Ci_new = []

        # Next in cycle mod(dim - 1)
        gci_n = (self.generation_cycle + 1) % (self.dim - 1)
        gci = gci_n
        self.generation_cycle = gci
        # TODO: Instead of looping:
        # Track generation here using generator indexes
        # Once the first generation is exhausted move on the
        # second generation of simplexes
        # continue untill desired sampling points are met
        # Quick hack:: We can use a pool and fudge exception
        # handling to generate new ones as required
        #
        # Pop the current simplex from old Ci
        # Keep the rest in memory when rebuilding
        for i, Si in enumerate(Ci):  # For every simplex in C:

            V_new = (V[Si[0]] + V[Si[-1]]) / 2.0
            i_new = self.generate_vertex(V_new)

            # TODO: Find N dimensionals simplices which
            # will split the edges connecting to V = [1, 1, 1]
            # in 3 dimensions
            # Noting that the first and last index is the edge to be split;
            #

            # New "lower" simplex
            Si_new_l = []
            Si_new_l.append(Si[0])
            Si_new_l.append(i_new)
            for ind in Si[1:-1]:
                Si_new_l.append(ind)

            # New "upper" simplex
            Si_new_u = []
            Si_new_u.append(Si[gci + 1])

            for k, ind in enumerate(Si[1:-1]):  # iterate through inner vertices
                #for easier k / gci tracking
                k += 1
                #if k == 0:
                #    continue  # We do this rather than S[1:-1]
                              # for easier k / gci tracking
                if k == (gci + 1):
                    Si_new_u.append(i_new)
                else:
                    Si_new_u.append(ind)

            Si_new_u.append(Si[-1])

            # Append to new complex
            Ci_new.append(Si_new_l)
            Ci_new.append(Si_new_u)

        self.Ci = Ci_new
        if not build_complex_array:
            return Ci_new

        C_new = self.construct_new_complex(Ci_new, V)
        return Ci_new, C_new

    def construct_new_complex(self, Ci_new, V):
        # Stack complexes from Ci
        # (mostly used for plots; not needed
        # for practical optimization)
        import numpy
        C_new = []
        for i, ci in enumerate(Ci_new):
            C_new.append(numpy.zeros([self.dim + 1, self.dim]))
            for k, ind in enumerate(ci):
                C_new[i][k, :] = V[ind]
        # Construct indexes
        self.C = C_new
        return C_new

    def initial_vertices(self, C, dim):
        # Fit n_cube result into vertices
        for S in C:
            self.generate_simplex(S, k=dim - 1)
            for i, x in enumerate(S):
                if i < dim:
                    self.generate_vertex(x)
                    # We want the last index of the initial cube to have
                    # the last index in the initial generation

        self.generate_vertex(S[-1])
        return

    def index_simplices(self, C):
        # Construct the index simplices
        Ci = []
        for i, S in enumerate(C):
            Ci.append([])  # = Si
            for x in S:
                for j, v in enumerate(self.V):
                    if (x == v).all():
                        Ci[i].append(j)
        self.Ci = Ci
        return Ci

    def connected_vertices(self, ind, Ci):
        # Find all vertices connected to V[ind]
        # where V is the set of all vertex vectors
        # contained in the complex of index simplices Ci
        import numpy
        V_connected = []
        for Si in Ci:
            if ind in Si:
                V_connected.append(Si)

        # Return unique elements
        V_connected = numpy.unique(V_connected)

        # Remove central index
        V_connected = list(V_connected)
        V_connected = list(filter(lambda a: a != ind, V_connected))

        return V_connected

    def generate_vertex(self, x, check=True):
        """
        x: vector of cartesian coordinates
        check_for_unique : Boolean, If true all vertices are looped through
                                    to ensure that x is not already in self.V
        """
        if check:
            for i, v in enumerate(self.V):
                if (x == v).all():
                    return i  # return if vertex is found without generating

        self.V.append(x)
        self.i_current[0] = next(self.i_gen[0])
        logging.info('self.i_current[0] = {}'.format(self.i_current[0]))
        self.I[0].append(self.i_current[0])
        i = self.i_current[0]
        return i

    def generate_simplex(self, V_i, k=1):
        """
        V_i: Tuple containing the indexes of vertices to connect
        """
        self.S[k].append(V_i)
        self.i_current[k] = next(self.i_gen[k])
        logging.info('self.i_current[k] = {}'.format(self.i_current[k]))
        self.I[k].append(self.i_current[k])

    def destroy_simplex(self, ind, k=1):
        """
        Delete faces from lists to free up memory
        """
        del self.S[k][ind]

    def index_gen(self):
        ind = 0
        while True:
            yield ind
            ind += 1

    # incidence arrays
    # TODO; might be useful in future

    # plots and prints
    def plot_vertexes(self, V):
        return

    def plot_graph(self):  # TODO: Deprecate this
        from matplotlib import pyplot
        pyplot.figure()
        for v in HC.V:
            pyplot.plot([v[0]], [v[1]], 'o')

        for f in HC.S[1]:
            # print(HC.V[f[0]], HC.V[f[1]])
            pyplot.plot([HC.V[f[0]][0], HC.V[f[1]][0]],
                        [HC.V[f[0]][1], HC.V[f[1]][1]], 'r-')

        pyplot.ylim([-1e-2, 1 + 1e-2])
        pyplot.xlim([-1e-2, 1 + 1e-2])
        pyplot.show()
        return

    def plot_complex(self, C):
        """
        Here C is the LIST of simplexes S in the
        2 or 3 dimensional complex

        To plot a single simplex S in a set C, use ex. [C[0]]
        """
        from matplotlib import pyplot

        dims = len(C[0][0])
        if dims == 2:
            pyplot.figure()
            for s in C:
                xlines = []
                ylines = []
                for v in s:
                    pyplot.plot([v[0]], [v[1]], 'o')
                    xlines.append(v[0])
                    ylines.append(v[1])

                xlines.append(s[0][0])
                ylines.append(s[0][1])
                pyplot.plot(xlines, ylines)

            pyplot.ylim([-1e-2, 1 + 1e-2])
            pyplot.xlim([-1e-2, 1 + 1e-2])
            #pyplot.show()

        elif dims == 3:
            from mpl_toolkits.mplot3d import Axes3D
            fig = pyplot.figure()
            ax = fig.add_subplot(111, projection='3d')

            for s in C:
                x = []
                y = []
                z = []
                for v in range(4):
                    x.append(s[v, 0])
                    y.append(s[v, 1])
                    z.append(s[v, 2])

                # Lines from first with thrid vertex
                x.append(s[0][0])
                y.append(s[0][1])
                z.append(s[0][2])
                x.append(s[2][0])
                y.append(s[2][1])
                z.append(s[2][2])
                # Lines from last to second vertex
                x.append(s[3][0])
                y.append(s[3][1])
                z.append(s[3][2])
                x.append(s[1][0])
                y.append(s[1][1])
                z.append(s[1][2])

                ax.plot(x, y, z, label='simplex')

            #pyplot.show()
        else:
            print("dimension higher than 3 or wrong S format")
        return

    def print_complex(self, permgroups, C):
        for i, tau in enumerate(permgroups):
            print('Tau: {}'.format(tau))
            print('Simpex {}:'.format(i))
            print(C[i])

        return

    def print_complex_set(self):
        print('Index set I = {}'.format(self.I))
        print('Vertices V = {}'.format(self.V))
        print('Simplices S = {}'.format(self.S))

if __name__ == '__main__':
    dim = 4
    HC = Complex(dim)
    C = HC.n_cube(dim, printout=False, symmetry=False)
    HC.initial_vertices(C, dim)
    Ci = HC.index_simplices(C)  # = HC.Ci
    print(len(HC.V))
    for ind in range(dim**2 - 1):
        print("Vertex: {}".format(HC.V[ind]))
        constr = 'Connections:'
        for i in HC.connected_vertices(ind, Ci):
            constr += ' ' + str(HC.V[i])

        print(constr)
    #print(HC.connected_vertices(ind, Ci))
    #HC.V[

    if 0:
        from matplotlib import pyplot
        #for D in range(60):
        dim = 60
        HC = Complex(dim)
        C = HC.n_cube(dim, printout=False, symmetry=True)
        HC.initial_vertices(C, dim)
        Ci = HC.index_simplices(C)  # = HC.Ci

        for i in range(22):
            print(HC.C[-1])
            # HC.plot_complex([HC.C[1]])
            Ci_new = HC.split_generation(HC.Ci, HC.V,
                                                build_complex_array=False)
            #print(Ci_new)
            #print(HC.V)
            print('len(HC.V) = {}'.format(len(HC.V)))
            print('len(Ci_new) = {}'.format(len(Ci_new)))
            print(".generation_cycle = {}".format(HC.generation_cycle))

        if 0:
            for i in range(10):
                HC.plot_complex(HC.C)
                HC.plot_complex([HC.C[0]])
                print(HC.C[-1])
                # HC.plot_complex([HC.C[1]])
                Ci_new, C_new = HC.split_generation(HC.Ci, HC.V,
                                                    build_complex_array=True)
                print(".generation_cycle = {}".format(HC.generation_cycle))

            pyplot.show()

    if 0:
        from matplotlib import pyplot
        # Generate intial simplex
        dim = 2
        HC = Complex(dim)
        #print('HC.dim = {}'.format(HC.dim))
        C = HC.n_cube(dim, printout=False)
        HC.initial_vertices(C, dim)
        Ci = HC.index_simplices(C)  # = HC.Ci

        for i in range(10):
            HC.plot_complex(HC.C)
            HC.plot_complex([HC.C[0]])
            print(HC.C[-1])
            # HC.plot_complex([HC.C[1]])
            Ci_new, C_new = HC.split_generation(HC.Ci, HC.V,
                                                build_complex_array=True)
            print(".generation_cycle = {}".format(HC.generation_cycle))

        pyplot.show()