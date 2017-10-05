from _shgo_sobol import *
import matplotlib
from matplotlib import pyplot as plot
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

def f(x):  # Alpine2
    prod = 1
    for i in range(numpy.shape(x)[0]):
        prod = prod * numpy.sqrt(x[i]) * numpy.sin(x[i])

    return prod

bounds = [(0, 5), (0, 5)]
n = 10

def f(x):  # Ursem01
    import numpy
    return -numpy.sin(2 * x[0] - 0.5 *numpy.pi)  - 3 * numpy.cos(x[1]) - 0.5 * x[0]

bounds = [(-2.5, 3), (-2, 2)]
bounds = [(-3, 3), (-3, 3)]
n = 10
SHc = SHGO(f, bounds, n = n)


# %matplotlib nbagg
import matplotlib.pyplot as plt


def build_contour(SHc, surface=True, contour=True):
    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    from matplotlib import cm

    X = SHc.C[:, 0]
    X = numpy.linspace(SHc.bounds[0][0], SHc.bounds[0][1])
    Y = SHc.C[:, 1]
    Y = numpy.linspace(SHc.bounds[1][0], SHc.bounds[1][1])
    xg, yg = numpy.meshgrid(X, Y)
    Z = numpy.zeros((xg.shape[0],
                     yg.shape[0]))

    for i in range(xg.shape[0]):
        for j in range(yg.shape[0]):
            Z[i, j] = SHc.func([xg[i, j], yg[i, j]])

    if surface:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0,
                        antialiased=True, alpha=1.0, shade=True)

        ax.set_xlabel('$x_1$')
        ax.set_ylabel('$x_2$')
        ax.set_zlabel('$f$')

    if contour:
        plt.figure()
        cs = plt.contour(xg, yg, Z, cmap='binary_r', color='k')
        plt.clabel(cs)


#build_contour(SHc)


def direct_2d(ax, func, V1, V2, vertex_plot_size=0.00):
    """Draw a directed graph arrow between two vertices"""

    # NOTE: Can retrieve from stored class
    f_1 = func(V1)
    f_2 = func(V2)

    def vertex_diff(V_low, V_high, vertex_plot_size):
        # Note assumes bounds in R+ (0, inf)
        dV = [0, 0]
        for i in [0, 1]:
            if V_low[i] < V_high[i]:
                dV[i] = -(V_high[i] - V_low[i])  # + vertex_plot_size
            else:
                dV[i] = V_low[i] - V_high[i]  # - vertex_plot_size

            if dV[i] > 0:
                dV[i] -= vertex_plot_size
            else:
                dV[i] += vertex_plot_size

        return dV

    if f_1 > f_2:  # direct V2 --> V1
        dV = vertex_diff(V1, V2, vertex_plot_size)
        # print(dV)
        # ax.arrow(V2[0], V2[1], dV[0], dV[1], head_width=0.2, head_length=0.05, fc='k', ec='k', color='b')
        ax.arrow(V2[0], V2[1], 0.6 * dV[0], 0.6 * dV[1], head_width=0.05, head_length=0.2, fc='k', ec='k', color='b')

    elif f_1 < f_2:  # direct V1 --> V2
        pass
        # ax.arrow(V2[0], V2[1], dV[0], dV[1], head_width=0.2, head_length=0.05, fc='k', ec='k', color='b')

    return f_1, f_2  # TEMPORARY USE IN LOOP


# %matplotlib nbagg


def build_complex(n=11, lb1=-3, lb2=-3, ub1=3, ub2=3, labels=True):
    import numpy
    bounds = [(lb1, ub1), (lb2, ub2)]
    SHc = SHGO(f, bounds, n=n)
    SHc.construct_complex_sobol()
    SHc.minimise_pool()
    SHc.sort_result()

    if 1:  # Prints for paper
        print(SHc.C)
        print(SHc.F)
        vstr = ''
        for i, v_c in enumerate(SHc.C):
            vstr += 'v_{' + '{}'.format(i) + '} = ' + '({}, '.format(v_c[0]) + '{}) '.format(v_c[1]) + '\\\\'
            vstr += '\n'

        print(vstr)
        fstr = ''
        for i, fun_out in enumerate(SHc.F):
            fstr += 'f_{' + '{}'.format(i) + '} = ' + '{}'.format(fun_out) + '\\\\'
            fstr += '\n'

        print(fstr)

    import matplotlib.pyplot as plt
    from scipy.spatial import Delaunay
    # plt.figure()

    points = SHc.C
    tri = Delaunay(points)

    # Label edges
    edges = numpy.array(tri.simplices)  # numpy.zeros(len(tri.simplices))
    constructed_edges = []
    incidence_array = numpy.zeros([numpy.shape(points)[0], numpy.shape(edges)[0]])

    # contour
    build_contour(SHc, surface=False, contour=True)

    # graph
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy(), color='k')
    plt.plot(points[:, 0], points[:, 1], 'k.')
    # plt.plot(points[:, 0], points[:, 1], 'ro')
    plt.plot(SHc.C[SHc.minimizer_pool, :][:, 0], SHc.C[SHc.minimizer_pool, :][:, 1], 'ro')
    print(SHc.C[SHc.minimizer_pool, :])

    # directed
    ax = plt.axes()
    for i in range(points.shape[0]):
        for i2 in SHc.find_neighbors_delaunay(i, SHc.Tri):
            # Draw arrow
            f_1, f_2 = direct_2d(ax, f, SHc.Tri.points[i, :], SHc.Tri.points[i2, :])

            # Find incidence on an edge
            for edge, e in zip(edges, range(numpy.shape(edges)[0])):
                # print(edge)
                if e not in constructed_edges:
                    if i in edge:
                        if f_1 < f_2:
                            incidence_array[i, e] += 1
                        elif f_1 > f_2:
                            incidence_array[i, e] -= 1
                    if i2 in edge:
                        if f_2 < f_1:
                            incidence_array[i2, e] += 1
                        elif f_2 > f_1:
                            incidence_array[i2, e] -= 1

                    constructed_edges.append(e)

                    # f_1 = func(V1)
                    # f_2 = func(V2)

    if 0:  # Simplx shades to demonstrate star notation
        x = [SHc.C[9][0], SHc.C[11][0], SHc.C[4][0], SHc.C[14][0]]
        y = [SHc.C[9][1], SHc.C[11][1], SHc.C[4][1], SHc.C[14][1]]
        plt.fill(x, y, color='grey', lw=1, alpha=0.6)#,  hatch= "/ ")

    if 1:  # Simplex shades for Sperner proof
        # Fills
        # define corner points


        plt.plot([SHc.C[7][0], 0], [SHc.C[7][1], 2.5], 'k--', mew=1.5, markersize=5)  # line

        x4 = [SHc.C[8][0], SHc.C[3][0], SHc.C[7][0],  0, SHc.C[0][0]]
        y4 = [SHc.C[8][1], SHc.C[3][1], 2.5,         2.5, SHc.C[0][1]]
        plt.fill(x4, y4, color='C3', lw=1, alpha=0.2,  hatch= "/ ")


        #vel = SHc.C[9][0] - SHc.C[1][0]

        v1 = 0.5 * (SHc.C[9] - SHc.C[1]) + SHc.C[1]
        v2 = 0.3 * (SHc.C[11] - SHc.C[1]) + SHc.C[1]
        print('v1 = {}'.format(v1))

        x2 = [v2[0], v1[0], SHc.C[14][0]]
        y2 = [v2[1], v1[1], SHc.C[14][1]]
        plt.fill(x2, y2, color='C1', lw=2, alpha=0.3, hatch='|')

        x3 = [SHc.C[9][0], SHc.C[10][0], SHc.C[5][0]]
        y3 = [SHc.C[9][1], SHc.C[10][1], SHc.C[5][1]]

        plt.fill(x3, y3, color='grey', lw=2, alpha=0.3, hatch='//')

        #ax.annotate('$\mathbf{1}$', xy=(SHc.C[5]) - [0.2,0.2], fontsize=20, fontweight='bold', color='blue')  # ,

        x = [SHc.C[8][0], SHc.C[3][0], 0]
        y = [SHc.C[8][1], SHc.C[3][1], 2.5]
        plt.fill(x, y, color='C0', lw=2, alpha=0.3, hatch="""\ \ """)
        

        for xl in SHc.res.xl:
            plt.plot(xl[0], xl[1], 'go', markersize=10, alpha=0.5)
            plt.plot(xl[0], xl[1], 'kx', mew=1.5, markersize=5)


    if labels:
        for i, v in enumerate(SHc.C):
            print(v)
            point = v
            if 1: # Manual vertex shifting
                if i is 0:
                    point[0] += 0.1  # x
                    point[1] += 0.18  # y

                if i is 1:
                    point[0] -= 0.18
                    point[1] += 0.2

                if i is 2:
                    point[0] += 0.1

                if i is 3:
                    point[1] += 0.15

                if i is 4:
                    point[0] += 0.2
                    point[1] -= 0.1

                if i is 5:
                    point[0] += 0.05
                    point[1] += 0.05

                if i is 6:
                    point[0] += 0.0
                    point[1] -= 0.2

                if i is 7:
                    point[0] -= 0.5
                    point[1] -= 0.1

                if i is 8:
                    point[0] -= 0.42
                    point[1] += 0

                if i is 9:
                    point[0] -= 0.08
                    point[1] -= 0.25

                if i is 10:
                    point[0] += 0.04

                if i is 11:
                    point[0] -= 0.25
                    point[1] += 0.18

                if i is 12:
                    point[0] += 0.1
                    point[1] += 0.02

                if i is 13:
                    point[0] += 0.1
                    point[1] -= 0.02

                if i is 14:
                    point[0] += 0.2
                    point[1] -= 0.05

            ax.annotate('$v_{' +'{}'.format(i) + '}$',
                          xy=(point), fontsize=14)  # ,



    # interact(build_complex, n=(3, 100), lb1=(-100, 100), lb2=(-100, 100), ub1=(-45, 55), ub2=(-45, 55))

#build_complex(n=15, lb1=0, lb2=-2.5, ub1=9.2, ub2=2.5)
if 1:
    build_complex(n=15, lb1=0, lb2=-2.5, ub1=9.2, ub2=2.5)

if 0:
    build_complex(n=150, lb1=0, lb2=-2.5, ub1=9.2, ub2=2.5, labels=False)
plt.xlabel('$x_1$', fontsize=14)
plt.ylabel('$x_2$', fontsize=14, rotation=1)
plt.show()