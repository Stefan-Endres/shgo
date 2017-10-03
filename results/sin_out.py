#from _shgo import *
from _tgo import *
import numpy
import matplotlib.pyplot as plot
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True


def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + ' & '.join(l.split()) + r'\\' for l in lines]
    rv += [r'\end{bmatrix}']
    return '\n'.join(rv)

    #print(bmatrix(A) + '\n')


def plot_results(func, bounds, n):
    import numpy
    xr = numpy.linspace(bounds[0][0], bounds[0][1], 1000)
    plot.figure()
    plot.plot(xr, func(xr), 'k')
    # Init tgo class
    # Note: Using ints for irrelevant class inits like func

    #TGOc = TGO(func, bounds, n=n)
    #TGOc.n = n
    #TGOc.sampling()

    #res2, TGOc = tgo(func, bounds, n=n, k_t=1)
    res2, TGOc = tgo(func, bounds, n=n)
    print('TGO results:')
    print('============')
    print("Sampled points TGOc.C = {}".format(TGOc.C))
    Sampling = numpy.chararray(n, itemsize=20)
    Topospoints = numpy.zeros([n, n])
    Topospoints[:, 0] = range(n)
    Topospoints[:, 1:] = TGOc.A
    #Topospoints[:, 1:] = TGOc.K_opt
    #Topos = numpy.chararray([n, n], itemsize=7)
    Topos = numpy.chararray([n, n-1], itemsize=8)
    Topos = numpy.chararray([n, n], itemsize=8)
    if 1:  #TODO: Python 3
        for i in range(Topos.shape[0]):
            Sampling[i] = "p_{"
            Sampling[i] += "{}".format(str(i + 1))
            Sampling[i] += "} = "
            Sampling[i] += str(TGOc.C[i])
            Sampling[i] += ", "
            for j in range(Topos.shape[1]):
                if j == 0:
                    Topos[i, j] = " p_{"
                    Topos[i, j] += "{}".format(str(int(Topospoints[i, j]+1)))
                    Topos[i, j] += "}"
                elif TGOc.T[i, j-1]:
                    #print("$+P_{}$".format(str(TGOc.A[i, j])))
                    Topos[i, j] ="+P_{"
                    Topos[i, j] += "{}".format(str(int(Topospoints[i, j]+1)))
                    Topos[i, j] += "}"
                else:
                    #print("$-P_{}$".format(str(TGOc.A[i, j])))
                    Topos[i, j] ="-P_{"
                    Topos[i, j] += "{}".format(str(int(Topospoints[i, j]+1)))
                    Topos[i, j] += "}"


    if False:
        print("Topos = {}".format(Topos))
        #print("Sampling = {}".format(Sampling))
    print(bmatrix(Topos))
    print(bmatrix(Sampling.T))
    #print(bmatrix(func(TGOc.C)))


    print("Topograph = TGOc.T = {}".format(TGOc.T))
    print(res2)

    print(len(res2.xl))
    print('=============')


    #print('SHGO results:')
    #print('=============')
    #res3 = shgo(func, bounds, n=n)
    #print(len(res3.xl))
    #print(res3)

    #fig, ax = plot.subplots()
    #ax = plot.subplot(111)
    for i, P, F in zip(range(len(TGOc.F)), TGOc.C, TGOc.F):
        print(P, F)
        strout = '$f(p_{'+'{}'.format(i + 1)+'})$'
        if (i+1) in [2, 5, 8, 10]:
            py = 0.04
            px = 0
        elif (i+1) in [1]:
            py = 0
            px = 0.5
        elif (i + 1) in [6, 7, 9]:
            py = 0.04
            px = -0.65
        elif (i + 1) in [3]:
            py = 0.02
            px = 0
        elif (i + 1) in [4]:
            py = 0.025
            px = -1.1
        elif (i + 1) in [7, 4]:
            py = 0
            px = 0

        else:
            py = 0
            px = 0
        plot.annotate(strout,
                      xy=(P + px, F + py), fontsize=10)#,
                 #arrowprops=dict(arrowstyle="->"))


    plot.plot(TGOc.C, func(TGOc.C), 'kx')
    plot.xlabel('$x_1$', fontsize=14)
    plot.ylabel('$f$', fontsize=14, rotation=1)



def f(x):
    return numpy.sin(x * 10) * numpy.e ** -x

def f(x):
    return numpy.sin(x * 2) /x#* numpy.e ** -x

bounds = [(0, 9)]

def f(x):
    return numpy.sin(x) /x


bounds = [(1e-10, 20)]
bounds = [(1, 20)]
bounds = [(1, 40)]
bounds = [(1e-10, 20)]
bounds = [(1, 20)]
n = 100
n = 10

#plot_results(f, bounds, n)

def f2(x):
    return x**3 -20* x**2 + x

from numpy import sin, exp, log
def f2(x):
    return -x * sin(x)


bounds2 = [(0, 10)]
bounds2 = [(0, 100)]
bounds2 = [(0, 80)]
n = 100
n = 50
n = 40
plot_results(f2, bounds2, n)





# 2D
if False:
    def build_contour(SHc, func, surface=True, contour=True):
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
            if False:
                cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
                cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
                cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

            ax.set_xlabel('$x_1$')
            ax.set_ylabel('$x_2$')
            ax.set_zlabel('$f$')

        if contour:
            plt.figure()
            cs = plt.contour(xg, yg, Z, cmap='binary_r')
            plt.clabel(cs)
            plt.plot(TGOc.C[:, 0], TGOc.C[:, 1], 'kx')

        res = tgo(func, bounds, n=n, k_t=1)
        print('TGO results:')
        print(res)

    def f(x):
        return x[0]**2 + x[1]**2

    bounds = [(-1, 1), (-1, 1)]
    n = 10
    TGOc = TGO(f, bounds, n=n)
    TGOc.sampling()
    build_contour(TGOc, f, surface=False)










plot.show()