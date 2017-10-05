from _shgo import *
from _tgo import *
import numpy
import matplotlib.pyplot as plot
import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True

def plot_mincount(func, bounds, n, plotit=True):
    import numpy
    k_pool = []
    min_pool = []
    for k in range(1, n):
        res2, TGOc = tgo(func, bounds, n=n, k_t=k)
        #print('TGO results:')
        #print('============')
        #print(TGOc.minimizers(TGOc.K_opt))
        #print(res2)

        #print(len(res2.xl))
        #print('=============')
        k_pool.append(k)
        min_pool.append(len(TGOc.minimizers(TGOc.K_opt)))

    if plotit:
        plot.plot(k_pool, min_pool, 'ko-', linewidth=1.0, markersize=4.0)
        plot.xlabel('$k$', fontsize=10)
        plot.ylabel('Minimisers              ', fontsize=10, rotation=1)

        plot.ylabel('$|\mathcal{M}^k|~~~~~~~$       ', fontsize=10, rotation=1)
        #plot.ylim([0, 6])
        #plot.xlim([0, 10])
    return

def minima_maps(func, bounds, nrange=[(2, 50)], kr=[1, 2, 3, 4, 5, 6, 8, 9, 10]):
    plot.figure()
    nr = []
    ks = []
    for i in range(len(kr)):
        ks.append([])

    for n in range(nrange[0][0], nrange[0][1] + 1):
        #res2, TGOc = tgo(func, bounds, n=n, k_t=k)
        nr.append(n)
        for i, k in zip(range(len(kr)), kr):
            res2, TGOc = tgo(func, bounds, n=n, k_t=k)
            ks[i].append(len(TGOc.minimizers(TGOc.K_opt)))

    print(ks)
    styles = ['o-', 'x-', '^-', 's-', '.-']
    if len(kr) > len(styles):
        for i in range(len(kr)):
            styles.append(styles[0])

    for i, k in zip(range(len(kr)), kr):
        plot.plot(nr, ks[i], styles[i], label='$k = {}$'.format(k), linewidth=1.0, markersize=3.0)

    plot.xlabel('$N$', fontsize=10)
    plot.ylabel('Minimisers              ', fontsize=10, rotation=1)
    plot.ylabel('$|\mathcal{M}^k|$       ', fontsize=10, rotation=1)
    plot.legend(loc=0)
    return nr, ks


def f(x):
    return numpy.sin(x) /x

#bounds = [(1, 20)]
bounds = [(1, 25)]
bounds = [(1, 40)]
bounds = [(1, 20)]

n = 100
n = 10

plot_mincount(f, bounds, n)
nr, ks = minima_maps(f, bounds, nrange=[(2, 50)])


from numpy import sin, exp, log
bounds2 = [(0, 80)]
def f2(x):
    return -x * sin(x)

nr, ks = minima_maps(f2, bounds2, nrange=[(2, 100)])



def fev_maps(func, bounds, nrange=[(2, 50)]):
    nr = []
    fev = []


    for n in range(nrange[0][0], nrange[0][1] + 1):
        res2, TGOc = tgo(func, bounds, n=n, k_t=3)
        fev.append(res2.nfev)
        nr.append(n)

    print(fev)
    plot.plot(nr, fev)

    plot.xlabel('$n$', fontsize=14)
    plot.ylabel('nfev', fontsize=14, rotation=1)
    plot.legend(loc=0)
    return

#fev_maps(f2, bounds2, nrange=[(2, 1000)])
#fev_maps(f, bounds, nrange=[(2, 1000)])
plot.show()