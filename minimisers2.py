from _shgo import *
from _tgo import *
import numpy
import matplotlib.pyplot as plot

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
        plot.plot(k_pool, min_pool, 'ko-')
        plot.xlabel('$k$', fontsize=14)
        plot.ylabel('Minimisers              ', fontsize=14, rotation=1)
        #plot.ylim([0, 6])
        #plot.xlim([0, 10])
    return

def minima_maps(func, bounds, nrange=[(2, 50)], kr=[1, 2, 3, 4], skip=1):
    nr = []
    ks = []
    kc = []
    kshgo = []
    for i in range(len(kr)):
        ks.append([])

    for n in range(nrange[0][0], nrange[0][1] + 1, skip):
        #res2, TGOc = tgo(func, bounds, n=n, k_t=k)
        print(n)
        nr.append(n)
        for i, k in zip(range(len(kr)), kr):
            #res2, TGOc = tgo(func, bounds, n=n, k_t=k)
            res2 = tgo(func, bounds, n=n, k_t=k)
            #ks[i].append(len(TGOc.minimizers(TGOc.K_opt)))
            ks[i].append(len(res2.funl))


        #res_kc, TGOc_kc = tgo(func, bounds, n=n)
        res_kc = tgo(func, bounds, n=n)
        #kc.append(len(TGOc_kc.minimizers(TGOc_kc.K_opt)))
        kc.append(len(res_kc))

        #res_shgo, SHGOc = shgo(func, bounds, n=n)
        res_shgo = shgo(func, bounds, n=n)
        #
        kshgo.append(len(res_shgo.funl))

    #print(ks)
    styles = ['o-', 'x-', '^-', 's-', '.-', '<-']
    styles = ['x-', 's-', '.-', '<-']
    if len(kr) > len(styles):
        for i in range(len(kr)):
            styles.append(styles[0])

    for i, k in zip(range(len(kr)), kr):
        plot.plot(nr, ks[i], styles[i], label='TGO $k = {}$'.format(k))

    plot.plot(nr, kc, '^-', label='TGO $k_c$')

    plot.plot(nr, kshgo, 'o-', label='SHGO')

    plot.xlabel('$N$', fontsize=14)
    plot.ylabel('Minimisers              ', fontsize=14, rotation=1)
    plot.ylabel('|$\mathcal{M}^k$|       ', fontsize=14, rotation=1)
    plot.legend(loc=0)
    return nr, ks, kc, kshgo

#bounds = [(1, 20)]
bounds = [(0, 5), (0, 5)]

n = 100

#plot_mincount(f, bounds, n)
#nr, ks = minima_maps(f, bounds, nrange=[(2, 50)])


from numpy import sin, exp, log
bounds2 = [(0, 5), (0, 5)]
from numpy import abs, cos, exp, pi, prod, sin, sqrt, sum
def f2(x):
    from numpy import prod, sin, sqrt
    return prod(sqrt(x) * sin(x))
    #import math
    #return math.sqrt(x[0]) * math.sin(x[0]) * math.sqrt(x[1]) * math.sin(x[1])

#nr, ks = minima_maps(f2, bounds2, nrange=[(2, 100)])
#nr, ks, kc, kshgo = minima_maps(f2, bounds2, nrange=[(15, 1000)], kr=[3], skip=5)
#nr, ks, kc, kshgo = minima_maps(f2, bounds2, nrange=[(3, 100)], kr=[3], skip=3)



####################3
N = 2
def f2(x, *args):
    return x[0] ** 2 + x[1] ** 2 + 25 * (sin(x[0]) ** 2 + sin(x[1]) ** 2)

bounds2 = list(zip([-5.0] * N, [5.0] * N))

nr, ks, kc, kshgo = minima_maps(f2, bounds2, nrange=[(3, 1000)], kr=[], skip=3)




def fev_maps(func, bounds, nrange=[(2, 50)]):
    from numpy import abs, cos, exp, pi, prod, sin, sqrt, sum
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