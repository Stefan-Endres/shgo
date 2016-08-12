#from tgo import *
from _shgo import *
import numpy
n = 23
#n = 26
n = 9
#n = 90
#n = 5000
n = 50
n = 6
#n = 600
bounds = [(0, 4)]
bounds2 = [(0, 4), (0, 4)]
bounds3 = [(0, 2), (0, 2)]
#bounds3 = [(-1, 1), (-1, 1)]
bounds3 = [(-1, 1), (-1, 1)]

import matplotlib.pyplot as plot
#from matplotlib import pyplot as plot
xr = numpy.linspace(0, bounds[0][1] + 2, 1000)
def f(x):
    return (x - 30) * numpy.sin(x)

#def f(x):
#    return x**2

def f2(x):
    return (x[0] - 30) * numpy.sin(x[1])

def f3(x):
    return x[0]**2 + x[1]**2


def f4(x):  #Eggholder
    return (-(x[1] + 47.0)
            * numpy.sin(numpy.sqrt(abs(x[0] / 2.0 + (x[1] + 47.0))))
            - x[0] * numpy.sin(numpy.sqrt(abs(x[0] - (x[1] + 47.0))))
            )

bounds4 = [(-50, 50), (-50, 50)]
bounds4 = [(0, 100), (0, 100)]
bounds4 = [(0, 80), (0, 80)]
bounds4 = [(0, 50), (0, 50)]
bounds4 = [(-25, 25), (-25, 25)]
bounds4 = [(-30, 30), (-30, 30)]

def f6(x):  # Alpine2
    prod = 1
    for i in range(numpy.shape(x)[0]):
        prod = prod * numpy.sqrt(x[i]) * numpy.sin(x[i])

    return prod

bounds6 = [(0, 10), (0, 10)]
bounds6 = [(0, 6), (0, 6)]
bounds6 = [(0, 5), (0, 5)]
#bounds6 = [(0, 5),] * 50
#n = 20000


def f5(x):  # Ackley
    arg1 = -0.2 * numpy.sqrt(0.5 * (x[0] ** 2 + x[1] ** 2))
    arg2 = 0.5 * (numpy.cos(2. * numpy.pi * x[0])
                  + numpy.cos(2. * numpy.pi * x[1]))
    return -20. * numpy.exp(arg1) - numpy.exp(arg2) + 20. + numpy.e

bounds5 = [(2, 5), (2, 5)]



if False:  # bounds bounds4 = [(-30, 30), (-30, 30)]
    x =  ([-30.,  30.])
    xl = ([[-30.        ,  30.        ],  # -30, 30
           [  8.45693338,  15.65091625],   # 8.45, 15.65
           [ 30.        , -13.42401816],   # 30, -13.42
           [ 30.        , -18.8429904 ],   # 30, -18.843
           [-20.0047695 , -30.        ]])  # -20, 30

if False: # Ans for bounds4 = [(0, 60), (0, 60)]
    x =  ([ 49.5072722, 0.       ])
    xl = ([[ 49.5072722 , 0.        ],  # 49, 0
           [ 44.61020505, 0.        ],  # 44.61, 0
           [ 44.6102052 , 0.        ],
           [  8.45693584,  15.65091715],  # 8.46, 15.65
           [  8.45693933,  15.65091923],
           [ 54.34558215,  60.        ]])  # 54.35, 60.0

if False:  # Ans for bounds4 = [(0, 80), (0, 80)]
    x = ([75.18031803, 80.])
    xl = ([[7.51803180e+01, 8.00000000e+01],  # 75; 80
           [4.95072722e+01, 3.54517001e-16],  # 4.95; 0
           [4.46102051e+01, 2.25757165e-18],  # 4.46; 0
           [4.46102050e+01, 0.00000000e+00],
           [4.46102051e+01, 0.00000000e+00],
           [4.46102052e+01, 0.00000000e+00],
           [8.45693587e+00, 1.56509137e+01],  # 8.45; 15.65
           [8.00000000e+01, 3.53775168e+01],  # 8.0; 35.38
           [8.00000000e+01, 0.00000000e+00],  # 8.0; 0
           [0.00000000e+00, 8.00000000e+01]]) # 0; 8


#bounds4 = [(-200, 200), (-200, 200)]
#bounds4 = [(-300, 300), (-300, 300)]
#plot.figure(1)
#plot.plot(xr, f(xr), 'k')



#print(a(f, bounds, n = 5))
#a(f, bounds, n = 30)
#ares = a(f3, bounds3, n=n, iter=None)
#ares = a(f4, bounds4, n=n)#, iter=None)
ares = shgo(f6, bounds6, n=n)#, iter=None)
#from _tgo import tgo
#print(tgo(f4, bounds4, n=n))
#plot.plot(TGOc.C, f(TGOc.C), 'rx')
#plot.show()




#ares = a(f5, bounds5, n=n)#, iter=None)
print(ares)
import matplotlib.pyplot as plt
plt.show()



if False:
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import cm

    x_r = 100
    x_range = numpy.linspace(-1.0, 1.0, x_r)
    y_range = numpy.linspace(-1.0, 1.0, x_r)
    xg, yg = numpy.meshgrid(x_range, y_range)
    func_r = numpy.zeros((x_r, x_r))
    for i in range(xg.shape[0]):
        for j in range(yg.shape[0]):
            X = [xg[i, j], yg[i, j]]  # [x_1, x_2]
            f_out = f3(X)  # Scalar outputs
            func_r[i, j] = numpy.float64(f_out)

    # Plots
    fig = plot.figure()
    ax = fig.gca(projection='3d')
    X, Y = xg, yg

    # Gibbs phase surfaces
    Z = func_r

    if True:

        cset = ax.contourf(X, Y, Z, zdir='z', offset=numpy.nanmin(Z) - 0.05,
                           cmap=cm.coolwarm)
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.3,
                        cmap=cm.coolwarm)
    if False:
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=cm.coolwarm, linewidth=0,
                               antialiased=True, alpha=0.5, shade=True)

        fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$\Delta g$', rotation=90)

    plot.show()




if False:
    # Init tgo class
    # Note: Using ints for irrelevant class inits like func
    #bounds = [(0, 15)]
    #bounds = [(0, 4), (0, 4)]
    TGOc = SHGO(f, bounds)
    TGOc2 = SHGO(f2, bounds2)

    # generate sampling points
    #TGOc.C = numpy.linspace(0, 15, 16)

    #TGOc.n = 15
    TGOc.n = n
    TGOc2.n = n
    TGOc.sampling()
    TGOc2.sampling()

    #plot.plot(TGOc2.C, f(TGOc2.C), 'rx')
    plot.plot(TGOc.C, f(TGOc.C), 'rx')


    # initiate global storage containers for all minima

    TGOc.sorted_samples()
    TGOc2.sorted_samples()
    TGOc.ax_subspace()
    TGOc2.ax_subspace()

    TGOc.fun_ref()
    TGOc.surface_topo_ref()
    TGOc2.fun_ref()
    TGOc2.surface_topo_ref()

#x_l = 1.534
#TGOc2.ax_basin(x_l)

if False:
    # print('TGc2.Ft = {}'.format(TGOc2.Ft))
    # print('TGc2.Ftm = {}'.format(TGOc2.Ftm))
    # print('TGc2.Ftp = {}'.format(TGOc2.Ftp))

    print('TGOc.Ft = {}'.format(TGOc.Ft))
    print('TGOc.Ftp = {}'.format(TGOc.Ftp))
    print('TGOc.Ftm = {}'.format(TGOc.Ftm))

    # print('len(TGOc.Ft) = {}'.format(len(TGOc.Ft)))
    # print('len(TGOc.Ftp) = {}'.format(len(TGOc.Ftp)))
    print('TGc.I = {}'.format(TGOc.I))
    print('TGc.Ii[0] = {}'.format(TGOc.Ii[0]))
    #print('TGOc.sample_topo(18) = {}'.format(TGOc2.sample_topo(32)))
    #print('TGOc.sample_topo(18) = {}'.format(TGOc.sample_topo(11)))
    print('='*100)
    # print('TGOc.Ftm + 1e-1 * TGOc.Ftp  = {}'.format(TGOc.Ftm = TGOc.Ftp ))
    # print('TGc.Ftp = {}'.format(TGOc.Ftp))


    for ind in range(TGOc.n):  # ANS index 4
        #print('='*30)
        Min_bool = TGOc.sample_topo(ind)
        print('TGOc.C[{}] = {}, {}'.format(ind, TGOc.C[ind], Min_bool))
        #print('TGOc2.C[{}] = {}, {}'.format(ind, TGOc2.C[ind], Min_bool))
        #TGOc2.sample_topo(ind)



if False:
    #print(TGOc2.C[:, 0])
    print('TGOc2.C = {}'.format(TGOc2.C))
    I = numpy.argsort(TGOc2.C, axis=0)
    S = numpy.sort(TGOc2.C, axis=0)
    print('I = {}'.format(I))
    print('S = {}'.format(S))
    # S = TGOc2.C[I]  #TODO: I mask not working?

if False:
    C1 = TGOc2.C[:, 0]
    S1 = C1[I[:, 0]]
    print(TGOc2.C[:, 0])
    print(I[:, 0])
    print(TGOc2.C[:, 0][I[:, 0]])
    #print('S = {}'.format(S))
    print('S1 = {}'.format(S1))
    print('len(S1) = {}'.format(numpy.size(S1)))
    cut_pos = numpy.searchsorted(S1, 10.0)
    print('cut S1 at 10.0 = {}'.format(cut_pos))
    S1_n = S1[:cut_pos]
    S1_p = S1[cut_pos:]
    print('S1_p = {}'.format(S1_p))
    print('S1_n = {}'.format(S1_n))
    I1 = I[:, 0]
    print('I1 = {}'.format(I1))
    I1_n = I1[:cut_pos]
    I1_p = I1[cut_pos:]
    print('I1_p = {}'.format(I1_p))
    print('I1_n = {}'.format(I1_n))
    print('Now use reference table to fit function values, the cut the data again')
    #print('I[:, 0] = {}'.format(I[:, 0]))

    print('F = {}'.format(F))
    print('numpy.argmin(F) = {}'.format(numpy.argmin(F)))
    print('F[numpy.argmin(F)] = {}'.format(F[numpy.argmin(F)]))


    # Construct F*:
    Ft = F[I]
    print('Ft = F[I] = {}'.format(Ft))
    minFt = numpy.diff(Ft, axis=0)
    print('minFt = {}'.format(minFt))
    posFt = numpy.diff(Ft[::-1], axis=0)[::-1]
    print('posFt = {}'.format(posFt))

    #print('S = {}'.format(S))

    # Find minimum
    # ---> 1.53

    cut_pos = numpy.searchsorted(S1, 1.53)
    print('cut S1 at 10.0 = {}'.format(cut_pos))
    S1_n = S1[:cut_pos]
    S1_p = S1[cut_pos:]
    print('S1_p = {}'.format(S1_p))
    print('S1_n = {}'.format(S1_n))
    I1 = I[:, 0]
    print('I1 = {}'.format(I1))
    I1_n = I1[:cut_pos]
    I1_p = I1[cut_pos:]

if False:
    list = [0.0, 0.46875, 0.703125, 0.9375, 1.40625, 1.875]
    print('list = {}'.format(list))
    print(numpy.searchsorted(list, 0.5))
    print(numpy.searchsorted(list, 1.0))



#plot.show()









