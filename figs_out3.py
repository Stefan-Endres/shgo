from _shgo_sobol import *


def f(x):  # s231
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

def g1(x):
    return x[0]/3 + x[1] + 0.1

def g2(x):
    return -1*x[0]/3 + x[1] + 0.1

g = (g1,g2)

bounds = [(-2, 2), ] * 2
n = 10

SHc = SHGO(f, bounds, n = n, g_cons=g)
SHc.construct_complex_sobol()
# Minimise the pool of minisers with local minimisation methods
SHc.minimise_pool()
# Sort results and build the global return object
SHc.sort_result()

import matplotlib.pyplot as plt
def build_contour(SHc, surface=True, contour=True, both=True):
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

    if surface and not both:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0,
                        antialiased=True, alpha=1.0, shade=True)
        if False:
            cset = ax.contour(X, Y, Z, zdir='z', offset=-100, cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='x', offset=-40, cmap=cm.coolwarm)
            cset = ax.contour(X, Y, Z, zdir='y', offset=40, cmap=cm.coolwarm)

        ax.set_xlabel('$x_1$', fontsize=14)
        ax.set_ylabel('$x_2$', fontsize=14)
        ax.set_zlabel('$f$', fontsize=14)

    if contour and not both:
        plt.figure()
        cs = plt.contour(xg, yg, Z, cmap='binary_r', color='k')
        plt.clabel(cs)

    if both:
        plt.figure()
        # Surface
        #fig = plt.subplot(1, 2, 1)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        #ax = fig.gca(projection='3d')
        ax.plot_surface(xg, yg, Z, rstride=1, cstride=1,
                        cmap=cm.coolwarm, linewidth=0,
                        antialiased=True, alpha=1.0, shade=True)

        # Contour
        #plt.subplot(1, 2, 2)
       # plt.figure()
        ax = fig.add_subplot(1, 2, 2)
        cs = plt.contour(xg, yg, Z, cmap='binary_r', color='k')
        plt.clabel(cs)


build_contour(SHc, contour=False, both=False)


if 0:
    # Plot
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(tspan, sstore[:, m_A], label='$m_A$')
    plt.ylabel('$m_A$ / kg')

    plt.subplot(3, 1, 2)
    plt.plot(tspan, sstore[:, x_A], label='$x_A$')
    plt.plot(tspan, x_A_Dstore, label='$x_A$ delayed')
    plt.legend()
    plt.ylabel('$x_A$ / ppm')

    plt.subplot(3, 1, 3)
    plt.plot(tspan, P_B_instore, label='$m_A$')
    plt.axhline(100, color='red', linestyle='--')
    plt.axhline(20.0, color='red', linestyle='--')
    plt.ylim(10, 110)
    plt.ylabel('$P_{B, in}$ / Pa')

    plt.xlabel('time / min')
    plt.show()


plt.show()