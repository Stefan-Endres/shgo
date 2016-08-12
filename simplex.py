from _shgo import *

bounds = [(-1, 1), (-1, 1)]
n = 10


# ares = a(f, bounds, n=n)

# print(ares)


def f1(x):
    return x ** 2


def simplex1d(f, V0):
    import scipy.optimize
    import numpy
    res = scipy.optimize.OptimizeResult()
    V = numpy.array(V0, dtype=float)

    res.nfev = 1
    F = f(V)
    # Find vertex with lowest objective function value
    mask = numpy.argsort(F.T)
    V_sorted = V[mask]
    F_sorted = F[mask]

    # Find finite differences
    V = V_sorted[0]
    F = F_sorted[0]
    FD1 = numpy.abs(F[0] - F[1]) / numpy.abs(V[0, :] - V[1, :])
    FD2 = numpy.abs(F[0] - F[2]) / numpy.abs(V[0, :] - V[2, :])
    print(V)
    print(FD1)
    print(FD2)
    if FD1 <= FD2:
        V_new = (V[0, :] + V[1, :]) / 2.0
    else:
        V_new = (V[0, :] + V[2, :]) / 2.0

    print(V_new)
    F_new = f(V_new)
    print(F_new)

    print(V[1:, :])
    V[1:, :] = V[:-1, :]
    V[0, :] = V_new
    # V = V + 0.5
    print(V)
    F[1:] = F[:-1]
    F[0] = F_new
    print(F)
    F_old = F_new

    epsilon = 1e50
    # Start iterating
    while abs(epsilon) >= 1e-9:
        FD1 = numpy.abs(F[0] - F[1]) / numpy.abs(V[0, :] - V[1, :])
        FD2 = numpy.abs(F[0] - F[2]) / numpy.abs(V[0, :] - V[2, :])
        if FD1 <= FD2:
            V_new = (V[0, :] + V[1, :]) / 2.0
        else:
            V_new = (V[0, :] + V[2, :]) / 2.0

        F_new = f(V_new)
        V[1:, :] = V[:-1, :]
        V[0, :] = V_new
        # V = V + 0.5
        print(V)
        F[1:] = F[:-1]
        F[0] = F_new
        print(F)
        # F_old = F_new


        epsilon = F_new - F_old
        F_old = F_new

        res.nfev += 1

    # FD =

    print('epsilon = {}'.format(epsilon))
    print('F final = {}'.format(F[0]))
    print('V final = {}'.format(V[0, :]))
    print('res.nfev = {}'.format(res.nfev))
    return None


V0 = [[-5], [-2], [3]]
V0 = [[-500], [-20], [30]]


# simplex1d(f1, V0)

def simplex(f, V0, args=()):
    """
    A local minimization routine [TODO with cubic assumptions]

    Requires a starting simplex with 3 vertexes

    Parameters
    ----------
    V : Ordered list of starting vertexes
        ex. a simplex in 4 dimensional space: [[0.1, 0.2, 0.3, 0.1],
                                               [0.5, 0.6, 0.8, 0.19],
                                               [0.3, 1.0, 0.4, 0.2]
                                               ]

    Returns
    -------

    """
    import scipy.optimize
    import numpy
    res = scipy.optimize.OptimizeResult()

    V = numpy.array(V0, dtype=float)
    dim = numpy.shape(V)[1]

    res.nfev = 1
    F = f(V.T)

    # Find vertex with lowest objective function value
    mask = numpy.argsort(F.T)
    print(mask)
    V_sorted = V[mask, :]
    F_sorted = F[mask]

    # Find finite differences
    V = V_sorted
    F = F_sorted

    FD = numpy.abs(F[0] - F[1:])
    print(FD)
    VD = numpy.abs(V[0, :] - V[1:, :])
    # print(V)
    VDi = numpy.reciprocal(VD)  # (1/VD)
    print(VDi)
    FDD = numpy.multiply(VDi.T, FD).T
    print(FDD)
    mask2 = numpy.argmin(FDD, axis=0)
    mask2 = 1 + mask2
    V_new = []
    for d in range(dim):
        V_new.append((V[0, d] + V[mask2[d], d]) / 2)

    V_new = numpy.array(V_new)

    print('V_new = {}'.format(V_new))
    print('V = {}'.format(V))

    F_new = f(V_new)
    print('F_new  = {}'.format(F_new))
    print('F  = {}'.format(F))
    # print(F_new)

    ### SHIT BELOW IS UNTESTED

    # print(V[1:, :])
    V[1:, :] = V[:-1, :]
    V[0, :] = V_new
    # V = V + 0.5
    # print(V)
    F[1:] = F[:-1]
    F[0] = F_new
    # print(F)
    F_old = F_new

    epsilon = 1e50
    while abs(epsilon) >= 1e-9:
        FD = numpy.abs(F[0] - F[1:])
        VD = numpy.abs(V[0, :] - V[1:, :])
        VDi = numpy.reciprocal(VD)  # (1/VD)
        FDD = numpy.multiply(VDi.T, FD).T
        mask2 = numpy.argmin(FDD, axis=0)
        mask2 = 1 + mask2
        V_new = []
        for d in range(dim):
            V_new.append((V[0, d] + V[mask2[d], d]) / 2)

        V_new = numpy.array(V_new)

        F_new = f(V_new)
        V[1:, :] = V[:-1, :]
        V[0, :] = V_new
        # V = V + 0.5
        print(V)
        F[1:] = F[:-1]
        F[0] = F_new
        print(F)
        # F_old = F_new


        epsilon = F_new - F_old
        F_old = F_new

        res.nfev += 1

    print('epsilon = {}'.format(epsilon))
    print('F final = {}'.format(F[0]))
    print('V final = {}'.format(V[0, :]))
    print('res.nfev = {}'.format(res.nfev))

    if False:

        epsilon = 1e50
        # Start iterating
        while abs(epsilon) >= 1e-9:
            FD1 = numpy.abs(F[0] - F[1]) / numpy.abs(V[0, :] - V[1, :])
            FD2 = numpy.abs(F[0] - F[2]) / numpy.abs(V[0, :] - V[2, :])
            if FD1 <= FD2:
                V_new = (V[0, :] + V[1, :]) / 2.0
            else:
                V_new = (V[0, :] + V[2, :]) / 2.0

            F_new = f(V_new)
            V[1:, :] = V[:-1, :]
            V[0, :] = V_new
            # V = V + 0.5
            print(V)
            F[1:] = F[:-1]
            F[0] = F_new
            print(F)
            # F_old = F_new


            epsilon = F_new - F_old
            F_old = F_new

            res.nfev += 1

        # FD =

        print('epsilon = {}'.format(epsilon))
        print('F final = {}'.format(F[0]))
        print('V final = {}'.format(V[0, :]))
        print('res.nfev = {}'.format(res.nfev))
    return None


def f(x):
    return x[0] ** 2 + x[1] ** 2


V0 = [[0.1, 0.2],
      [-0.1, -0.3],
      [0.0, 0.3],
      [-1.0, 5.0]]

simplex(f, V0)


def f3(x):
    return x[0] ** 2 + x[1] ** 2 + x[2] ** 2  # + x[2] * x[1] + 5 * x[0]


V03 = [[0.1, 0.2, 0.2],
       [-0.1, -0.3, -0.3],
       [0.0, 0.3, 0.3],
       [-1.0, 5.0, 5.0]]

simplex(f3, V03)
test = True * 3
