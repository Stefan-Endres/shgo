from _shgo import *
from numpy import *
if __name__ == '__main__':
    '''
    Temporary dev work:
    '''
    if 0:  # "Churchill's problem"
        Voc = 32.9
        Isc = 8.21
        Vmp = 26.3
        Imp = 7.61

        Ns = 54
        q = 1.602e-19
        k = 1.381e-23
        T = 298  # K
        Vt = k * T / q


        def f(variables):
            (Rs, Rsh, a) = variables
            x = (Imp / Vmp) - (1 / (a * Vt)) * (1 - Rs * (Imp / Vmp)) * ((-Voc + (Rs + Rsh) * Isc) / Rsh) * numpy.exp(
                (Vmp - Voc + Rs * Imp) / (a * Vt)) - (1 / Rsh) * (1 - Rs * (Imp / Vmp))
            y = -Imp * (1 + (Rs / Rsh)) + ((-Voc + (Rs + Rsh) * Isc) / Rsh) * (
            1 - numpy.exp((Vmp - Voc + Rs * Imp) / (a * Vt))) + (Voc - Vmp) / Rsh
            z = (-Rs / Rsh) + (Rsh - Rs) / (a * Vt) + ((-Voc + (Rs + Rsh) * Isc) / Rsh) * numpy.exp(
                (Rs * Isc - Voc) / (a * Vt))
            #return x, y, z
            return numpy.sum([numpy.abs(x), numpy.abs(y), numpy.abs(z)])
            #return numpy.sum([x**2, y**2, z**2])

        bounds = [(1e-5, 100), (1e-5, 10000), (1e-5, 10000)]
        bounds = [(1e-3, 2), (1e-1, 10000), (1e-1, 1000)]

        if 0:
            print(shgo(f, bounds))
        if 1:
            from _tgo import tgo
            print(tgo(f, bounds, n=5000))
        if 0:
            scipy.optimize.fsolve(f, [0.217 * 1e-1, 952* 1e-1, 76* 1e-1])

    # Eggcrate
    if 1:
        N = 2
        def fun(x, *args):
            return x[0] ** 2 + x[1] ** 2 + 25 * (sin(x[0]) ** 2 + sin(x[1]) ** 2)

        def g_cons1(x):
            return x[0] # -x[0] ** 2 - x[1] ** 2 + 2.5

        def g_cons2(x):
            return x[0] # -x[0] ** 2 - x[1] ** 2 + 2.5

        g_cons = (g_cons1, g_cons2)
        bounds = list(zip([-5.0] * N, [5.0] * N))

        if 0:
            options = {'disp': True}
            SHGOc3 = SHGO(fun, bounds, options=options)
            SHGOc3.construct_complex_simplicial()
            SHGOc3.simplex_minimizers()
            SHGOc3.minimise_pool()
            SHGOc3.sort_result()
            print('SHGOc3.res = {}'.format(SHGOc3.res))

        #print(shgo(fun, bounds, g_cons=g_cons))
        #print(shgo(fun, bounds, g_cons=g_cons, iter=4))
        print(shgo(fun, bounds, iters=4))
        #print(shgo(fun, bounds))#, g_cons=g_cons))


        #SHGOc3.HC.plot_complex()


    # Apline2
    if 0:

        def f(x):  # Alpine2
            prod = 1
            for i in range(numpy.shape(x)[0]):
                prod = prod * numpy.sqrt(x[i]) * numpy.sin(x[i])

            return prod

        bounds = [(0, 10), (0, 10)]
        bounds = [(0, 10), (0, 10)]
        #bounds = [(0, 5), (0, 5)]
        bounds = [(0, 1), (0, 1)]
        bounds = [(3, 4), (3, 4)]
        bounds = [(2, 4), (2, 4)]
        bounds = [(0, 10), (0, 10)]
        #bounds = [(1, 6), (1, 6)]

        SHGOc1 = SHGO(f, bounds)
        SHGOc1.construct_complex_simplicial()
        print(shgo(f, bounds))
        SHGOc1.HC.plot_complex()


    if 0:
        N = 2

        def fun(x):  # Damavand
            import numpy
            try:
                num = sin(pi * (x[0] - 2.0)) * sin(pi * (x[1] - 2.0))
                den = (pi ** 2) * (x[0] - 2.0) * (x[1] - 2.0)
                factor1 = 1.0 - (abs(num / den)) ** 5.0
                factor2 = 2 + (x[0] - 7.0) ** 2.0 + 2 * (x[1] - 7.0) ** 2.0
                return factor1 * factor2
            except ZeroDivisionError:
                return numpy.nan


        bounds = list(zip([0.0] * N, [14.0] * N))

        SHGOc2 = SHGO(fun, bounds)
        SHGOc2.construct_complex_simplicial()
        print(shgo(fun, bounds))


    if 0:
        def f(x):  # sin
            return numpy.sin(x)
        bounds = [(0, 5)]

        SHGOc2 = SHGO(f, bounds)
        SHGOc2.construct_complex_simplicial()
        print(shgo(f, bounds))

    #print(SHGOc1.disp)


    # Ursem01
    if 0:
        def f(x):
            return -numpy.sin(2 * x[0] - 0.5 * math.pi) - 3 * numpy.cos(x[0]) - 0.5 * x[0]

        def f2(x):
            return x[0]**2 + x[1]**2

        bounds = [(-1, 1), (-1, 2)]
        bounds = [(0, 9), (-2.5, 2.5)]
        bounds = [(0, 9), (-18, 2.5)]
        bounds = [(0, 1), (0, 1)]
        #bounds = [(-15, 1), (-1, 1)]

        #SHGOc2 = SHGO(f, bounds)
        #SHGOc2.construct_complex_simplicial()
        print(shgo(f, bounds))
        #SHGOc2.HC.plot_complex()

    if 0:
        print(shgo(f, bounds, iters=1,
                   sampling_method='simplicial'))
        print('='*100)
        print('Sobol shgo:')
        print('===========')
        print(shgo(f, bounds))