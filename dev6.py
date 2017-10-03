from _shgo import *
from numpy import *

if __name__ == '__main__':
    if 1: # functions
        N = 2
        def fun(x, *args): # Eggcrate
            return x[0] ** 2 + x[1] ** 2 + 25 * (sin(x[0]) ** 2 + sin(x[1]) ** 2)

        def g_cons1(x):
            return x[0]  # -x[0] ** 2 - x[1] ** 2 + 2.5

        def g_cons2(x):
            return x[0]  # -x[0] ** 2 - x[1] ** 2 + 2.5

        g_cons = (g_cons1, g_cons2)
        bounds = list(zip([-5.0] * N, [5.0] * N))

    #print(shgo(fun, bounds, iter=3, g_cons=g_cons))

    #SHc = SHGOh(fun, bounds)

    if 1:
        #SHc = SHGOh(fun, bounds, g_cons, iter=3)
        #SHc.construct_complex_iteratively()
        #SHc.construct_complex_simplicial()
        #SHc.iterate()


        #print(SHc.simplex_minimizers())
        #SHc.minimise_pool()
        #SHc.sort_result()
        #print(f'SHc.res = {SHc.res}')

        #res = shgo(fun, bounds, g_cons, iter=3)
        res = shgo(fun, bounds, g_cons, n=15)
        print(res)

    if 0:
        options = {'disp': True}
        SHGOc3 = SHGO(fun, bounds, options=options)
        SHGOc3.construct_complex_simplicial()
        SHGOc3.simplex_minimizers()
        SHGOc3.minimise_pool()
        SHGOc3.sort_result()
        print('SHGOc3.res = {}'.format(SHGOc3.res))