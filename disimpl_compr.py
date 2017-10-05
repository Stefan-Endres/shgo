from _shgo import *
import numpy

def run_test(test, args=(), g_args=(), test_atol=1e-5, n=100, iter=None,
             callback=None, minimizer_kwargs=None, options=None,
              sampling_method='sobol'):

    res = shgo(test.f, test.bounds, args=args, g_cons=test.g,
               g_args=g_args, n=n, iters=iter, callback=callback,
               minimizer_kwargs=minimizer_kwargs, options=options,
               sampling_method=sampling_method)

if __name__ == '__main__':
    run_test