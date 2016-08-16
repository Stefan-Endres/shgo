#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" execfile('tgo.py')
"""
from __future__ import division, print_function, absolute_import
import numpy
import scipy.spatial
import scipy.optimize
import logging
#from . import __file__

try:
    pass
    #from multiprocessing_on_dill import Pool
except ImportError:
    from multiprocessing import Pool

def shgo(func, bounds, args=(), g_cons=None, g_args=(), n=30, iter=None,
         callback=None, minimizer_kwargs=None, options=None,
         multiproc=False):  #TODO: Update documentation
    """
    Finds the global minima of a function using topograhphical global
    optimisation.

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.

    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
        Use ``None`` for one of min or max when there is no bound in that
        direction. By default bounds are ``(None, None)``.

    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.

    g_cons : sequence of callable functions, optional
        Function(s) used to define a limited subset to defining the feasible
        set of solutions in R^n in the form g(x) <= 0 applied as g : R^n -> R^m

        NOTE: If the ``constraints`` sequence used in the local optimization
              problem is not defined in ``minimizer_kwargs`` and a constrained
              method is used then the ``g_cons`` will be used.
              (Defining a ``constraints`` sequence in ``minimizer_kwargs``
               means that ``g_cons`` will not be added so if equality
               constraints and so forth need to be added then the inequality
               functions in ``g_cons`` need to be added to ``minimizer_kwargs``
               too).

    g_args : sequence of tuples, optional
        Any additional fixed parameters needed to completely specify the
        feasible set functions ``g_cons``.
        ex. g_cons = (f1(x, *args1), f2(x, *args2))
        then
            g_args = (args1, args2)

    n : int, optional
        Number of sampling points used in the construction of the topography
        matrix.

    k_t : int, optional
        Defines the number of columns constructed in the k-t matrix. The higher
        k is the lower the amount of minimisers will be used for local search
        routines. If None the empirical model of Henderson et. al. (2015) will
        be used. (Note: Lower ``k_t`` values increase the number of local
        minimisations that need need to be performed, but could potentially be
        more robust depending on the local solver used due to testing more
        local minimisers on the function hypersuface)

    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the minimizer
        ``scipy.optimize.minimize`` Some important options could be:

            method : str
                The minimization method (e.g. ``SLSQP``)
            args : tuple
                Extra arguments passed to the objective function (``func``) and
                its derivatives (Jacobian, Hessian).

            options : {ftol: 1e-12}

    callback : callable, optional
        Called after each iteration, as ``callback(xk)``, where ``xk`` is the
        current parameter vector.

    options : dict, optional
        A dictionary of solver options. All methods in scipy.optimize.minimize
        accept the following generic options:

            maxiter : int
                Maximum number of iter to perform.
            disp : bool
                Set to True to print convergence messages.

        The following options are also used in the global routine:

            maxfev : int
                Maximum number of iter to perform in local solvers.
                (Note only methods that support this option will terminate
                tgo at the exact specified value)

    multiproc : boolean, optional
        If True the local minimizations of the minimizer points will be pooled
        and processed in parallel using the multiprocessing module. This could
        significantly speed up slow optimizations.

    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are:
        ``x`` the solution array corresponding to the global minimum,
        ``fun`` the function output at the global solution,
        ``xl`` an ordered list of local minima solutions,
        ``funl`` the function output at the corresponding local solutions,
        ``success`` a Boolean flag indicating if the optimizer exited
        successfully and
        ``message`` which describes the cause of the termination,
        ``nfev`` the total number of objective function evaluations including
        the sampling calls.
        ``nlfev`` the total number of objective function evaluations
        culminating from all local search optimisations.

    Notes
    -----
    Global optimization using the Topographical Global Optimization (TGO)
    method first proposed by Törn (1990) [1] with the the semi-empirical
    correlation by Hendorson et. al. (2015) [2] for k integer defining the
    k-t matrix.

    The TGO is a clustering method that uses graph theory to generate good
    starting points for local search methods from points distributed uniformly
    in the interior of the feasible set. These points are generated using the
    Sobol (1967) [3] sequence.

    The local search method may be specified using the ``minimizer_kwargs``
    parameter which is inputted to ``scipy.optimize.minimize``. By default
    the ``SLSQP`` method is used. In general it is recommended to use the
    ``SLSQP`` or ``COBYLA`` local minimization if inequality constraints
    are defined for the problem since the other methods do not use constraints.

    Performance can sometimes be improved by either increasing or decreasing
    the amount of sampling points ``n`` depending on the system. Increasing the
    amount of sampling points can lead to a lower amount of minimisers found
    which requires fewer local optimisations. Forcing a low ``k_t`` value will
    nearly always increase the amount of function evaluations that need to be
    performed, but could lead to increased robustness.

    The primitive polynomials and various sets of initial direction numbers for
    generating Sobol sequences is provided by [4] by Frances Kuo and
    Stephen Joe. The original program sobol.cc is available and described at
    http://web.maths.unsw.edu.au/~fkuo/sobol/ translated to Python 3 by
    Carl Sandrock 2016-03-31

    Examples
    --------
    First consider the problem of minimizing the Rosenbrock function. This
    function is implemented in `rosen` in `scipy.optimize`

    >>> from scipy.optimize import rosen, shgo
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = shgo(rosen, bounds)
    >>> result.x, result.fun
    (array([ 1.,  1.,  1.,  1.,  1.]), 2.9203923741900809e-18)

    Note that bounds determine the dimensionality of the objective
    function and is therefore a required input, however you can specify
    empty bounds using ``None`` or objects like numpy.inf which will be
    converted to large float numbers.

    >>> bounds = [(None, None), (None, None), (None, None), (None, None)]
    >>> result = shgo(rosen, bounds)
    >>> result.x
    array([ 0.99999851,  0.99999704,  0.99999411,  0.9999882 ])

    Next we consider the Eggholder function, a problem with several local
    minima and one global minimum.
    (https://en.wikipedia.org/wiki/Test_functions_for_optimization)

    >>> from scipy.optimize import shgo
    >>> import numpy as np
    >>> def eggholder(x):
    ...     return (-(x[1] + 47.0)
    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    ...             )
    ...
    >>> bounds = [(-512, 512), (-512, 512)]
    >>> result = shgo(eggholder, bounds)
    >>> result.x, result.fun
    (array([ 512.        ,  404.23180542]), -959.64066272085051)

    ``tgo`` also has a return for any other local minima that was found, these
     can be called using:

    >>> result.xl, result.funl
    (array([[ 512.        ,  404.23180542],
           [-456.88574619, -382.6233161 ],
           [ 283.07593402, -487.12566542],
           [ 324.99187533,  216.0475439 ],
           [-105.87688985,  423.15324143],
           [-242.97923629,  274.38032063],
           [-414.8157022 ,   98.73012628],
           [ 150.2320956 ,  301.31377513],
           [  91.00922754, -391.28375925],
           [ 361.66626134, -106.96489228]]),
           array([-959.64066272, -786.52599408, -718.16745962, -582.30628005,
           -565.99778097, -559.78685655, -557.85777903, -493.9605115 ,
           -426.48799655, -419.31194957]))

    Now suppose we want to find a larger amount of local minima, this can be
    accomplished for example by increasing the amount of sampling points...

    >>> result_2 = shgo(eggholder, bounds, n=1000)
    >>> len(result.xl), len(result_2.xl)
    (10, 60)

    To demonstrate solving problems with non-linear constraints consider the
    following example from [5] (Hock and Schittkowski problem 18):

    Minimize: f = 0.01 * (x_1)**2 + (x_2)**2

    Subject to: x_1 * x_2 - 25.0 >= 0,
                (x_1)**2 + (x_2)**2 - 25.0 >= 0,
                2 <= x_1 <= 50,
                0 <= x_2 <= 50.

    Approx. Answer:
        f([(250)**0.5 , (2.5)**0.5]) = 5.0

    >>> from scipy.optimize import shgo
    >>> def f(x):
    ...     return 0.01 * (x[0])**2 + (x[1])**2
    ...
    >>> def g1(x):
    ...     return x[0] * x[1] - 25.0
    ...
    >>> def g2(x):
    ...     return x[0]**2 + x[1]**2 - 25.0
    ...
    >>> g = (g1, g2)
    >>> bounds = [(2, 50), (0, 50)]
    >>> result = shgo(f, bounds, g_cons=g)
    >>> result.x, result.fun
    (array([ 15.81138847,   1.58113881]), 4.9999999999996252)


    References
    ----------
    .. [1] Törn, A (1990) "Topographical global optimization", Reports on
           Computer Science and Mathematics Ser. A, No 199, 8p. Abo Akademi
           University, Sweden
    .. [2] Henderson, N, de Sá Rêgo, M, Sacco, WF, Rodrigues, RA Jr. (2015) "A
           new look at the topographical global optimization method and its
           application to the phase stability analysis of mixtures",
           Chemical Engineering Science, 127, 151-174
    .. [3] Sobol, IM (1967) "The distribution of points in a cube and the
           approximate evaluation of integrals. USSR Comput. Math. Math. Phys.
           7, 86-112.
    .. [4] S. Joe and F. Y. Kuo (2008) "Constructing Sobol sequences with
           better  two-dimensional projections", SIAM J. Sci. Comput. 30,
           2635-2654
    .. [5] Hoch, W and Schittkowski, K (1981) "Test examples for nonlinear
           programming codes." Lecture Notes in Economics and mathematical
           Systems, 187. Springer-Verlag, New York.
           http://www.ai7.uni-bayreuth.de/test_problem_coll.pdf
    """
    # Initiate TGO class
    SHc= SHGO(func, bounds, args=args, g_cons=g_cons, g_args=g_args, n=n,
              iter=iter, callback=callback, minimizer_kwargs=minimizer_kwargs,
              options=options, multiproc=multiproc)

    # Generate sampling points
    sample = True
    if SHc.disp:
        print('Generating sampling points')

    while sample:
        SHc.sampling()

        # Find subspace of feasible points
        if g_cons is not None:
            SHc.sampling_subspace()
        else:
            SHc.fn = SHc.n

        # Sort remaining samples
        SHc.sorted_samples()

        # Find objective function references
        SHc.fun_ref()

        # initiate global storage containers for all minima
        SHc.x_min_glob = []
        SHc.fun_min_glob = []

        # Find minimiser pool
        # DIMENSIONS self.dim
        if SHc.dim < 2:  # Scalar objective functions
            if SHc.disp:
                print('Constructing 1D minimizer pool')

            SHc.ax_subspace()
            SHc.surface_topo_ref()
            SHc.X_min = SHc.minimizers()

        else:  # Multivariate functions.
            if SHc.disp:
                print('Constructing Gabrial graph and minimizer pool')

            SHc.delaunay_triangulation()
            if SHc.disp:
                print('Triangulation completed, building minimizer pool')

            SHc.X_min = SHc.delaunay_minimizers()

        logging.info("Minimiser pool = SHGO.X_min = {}".format(SHc.X_min))

        #TODO: Keep sampling until n feasible points in non-linear constraints
        # self.fn < self.n ---> self.n - self.fn
        if len(SHc.minimizer_pool) == 0:
            if SHc.disp:
                print('No minimizers found. Increasing sampling space.')
            n_add = 100
            if options is not None:
                if 'maxiter' in options.keys():
                    n_add = int((options['maxiter'] - SHc.fn) / 1.618)
                    if n_add < 1:
                        SHc.res.message = ("Failed to find a minimizer "
                                            "within the maximum allowed "
                                            "function evaluations.")

                        if SHc.disp:
                            print(SHc.res.message + " Breaking routine...")

                        SHc.break_routine = True
                        SHc.res.success = False
                        sample = False

            SHc.n += n_add
            SHc.res.nfev = SHc.fn

        else:  # If good values are found stop while loop
            # Include each sampling point as func evaluation:
            SHc.res.nfev = SHc.fn
            sample = False

    if not SHc.break_routine:
        if SHc.disp:
            print("Succesfully completed construction of complex.")

        # Find first local minimum
        # NOTE: Since we always minimize this value regardless it is a waste to
        # build the topograph first before minimizing
        lres_f_min = SHc.minimize(SHc.X_min[[0]])

        # Trim minimised point from current minimiser set
        SHc.trim_min_pool(0)

        # Minimise the global
        while not SHc.stopiter:
            if SHc.iter is not None:  # Note first iteration is outside loop
                logging.info('SHGO.iter = {}'.format(SHc.iter))
                SHc.iter -= 1
                if __name__ == '__main__':
                    if SHc.iter == 0:
                        SHc.stopiter = True
                        break
                    #TODO: Test usage of iterative features

            if numpy.shape(SHc.X_min)[0] == 0:
                SHc.stopiter = True
                break

            # NOTE: This routine processes the furthest minimizers and is only
            # useful for runs with a maximum amount of function evaluations to
            # find the best minimimizers. #TODO: Note useful for SHGO?

            # Construct topograph from current minimiser set
            SHc.g_topograph(lres_f_min.x, SHc.X_min)

            # Find local minimum at the miniser with the greatest euclidean
            # distance from the current solution
            ind_xmin_l = SHc.Z[:, -1]
            lres_f_min = SHc.minimize(SHc.Ss[:, -1])

            # Trim minimised point from current minimiser set
            SHc.trim_min_pool(ind_xmin_l)


        # Sort results and save
        SHc.x_min_glob = numpy.array(SHc.x_min_glob)
        SHc.fun_min_glob = numpy.array(SHc.fun_min_glob)

        # Sort and save
        # Sorted indexes in Func_min
        ind_sorted = numpy.argsort(SHc.fun_min_glob)

        # Save ordered list of minima
        SHc.res.xl = SHc.x_min_glob[ind_sorted]  # Ordered x vals #TODO: Check
        SHc.fun_min_glob = numpy.array(SHc.fun_min_glob)
        SHc.res.funl = SHc.fun_min_glob[ind_sorted]
        SHc.res.funl = SHc.res.funl.T

        # Find global of all minimisers
        SHc.res.x = SHc.x_min_glob[ind_sorted[0]]  # Save global minima
        SHc.res.fun = SHc.fun_min_glob[ind_sorted[0]] # Save global fun value

    # Confirm the routine ran succesfully
    if not SHc.break_routine:
        SHc.res.message = 'Optimization terminated successfully.'
        SHc.res.success = True

    # Add local func evals to sampling func evals
    SHc.res.nfev += SHc.res.nlfev

    return SHc.res


# %% Define tgo class
class SHGO(object):
    """
    This class implements the shgo routine
    """

    def __init__(self, func, bounds, args=(), g_cons=None, g_args=(), n=100,
                 iter=None, callback=None, minimizer_kwargs=None,
                 options=None, multiproc=False):

        self.func = func
        self.bounds = bounds
        self.args = args
        if type(g_cons) is not tuple and type(g_cons) is not list:
            self.g_func = (g_cons,)
        else:
            self.g_func = g_cons

        self.g_args = g_args
        self.n = n
        self.n_sampled = 0  # To track sampling points already evaluated
        self.fn = n  # Number of feasible samples remaining
        self.iter = iter

        self.callback = callback
        self.maxfev = None
        self.disp = False
        if options is not None:
            if 'maxfev' in options:
                self.maxfev = options['maxfev']
            if 'disp' in options:
                self.disp = options['disp']

        # set bounds
        abound = numpy.array(bounds, float)
        self.dim = numpy.shape(abound)[0]  # Dimensionality of problem
        # Check if bounds are correctly specified
        bnderr = numpy.where(abound[:, 0] > abound[:, 1])[0]
        # Set none finite values to large floats
        infind = ~numpy.isfinite(abound)
        abound[infind[:, 0], 0] = -1e50  # e308
        abound[infind[:, 1], 1] = 1e50  # e308
        if bnderr.any():
            raise ValueError('Error: lb > ub in bounds %s.' %
                             ', '.join(str(b) for b in bnderr))

        self.bounds = abound

        # Define constraint function used in local minimisation
        if g_cons is not None:
            self.min_cons = []
            for g in self.g_func:
                self.min_cons.append({'type': 'ineq',
                                      'fun': g})

        # Define local minimization keyword arguments
        if minimizer_kwargs is not None:
            self.minimizer_kwargs = minimizer_kwargs
            if 'args' not in minimizer_kwargs:
                self.minimizer_kwargs['args'] = self.args

            if 'method' not in minimizer_kwargs:
                self.minimizer_kwargs['method'] = 'SLSQP'

            if 'bounds' not in minimizer_kwargs:
                self.minimizer_kwargs['bounds'] = self.bounds

            if 'options' not in minimizer_kwargs:
                minimizer_kwargs['options'] = {'ftol': 1e-12}

                if options is not None:
                    if 'ftol' in options:
                        self.minimizer_kwargs['options']['ftol'] = \
                            options['ftol']
                    if 'maxfev' in options:
                        self.minimizer_kwargs['options']['maxfev'] = \
                            options['maxfev']
                    if 'disp' in options:
                        self.minimizer_kwargs['options']['disp'] = \
                            options['disp']

            if 'callback' not in minimizer_kwargs:
                minimizer_kwargs['callback'] = self.callback

            if self.minimizer_kwargs['method'] == 'SLSQP' or \
                            self.minimizer_kwargs['method'] == 'COBYLA':
                if 'constraints' not in minimizer_kwargs:
                    minimizer_kwargs['constraints'] = self.min_cons
        else:
            self.minimizer_kwargs = {'args': self.args,
                                     'method': 'SLSQP',
                                     'bounds': self.bounds,
                                     'options': {'ftol': 1e-12},
                                     'callback': self.callback
                                     }

            if g_cons is not None:
                self.minimizer_kwargs['constraints'] = self.min_cons

            if options is not None:
                if 'ftol' in options:
                    self.minimizer_kwargs['options']['ftol'] = \
                        options['ftol']
                if 'maxfev' in options:
                    self.minimizer_kwargs['options']['maxfev'] = \
                        options['maxfev']
                if 'disp' in options:
                    self.minimizer_kwargs['options']['disp'] = options['disp']

        # Algorithm controls
        self.stopiter = False
        self.break_routine = False
        self.multiproc = multiproc

        # Initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.nfev = 0  # Include each sampling point as func evaluation
        self.res.nlfev = 0  # Local function evals for all minimisers
        self.res.nljev = 0  # Local jacobian evals for all minimisers

    def construct_complex(self):
        pass

    def sobol_points(self, N, D):
        """
        sobol.cc by Frances Kuo and Stephen Joe translated to Python 3 by
        Carl Sandrock 2016-03-31

        The original program is available and described at
        http://web.maths.unsw.edu.au/~fkuo/sobol/
        """
        import gzip
        import os
        path = os.path.join(os.path.dirname(__file__), 'new-joe-kuo-6.21201.gz')
        with gzip.open(path) as f:
            unsigned = "uint64"
            # swallow header
            buffer = next(f)

            L = int(numpy.log(N) // numpy.log(2.0)) + 1

            C = numpy.ones(N, dtype=unsigned)
            for i in range(1, N):
                value = i
                while value & 1:
                    value >>= 1
                    C[i] += 1

            points = numpy.zeros((N, D), dtype='double')

            # XXX: This appears not to set the first element of V
            V = numpy.empty(L + 1, dtype=unsigned)
            for i in range(1, L + 1):
                V[i] = 1 << (32 - i)

            X = numpy.empty(N, dtype=unsigned)
            X[0] = 0
            for i in range(1, N):
                X[i] = X[i - 1] ^ V[C[i - 1]]
                points[i, 0] = X[i] / 2 ** 32

            for j in range(1, D):
                F_int = [int(item) for item in next(f).strip().split()]
                (d, s, a), m = F_int[:3], [0] + F_int[3:]

                if L <= s:
                    for i in range(1, L + 1): V[i] = m[i] << (32 - i)
                else:
                    for i in range(1, s + 1): V[i] = m[i] << (32 - i)
                    for i in range(s + 1, L + 1):
                        V[i] = V[i - s] ^ (
                        V[i - s] >> numpy.array(s, dtype=unsigned))
                        for k in range(1, s):
                            V[i] ^= numpy.array(
                                (((a >> (s - 1 - k)) & 1) * V[i - k]),
                                dtype=unsigned)

                X[0] = 0
                for i in range(1, N):
                    X[i] = X[i - 1] ^ V[C[i - 1]]
                    points[i, j] = X[i] / 2 ** 32  # *** the actual points

            return points

    def sampling(self):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        # Generate sampling points.
        #  TODO Assert if func output matches dims. found from bounds
        self.m = len(self.bounds)  # Dimensions

        # Generate uniform sample points in [0, 1]^m \subset R^m
        self.C = self.sobol_points(self.n, self.m)

        # Distribute over bounds
        # TODO: Find a better way to do this
        for i in range(len(self.bounds)):
            self.C[:, i] = (self.C[:, i] *
                            (self.bounds[i][1] - self.bounds[i][0])
                            + self.bounds[i][0])
        return self.C

    def sampling_subspace(self):
        """Find subspace of feasible points from g_func definition"""
        # Subspace of feasible points.
        for g in self.g_func:
            self.C = self.C[g(self.C.T, *self.g_args) >= 0.0]
            if self.C.size == 0:
                self.res.message = ('No sampling point found within the '
                                    + 'feasible set. Increasing sampling '
                                    + 'size.')
                #TODO: Write a unittest to see if algorithm is increasing
                # sampling correctly for both 1D and >1D cases
                if self.disp:
                    print(self.res.message)

        self.fn = numpy.shape(self.C)[0]
        return

    def sorted_samples(self):  # Validated
        """Find indexes of the sorted sampling points"""
        self.I = numpy.argsort(self.C, axis=0)
        # TODO Use self.I as mask to sort only once
        self.Xs = numpy.sort(self.C, axis=0)
        return self.I, self.Xs

    def ax_subspace(self):  # Validated
        """
        Finds the subspace vectors along each component axis.
        """
        self.Ci = []
        self.Xs_i = []
        self.Ii = []
        for i in range(self.m):
            self.Ci.append(self.C[:, i])
            self.Ii.append(self.I[:, i])
            self.Xs_i.append(self.Xs[:, i])

        return

    def fun_ref(self):
        """
        Find the objective function output reference table
        """
        #TODO: This process can be pooled
        # Obj. function returns to be used as reference table.:
        if self.n_sampled > 0:  # Store old function evaluations
            Ftemp = self.F

        self.F = numpy.zeros(numpy.shape(self.C)[0])
        for i in range(self.n_sampled, numpy.shape(self.C)[0]):
            self.F[i] = self.func(self.C[i, :], *self.args)

        if self.n_sampled > 0:  # Restore saved function evaluations
            self.F[0:self.n_sampled] = Ftemp

        self.n_sampled = numpy.shape(self.C)[0]

        return self.F

    def surface_topo_ref(self):  # Validated
        """
        Find the BD and FD finite differences along each component
        vector.
        """
        # Replace numpy inf, -inf and nan objects with floating point numbers
        # fixme: Find a better way to deal with numpy.nan values.
        # nan --> float
        self.F[numpy.isnan(self.F)] = numpy.inf
        # inf, -inf  --> floats
        self.F = numpy.nan_to_num(self.F)

        self.Ft = self.F[self.I]
        self.Ftp = numpy.diff(self.Ft, axis=0)  # FD
        self.Ftm = numpy.diff(self.Ft[::-1], axis=0)[::-1]  # BD
        return

    def sample_topo(self, ind):
        # Find the position of the sample in the component axial directions
        self.Xi_ind_pos = []
        self.Xi_ind_topo_i = []

        for i in range(self.m):
            for x, I_ind in zip(self.Ii[i], range(len(self.Ii[i]))):
                if x == ind:
                     self.Xi_ind_pos.append(I_ind)

            # Use the topo reference tables to find if point is a minimizer on
            # the current axis

            # First check if index is on the boundary of the sampling points:
            if self.Xi_ind_pos[i] == 0:
                if self.Ftp[:, i][0] > 0:  # if boundary is in basin
                    BoundBasin = True
                    self.Xi_ind_topo_i.append(True)
                    #self.Xi_ind_topo_i.append(False)
                else:
                    self.Xi_ind_topo_i.append(False)

            elif self.Xi_ind_pos[i] == self.fn - 1:
                # Largest value at sample size
                if self.Ftp[:, i][self.fn - 2] < 0:
                    self.Xi_ind_topo_i.append(True)
                else:
                    self.Xi_ind_topo_i.append(False)

            # Find axial reference for other points
            else:
                if self.Ftp[:, i][self.Xi_ind_pos[i]] > 0:
                    Xi_ind_top_p = True
                else:
                    Xi_ind_top_p = False

                if self.Ftm[:, i][self.Xi_ind_pos[i] - 1] > 0:
                    Xi_ind_top_m = True
                else:
                    Xi_ind_top_m = False

                if Xi_ind_top_p and Xi_ind_top_m:
                    self.Xi_ind_topo_i.append(True)
                else:
                    self.Xi_ind_topo_i.append(False)

        if numpy.array(self.Xi_ind_topo_i).all():
            self.Xi_ind_topo = True
        else:
            self.Xi_ind_topo = False

        return self.Xi_ind_topo

    def minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        self.minimizer_pool = []
        #TODO: Can be parralized
        for ind in range(self.fn):
            Min_bool = self.sample_topo(ind)
            if Min_bool:
                self.minimizer_pool.append(ind)


        self.minimizer_pool_F = self.F[self.minimizer_pool]

        # Sort to find minimum func value in min_pool
        self.sort_min_pool()

        self.X_min = self.C[self.minimizer_pool]

        return self.X_min

    def sort_min_pool(self):
        # Sort to find minimum func value in min_pool
        self.ind_f_min = numpy.argsort(self.minimizer_pool_F)
        self.minimizer_pool = numpy.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = self.minimizer_pool_F[self.ind_f_min]
        return

    def trim_min_pool(self, trim_ind):
        self.X_min = numpy.delete(self.X_min, trim_ind, axis=0)
        self.minimizer_pool_F = numpy.delete(self.minimizer_pool_F, trim_ind)
        return

    def g_topograph(self, x_min, X_min):
        """
        Returns the topographical vector stemming from the specified value
        value 'x_min' for the current feasible set 'X_min' with True boolean
        values indicating positive entries and False ref. values indicating
        negative values.
        """
        x_min = numpy.array([x_min])
        self.Y = scipy.spatial.distance.cdist(x_min,
                                              X_min,
                                              'euclidean')
        # Find sorted indexes of spatial distances:
        self.Z = numpy.argsort(self.Y, axis=-1)

        self.Ss = X_min[self.Z]
        return self.Ss


    def minimize(self, x_min):
        """
        This function is used to calculate the local minima using the specified
        sampling point as a starting value.

        Parameters
        ----------
        x_min : vector of floats
            Current starting point to minimise.

        Returns
        -------
        lres : OptimizeResult
            The local optimization result represented as a `OptimizeResult`
            object.
        """
        if self.callback is not None:
            print('Callback for '
                  'minimizer starting at {}:'.format(x_min))

        if self.disp:
            print('Starting '
                  'minimization at {}...'.format(x_min))

        lres = scipy.optimize.minimize(self.func, x_min,
                                       **self.minimizer_kwargs)

        if self.disp:
            print('lres = {}'.format(lres))
        # Local function evals for all minimisers
        self.res.nlfev += lres.nfev
        self.x_min_glob.append(lres.x)
        try:  # Needed because of the brain dead 1x1 numpy arrays
            self.fun_min_glob.append(lres.fun[0])
        except IndexError:
            self.fun_min_glob.append(lres.fun)

        return lres

    def delaunay_triangulation(self):
        from scipy.spatial import Delaunay
        self.Tri = Delaunay(self.C)
        return self.Tri

    def find_neighbors(self, pindex, triang):
        """
        Returns the indexes of points connected to ``pindex``  on the Gabriel
        chain subgraph of the Delaunay triangulation.

        """
        return triang.vertex_neighbor_vertices[1][
               triang.vertex_neighbor_vertices[0][pindex]:
               triang.vertex_neighbor_vertices[0][pindex + 1]]

    def sample_delaunay_topo(self, ind):
        self.Xi_ind_topo_i = []

        # Find the position of the sample in the component Gabrial chain
        G_ind = self.find_neighbors(ind, self.Tri)

        # Find finite deference between each point
        for g_i in G_ind:
            rel_topo_bool = self.F[ind] < self.F[g_i]
            self.Xi_ind_topo_i.append(rel_topo_bool)

        # Check if minimizer
        if numpy.array(self.Xi_ind_topo_i).all():
            self.Xi_ind_topo = True
        else:
            self.Xi_ind_topo = False

        return self.Xi_ind_topo

    def delaunay_minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        pass
        self.minimizer_pool = []
        # TODO: Can easily be parralized
        for ind in range(self.fn):
            Min_bool = self.sample_delaunay_topo(ind)
            if Min_bool:
                self.minimizer_pool.append(ind)

        self.minimizer_pool_F = self.F[self.minimizer_pool]

        # Sort to find minimum func value in min_pool
        self.sort_min_pool()

        self.X_min = self.C[self.minimizer_pool]

        return self.X_min

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    from numpy import *
    #exec(open('./shgo_tests.py').read())




