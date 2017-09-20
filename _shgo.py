#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" execfile('tgo.py')
"""
from __future__ import division, print_function, absolute_import
import numpy
import scipy.spatial
import scipy.optimize
import logging
from triangulation import *
from sobol_seq import *

try:
    pass
    #from multiprocessing_on_dill import Pool
except ImportError:
    from multiprocessing import Pool

def shgo(func, bounds, args=(), g_cons=None, g_args=(), n=None, iter=None,
         callback=None, minimizer_kwargs=None, options=None,
         sampling_method='simplicial'):
    #TODO: Update documentation

    # sampling_method: str, options = 'sobol', 'simplicial'
    """
    Finds the global minima of a function using simplicial homology global
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
    # Initiate SHGO class
    if sampling_method == 'simplicial':
        SHc = SHGOh(func, bounds, args=args, g_cons=g_cons, g_args=g_args, n=n,
                    iter=iter, callback=callback, minimizer_kwargs=minimizer_kwargs,
                    options=options, sampling_method=sampling_method)

    elif sampling_method == 'sobol':
        SHc = SHGOs(func, bounds, args=args, g_cons=g_cons, g_args=g_args, n=n,
                    iter=iter, callback=callback, minimizer_kwargs=minimizer_kwargs,
                    options=options, sampling_method=sampling_method)

    else:
        raise IOError("""Unknown sampling_method specified, use either 
                         'sobol' or 'simplicial' """)

    # Generate sampling points
    if SHc.disp:
        print('Generating sampling points')

    # Construct directed complex.
    if not SHc.break_routine:
        SHc.construct_complex()

    if not SHc.break_routine:
        if SHc.disp:
            print("Succesfully completed construction of complex.")

        if sampling_method == 'simplicial':
            # Build minimiser pool
            SHc.simplex_minimizers()

    if len(SHc.X_min) == 0:
        # If sampling failed to find pool, return lowest sampled point
        # with a warning
        #TODO: Implement warning and lowest sampling return
        SHc.break_routine = True
        SHc.fail_routine(mes="Failed to find a feasible minimiser point. "
                             "Lowest sampling point =")

    if not SHc.break_routine:
        # Minimise the pool of minisers with local minimisation methods
        # Note that if Options['local_iter'] is an `int` instead of default
        # value False then only that number of candidates will be minimised
        SHc.minimise_pool(SHc.local_iter)

    # Sort results and build the global return object
    if not SHc.break_routine:
        SHc.sort_result()

    # Confirm the routine ran succesfully
    if not SHc.break_routine:
        SHc.res.message = 'Optimization terminated successfully.'
        SHc.res.success = True

    return SHc.res

# %% Define the base SHGO class inherited by the different methods
class SHGO(object):
    def __init__(self, func, bounds, args=(), g_cons=None, g_args=(), n=None,
                 iter=None, callback=None, minimizer_kwargs=None,
                 options=None, sampling_method='sobol'):

        self.func = func
        #  TODO Assert if func output matches dims. found from bounds
        self.bounds = bounds
        self.m = len(self.bounds)  # Dimensions
        self.args = args
        self.g_cons = g_cons
        if g_cons is not None:
            if (type(g_cons) is not tuple) and (type(g_cons) is not list):
                self.g_func = (g_cons,)
                self.g_cons = (g_cons,)
            else:
                self.g_func = g_cons
        else:
            self.g_func = None

        self.g_args = g_args

        self.callback = callback

        # Process options dict
        self.options = options
        self.maxfev = None
        self.disp = False
        self.symmetry = False
        self.local_iter = False
        if options is not None:
            if 'maxfev' in options:
                # Maximum number of function evaluations in the feasible domain
                self.maxfev = options['maxfev']
            else:
                self.maxfev = None
            if 'maxev' in options:
                # Maximum number of sampling evaluations (includes searching in
                # infeasible points
                self.maxev = options['maxev']
            else:
                self.maxev = None

            if 'disp' in options:
                self.disp = options['disp']
            if 'symmetry' in options:
                self.symmetry = True
            else:
                self.symmetry = False
            if 'minimize_every_iter' in options:
                self.minimize_every_iter = True
            else:
                self.minimize_every_iter = False
            if 'local_iter' in options:  # Only evaluate a few of the best candiates
                self.local_iter = options['local_iter']
            else:  # Evaluate all minimisers
                self.local_iter = False
            if 'local_fglob' in options:  #TODO: Temporary dev work
                self.local_fglob = options['local_fglob']
                if 'local_f_tol' in options:
                    self.local_f_tol = options['local_f_tol']
                else:
                    self.local_f_tol = 1e-5
            else:
                self.local_fglob = None
            if 'min_iter' in options:
                self.min_iter = options['min_iter']
            if 'max_iter' in options:
                self.min_iter = options['max_iter']
            if 'min_hgrd' in options:
                self.min_hgrd = options['min_hgrd']
            else:
                self.min_hgrd = 0

            if 'f_min' in options:
                self.f_min_true = options['f_min']
                if 'f_tol' in options:
                    self.f_tol = options['f_tol']
                else:
                    self.f_tol = 1e-4

            if 'multiproc' in options:
                self.multiproc = options['multiproc']

            #self.options = None
        else:
            self.f_min_true = None
            self.maxev = None
            self.maxfev = None
            self.minimize_every_iter = False
            self.local_fglob = None # dev

        # set bounds
        abound = numpy.array(bounds, float)
        self.dim = numpy.shape(abound)[0]  # Dimensionality of problem
        # Check if bounds are correctly specified
        #bnderr = numpy.where(abound[:, 0] > abound[:, 1])[0]
        bnderr = abound[:, 0] > abound[:, 1]
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
                                     'options': {'ftol': 1e-12
                                                 # ,'eps': 1e-15
                                                 },
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
            if 'f_min' not in options:
                self.f_min_true = None

        ## Algorithm controls
        # Global controls
        self.iter = iter
        self.n = n
        self.n_sampled = 0  # To track sampling points already evaluated
        self.fn = n  # Number of feasible samples remaining

        # Stop the algorithm if multiple stopping criteria are specified
        if (self.n is not None) and (self.iter is not None):
            raise IOError('Ambiguous input: specify either' 
                    """`iter` finite iterations """
                    """or `n` finite sampling points""")
        elif (self.n is not None) and (self.f_min_true is not None):
            raise IOError("""Ambiguous input: specify either """
                          """`n` finite sampling points"""
                          """or options['f_min'] known function global minima""")
        elif (self.iter is not None) and (self.f_min_true is not None):
            raise IOError("""Ambiguous input: specify either """
                          """`iter` finite iterations """
                          """or options['f_min'] known function global minima""")

        #else:  # else only for collapse-able syntax
        if (self.iter is not None):
            self.stop_global = False
            # Define stop iteration method
            self.stop_iter_m = self.finite_iterations
        elif (self.n is not None):
            self.stop_global = False
            # Define stop iteration method
            self.stop_iter_m = self.finite_sampling
        elif (sampling_method == 'simplicial') and (self.iter is None):
            self.stop_iter_m = self.finite_iterations
            self.iter = 1
        elif options is not None:
            if 'f_min' in options:
                self.stop_global = False
                self.stop_iter_m = self.finite_precision
        else: # Choose a default strategy if none is specified
            self.n = 100
            self.stop_iter_m = None

        # Local controls
        self.stop_l_iter = False  # Local minimisation iterations
        self.stop_complex_iter = False  # Sampling iterations
        self.break_routine = False

        self.stop_evaluation = None  #TODO Assign a function based on inputs ex. maxeval or max n

        # SOBOL SAMPLING v SIMPLICIAL
        # self.sampling = sampling
        self.sampling_method = sampling_method

        ## Set complex construction mode:
        if (iter is not None):
            if sampling_method == 'sobol':
                self.construct_complex = self.construct_complex_sobol_iter
            elif sampling_method == 'simplicial':
                self.construct_complex = self.construct_complex_iteratively
        elif (self.n is not None):  # Finite sampling points
            if sampling_method == 'sobol':
                self.construct_complex = self.construct_complex_sobol
                if options is not None:
                    if 'infty constraints' in options:
                        self.construct_complex = self.construct_complex_sobol_inf

            elif sampling_method == 'simplicial':
                self.construct_complex = self.construct_complex_iteratively
        elif (self.f_min_true is not None):
            if sampling_method == 'sobol':
                #self.construct_complex = self.construct_complex_sobol_iter
                #self.construct_complex = self.construct_complex_iteratively
                self.n = self.dim + 1  #TODO: Long unittest, fix
                self.construct_complex = self.construct_complex_sobol
            elif sampling_method == 'simplicial':
                self.construct_complex = self.construct_complex_iteratively

        # Initiate storate objects used in alorithm classes
        self.x_min_glob = []
        self.fun_min_glob = []

        # Initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.nfev = 0  # Include each sampling point as func evaluation
        self.res.nlfev = 0  # Local function evals for all minimisers
        self.res.nljev = 0  # Local jacobian evals for all minimisers

    # Iteration properties
    # Stopping criteria functions:
    def finite_iterations(self):
        self.iter -= 1
        if self.iter <= 0:
            self.stop_global = True
        elif self.maxev is not None:
            if len(self.HC.V.cache) >= self.maxev:  # Stop for infeasible sampling
                self.stop_global = True
                self.fail_routine(mes=("Failed to find a feasible "
                                       "sampling point within the "
                                       "maximum allowed evaluations."))
        return self.stop_global

    def finite_sampling(self):
        # This is for the simplicial complex since Sobol has it's own finite generation
        #self.fn -= 1
        #print(f'self.fn = {self.fn}')
        if self.disp:
            logging.info(f'len(self.HC.V.cache)= {len(self.HC.V.cache)}')
            logging.info(f'self.HC.V.nfev = {self.HC.V.nfev}')
        if self.HC.V.nfev >= self.n:
            self.stop_global = True
        elif self.maxev is not None:
            if len(self.HC.V.cache) >= self.maxev:  # Stop for infeasible sampling
                self.stop_global = True
                self.fail_routine(mes=("Failed to find a feasible "
                                       "sampling point within the "
                                       "maximum allowed evaluations."))
            #elif self.HC.V.nfev >= self.maxfev:  # STOP FOR INFEASIBLE SAMPLING
            #    self.stop_global = True
            #    self.fail_routine(mes=("Failed to find a feasible "
            #                           "sampling point within the "
            #                           "maximum allowed evaluations."))


        return self.stop_global

    def finite_precision(self):
        # Stop the algorithm if the final function value is known
        # Specify in options (with self.f_min_true = options['f_min'])
        #  and the tolerance with f_tol = options['f_tol']
        if self.f_lowest == 0:
            if self.f_lowest <= self.f_tol:
                self.stop_global = True
        else:
            pe = (self.f_min_true - self.f_lowest)/abs(self.f_lowest)
            if pe <= self.f_tol:
                self.stop_global = True
        return self.stop_global

    # Minimiser pool processing
    def minimise_pool(self, force_iter=False):
        """
        This processing method can optionally minimise only the best candidate
        solutions in the minimiser pool

        Parameters
        ----------

        force_iter : int
                     Number of starting minimisers to process (can be sepcified
                     globally or locally)

        """

        # Find first local minimum
        # NOTE: Since we always minimize this value regardless it is a waste to
        # build the topograph first before minimizing
        lres_f_min = self.minimize(self.X_min[[0]])

        # Trim minimised point from current minimiser set
        self.trim_min_pool(0)

        # Force processing to only
        if force_iter:
            self.local_iter = force_iter

        while not self.stop_l_iter:
            #======Dev========
            # Global stopping criteria:
            if self.local_fglob is not None:
                if abs(lres_f_min.fun - self.local_fglob) <= self.local_f_tol:
                    self.stop_l_iter = True
                    break

            if self.f_min_true is not None:
                if abs(lres_f_min.fun - self.f_min_true) <= self.f_tol:
                    self.stop_l_iter = True
            # ======Dev========


            if self.local_iter is not None:  # Note first iteration is outside loop
                if self.disp:
                    logging.info('SHGO.iter in function minimise_pool = {}'.format(self.local_iter))
                self.local_iter -= 1
                #if __name__ == '__main__':
                if self.local_iter == 0:
                    self.stop_l_iter = True
                    break
                    #TODO: Test usage of iterative features

            if numpy.shape(self.X_min)[0] == 0:
                self.stop_l_iter = True
                break

            # Construct topograph from current minimiser set
            # (NOTE: This is a very small topograph using only the miniser pool
            #        , it might be worth using some graph theory tools instead.
            self.g_topograph(lres_f_min.x, self.X_min)

            # Find local minimum at the miniser with the greatest euclidean
            # distance from the current solution
            ind_xmin_l = self.Z[:, -1]
            lres_f_min = self.minimize(self.Ss[:, -1])

            # Trim minimised point from current minimiser set
            self.trim_min_pool(ind_xmin_l)

        return

    # Local bound functions
    def contstruct_lcb(self, v_min):
        """
        Construct locally (approximately) convex bounds

        Parameters
        ----------
        v_min : Vertex object
                The minimiser vertex
        Returns
        -------
        cbounds : List of size dim with tuple of bounds for each dimension
        """
        cbounds = []
        for x_b_i in self.bounds:
            cbounds.append([x_b_i[0], x_b_i[1]])
        # Loop over all bounds
        for vn in v_min.nn:
            #for i, x_i in enumerate(vn.x):
            for i, x_i in enumerate(vn.x_a):
                # Lower bound
                if (x_i < v_min.x_a[i]) and (x_i > cbounds[i][0]):
                    cbounds[i][0] = x_i

                # Upper bound
                if (x_i > v_min.x_a[i]) and (x_i < cbounds[i][1]):
                    cbounds[i][1] = x_i
        if self.disp:
            logging.info(f'cbounds found for v_min.x_a = {v_min.x_a} ')
            logging.info(f'cbounds = {cbounds}')
        return cbounds

    # Minimize a starting point locally
    def minimize(self, x_min, cache_position=None):
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

        # TODO: Optionally construct bounds if minimizer_kwargs is a
        #      solver that accepts bounds
        if self.sampling_method == 'simplicial': #TODO: Arbitrary sampling input
            x_min_t = tuple(x_min[0])
            # Find the normalized tuple in the Vertex cache:
            x_min_t_norm = self.X_min_cache[tuple(x_min_t)]

            x_min_t_norm = tuple(x_min_t_norm)

            self.minimizer_kwargs['bounds'] = \
                self.contstruct_lcb(self.HC.V[x_min_t_norm])

            if self.disp:
                print('bounds in kwarg:')
                print(self.minimizer_kwargs['bounds'])

        lres = scipy.optimize.minimize(self.func, x_min,
                                       **self.minimizer_kwargs)

        if self.disp:
            print('lres = {}'.format(lres))
        # Local function evals for all minimisers
        self.res.nlfev += lres.nfev
        self.x_min_glob.append(lres.x)
        try:  # Needed because of the brain dead 1x1 numpy arrays
            self.fun_min_glob.append(lres.fun[0])
        except (IndexError, TypeError):
            self.fun_min_glob.append(lres.fun)

        return lres

    # Post local minimisation processing
    def sort_result(self):
        """
        Sort results and build the global return object
        """
        import numpy
        # Sort results and save
        self.x_min_glob = numpy.array(self.x_min_glob)
        self.fun_min_glob = numpy.array(self.fun_min_glob)

        # Sorted indexes in Func_min
        ind_sorted = numpy.argsort(self.fun_min_glob)

        # Save ordered list of minima
        self.res.xl = self.x_min_glob[ind_sorted]  # Ordered x vals #TODO: Check
        self.fun_min_glob = numpy.array(self.fun_min_glob)
        self.res.funl = self.fun_min_glob[ind_sorted]
        self.res.funl = self.res.funl.T

        # Find global of all minimisers
        self.res.x = self.x_min_glob[ind_sorted[0]]  # Save global minima
        self.res.fun = self.fun_min_glob[ind_sorted[0]] # Save global fun value

        # Add local func evals to sampling func evals
        self.res.nfev += self.res.nlfev
        return

    # Algorithm controls
    def fail_routine(self, mes=("Failed to converge")):
        self.break_routine = True
        self.res.success = False
        self.X_min = [None]
        self.res.message = mes
        return

# %% Define shgo class using simplicial hypercube sampling
class SHGOh(SHGO):
    """
    This class implements the shgo routine
    """
    def __init__(self, func, bounds, args=(), g_cons=None, g_args=(), n=None,
                 iter=None, callback=None, minimizer_kwargs=None,
                 options=None, sampling_method='sobol'):

        SHGO.__init__(self, func, bounds, args=args, g_cons=g_cons, g_args=g_args, n=n,
                    iter=iter, callback=callback, minimizer_kwargs=minimizer_kwargs,
                    options=options, sampling_method=sampling_method)


    def construct_initial_complex(self):
        if self.disp:
            print('Building initial complex')

        self.HC = Complex(self.dim, self.func, self.args,
                          self.symmetry, self.bounds, self.g_func, self.g_args)
        return

    def iterate(self):
        """
        Iterate a subdivision of the complex
         (globally biased)
        """
        self.HC.split_generation()
        return

    def construct_complex_iteratively(self):
        """
        Stop iterations when stopping criteria (sampling points or
        processing time) have been met.

        """
        self.construct_initial_complex()

        if self.disp:
            print('Splitting first generation')

        if 1: #TODO: IF CERTAIN METHOD TO EVAL every iter
            self.minimize_locally = True

        while not self.stop_iter_m():
            self.iterate()

            # Specify in options['minimize_every_iter'] = True to use
            if self.minimize_every_iter: #TODO: TEST THIS ROUTINE
                self.simplex_minimizers()
                if len(self.X_min) is not 0:
                    self.minimise_pool(self.local_iter)
                    self.sort_result()
                    self.f_lowest = self.res.fun

        # Algorithm updates
        # Count the number of vertices and add to function evaluations:
        self.res.nfev += self.HC.V.nfev
        if self.disp:
            print(f'self.res.nfev = {self.res.nfev}')
        return

    def iterate_diff(self):
        """
        Iterate a subdivision of the complex, but skip
        unpromising cells (locally biased)

        (UNFINISHED)

        Ideas: We will search for sub-cells which total rank
               contains the same number of minimisers ar the
               the parent cell and then stop searching in
               this region. For a first attempt, use the
               heuristic of 2 generations with no change.
        """
        self.HC.split_generation()
        return

    def simplex_minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        self.minimizer_pool = []
        # TODO: Can easily be parralized
        for x in self.HC.V.cache:
            if self.HC.V[x].minimiser():
                if self.disp:
                    logging.info('=' * 60)
                    logging.info('v.x = {} is minimiser'.format(self.HC.V[x].x_a))
                    logging.info('v.f = {} is minimiser'.format(self.HC.V[x].f))
                    logging.info('=' * 30)

                if self.HC.V[x] not in self.minimizer_pool:
                    self.minimizer_pool.append(self.HC.V[x])

                if self.disp:
                    logging.info('Neighbours:')
                    logging.info('='*30)
                    for vn in self.HC.V[x].nn:
                        logging.info('x = {} || f = {}'.format(vn.x, vn.f))

                    logging.info('=' * 60)

        self.minimizer_pool_F = []
        self.X_min = []
        if self.sampling_method == 'simplicial':
            # normalized tuple in the Vertex cache
            self.X_min_cache = {}  # Cache used in hypercube sampling

        for v in self.minimizer_pool:
            self.X_min.append(v.x_a)
            self.minimizer_pool_F.append(v.f)
            if self.sampling_method == 'simplicial':
                self.X_min_cache[tuple(v.x_a)] = v.x

        self.minimizer_pool_F = numpy.array(self.minimizer_pool_F)
        self.X_min = numpy.array(self.X_min)

        # TODO: Only do this if global mode
        self.sort_min_pool()

        return self.X_min

    def sort_min_pool(self):
        # Sort to find minimum func value in min_pool
        self.ind_f_min = numpy.argsort(self.minimizer_pool_F)
        self.minimizer_pool = numpy.array(self.minimizer_pool)[self.ind_f_min]
        self.minimizer_pool_F = numpy.array(self.minimizer_pool_F)[self.ind_f_min]
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


# %% Define shgo class using arbitrary (ex Sobol) sampling
class SHGOs(SHGO):
    """
    This class implements the shgo routine
    """
    def __init__(self, func, bounds, args=(), g_cons=None, g_args=(), n=None,
                 iter=None, callback=None, minimizer_kwargs=None,
                 options=None, sampling_method='sobol'):

        if (n is None) and (iter is None):
            if options is None:
                n = 100  # Define arbitrary sampling if user provided none
            elif 'f_min' in options:
                pass

        SHGO.__init__(self, func, bounds, args=args, g_cons=g_cons, g_args=g_args, n=n,
                    iter=iter, callback=callback, minimizer_kwargs=minimizer_kwargs,
                    options=options, sampling_method=sampling_method)

    def construct_complex_sobol(self):
        """
        Construct a complex based on the Sobol sequence
        """
        sample = True
        while sample:
            self.sampling()

            # Find subspace of feasible points
            if self.g_cons is not None:
                self.sampling_subspace()
            else:
                self.fn = self.n

            # Sort remaining samples
            self.sorted_samples()

            # Find objective function references
            self.fun_ref()

            # Find minimiser pool
            # DIMENSIONS self.dim
            if self.fn >= (self.dim + 1):
                if self.dim < 2:  # Scalar objective functions
                    if self.disp:
                        print('Constructing 1D minimizer pool')

                    self.ax_subspace()
                    self.surface_topo_ref()
                    self.X_min = self.minimizers()

                else:  # Multivariate functions.
                    if self.disp:
                        print('Constructing Gabrial graph and minimizer pool')

                    self.delaunay_triangulation()
                    if self.disp:
                        print('Triangulation completed, building minimizer pool')

                    self.X_min = self.delaunay_minimizers()

                logging.info("Minimiser pool = SHGO.X_min = {}".format(self.X_min))
            else:
                if self.disp:
                    print('Not enough sampling points found in the feasible domain.')
                self.minimizer_pool = [None]

            # TODO: Keep sampling until n feasible points in non-linear constraints
            # self.fn < self.n ---> self.n - self.fn
            if (len(self.minimizer_pool) == 0) or (self.fn == 0) or (self.fn < (self.dim + 1)):
                if self.disp:
                    if len(self.minimizer_pool) == 0:
                        print('No minimizers found. Increasing sampling space.')
                    if self.fn == 0:
                        print('No feasible points found. Increasing sampling space.')
                n_add = 100
                if self.options is not None:
                    if 'maxfev' in self.options.keys():
                        n_add = int((self.options['maxfev'] - self.fn) / 1.618)
                        if (n_add < 1) or ((self.n + n_add) >= self.options['maxfev']):
                            self.res.message = ("Failed to find a minimizer "
                                                "within the maximum allowed "
                                                "function evaluations.")

                            if self.disp:
                                print(self.res.message + " Breaking routine...")

                            self.break_routine = True
                            self.X_min = [None]
                            self.res.success = False
                            sample = False

                self.n += n_add
                self.res.nfev = self.fn

            else:  # If good values are found stop while loop
                # Include each sampling point as func evaluation:
                self.res.nfev = self.fn
                sample = False

    def construct_complex_sobol_inf(self):
        """
        Construct a complex based on the Sobol sequence.
        Use infinity evals on vertices outside constraints
        """
        sample = True
        while sample:
            self.sampling()

            # Sort remaining samples
            self.sorted_samples()

            # Find objective function references
            self.fun_ref_inf()  # Use numpy.inf on values < g(x)

            # Find minimiser pool
            # DIMENSIONS self.dim
            if self.fn >= (self.dim + 1):
                if self.dim < 2:  # Scalar objective functions
                    if self.disp:
                        print('Constructing 1D minimizer pool')

                    self.ax_subspace()
                    self.surface_topo_ref()
                    self.X_min = self.minimizers()

                else:  # Multivariate functions.
                    if self.disp:
                        print('Constructing Gabrial graph and minimizer pool')

                    self.delaunay_triangulation()
                    if self.disp:
                        print('Triangulation completed, building minimizer pool')

                    self.X_min = self.delaunay_minimizers()
                if self.disp:
                    logging.info("Minimiser pool = SHGO.X_min = {}".format(self.X_min))
            else:
                if self.disp:
                    print('Not enough sampling points found in the feasible domain.')
                self.minimizer_pool = [None]

            # TODO: Keep sampling until n feasible points in non-linear constraints
            # self.fn < self.n ---> self.n - self.fn
            if (len(self.minimizer_pool) == 0) or (self.fn == 0) or (self.fn < (self.dim + 1)):
                if self.disp:
                    if len(self.minimizer_pool) == 0:
                        print('No minimizers found. Increasing sampling space.')
                    if self.fn == 0:
                        print('No feasible points found. Increasing sampling space.')
                n_add = 100
                if self.options is not None:
                    if 'maxfev' in self.options.keys():
                        n_add = int((self.options['maxfev'] - self.fn) / 1.618)
                        if (n_add < 1) or ((self.n + n_add) >= self.options['maxfev']):
                            self.res.message = ("Failed to find a minimizer "
                                                "within the maximum allowed "
                                                "function evaluations.")

                            if self.disp:
                                print(self.res.message + " Breaking routine...")

                            self.break_routine = True
                            self.X_min = [None]
                            self.res.success = False
                            sample = False
                    else:  # TEMPORARY DEV
                        self.break_routine = True
                        self.X_min = [None]
                        self.res.success = False
                        sample = False

                self.n += n_add
                self.res.nfev = self.fn

            else:  # If good values are found stop while loop
                # Include each sampling point as func evaluation:
                self.res.nfev = self.fn
                sample = False


    def construct_complex_sobol_iter(self, n_growth_init=30):
        """
        Construct a complex based on the Sobol sequence
        """
        # Construct initial sampling pool
        self.n = self.dim + 1 # Minimum vertexes required to build complex
        self.n = self.dim + 2 # Minimum vertexes required to build complex
                                # (only 2D test functions at the moment)

        self.n = self.dim**2 + 2
        n_growth_init = 2 * self.n #TODO: TESTING THIS

        logging.info("self.dim = {}".format(self.dim))
        logging.info("Constructing initial complex"
                     " with self.n = {}".format(self.n ))
        # ^Run initial contructor for table top avoidance
    #    logging.info('Run initial contructor for table top avoidance')
    #    self.sampling()
    #    logging.info('self.C = {}'.format(self.C))
    #    self.construct_complex_sobol()
    #    logging.info('First Sobol iteration grown')
    #    self.processed_n = self.n

        # Start iterative complex growth
        self.processed_n = 0
        self.X_min_all = []
        self.minimizer_pool_F_all = []
        grow_complex = True
        #homology_group_differential = n_growth_init
        n_pool = n_growth_init
        n_growth = 0
        homology_group_prev = 0#len(self.minimizer_pool)
        homology_group_differential = 0
    #    self.n = n_growth_init

        while grow_complex:
            sample_loop = True
            # Sample untill enough points in subspace is found
            self.n_cons = self.n  # Desired sampling points
            while sample_loop:
                self.sampling()
            #    self.C = self.C[self.processed_n:, :]

                # Find subspace of feasible points
                if self.g_cons is not None:
                    #TODO: Adapt sampling_subpace to not process old points
                    self.sampling_subspace()
                    self.fn = numpy.shape(self.C)[0]
                else:
                    self.fn = self.n

                if numpy.shape(self.C)[0] < self.n_cons:
                    self.n += 1
                    #logging.info('Subspace search increased to '
                    #             'self.n = {}'.format(self.n ))
                else:
                    sample_loop = False

            # Reset to actual sampling points used
            self.n = self.n_cons

            # Sort remaining samples
            self.sorted_samples()

            # Find objective function references
            self.fun_ref()

            # Find minimiser pool
            # DIMENSIONS self.dim
            if self.dim < 2:  # Scalar objective functions
                if self.disp:
                    print('Constructing 1D minimizer pool')

                self.ax_subspace()
                self.surface_topo_ref()
                self.X_min = self.minimizers()

            else:  # Multivariate functions.
                if self.disp:
                    print('Constructing Gabrial graph and minimizer pool')

                logging.info('Adding vertex to complex, current size = '
                             '{}'.format(self.processed_n))

                self.delaunay_triangulation(grow=True,
                                            n_prc=self.processed_n)
                if self.disp:
                    print('Triangulation completed, building minimizer pool')

                logging.info('Complex constructed...'
                             'looking for minimizer pool:')

                self.X_min = self.delaunay_minimizers()

            #logging.info("Minimiser pool = SHGO.X_min = {}".format(self.X_min))

            if not len(self.minimizer_pool) == 0:
                #self.X_min_all.append(self.X_min)
                self.minimizer_pool_F_all.append(self.minimizer_pool_F)
                homology_group = len(self.minimizer_pool)
            else:
                homology_group = 0

            if self.disp:
                logging.info('homology_group ='
                             ' {}'.format(homology_group))
                logging.info('homology_group_differential ='
                             ' {}'.format(homology_group_differential))
                logging.info('n_pool ='
                             ' {}'.format(n_pool))

            if homology_group > homology_group_prev:
                #homology_group_differential += homology_group * 10 #hyperparam
                #homology_group_differential += 10
                #homology_group - homology_group_prev
                hgd_new = self.n - n_growth
                homology_group_differential = max(hgd_new,
                                                  homology_group_differential)
                homology_group_prev = homology_group

                if self.fn < n_growth_init:
                    n_pool += homology_group_differential
                else:
                    n_pool = homology_group_differential #* self.dim
                #^^ Should be set equal to this?
                n_growth = self.n
            else:
                n_pool -= 1
                self.n += 1

            if n_pool== 0:
                grow_complex = False
            #if homology_group == 0:
            #    pass


            #else:  # If good values are found stop while loop
            self.processed_n = self.fn#self.n
            #self.n += 1

            #Global mode check:
            if self.stop_iter_m is not None:
                if self.f_min_true is not None:
                    if self.stop_iter_m():
                        grow_complex = False

            # Break if no final minima found
        #logging.info('self.X_min = {}'.format(self.X_min))
        if len(self.X_min) == 0:
            self.break_routine = True
            self.res.message = 'No minimizers found in iterative growth mode'
            self.res.success = False
            if self.disp:
                print("Failed to find non-zero set of minimzers in iterative"
                      "growrth mode complex construction")

        # After the complex is completed count the function evaluations used:
        if self.dim > 1:
            logging.info('self.Tri.points = {}'.format(len(self.Tri.points)))
            self.res.nfev = len(self.Tri.points)
            #logging.info('self.Tri.points '
            #             '= {}'.format(self.Tri.points))
        else:
            # Include each sampling point as func evaluation:
            self.res.nfev = self.fn

    def sobol_points_old(self, N, D):
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

    def sobol_points(self, N, D):
        """
        Wrapper for sobol_seq.i4_sobol_generate

        Generate N sampling points in D dimensions
        """
        points = i4_sobol_generate(D, N, skip=0)

        return points

    def sampling(self, method='sobol'):
        """
        Generates uniform sampling points in a hypercube and scales the points
        to the bound limits.
        """
        # Generate sampling points.
        #  TODO Assert if func output matches dims. found from bounds
        self.m = len(self.bounds)  # Dimensions

        # Generate uniform sample points in [0, 1]^m \subset R^m
        if method == 'sobol':
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
        #NOTE: It might be easier to replace this with a cached
        #      objective function
        for i in range(self.n_sampled, numpy.shape(self.C)[0]):
            eval_f = True
            if self.g_func is not None:
                for g in self.g_func:
                    if g(self.C[i, :], *self.args) < 0.0:
                        self.F[i] = numpy.inf
                        eval_f = False
                        break  # Breaks the g loop

            if eval_f:
                self.F[i] = self.func(self.C[i, :], *self.args)

        if self.n_sampled > 0:  # Restore saved function evaluations
            self.F[0:self.n_sampled] = Ftemp

        self.n_sampled = numpy.shape(self.C)[0]

        return self.F

    def fun_ref_inf(self):
        """
        Find the objective function output reference table
        Use inf on values outside constraints
        """
        #TODO: This process can be pooled
        # Obj. function returns to be used as reference table.:
        if self.n_sampled > 0:  # Store old function evaluations
            Ftemp = self.F

        self.F = numpy.zeros(numpy.shape(self.C)[0])
        #NOTE: It might be easier to replace this with a cached
        #      objective function
        for i in range(self.n_sampled, numpy.shape(self.C)[0]):
            self.F[i] = self.func(self.C[i, :], *self.args)
            if 0:
                self.F[i] = self.f = numpy.inf

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
        if not len(self.minimizer_pool) == 0:
            self.X_min = self.C[self.minimizer_pool]
            # If function is called again and pool is found unbreak:
        else:
            return []

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

    def delaunay_triangulation(self, grow=False, n_prc=0):
        from scipy.spatial import Delaunay
        if not grow:
            self.Tri = Delaunay(self.C)
        else:
            try:
                self.Tri.add_points(self.C[n_prc:, :])
            except AttributeError:  #TODO: Fix in main algorithm
                self.Tri = Delaunay(self.C, incremental=True)

        return self.Tri

    def find_neighbors_delaunay(self, pindex, triang):
        """
        Returns the indexes of points connected to ``pindex``  on the Gabriel
        chain subgraph of the Delaunay triangulation.

        """
    #    logging.info('triang.vertices = {}'.format(triang.vertices))
    #    logging.info('triang.points = {}'.format(triang.points))
    #    logging.info('pindex = {}'.format(pindex))
        return triang.vertex_neighbor_vertices[1][
               triang.vertex_neighbor_vertices[0][pindex]:
               triang.vertex_neighbor_vertices[0][pindex + 1]]

    def sample_delaunay_topo(self, ind):
        self.Xi_ind_topo_i = []

        # Find the position of the sample in the component Gabrial chain
        G_ind = self.find_neighbors_delaunay(ind, self.Tri)

        # Find finite deference between each point
        for g_i in G_ind:
        #    logging.info('self.F[g_i] ={}'.format(self.F[g_i]))
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
        self.minimizer_pool = []
        # TODO: Can easily be parralized
    #    logging.info('self.fn = {}'.format(self.fn))
    #    logging.info('self.n = {}'.format(self.n))
    #    logging.info('numpy.shape(self.C)'
    #                 ' = {}'.format(numpy.shape(self.C)))
        for ind in range(self.fn):
        #    logging.info('ind = {}'.format(ind))
            Min_bool = self.sample_delaunay_topo(ind)
            if Min_bool:
                self.minimizer_pool.append(ind)

        self.minimizer_pool_F = self.F[self.minimizer_pool]

        # Sort to find minimum func value in min_pool
        self.sort_min_pool()
        if self.disp:
            logging.info('self.minimizer_pool = {}'.format(self.minimizer_pool))
        if not len(self.minimizer_pool) == 0:
            self.X_min = self.C[self.minimizer_pool]
        else:
            self.X_min = []  # Empty pool breaks main routine
        return self.X_min

if __name__ == '__main__':
    import doctest
    #doctest.testmod()