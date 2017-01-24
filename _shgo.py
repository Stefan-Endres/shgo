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
#from . import __file__

try:
    pass
    #from multiprocessing_on_dill import Pool
except ImportError:
    from multiprocessing import Pool

def shgo(func, bounds, args=(), g_cons=None, g_args=(), n=30, iter=None,
         callback=None, minimizer_kwargs=None, options=None,
         multiproc=False, crystal_mode=False, sampling_method='simplicial'):
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
    # Initiate TGO class
    SHc= SHGO(func, bounds, args=args, g_cons=g_cons, g_args=g_args, n=n,
              iter=iter, callback=callback, minimizer_kwargs=minimizer_kwargs,
              options=options, multiproc=multiproc)

    # Generate sampling points
    if SHc.disp:
        print('Generating sampling points')

    # Construct directed complex.
    SHc.construct_complex_simplicial()

    if not SHc.break_routine:
        if SHc.disp:
            print("Succesfully completed construction of complex.")

        # Minimise the pool of minisers with local minimisation methods
        SHc.minimise_pool()

    # Sort results and build the global return object
    SHc.sort_result()

    # Confirm the routine ran succesfully
    if not SHc.break_routine:
        SHc.res.message = 'Optimization terminated successfully.'
        SHc.res.success = True

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
        self.g_cons = g_cons
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
            if 'symmetry' in options:
                self.symmetry = True
            else:
                self.symmetry = False
            if 'min_iter' in options:
                self.min_iter = options['min_iter']
            if 'min_hgrd' in options:
                self.min_hgrd = options['min_hgrd']
            else:
                self.min_hgrd = 0


            self.options = None

        else:
            self.symmetry = False
            self.crystal_iter = 1

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

        # Initiate storate objects used in alorithm classes
        self.x_min_glob = []
        self.fun_min_glob = []

        # Initialize return object
        self.res = scipy.optimize.OptimizeResult()
        self.res.nfev = 0  # Include each sampling point as func evaluation
        self.res.nlfev = 0  # Local function evals for all minimisers
        self.res.nljev = 0  # Local jacobian evals for all minimisers

    def construct_complex_simplicial(self):
        if self.disp:
            print('Building initial complex')

        self.HC = Complex(self.dim, self.func, self.args,
                          self.symmetry, self.bounds, self.g_cons, self.g_args)

        if self.disp:
            print('Splitting first generation')

        self.HC.C0.hgr = self.HC.C0.homology_group_rank()#split_generation()
        print('self.HC.C0.hg_ns = {}'.format(self.HC.C0.hg_n))


        #TODO: If minimum subspace tolerance is specified split a finite
        #      amount of generations
        if 0:
            # while cell_space < tolerance:
            for Cell in self.HC.H[self.HC.gen]:
                Cell.homology_group_rank()

            self.HC.split_generation()

             # Find the smallest subspace covered by a cell
            sup = self.HC.H[self.HC.gen][0].suprenum
            origin = self.HC.H[self.HC.gen][0].origin
            cell_space = numpy.linalg.norm(numpy.array(sup)
                                           - numpy.array(origin))

            print('cell_space = {}'.format(cell_space))

        #TODO: We can also implement a maximum tolerance to ensure the algorithm
        #      terminates in finite time for a function with infinite minima
        else:
            self.HC.C0.homology_group_rank()

            # self.HC[1].homology_group_rank()

            # print('complex_homology_group_rank(self) = {}'.format(
            #     self.HC.complex_homology_group_rank()
            # ))
            #  print('HC.hgrd = {}'.format(self.HC.hgrd))


            gen = 1
            Stop = False
            hgr_diff_iter = 1  # USER INPUT?
            #hgr_diff_iter = 1  # USER INPUT?

            # Split first generation
            self.HC.split_generation()
            #self.HC.split_generation() #TODO REMOVE THIS
            #gen +=1
            while not Stop:
                #self.HC.split_generation()
                print('TEST')

                # Split all cells except for those with hgr_d < 0
                try:
                    Cells_in_gen = self.HC.H[gen]
                except IndexError:  # No cells in index range
                    logging.warning("INDEXERROR")
                    pass

                for Cell in Cells_in_gen:
                    print('TEST')
                    print('gen = {}'.format(gen))
                    Cell.homology_group_rank()
                    if Cell.homology_group_differential() >= 0:
                        self.HC.sub_generate_cell(Cell, gen + 1)



                print('TEST')
                # Find total complex group:
                self.HC.complex_homology_group_rank()
                logging.info('self.HC.hgrd = {}'.format(self.HC.hgrd))
                if self.HC.hgrd <= 0:
                    hgr_diff_iter -= 1
                    if hgr_diff_iter == 0:
                        Stop = True

                # Increase generation counter
                gen +=1

                # Homology group iterations with no tolerance:
        #self.max_hgr_h = -1 # TODO: THIS WILL BE AN OPTIONAL INPUT

        #TODO: Define a permutaiton function that calls itself after a split
               # for every cell with a non-zero differential
        if 0:
            hgr_h = self.HC.C0.hg_n
            for Cell in self.HC.H[1]:
                Cell.homology_group_rank()
                hgr_h += Cell.hg_n

            for Cell in self.HC.H[1]:
                Cell.p_hgr_h = hgr_h
            for Cell in self.HC.H[self.HC.gen]:
                pass

        # homology_group_rank(self)
        # homology_group_differential(self)

        # Cell.p_hgr_h = Cell.p_hgr_h + Cell.hgd

        return


        if 0:
            # Initiate complex
            self.n = self.dim + 1
            HC = Complex(self.dim)
            self.HC = HC

            if self.symmetry:
                C = self.HC.n_cube(self.dim, symmetry=True)
            else:
                C = self.HC.n_cube(self.dim)

            self.HC.initial_vertices(C, self.dim)
            Ci = self.HC.index_simplices(C)
            self.C = self.HC.V[0]

            for i in range(len(self.HC.V) - 1):
                self.C = numpy.vstack((self.C, self.HC.V[i + 1]))

            # stretch values
            for i in range(len(self.bounds)):
                self.C[:, i] = (self.C[:, i] *
                                (self.bounds[i][1] - self.bounds[i][0])
                                + self.bounds[i][0])

            # Stretch values in complex class
            self.HC.V = []
            for vi in range(numpy.shape(self.C)[0]):
                self.HC.V.append(self.C[vi, :])

            self.Ci = Ci
            # Containers
            self.X_min_all = []
            self.minimizer_pool_F_all = []

            grow_complex = True
            homology_group = 0
            homology_group_prev = 0
            #iter = 2  # Max iterations with no pool growth
            iter = self.crystal_iter
            while grow_complex:
                Ci_new = self.HC.split_generation(self.HC.Ci, self.HC.V,
                                                  build_complex_array=False)

                #print(Ci_new)
                self.Ci = Ci_new
                self.HC.connected_vertices(0, Ci_new)
                #print(HC.V)
                self.C = HC.V[0]

                # Stack self.C:
                for i in range(len(HC.V)-1):
                    self.C = numpy.vstack((self.C, HC.V[i + 1]))

                if self.g_cons is not None:
                    self.sampling_subspace()
                    self.fn = numpy.shape(self.C)[0]
                else:
                    self.fn = numpy.shape(self.C)[0]

                # Sort remaining samples
                self.sorted_samples()

                # Find objective function references
                self.fun_ref()

                # Build minimiser pool
                # DIMENSIONS self.dim
                if self.dim < 2: #UNTESTED
                    self.ax_subspace()
                    self.surface_topo_ref()
                    self.X_min = self.minimizers()

                else:  # Multivariate functions.
                    self.X_min = self.simplex_minimizers()

               # Continue loop if pool is zero to iterate
                print('self.X_min = {}'.format(self.X_min ))
                if len(self.minimizer_pool) == 0:
                    homology_group = 0
                    continue
                else:
                    homology_group = len(self.minimizer_pool)
                    print('homology_group = {}'.format(homology_group))
                    print('homology_group_prev = {}'.format(homology_group_prev))
                    if homology_group > homology_group_prev:
                        homology_group_prev = homology_group
                        continue

                print(iter)
                if iter > 0:
                    iter -= 1
                else:
                    # Stop growth if no new minimisers found:
                    grow_complex = False

            self.res.nfev = self.fn
            print('self.res.nfev = {}'.format(self.res.nfev))
            self.processed_n = self.fn
            self.n = numpy.shape(self.C)[0]

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
            self.iter = force_iter

        while not self.stopiter:
            if self.iter is not None:  # Note first iteration is outside loop
                logging.info('SHGO.iter = {}'.format(self.iter))
                self.iter -= 1
                if __name__ == '__main__':
                    if self.iter == 0:
                        self.stopiter = True
                        break
                    #TODO: Test usage of iterative features

            if numpy.shape(self.X_min)[0] == 0:
                self.stopiter = True
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
        except (IndexError, TypeError):
            self.fun_min_glob.append(lres.fun)

        return lres


    def simplex_minimizers(self):
        """
        Returns the indexes of all minimizers
        """
        self.minimizer_pool = []
        # TODO: Can easily be parralized

        for ind in range(self.fn):
            Min_bool = self.sample_simplex_topo(ind)
            if Min_bool:
                self.minimizer_pool.append(ind)

        self.minimizer_pool_F = self.F[self.minimizer_pool]

        # Sort to find minimum func value in min_pool
        self.sort_min_pool()
        logging.info('self.minimizer_pool = {}'.format(self.minimizer_pool))
        if not len(self.minimizer_pool) == 0:
            self.X_min = self.C[self.minimizer_pool]
        else:
            self.X_min = []

        return self.X_min

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


if __name__ == '__main__':
    import doctest
    #doctest.testmod()
    from numpy import *

    '''
    Temporary dev work:
    '''
    # Eggholder
    if 1:
        N = 2
        def fun(x, *args):
            return x[0] ** 2 + x[1] ** 2 + 25 * (sin(x[0]) ** 2 + sin(x[1]) ** 2)


        bounds = list(zip([-5.0] * N, [5.0] * N))

        SHGOc3 = SHGO(fun, bounds)
        SHGOc3.construct_complex_simplicial()

        SHGOc3.HC.plot_complex()

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


    if 0:
        def f(x):  # sin
            return numpy.sin(x)
        bounds = [(0, 5)]

        SHGOc2 = SHGO(f, bounds)
        SHGOc2.construct_complex_simplicial()


    #print(SHGOc1.disp)

    #SHGOc2 = SHGO(f, bounds,
    #              crystal_mode=True, sampling_method='simplicial')
    #SHGOc2.construct_complex_simplicial()


    if 0:
        print(shgo(f, bounds,
                   crystal_mode=True,
                   sampling_method='simplicial'))
        print('='*100)
        print('Sobol shgo:')
        print('===========')
        print(shgo(f, bounds))