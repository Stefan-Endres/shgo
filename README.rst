.. image:: https://travis-ci.org/Stefan-Endres/shgo.svg?branch=master
    :target: https://travis-ci.org/Stefan-Endres/shgo
    
.. image:: https://coveralls.io/repos/Stefan-Endres/shgo/badge.png?branch=master
    :target: https://coveralls.io/r/Stefan-Endres/shgo?branch=master

Description
-----------

Finds the global minimum of a function using simplicial homology global
optimisation. Appropriate for solving general purpose NLP and blackbox
optimisation problems to global optimality (low dimensional problems).
The general form of an optimisation problem is given by:

::

    minimize f(x) subject to

    g_i(x) >= 0,  i = 1,...,m
    h_j(x)  = 0,  j = 1,...,p

where x is a vector of one or more variables. ``f(x)`` is the objective
function ``R^n -> R`` ``g_i(x)`` are the inequality constraints.
``h_j(x)`` are the equality constrains.


Installation
------------
.. code::

    pip install shgo


Examples
--------

First consider the problem of minimizing the Rosenbrock function. This
function is implemented in ``rosen`` in ``scipy.optimize``

.. code:: python

    >>> from scipy.optimize import rosen
    >>> from shgo import shgo
    >>> bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
    >>> result = shgo(rosen, bounds)
    >>> result.x, result.fun
    (array([ 1.,  1.,  1.,  1.,  1.]), 2.9203923741900809e-18)

Note that bounds determine the dimensionality of the objective function
and is therefore a required input, however you can specify empty bounds
using ``None`` or objects like numpy.inf which will be converted to
large float numbers.

.. code:: python

    >>> bounds = [(None, None), ]*4
    >>> result = shgo(rosen, bounds)
    >>> result.x
    array([ 0.99999851,  0.99999704,  0.99999411,  0.9999882 ])

Next we consider the Eggholder function, a problem with several local
minima and one global minimum. We will demonstrate the use of arguments
and the capabilities of shgo.
(https://en.wikipedia.org/wiki/Test\_functions\_for\_optimization)

.. code:: python

    >>> from shgo import shgo
    >>> import numpy as np
    >>> def eggholder(x):
    ...     return (-(x[1] + 47.0)
    ...             * np.sin(np.sqrt(abs(x[0]/2.0 + (x[1] + 47.0))))
    ...             - x[0] * np.sin(np.sqrt(abs(x[0] - (x[1] + 47.0))))
    ...             )
    ...
    >>> bounds = [(-512, 512), (-512, 512)]

shgo has two built-in low discrepancy sampling sequences. First we will
input 30 initial sampling points of the Sobol sequence

.. code:: python

    >>> result = shgo(eggholder, bounds, n=30, sampling_method='sobol')
    >>> result.x, result.fun
    (array([ 512.    ,  404.23180542]), -959.64066272085051)

``shgo`` also has a return for any other local minima that was found,
these can be called using:

.. code:: python

    >>> result.xl
    array([[ 512., 404.23180542], [ 283.07593402, -487.12566542], [-294.66820039, -462.01964031], [-105.87688985,  423.15324143], [-242.97923629,  274.38032063], [-506.25823477, 6.3131022 ], [-408.71981195, -156.10117154], [150.23210485,  301.31378508], [91.00922754, -391.28375925], [ 202.8966344, -269.38042147], [361.66625957, -106.96490692], [-219.40615102, -244.06022436], [ 151.59603137, -100.61082677]])
    >>> result.funl
    array([-959.64066272, -718.16745962, -704.80659592, -565.99778097, -559.78685655, -557.36868733, -507.87385942, -493.9605115, -426.48799655, -421.15571437, -419.31194957, -410.98477763, -202.53912972])

These results are useful in applications where there are many global
minima and the values of other global minima are desired or where the
local minima can provide insight into the system such are for example
morphologies in physical chemistry [5]

Now suppose we want to find a larger number of local minima, this can be
accomplished for example by increasing the amount of sampling points or
the number of iterations. We'll increase the number of sampling points
to 60 and the number of iterations to 3 increased from the default 100
for a total of 60 x 3 = 180 initial sampling points.

.. code:: python

    >>> result_2 = shgo(eggholder, bounds, n=60, iters=3, sampling_method='sobol')
    >>> len(result.xl), len(result_2.xl)
    (13, 33)

Note that there is a difference between specifying arguments for ex.
``n=180, iters=1`` and ``n=60, iters=3``. In the first case the
promising points contained in the minimiser pool is processed only once.
In the latter case it is processed every 60 sampling points for a total
of 3 times.

To demonstrate solving problems with non-linear constraints consider the
following example from Hock and Schittkowski problem 73 (cattle-feed)
[4]:

::

    minimize: f = 24.55 * x_1 + 26.75 * x_2 + 39 * x_3 + 40.50 * x_4

    subject to: 2.3 * x_1 + 5.6 * x_2 + 11.1 * x_3 + 1.3 * x_4 - 5      >= 0,

                12 * x_1 + 11.9 * x_2 + 41.8 * x_3 + 52.1 * x_4 - 21
                    -1.645 * sqrt(0.28 * x_1**2 + 0.19 * x_2**2 +
                                  20.5 * x_3**2 + 0.62 * x_4**2)        >= 0,

                x_1 + x_2 + x_3 + x_4 - 1                               == 0,

                1 >= x_i >= 0 for all i

Approx. Answer [4]: f([0.6355216, -0.12e-11, 0.3127019, 0.05177655]) =
29.894378

.. code:: python

        >>> from scipy.optimize import shgo
        >>> import numpy as np
        >>> def f(x):  # (cattle-feed)
        ...     return 24.55*x[0] + 26.75*x[1] + 39*x[2] + 40.50*x[3]
        ...
        >>> def g1(x):
        ...     return 2.3*x[0] + 5.6*x[1] + 11.1*x[2] + 1.3*x[3] - 5  # >=0
        ...
        >>> def g2(x):
        ...     return (12*x[0] + 11.9*x[1] +41.8*x[2] + 52.1*x[3] - 21
        ...             - 1.645 * np.sqrt(0.28*x[0]**2 + 0.19*x[1]**2
        ...                             + 20.5*x[2]**2 + 0.62*x[3]**2)
        ...             ) # >=0
        ...
        >>> def h1(x):
        ...     return x[0] + x[1] + x[2] + x[3] - 1  # == 0
        ...
        >>> cons = ({'type': 'ineq', 'fun': g1},
        ...         {'type': 'ineq', 'fun': g2},
        ...         {'type': 'eq', 'fun': h1})
        >>> bounds = [(0, 1.0),]*4
        >>> res = shgo(f, bounds, iters=2, constraints=cons)
        >>> res
             fun: 29.894378159142136
            funl: array([ 29.89437816])
         message: 'Optimization terminated successfully.'
            nfev: 119
             nit: 2
           nlfev: 40
           nljev: 0
         success: True
               x: array([  6.35521569e-01,   1.13700270e-13,   3.12701881e-01,
                 5.17765506e-02])
              xl: array([[  6.35521569e-01,   1.13700270e-13,   3.12701881e-01,
                  5.17765506e-02]])
        >>> g1(res.x), g2(res.x), h1(res.x)
        (-5.0626169922907138e-14, -2.9594104944408173e-12, 0.0)


Parameters
----------

::

    func : callable

The objective function to be minimized. Must be in the form
``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
and ``args`` is a tuple of any additional fixed parameters needed to
completely specify the function.

--------------

::

    bounds : sequence

Bounds for variables. ``(min, max)`` pairs for each element in ``x``,
defining the lower and upper bounds for the optimizing argument of
``func``. It is required to have ``len(bounds) == len(x)``.
``len(bounds)`` is used to determine the number of parameters in ``x``.
Use ``None`` for one of min or max when there is no bound in that
direction. By default bounds are ``(None, None)``.

--------------

::

    args : tuple, optional

Any additional fixed parameters needed to completely specify the
objective function.

--------------

::

    constraints : dict or sequence of dict, optional

Constraints definition. Function(s) R^n in the form g(x) <= 0 applied as
g : R^n -> R^m h(x) == 0 applied as h : R^n -> R^p

Each constraint is defined in a dictionary with fields:

::

    * type : str
        Constraint type: 'eq' for equality, 'ineq' for inequality.
    * fun : callable
        The function defining the constraint.
    * jac : callable, optional
        The Jacobian of `fun` (only for SLSQP).
    * args : sequence, optional
        Extra arguments to be passed to the function and Jacobian.

Equality constraint means that the constraint function result is to be
zero whereas inequality means that it is to be non-negative. Note that
COBYLA only supports inequality constraints.

NOTE: Only the COBYLA and SLSQP local minimize methods currently support
constraint arguments. If the ``constraints`` sequence used in the local
optimization problem is not defined in ``minimizer_kwargs`` and a
constrained method is used then the global ``constraints`` will be used.
(Defining a ``constraints`` sequence in ``minimizer_kwargs`` means that
``constraints`` will not be added so if equality constraints and so
forth need to be added then the inequality functions in ``constraints``
need to be added to ``minimizer_kwargs`` too).

--------------

::

    n : int, optional

Number of sampling points used in the construction of the simplicial
complex. Note that this argument is only used for ``sobol`` and other
arbitrary sampling\_methods.

--------------

::

    iters : int, optional

Number of iterations used in the construction of the simplicial complex.

--------------

::

    callback : callable, optional

Called after each iteration, as ``callback(xk)``, where ``xk`` is the
current parameter vector.

--------------

::

    minimizer_kwargs : dict, optional

Extra keyword arguments to be passed to the minimizer
``scipy.optimize.minimize`` Some important options could be:

::

    * method : str
        The minimization method (e.g. ``SLSQP``)
    * args : tuple
        Extra arguments passed to the objective function (``func``) and
        its derivatives (Jacobian, Hessian).

    options : {ftol: 1e-12}

--------------

::

    options : dict, optional

A dictionary of solver options. Many of the options specified for the
global routine are also passed to the scipy.optimize.minimize routine.
The options that are also passed to the local routine are marked with an
(L)

Stopping criteria, the algorithm will terminate if any of the specified
criteria are met. However, the default algorithm does not require any to
be specified:

::

    * maxfev : int (L)
        Maximum number of function evaluations in the feasible domain.
        (Note only methods that support this option will terminate
        the routine at precisely exact specified value. Otherwise the
        criterion will only terminate during a global iteration)
    * f_min
        Specify the minimum objective function value, if it is known.
    * f_tol : float
        Precision goal for the value of f in the stopping
        criterion. Note that the global routine will also
        terminate if a sampling point in the global routine is
        within this tolerance.
    * maxiter : int
        Maximum number of iterations to perform.
    * maxev : int
        Maximum number of sampling evaluations to perform (includes
        searching in infeasible points).
    * maxtime : float
        Maximum processing runtime allowed
    * maxhgrd : int
        Maximum homology group rank differential. The homology group of the
        objective function is calculated (approximately) during every
        iteration. The rank of this group has a one-to-one correspondence
        with the number of locally convex subdomains in the objective
        function (after adequate sampling points each of these subdomains
        contain a unique global minima). If the difference in the hgr is 0
        between iterations for ``maxhgrd`` specified iterations the
        algorithm will terminate.

Objective function knowledge:

::

    * symmetry : bool
       Specify True if the objective function contains symmetric variables.
       The search space (and therfore performance) is decreased by O(n!).

Algorithm settings:

::

    * minimize_every_iter : bool
        If True then promising global sampling points will be passed to a
        local minimisation routine every iteration. If False then only the
        final minimiser pool will be run.
    * local_iter : int
        Only evaluate a few of the best minimiser pool candiates every
        iteration. If False all potential points are passed to the local
        minimsation routine.
    * infty_constraints: bool
        If True then any sampling points generated which are outside will
        the feasible domain will be saved and given an objective function
        value of numpy.inf. If False then these points will be discarded.
        Using this functionality could lead to higher performance with
        respect to function evaluations before the global minimum is found,
        specifying False will use less memory at the cost of a slight
        decrease in performance.

Feedback:

::

    * disp : bool (L)
        Set to True to print convergence messages.

--------------

::

    sampling_method : str or function, optional

Current built in sampling method options are ``sobol`` and
``simplicial``. The default ``simplicial`` uses less memory and provides
the theoretical guarantee of convergence to the global minimum in finite
time. The ``sobol`` method is faster in terms of sampling point
generation at the cost of higher memory resources and the loss of
guaranteed convergence. It is more appropriate for most "easier"
problems where the convergence is relatively fast. User defined sampling
functions must accept two arguments of ``n`` sampling points of
dimension ``dim`` per call and output an array of s ampling points with
shape ``n x dim``. See SHGO.sampling\_sobol for an example function.

Returns
-------

::

    res : OptimizeResult

The optimization result represented as a ``OptimizeResult`` object.
Important attributes are: ``x`` the solution array corresponding to the
global minimum, ``fun`` the function output at the global solution,
``xl`` an ordered list of local minima solutions, ``funl`` the function
output at the corresponding local solutions, ``success`` a Boolean flag
indicating if the optimizer exited successfully, ``message`` which
describes the cause of the termination, ``nfev`` the total number of
objective function evaluations including the sampling calls, ``nlfev``
the total number of objective function evaluations culminating from all
local search optimisations, ``nit`` number of iterations performed by
the global routine.

Notes
-----

Global optimisation using simplicial homology global optimisation [1].
Appropriate for solving general purpose NLP and blackbox optimisation
problems to global optimality (low dimensional problems).

In general, the optimisation problems are of the form:

::

    minimize f(x) subject to

    g_i(x) >= 0,  i = 1,...,m
    h_j(x)  = 0,  j = 1,...,p

where x is a vector of one or more variables. ``f(x)`` is the objective
function ``R^n -> R`` ``g_i(x)`` are the inequality constraints.
``h_j(x)`` are the equality constrains.

Optionally, the lower and upper bounds for each element in x can also be
specified using the ``bounds`` argument.

While most of the theoretical advantages of shgo are only proven for
when ``f(x)`` is a Lipschitz smooth function. The algorithm is also
proven to converge to the global optimum for the more general case where
``f(x)`` is non-continuous, non-convex and non-smooth iff the default
sampling method is used [1].

The local search method may be specified using the ``minimizer_kwargs``
parameter which is inputted to ``scipy.optimize.minimize``. By default
the ``SLSQP`` method is used. In general it is recommended to use the
``SLSQP`` or ``COBYLA`` local minimization if inequality constraints are
defined for the problem since the other methods do not use constraints.

The ``sobol`` method points are generated using the Sobol (1967) [2]
sequence. The primitive polynomials and various sets of initial
direction numbers for generating Sobol sequences is provided by [3] by
Frances Kuo and Stephen Joe. The original program sobol.cc (MIT) is
available and described at http://web.maths.unsw.edu.au/~fkuo/sobol/
translated to Python 3 by Carl Sandrock 2016-03-31.

References
----------

1. Endres, SC (2017) "A simplicial homology algorithm for Lipschitz
   optimisation".

2. Sobol, IM (1967) "The distribution of points in a cube and the
   approximate evaluation of integrals", USSR Comput. Math. Math. Phys.
   7, 86-112.

3. Joe, SW and Kuo, FY (2008) "Constructing Sobol sequences with better
   two-dimensional projections", SIAM J. Sci. Comput. 30, 2635-2654.

4. Hoch, W and Schittkowski, K (1981) "Test examples for nonlinear
   programming codes", Lecture Notes in Economics and mathematical
   Systems, 187. Springer-Verlag, New York.
   http://www.ai7.uni-bayreuth.de/test\_problem\_coll.pdf

5. Wales, DJ (2015) "Perspective: Insight into reaction coordinates and
   dynamics from the potential energy landscape", Journal of Chemical
   Physics, 142(13), 2015.

.. |Build Status| image:: https://travis-ci.org/Stefan-Endres/shgo.svg?branch=master
   :target: https://travis-ci.org/Stefan-Endres/shgo
.. |Build Status| image:: https://coveralls.io/repos/Stefan-Endres/shgo/badge.png?branch=master
   :target: https://coveralls.io/r/Stefan-Endres/shgo?branch=master
