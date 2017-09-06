#!/usr/bin/env python
"""
go_bench of linearly constrained functions
Run examples:
$ python go_bench.py -s shgo
$ python go_bench.py -s shgo tgo de bh
$ python go_bench.py -s shgo -debug True
"""

from __future__ import division, print_function, absolute_import
import numpy
import scipy.optimize
from _shgo import *
#from _shgo_sobol import shgo as shgo_sobol
from _tgo import *
from go_funcs.go_benchmark import Benchmark
import go_funcs_lc
import inspect
import time
import logging
import sys

logger = logging.getLogger()
logger.disabled = 0

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--solvers', nargs='+',
                    help='List of Global optimization routines to benchmark'
                         'current implemenations are: '
                         'shgo tgo de bh')

parser.add_argument('-debug', nargs=1, type=bool,
                    default=False,
                    help='Raise logging info level to logging.DEBUG')

args = parser.parse_args()


excluded = [
            'Horst6', # <--- Not correctly defined
            'Horst5', 'Horst7', 'Hs036', 'S250',  # <--- Slow
            #"Bukin06",  # <--- Working, but fail on all solvers + high nfev
            'Benchmark',  # Not a GO function
            ]

class GoRunner:
    def __init__(self, solvers=['shgo']):
        """
        Initiate with the list solvers to run, not implemented yet:
        solvers=['TGO', 'DE', 'BH']
        """

        self.solvers = solvers
        self.solvers_wrap = {'shgo': self.run_shgo,
                             'shgo_sobol': self.run_shgo_sobol,
                             'tgo': self.run_tgo,
                             'de': self.run_differentialevolution,
                             'bh': self.run_basinhopping}

        self.results = {'All': {}}
        self.results['Average'] = {}
        for solver in self.solvers:
            self.results['All'][solver] = {'nfev': 0,
                                           'iter': 0,
                                           # Number of function evaluations or iterations
                                           'nlmin': 0,
                                           'nulmin': 0,
                                           # Number of local minima
                                           'success rate': 0,
                                           # Total success rate over all functions
                                           'success count': 0,
                                           'eval count': 0,
                                           'total runtime': 0}

            self.results['Average'][solver] = {}

    def run_func(self, FuncClass, name):
        """
        Run the for all solvers
        """
        # Store the function class and its attributes here:
        self.function = FuncClass
        self.name = name
        self.results[name] = {}
        for solver in self.solvers:
            self.solver = solver
            self.solvers_wrap[self.solver]()
            self.update_results()

    def run_shgo(self):
        self.function.nfev = 0

        # Add exception handling here?
        if self.function.g == None:
            kwarg_args = {'options': None}
        else:
            kwarg_args = {'g_cons': self.function.g}


        success_l = False
        iters = 1
        while not success_l:
            self.function.nfev = 0
            t0 = time.time()

            res = shgo(self.function.fun, self.function._bounds,
                       iter=iters,
                       sampling_method='simplicial',
                       **kwarg_args)
            #print(res)
            nfev = self.function.nfev
            #nfev = res.nfev
            runtime = time.time() - t0
            if res.success is False:
                iters += 1
                continue
            try:
                #if self.function.success(res.x, tol=0.01):
                if abs(res.fun - self.function.fglob) < 0.01:
                    success_l = True
                else:
                    print(f'iter = {iters}')
                    print(f'nfev = {nfev}')
                    iters += 1
            except ValueError:
                iters += 1

# Prepare Return dictionary
        self.results[self.name]['shgo'] = \
            {'nfev': nfev,#self.function.nfev,
             'iter': int(iters),
             'nlmin': len(res.xl),
             'nulmin': self.unique_minima(res.xl),
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': int(self.function._dimensions),
             'name': 'shgo-simplicial'
             }
        return

    def run_shgo_sobol(self):
        self.function.nfev = 0

        # Add exception handling here?
        if self.function.g == None:
            kwarg_args = {'options': None}
        else:
            kwarg_args = {'g_cons': self.function.g}

        success_l = False
        n = self.function._dimensions + 1
        if self.name == 'Hs038':
            n = 6  # Strange Qhull error on self.function._dimensions + 1 points

        while not success_l:
            self.function.nfev = 0
            t0 = time.time()

            res = shgo(self.function.fun, self.function._bounds,
                       n=n,
                       sampling_method='sobol',
                       **kwarg_args)
            # print(res)
            nfev = self.function.nfev
            # nfev = res.nfev
            runtime = time.time() - t0
            if res.success is False:
                n += 1
                continue
            try:
                # if self.function.success(res.x, tol=0.01):
                if abs(res.fun - self.function.fglob) < 0.01:
                    success_l = True
                    if (res.fun - self.function.fglob) < -1e-6:
                        print("LOWER FUNCTION VALUE FOUND = {res.fun}")
                else:
                    print(f'n = {n}')
                    print(f'nfev = {nfev}')
                    n += 1
            except ValueError:
                n += 1

                # Prepare Return dictionary
        self.results[self.name]['shgo_sobol'] = \
            {'nfev': nfev,  # self.function.nfev,
             'iter': n,  # Total sampling including infeasible regions
             'nlmin': len(res.xl),
             'nulmin': self.unique_minima(res.xl),
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': int(self.function._dimensions),
             'name': 'shgo-sobol'
             }
        return

    def run_differentialevolution(self):
        """
        Do an optimization run for differential_evolution
        """
        self.function.nfev = 0

        t0 = time.time()

        res = scipy.optimize.differential_evolution(self.function.fun,
                                     self.function._bounds,
                                     popsize=20)

        runtime = time.time() - t0
        self.results[self.name]['de'] = \
            {'nfev': self.function.nfev,
             'nlmin': 0,  #TODO: Look through res object
             'nulmin': 0,
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions,
             'name': 'de'
            }

    def run_basinhopping(self):
        """
        Do an optimization run for basinhopping
        """
        self.function.nfev = 0

        if 0:  #TODO: Find out if these are important:
            kwargs = self.minimizer_kwargs
            if hasattr(self.fun, "temperature"):
                kwargs["T"] = self.function.temperature
            if hasattr(self.fun, "stepsize"):
                kwargs["stepsize"] = self.function.stepsize

        minimizer_kwargs = {"method": "L-BFGS-B"}

        x0 = self.function.initial_vector()

        # basinhopping - no gradient
        minimizer_kwargs['jac'] = False
        self.function.nfev = 0

        t0 = time.time()

        res = scipy.optimize.basinhopping(
            self.function.fun, x0,
            #accept_test=self.accept_test,
            minimizer_kwargs=minimizer_kwargs,
            #**kwargs
            )

        # Prepare Return dictionary
        runtime = time.time() - t0
        self.results[self.name]['bh'] = \
            {'nfev': self.function.nfev,
             'nlmin': 0,  #TODO: Look through res object
             'nulmin': 0,
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions,
             'name': 'bh'
            }

    def run_tgo(self):
        """
        Do an optimization run for tgo
        """
        self.function.nfev = 0

        t0 = time.time()
        # Add exception handling here?
        res = tgo(self.function.fun, self.function._bounds)
                  #, n=5000)
        runtime = time.time() - t0

        # Prepare Return dictionary
        self.results[self.name]['tgo'] = \
            {'nfev': self.function.nfev,
             'nlmin': len(res.xl),  # TODO: Find no. unique local minima
             'nulmin': self.unique_minima(res.xl),
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions,
             'name': 'tgo'
             }
        return

    def update_results(self):
        # Update global results let nlmin for DE and BH := 0
        self.results['All'][self.solver]['name'] = self.solver
        self.results['All'][self.solver]['nfev'] += \
            self.results[self.name][self.solver]['nfev']
        self.results['All'][self.solver]['iter'] += \
            self.results[self.name][self.solver]['iter']
        self.results['All'][self.solver]['nlmin'] += \
            self.results[self.name][self.solver]['nlmin']
        self.results['All'][self.solver]['nulmin'] += \
            self.results[self.name][self.solver]['nulmin']
        self.results['All'][self.solver]['eval count'] += 1
        self.results['All'][self.solver]['success count'] += \
            self.results[self.name][self.solver]['success']
        self.results['All'][self.solver]['success rate'] = \
            (100.0 * self.results['All'][self.solver]['success count']
             /float(self.results['All'][self.solver]['eval count']))
        self.results['All'][self.solver]['total runtime'] += \
            self.results[self.name][self.solver]['runtime']
        self.results['All'][self.solver]['ndim'] = int(0)
        return

    def unique_minima(self, xl, tol=1e-5):
        """
        Returns the number of points in `xl` that are unique to the default
        tolerance of numpy.allclose
        """
        import itertools

        uniql = len(xl)
        if uniql == 1:
            uniq = 1
        else:
            xll = len(xl)
            flag = []
            for i in range(xll):
                for k in range(i + 1, xll):
                    if numpy.allclose(xl[i], [xl[k]],
                                      rtol=tol,
                                      atol=tol):
                        flag.append(k)

            uniq = uniql - len(numpy.unique(numpy.array(flag)))
        return uniq


if __name__ == '__main__':
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.solvers is None:
        GR = GoRunner(solvers=['shgo'])
                             # , 'tgo'])
    else:
        GR = GoRunner(solvers=args.solvers)

    for name, obj in inspect.getmembers(go_funcs_lc):
        if inspect.isclass(obj):
            logging.info(obj)
            logging.info(name)
            if name not in excluded:
                FuncClass = obj()
                #try:
                GR.run_func(FuncClass, name)
                #except:
                #    pass

    # int being converted because 'All' is NaN
    #for key in GR.results.keys():
    #    if key is not "All":
    #        for solver in GR.results[key].keys():
    #            GR.results[key][solver]['ndim'] = int(GR.results[key][solver]['ndim'])

    # Process average values and print out total performance

    for solver in GR.results['All'].keys():
        print("=" * 60)
        print("Results for {}".format(solver))
        print("="*30)
        GR.results['All'][solver]['average runtime'] = (GR.results['All'][solver]['total runtime']
                                                /GR.results['All'][solver]['eval count'])
        GR.results['All'][solver]['average nfev'] = (GR.results['All'][solver]['nfev']
                                                /GR.results['All'][solver]['eval count'])


        GR.results['Average'][solver]['runtime'] = GR.results['All'][solver]['average runtime']
        GR.results['Average'][solver]['nfev'] = int(GR.results['All'][solver]['average nfev'])
        GR.results['Average'][solver]['ndim'] = int(0)
        GR.results['Average'][solver]['nlmin'] = (GR.results['All'][solver]['nlmin']
                                                  / GR.results['All'][solver]['eval count']
                                                  )
        GR.results['Average'][solver]['nulmin'] = (GR.results['All'][solver]['nulmin']
                                                  / GR.results['All'][solver]['eval count']
                                                  )
        GR.results['Average'][solver]['name'] = solver

        for key in GR.results['All'][solver].keys():
            print(key + ": " + str(GR.results['All'][solver][key]))
            #print(GR.results['All'][solver])
        print("=" * 60)
        print(f"""GR.results['All'][{solver}]['average nfev']"""
               f""" = {GR.results['All'][solver]['average nfev']}""")

        print(f"""GR.results['All'][{solver}]['nlmin']"""
               f""" = {GR.results['All'][solver]['nlmin']}""")

        print(f"""GR.results['All'][{solver}]['nulmin']"""
               f""" = {GR.results['All'][solver]['nulmin']}""")

        GR.results['All'][solver]['nlmin'] = int(GR.results['Average'][solver]['nlmin'])
        GR.results['Average'][solver]['nlmin'] = int(GR.results['Average'][solver]['nlmin'])
        GR.results['All'][solver]['nulmin'] = int(GR.results['Average'][solver]['nulmin'])
        GR.results['Average'][solver]['nulmin'] = int(GR.results['Average'][solver]['nulmin'])
        import json
        with open('results/results_lc.json', 'w') as fp:
            json.dump(GR.results, fp)
