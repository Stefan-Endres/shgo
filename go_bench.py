#!/usr/bin/env python
"""
Run examples:
$ python go_bench.py -s shgo
$ python go_bench.py -s shgo tgo de bh
$ python go_bench.py -s shgo -debug True
"""

from __future__ import division, print_function, absolute_import
import numpy
import scipy.optimize
from _shgo import *
from _tgo import *
from go_funcs.go_benchmark import Benchmark
import go_funcs
import inspect
import time
import logging
import sys

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


excluded = ['Cola', 'Paviani', 'Xor',  # <--- Fucked
            'AMGM', 'Csendes', "Infinity", "Plateau",  # <--- Partially Fucked
            'Benchmark'  # Not a GO function
            ]

class GoRunner:
    def __init__(self, solvers=['shgo']):
        """
        Initiate with the list solvers to run, not implemented yet:
        solvers=['TGO', 'DE', 'BH']
        """

        self.solvers = solvers
        self.solvers_wrap = {'shgo': self.run_shgo,
                             'tgo': self.run_tgo,
                             'de': self.run_differentialevolution,
                             'bh': self.run_basinhopping}

        self.results = {'All': {}}
        for solver in self.solvers:
            self.results['All'][solver] = {'nfev': 0,
                                           # Number of function evaluations
                                           'nlmin': 0,
                                           # Number of local minima
                                           'success rate': 0,
                                           # Total success rate over all functions
                                           'success count': 0,
                                           'eval count': 0,
                                           'Total runtime': 0}


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

        t0 = time.time()
        # Add exception handling here?
        res = shgo(self.function.fun, self.function._bounds, n=100)#, n=50, crystal_mode=False)
        runtime = time.time() - t0

        # Prepare Return dictionary
        self.results[self.name]['shgo'] = \
            {'nfev': self.function.nfev,
             'nlmin': len(res.xl),  #TODO: Find no. unique local minima
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions
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
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions
            }

    def run_basinhopping(self):
        """
        Do an optimization run for basinhopping
        """
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
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions
            }

    def run_tgo(self):
        """
        Do an optimization run for tgo
        """
        t0 = time.time()
        # Add exception handling here?
        res = tgo(self.function.fun, self.function._bounds)
        runtime = time.time() - t0

        # Prepare Return dictionary
        self.results[self.name]['tgo'] = \
            {'nfev': self.function.nfev,
             'nlmin': len(res.xl),  # TODO: Find no. unique local minima
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions
             }

        return

    def update_results(self):
        # Update global results let nlmin for DE and BH := 0
        self.results['All'][self.solver]['nfev'] += \
            self.results[self.name][self.solver]['nfev']
        self.results['All'][self.solver]['nlmin'] += \
            self.results[self.name][self.solver]['nlmin']
        self.results['All'][self.solver]['eval count'] += 1
        self.results['All'][self.solver]['success count'] += \
            self.results[self.name][self.solver]['success']
        self.results['All'][self.solver]['success rate'] = \
            (100.0 * self.results['All'][self.solver]['success count']
             /float(self.results['All'][self.solver]['eval count']))
        self.results['All'][self.solver]['Total runtime'] += \
            self.results[self.name][self.solver]['runtime']

        return


if __name__ == '__main__':
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.solvers is None:
        GR = GoRunner(solvers=['shgo', 'tgo', 'bh'])
    else:
        GR = GoRunner(solvers=args.solvers)

    for name, obj in inspect.getmembers(go_funcs):
        if inspect.isclass(obj):
            logging.info(obj)
            logging.info(name)
            if name not in excluded:
                FuncClass = obj()
                GR.run_func(FuncClass, name)

    for solver in GR.results['All'].keys():
        print("=" * 60)
        print("Results for {}".format(solver))
        print("="*30)
        for key in GR.results['All'][solver].keys():
            print(key + ": " + str(GR.results['All'][solver][key]))
            #print(GR.results['All'][solver])
        print("=" * 60)
