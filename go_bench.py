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
        self.results = {'All': {'nfev': 0,  # Number of function evaluations
                                'nlmin': 0,  # Number of local minima
                                'success rate': 0,
                                # Total success rate over all functions
                                'success count': 0,
                                'eval count': 0,
                                'Total runtime': 0}
                        }

        self.solvers = solvers
        self.solvers_wrap = {'shgo': self.run_shgo,
                             'tgo': self.run_tgo,
                             'de': self.run_differentialevolution,
                             'bh': self.run_basinhopping}

    def run_func(self, FuncClass, name):
        """
        Run the for all solvers
        """
        # Store the function class and its attributes here:
        self.funC = FuncClass
        self.name = name
        self.results[name] = {}
        for solver in self.solvers:
            self.solver = solver
            self.solvers_wrap[self.solver]()
            self.update_results()

    def run_shgo(self):
        t0 = time.time()
        # Add exception handling here?
        res = shgo(self.funC.fun,self.funC._bounds, n=50)#, n=50, crystal_mode=False)
        runtime = time.time() - t0

        # Prepare Return dictionary
        self.results[self.name]['shgo'] = \
            {'nfev': res.nfev,
             'nlmin': len(res.xl),  #TODO: Find no. unique local minima
             'runtime': runtime,
             'success': self.funC.success(res.x),
             'ndim': self.funC._dimensions
             }
        return


    def run_differentialevolution(self):
        """
        Do an optimization run for differential_evolution
        """
        self.function.nfev = 0

        t0 = time.time()

        res = scipy.optimize.differential_evolution(self.fun,
                                     self.bounds,
                                     popsize=20)

        t1 = time.time()
        res.success = self.function.success(res.x)
        res.nfev = self.function.nfev
        self.add_result(res, t1 - t0, 'DE')

    def run_basinhopping(self):
        """
        Do an optimization run for basinhopping
        """
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
            self.fun, x0, accept_test=self.accept_test,
            minimizer_kwargs=minimizer_kwargs,
            **kwargs)

        t1 = time.time()
        res.success = self.function.success(res.x)
        res.nfev = self.function.nfev

        self.add_result(res, t1 - t0, 'basinh.')

    def run_tgo(self):
        """
        Do an optimization run for tgo
        """
        self.function.nfev = 0

        t0 = time.time()

        res = scipy.optimize.tgo(self.fun, self.bounds)

        t1 = time.time()
        res.success = self.function.success(res.x)
        res.nfev = self.function.nfev
        self.add_result(res, t1 - t0, 'TGO')

    def update_results(self):
        # Update global results let nlmin for DE and BH := 0
        self.results['All']['nfev'] += \
            self.results[self.name][self.solver]['nfev']
        self.results['All']['nlmin'] += \
            self.results[self.name][self.solver]['nlmin']
        self.results['All']['eval count'] += 1
        self.results['All']['success count'] += \
            self.results[self.name][self.solver]['success']
        self.results['All']['success rate'] = \
            (100.0 * self.results['All']['success count']
             /float(self.results['All']['eval count']))
        self.results['All']['Total runtime'] += \
            self.results[self.name][self.solver]['runtime']

        return


if __name__ == '__main__':
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.solvers is None:
        GR = GoRunner(solvers=['shgo'])
    else:
        GR = GoRunner(solvers=args.solvers)

    for name, obj in inspect.getmembers(go_funcs):
        if inspect.isclass(obj):
            logging.info(obj)
            logging.info(name)
            if name not in excluded:
                FuncClass = obj()
                GR.run_func(FuncClass, name)

    print(GR.results['All'])
