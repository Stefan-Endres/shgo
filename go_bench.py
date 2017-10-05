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
#from _shgo_sobol import shgo as shgo_sobol
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


excluded = ['Cola', 'Paviani', 'Xor',
            #"Bukin06",  # <--- Working, but fail on all solvers + high nfev
            'Benchmark',  # Not a GO function
            'LennardJonesN', 'TIP4P',  # New function not in the original suite
            'matrix'  # Numpy tests ???
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
        for solver in self.solvers:
            self.results['All'][solver] = {'nfev': 0,
                                           # Number of function evaluations
                                           'nlmin': 0,
                                           'nulmin': 0,
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
        res = shgo(self.function.fun, self.function._bounds,
                   n=200,
                   #iter=3,
                   sampling_method='simplicial')#,
                   #,n=5000)#, n=50, crystal_mode=False)
        runtime = time.time() - t0

        # Prepare Return dictionary
        self.results[self.name]['shgo'] = \
            {'nfev': self.function.nfev,
             'nlmin': len(res.xl),
             'nulmin': self.unique_minima(res.xl),
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions,
             'name': 'shgo'
             }
        return

    def run_shgo_sobol(self):
        self.function.nfev = 0

        t0 = time.time()
        # Add exception handling here?
        res = shgo(self.function.fun, self.function._bounds, n=100, sampling_method='sobol')
        runtime = time.time() - t0

        # Prepare Return dictionary
        self.results[self.name]['shgo_sobol'] = \
            {'nfev': self.function.nfev,
             'nlmin': len(res.xl),  #TODO: Find no. unique local minima
             'nulmin': self.unique_minima(res.xl),
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': self.function._dimensions,
             'name': 'shgo_sobol'
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
        self.results['All'][self.solver]['Total runtime'] += \
            self.results[self.name][self.solver]['runtime']

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

    for name, obj in inspect.getmembers(go_funcs):
        #if name == 'Ackley01':
            #if name == 'Ackley03':
        #if name == 'Alpine02':
        #nif name == 'Wolfe':
            if inspect.isclass(obj):
                logging.info(obj)
                logging.info(name)
                if name not in excluded:
                    FuncClass = obj()
                    try:
                        GR.run_func(FuncClass, name)
                    except:
                        pass
    for solver in GR.results['All'].keys():
        print("=" * 60)
        print("Results for {}".format(solver))
        print("="*30)
        for key in GR.results['All'][solver].keys():
            print(key + ": " + str(GR.results['All'][solver][key]))
            #print(GR.results['All'][solver])
        print("=" * 60)

        import json
        with open('results/results.json', 'w') as fp:
            json.dump(GR.results, fp)
'''
============================================================

Results for tgo
==============================
nulmin: 744
eval count: 188
nfev: 219729
success rate: 81.91489361702128
nlmin: 897
success count: 154
Total runtime: 2.066362142562866
============================================================
Results for shgo
==============================
nulmin: 938
eval count: 188
nfev: 117100
success rate: 80.31914893617021
nlmin: 1037
success count: 151
Total runtime: 3.6703944206237793
============================================================
============================================================
'''

"""
CONSISTENCY TEST 28.08.2017

TEST 0 with old funcs
Results for shgo_sobol
==============================
nfev: 117245
nlmin: 1041
nulmin: 950
success rate: 80.42328042328042
success count: 152
eval count: 189
Total runtime: 3.899496078491211
name: shgo_sobol



TEST 1  with new funcs
Results for shgo_sobol
==============================
nfev: 117253
nlmin: 1037
nulmin: 946
success rate: 80.42328042328042
success count: 152
eval count: 189
Total runtime: 3.7815604209899902
name: shgo_sobol


TEST 2  with new funcs
Results for shgo_sobol
==============================
nfev: 117383
nlmin: 1039
nulmin: 948
success rate: 80.42328042328042
success count: 152
eval count: 189
Total runtime: 3.786623477935791
name: shgo_sobol
"""




"""
============================================================
========================================================================================================================
========================================================================================================================
========================================================================================================================
============================================================

"""

"""

PERFORMANCE TEST 28.08.2017

TEST 0 with iter=1 on simplicial
============================================================
Results for shgo
==============================
nfev: 55742
nlmin: 337
nulmin: 323
success rate: 53.591160220994475
success count: 97
eval count: 181
Total runtime: 4.5354509353637695
name: shgo
============================================================
============================================================
Results for shgo_sobol
==============================
nfev: 111372
nlmin: 986
nulmin: 898
success rate: 80.33707865168539
success count: 143
eval count: 178
Total runtime: 3.5806477069854736
name: shgo_sobol
============================================================
============================================================
Results for tgo
==============================
nfev: 94959
nlmin: 835
nulmin: 694
success rate: 80.33707865168539
success count: 143
eval count: 178
Total runtime: 2.1270735263824463
name: tgo
============================================================
============================================================
Results for de
==============================
nfev: 1030253
nlmin: 0
nulmin: 0
success rate: 86.51685393258427
success count: 154
eval count: 178
Total runtime: 40.487993240356445
name: de
============================================================
============================================================
Results for bh
==============================
nfev: 1434992
nlmin: 0
nulmin: 0
success rate: 58.42696629213483
success count: 104
eval count: 178
Total runtime: 23.928292274475098
name: bh
============================================================

"""


"""
============================================================
Results for shgo
==============================
nfev: 4360113
nlmin: 14938
nulmin: 12597
success rate: 80.0
success count: 148
eval count: 185
Total runtime: 657.9505994319916
name: shgo
============================================================
============================================================
Results for shgo_sobol
==============================
nfev: 116107
nlmin: 1018
nulmin: 927
success rate: 80.32786885245902
success count: 147
eval count: 183
Total runtime: 6.303086280822754
name: shgo_sobol
============================================================
============================================================
Results for tgo
==============================
nfev: 100931
nlmin: 871
nulmin: 725
success rate: 81.4207650273224
success count: 149
eval count: 183
Total runtime: 3.7384634017944336
name: tgo
============================================================

"""