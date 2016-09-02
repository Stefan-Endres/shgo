#!/usr/bin/env python
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
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


excluded = ['Cola', 'Paviani', 'Xor',  # <--- Fucked
            'AMGM', 'Csendes', "Infinity", "Plateau",  # <--- Partially Fucked
            'Benchmark'  # Not a GO function
            ]

results = {'All': {'nfev': 0,  # Number of function evaluations
                   'nlmin': 0,  # Number of local minima
                   'success rate': 0,  # Total success rate over all functions
                   'success count': 0,
                   'eval count': 0,
                   'Total runtime': 0}
           }


class GoRunner:
    def __init__(self, solvers=['SHGO']):
        """
        Initiate with the list solvers to run
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
        self.solvers_wrap = {'SHGO': self.run_shgo,
                             'TGO': self.run_tgo,
                             'DE': self.run_differentialevolution,
                             'BH': self.run_basinhopping}

    def run_func(self, FuncClass, name):
        """
        Run the for all solvers
        """
        # Store the function class and its attributes here:
        self.funC = FuncClass
        self.name = name
        #self.function = FuncClass.function
        self.results[name] = {}
        for solver in self.solvers:
            self.solver = solver
            self.solvers_wrap[self.solver]()
            self.update_results()

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


    def run_shgo(self):
        t0 = time.time()
        # Add exception handling here?
        res = shgo(self.funC.fun,self.funC._bounds, n=50)#, n=50, crystal_mode=False)
        runtime = time.time() - t0
        # Prepare Return dictionary
        self.results[self.name]['SHGO'] = \
            {'nfev': res.nfev,
             'nlmin': len(res.xl),  #TODO: Find no. unique local minima
             'runtime': runtime,
             'success': self.funC.success(res.x),
             'ndim': self.funC._dimensions
             }
        return

    def update_results(self):
        # Update global results let nlmin for DE and BH := 0
        self.results['All']['nfev'] += \
            self.results[name][self.solver]['nfev']
        self.results['All']['nlmin'] += \
            self.results[name][self.solver]['nlmin']
        self.results['All']['eval count'] += 1
        self.results['All']['success count'] += \
            self.results[name][self.solver]['success']
        self.results['All']['success rate'] = \
            (100.0 * self.results['All']['success count']
             /float(self.results['All']['eval count']))
        self.results['All']['Total runtime'] += \
            self.results[name][self.solver]['runtime']

        return

def run_shgo(FuncClass, results, name):

    st = time.time()
    # Add exception handling here?
    res = shgo(FuncClass.fun, FuncClass._bounds, n =50)#, n=50, crystal_mode=False)
    runtime = time.time() - st
    # Success/fail:
    #print(FuncClass.success(res.x))

    # Prepare Return dictionary

    results[name]['SHGO'] = {'nfev': res.nfev,
                             'nlmin': len(res.xl),  #TODO: Find unique local minima
                             'runtime': runtime,
                             'success': FuncClass.success(res.x),
                             'ndim': FuncClass._dimensions
                             }

    # Update global results
    results['All']['nfev'] += results[name]['SHGO']['nfev']
    results['All']['nlmin'] += results[name]['SHGO']['nlmin']
    results['All']['eval count'] += 1
    results['All']['success count'] += results[name]['SHGO']['success']
    results['All']['success rate'] = (100.0 * results['All']['success count']
                                      /float(results['All']['eval count']))
    results['All']['Total runtime'] += results[name]['SHGO']['runtime']

    return results





def bench_run_global(self, numtrials=50, methods=None):
    """
    Run the optimization tests for the required minimizers.
    """

    if methods is None:
        methods = ['DE', 'basinh.', 'TGO', 'SHGO']

    method_fun = {'DE': self.run_differentialevolution,
                  'basinh.': self.run_basinhopping,
                  'TGO': self.run_tgo,
                  'SHGO': self.run_shgo}

    for i in range(numtrials):
        for m in methods:
            method_fun[m]()


if __name__ =='__main__':
    GR = GoRunner(solvers=['SHGO'])
    for name, obj in inspect.getmembers(go_funcs):
        if inspect.isclass(obj):
            print(obj)
            print(name)
            if name not in excluded:
                results[name] = {}
                FuncClass = obj()
                results = run_shgo(FuncClass, results, name)
                #GR.run_shgo_2(FuncClass, name)
                #GR.run_shgo()
                GR.run_func(FuncClass, name)
                #print(res)
                #print(Class.fun)


    print(results['All'])
    print(GR.results['All'])
'''
    Success
    rates
    == == == == == == =
    SHGO:  76.06382978723404 %
    TGO:  81.38297872340425 %
    basinh.:  59.57446808510638 %
    DE:  81.38297872340425 %

    Total
    function
    evaluations
    over
    all
    functions
    == == == == == == == == == == == == == == == == == == == == == == =
    SHGO:  20823.0
    TGO:  66749.0
    basinh.:  956611.0
    DE:  612382.0

'''

#{'success count': 143, 'nfev': 60505, 'nlmin': 633, 'eval count': 188, 'success rate': 76.06382978723404, 'Total runtime': 1.6896166801452637}
#{'success count': 143, 'nfev': 60647, 'nlmin': 636, 'eval count': 188, 'success rate': 76.06382978723404, 'Total runtime': 1.6997058391571045}

#{'nfev': 60527, 'eval count': 188, 'Total runtime': 1.70491361618042, 'success rate': 76.06382978723404, 'nlmin': 630, 'success count': 143}
#{'nfev': 60567, 'eval count': 188, 'Total runtime': 1.712622880935669, 'success rate': 76.06382978723404, 'nlmin': 632, 'success count': 143}

#{'eval count': 188, 'success count': 143, 'nfev': 60459, 'nlmin': 635, 'success rate': 76.06382978723404, 'Total runtime': 1.729461431503296}
#{'eval count': 188, 'success count': 143, 'nfev': 60682, 'nlmin': 637, 'success rate': 76.06382978723404, 'Total runtime': 1.7363505363464355}

#{'Total runtime': 1.680638313293457, 'success rate': 76.06382978723404, 'nfev': 60536, 'nlmin': 637, 'eval count': 188, 'success count': 143}
#{'Total runtime': 1.6847310066223145, 'success rate': 76.59574468085107, 'nfev': 60382, 'nlmin': 634, 'eval count': 188, 'success count': 144}

#{'nlmin': 638, 'eval count': 188, 'success rate': 76.06382978723404, 'success count': 143, 'nfev': 60562, 'Total runtime': 1.7481226921081543}
#{'nlmin': 632, 'eval count': 188, 'success rate': 76.59574468085107, 'success count': 144, 'nfev': 60409, 'Total runtime': 1.7493815422058105}

#{'success rate': 76.59574468085107, 'nfev': 60470, 'eval count': 188, 'success count': 144, 'Total runtime': 1.7047905921936035, 'nlmin': 632}
#{'success rate': 76.06382978723404, 'nfev': 60456, 'eval count': 188, 'success count': 143, 'Total runtime': 1.7081894874572754, 'nlmin': 637}