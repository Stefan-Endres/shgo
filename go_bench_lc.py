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
from go_funcs_lc.go_benchmark import Benchmark
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
            #'Horst6', # <--- Not correctly defined
            #'Hs038',  #
            #'Horst5', 'Horst7', 'Hs036', 'S250',  # <--- Slow
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
                                           'total runtime': 0
                                           }

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

        # Global mode
        kwarg_args['options'] = {}
        kwarg_args['options']['local_iter'] = 10
        kwarg_args['options']['local_fglob'] = self.function.fglob
        kwarg_args['options']['local_f_tol'] = 0.01
        #kwarg_args['options']['f_tol'] = 0.01

        #minimizer_kwargs = {'method': 'Nelder-Mead'
        #                    }

        success_l = False
        iters = 1
        while not success_l:
            self.function.nfev = 0
            t0 = time.time()

            res = shgo(self.function.fun, self.function._bounds,
                       iters=iters,
                       #minimizer_kwargs=minimizer_kwargs,
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
                    print(f'res.fun = {res.fun}')
                    print(f'res.fun = {res.x}')
                    print(f'f* = {self.function.fglob}')
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

        # Global mode
        kwarg_args['options'] = {}
        kwarg_args['options']['local_iter'] = 10
        kwarg_args['options']['local_fglob'] = self.function.fglob
        kwarg_args['options']['local_f_tol'] = 0.01
        kwarg_args['options']['infty constraints'] = True

        success_l = False
        n = self.function._dimensions + 1

        if self.name == 'Hs038':
            n = 458-1

        while not success_l:
            self.function.nfev = 0
            t0 = time.time()
            try:
                res = shgo(self.function.fun, self.function._bounds,
                           n=n,
                           sampling_method='sobol',
                           **kwarg_args)
            except scipy.spatial.qhull.QhullError:
                n += 1
                continue
            # print(res)
            # nfev = res.nfev
            runtime = time.time() - t0
            nfev = self.function.nfev
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

        # Add exception handling here?
        if self.function.g == None:
            kwarg_args = {'options': None}
        else:
            kwarg_args = {'g_cons': self.function.g}

        success_l = False
        n = self.function._dimensions + 1

        #if self.name == 'Hs038':  # difficult test
        #    n = 1000

        #n = 1000
        while not success_l:
            self.function.nfev = 0
            t0 = time.time()
            try:
                res = tgo(self.function.fun, self.function._bounds,
                           n=n,
                           **kwarg_args)
            except IndexError:
                n += 1
                continue
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
        self.results[self.name]['tgo'] = \
            {'nfev': self.function.nfev,
             'iter': n,  # Total sampling including infeasible regions
             'nlmin': len(res.xl),  # TODO: Find no. unique local minima
             'nulmin': self.unique_minima(res.xl),
             'runtime': runtime,
             'success': self.function.success(res.x),
             'ndim': int(self.function._dimensions),
             'name': 'tgo'
             }
        return

    def update_results(self):
        # Update global results let nlmin for DE and BH := 0
        self.results['All'][self.solver]['name'] = \
            self.results[self.name][self.solver]['name']
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

    def performance_profiles(self, tau_l=['nfev', 'runtime'],
                             solvers=['shgo', 'shgo_sobol', 'tgo'],
                             xlims=None):
        # Use tau performance index
        # for example function evaluations tau=`nfev`
        # other examples like processing time tau=`runtime`
        # If list is given subplots are generated
        import matplotlib
        from matplotlib import pyplot as plot
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.unicode'] = True
        fig = plot.figure()
        axes = []
        for tau_ind, tau in enumerate(tau_l):
            eval_funcs = self.results['All'][self.solver]['eval count']
            xranges = {}
            yranges = {}
            yranges_ext = {}
            for solver in solvers:
                yranges[solver] = numpy.array(range(1, eval_funcs + 1)) / (eval_funcs)# - 1.0)
                yranges_ext[solver] = numpy.array([])
                xranges[solver] = numpy.zeros(self.results['All'][self.solver]['eval count'])

            for solver in solvers:
                #xranges[solver] = numpy.zeros(self.results['All'][self.solver]['eval count'])
                i = 0
                for problem in self.results.keys():
                    if (problem is not 'All') and (problem is not 'Average'):
                        xranges[solver][i] = GR.results[problem][solver][tau]
                        i += 1

                xranges[solver] = numpy.sort(xranges[solver])

            # Find the total range of tau data points
            x_range = numpy.array([])
            for solver in solvers:
                x_range = numpy.append(x_range, xranges[solver])

            x_range = numpy.sort(x_range)

            # Find the total range of corresponding solved functions:
            for solver in solvers:
                yranges_ext[solver] = []
                solved = 0
                ind_s = 0
                for x in x_range:
                    if ind_s < xranges[solver].size:
                        if x == xranges[solver][ind_s]:
                            solved += 1
                            ind_s += 1
                            yranges_ext[solver].append(solved)
                            continue
                        else:
                            yranges_ext[solver].append(solved)
                    else:
                        yranges_ext[solver].append(solved)
                #print(yranges_ext[solver])
                yranges_ext[solver] = numpy.array(yranges_ext[solver])/float(xranges[solver].size)
                #print(yranges_ext[solver])

            if tau_ind == 0:
                #axes.append(plot.subplot(int('21{}'.format(tau_ind + 1))))
                axes.append(
                    fig.add_subplot(
                        plot.subplot(int('12{}'.format(tau_ind + 1)))))
                plot.ylabel(r"$\textrm{Fraction of functions solved}$", fontsize=14)
            else:
                axes.append(
                    fig.add_subplot(
                        plot.subplot(int('12{}'.format(tau_ind + 1)),
                                     sharey=axes[0])))
                plot.setp(axes[tau_ind].get_yticklabels(), visible=False)
            #plot.figure()

            line_types = ['-', '--', '-.', '-^', ':', '-o','-d']
            colours = []
            line_ind = 0
            for key in yranges_ext.keys():
                #plot.plot(x_range, yranges_ext[key], line_types[line_ind], label=key, linewidth=1.0)
                #axes[tau_ind].plot(x_range, yranges_ext[key], line_types[line_ind], label=key, linewidth=1.0)
                #ax1 = plot.subplot(int('2{}1'.format(tau_ind + 1)))
                solver_name = self.results['All'][key]['name']
                solv_label = r'$\textrm{' + solver_name + '}$'
                axes[tau_ind].plot(x_range, yranges_ext[key],
                                   line_types[line_ind], label=solv_label, linewidth=1.0)
                axes[tau_ind].grid(1, alpha=0.5)
                line_ind += 1
                if line_ind > len(line_types):
                    line_ind = 0

            plot.legend(fontsize=14)

            if tau == 'nfev':
                plot.xlabel(r"$\textrm{Function evaluations}$", fontsize=14)
            elif tau == 'runtime':
                plot.xlabel(r"$\textrm{Processing time (s)}$", fontsize=14)
            else:
                plot.xlabel(tau, fontsize=14)

        plot.tight_layout()
        plot.show()


if __name__ == '__main__':
    if args.debug:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

    if args.solvers is None:
        GR = GoRunner(solvers=['shgo'])
                             # , 'tgo'])
        args.solvers = ['shgo']
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
                                                /22.0)#GR.results['All'][solver]['eval count'])
        GR.results['All'][solver]['average nfev'] = (GR.results['All'][solver]['nfev']
                                                /22.0)#GR.results['All'][solver]['eval count'])

        # Process averages
        if 1:  # Average
            GR.results['Average'][solver]['runtime'] = GR.results['All'][solver]['average runtime']
            GR.results['Average'][solver]['nfev'] = int(GR.results['All'][solver]['average nfev'])
            GR.results['Average'][solver]['ndim'] = int(0)
            GR.results['Average'][solver]['nlmin'] = (GR.results['All'][solver]['nlmin']
                                                      / GR.results['All'][solver]['eval count']
                                                      )
            GR.results['Average'][solver]['nulmin'] = (GR.results['All'][solver]['nulmin']
                                                      / GR.results['All'][solver]['eval count']
                                                      )
            GR.results['Average'][solver]['name'] = GR.results['All'][solver]['name']
        # Print results
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

        GR.results['All'][solver]['nlmin'] = int(GR.results['All'][solver]['nlmin'])
        GR.results['Average'][solver]['nlmin'] = int(GR.results['Average'][solver]['nlmin'])
        GR.results['All'][solver]['nulmin'] = int(GR.results['All'][solver]['nulmin'])
        GR.results['Average'][solver]['nulmin'] = int(GR.results['Average'][solver]['nulmin'])


        import json
        with open('results/results_lc.json', 'w') as fp:
            json.dump(GR.results, fp)

    if 1:
        GR.performance_profiles(solvers=args.solvers)
        #GR.performance_profiles(solvers=['shgo', 'tgo'])