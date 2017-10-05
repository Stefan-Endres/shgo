import json
from go_bench_f_min import *
from pprint import pprint

with open('results_lc.json') as data_file:
    data = json.load(data_file)

with open('results_f_min.json') as data_file:
    data_f_min = json.load(data_file)

#pprint(data)
#pprint(data_f_min)

if __name__ == '__main__':
    GR = GoRunner(solvers=['shgo', 'shgo_sobol', 'tgo', 'de', 'bh'])
    GR = GoRunner(solvers=['shgo', 'shgo_sobol', 'tgo'])

    #with open('results_f_min.json') as data_file:
    with open('results_lc.json') as data_file:
        GR.results = json.load(data_file)


    if 0:
        GR.failure_performance(nfev_fail=1.0e5, runtime_fail=3000,
                                #solvers=['shgo', 'shgo_sobol', 'tgo', 'de', 'bh'])
                                solvers=['shgo', 'shgo_sobol', 'tgo'])

    if 0:
        xlims = {'nfev': [-10, 1000],
                 'runtime': [-0.01, 0.4]}
    else:
        xlims = None
    GR.performance_profiles(tau_l=['nfev', 'runtime'],
                            #solvers=['shgo', 'shgo_sobol', 'tgo', 'de', 'bh'],
                            solvers=['shgo', 'shgo_sobol', 'tgo'],
                            #solvers=['shgo', 'shgo_sobol', 'tgo'],
                             nfev_fail = 1.0e5, runtime_fail=3000,
                             xlims=xlims
                            )

    pprint(GR.results)