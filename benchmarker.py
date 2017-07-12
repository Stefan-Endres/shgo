import data_parsers, _tgo, time, scipy.optimize, numpy, datetime

class Benchmarker():
    """
    Runs various global optimisation algorithms against
    various benchmarks.

    The algorithms are required to take two inputs:
        fun : function
            The objective function to be optimised
        bounds : [(float, float)]
            List of bounds for each dimension

    The benchmarks are expected to be child objects of the
    scipy Benchmark class.
    """
    def __init__(self, algorithms, benchmarks):
        """
        Parameters
        ----------
        algorithms : [function]
            list of algorithms to be benchmarked
        
        benchmarks : [object]
            list of Benchmark objetcs
        """
        self.algorithms = algorithms
        self.benchmarks = benchmarks

    def run(self, disp=False, timer=False):
        """
        Parameters
        ----------
        disp : Bool
            If true, then results are printed to the console

        timer : Bool
            If true, then algorithms are timed

        Returns
        -------
        result : dict{ dict{} }
            A dictionary of algorithm names of dictionaries of benchmark names of benchmark results.
            If timer is True, then each algorithm also has a "time" key.
        """
        if disp:
            print("Running Benchmarks")

        self.results = {}
        for algorithm in self.algorithms:
            self.results[algorithm.__name__] = {}
            for benchmark in self.benchmarks:
                if disp:
                    print("Testing algorithm: {0} with benchmark {1}".format(algorithm.__name__, benchmark._name) )

                if timer:
                    t0 = time.time()
                result = algorithm(benchmark.fun, benchmark._bounds)
                if timer:
                    t = time.time()-t0

                self.results[algorithm.__name__][benchmark._name] = result
                if disp:
                    if benchmark.success(result.x, 1e-3):
                        print("Found global minimum {0} at \n{1}\nIn {2} evaluations".format(result.fun, result.x, result.nfev) )
                    else:
                        print("Unsuccessful Lowest value found {0} at\n{1}\nIn {2} evaluations".format(result.fun, result.x, result.nfev) )
                        print("Global Minimum is {0}".format(benchmark.fglob) )
                    if timer:
                        print("Time taken was {0}".format(datetime.timedelta(seconds=int(t))) )
                if timer:
                    self.results[algorithm.__name__]["time"] = t

        return self.results


#The rest of this file is an example of how the class is to be used

def tgo_wo(fun, bounds):
    options = {'symmetry': True,
           'disp': False,
           'crystal_iter': 1}
    return _tgo.tgo(fun, bounds, options=options, n=5000)


#benchmarks = [ data_parsers.LJ_parser("Data/LJ_3-150/" + str(i)) for i in range(3,18)] #data_parsers.BLJ_parser("Data/BLJ_5-100/1.3/13"),
benchmarks = [ data_parsers.TIP4P_parser("Data/TIP4P_2-21/TIP4P-4.xyz") ]

"""
Please note that the data for this run should be downloaded from:
https://bitbucket.org/darrenroos/shgo_data
All of the source folder should be placed within a folder called Data for the above code to run.
"""
algorithms = [tgo_wo]

B = Benchmarker(algorithms, benchmarks)
r = B.run(True, True)
print("\n\nData Dump\n")
print(r)
