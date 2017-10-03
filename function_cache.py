#TODO: Implement in _shgo.py for random sampling
class FunctionCache:
    def __init__(self, func, func_args=(), bounds=None, g_cons=None,
                 g_cons_args=(), indexed=True):

        self.cache = {}
        # self.cache = set()
        self.func = func
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.func_args = func_args
        self.bounds = bounds
        self.nfev = 0

        if indexed:
            self.Index = -1


    def __getitem__(self, x, indexed=True):
        try:
            return self.cache[x]
        except KeyError:
            if indexed:
                self.Index += 1
                xval = Vertex(x, bounds=self.bounds,
                              func=self.func, func_args=self.func_args,
                              g_cons=self.g_cons,
                              g_cons_args=self.g_cons_args,
                              I=self.Index)
            else:
                xval = Vertex(x, bounds=self.bounds,
                              func=self.func, func_args=self.func_args,
                              g_cons=self.g_cons,
                              g_cons_args=self.g_cons_args)

            #logging.info("New generated vertex at x = {}".format(x))
            #NOTE: Surprisingly high performance increase if logging is commented out
            self.cache[x] = xval
            if self.func is not None:
                if self.g_cons is not None:
                    #print(f'xval.feasible = {xval.feasible}')
                    if xval.feasible:
                        self.nfev += 1
                else:
                    self.nfev += 1

            return self.cache[x]