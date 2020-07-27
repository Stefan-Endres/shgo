import numpy as np
"""
Field cache objects
"""
class FieldCache(object):
    """
    Base class providing a cache for field computations
    """
    def __init__(self, vcache, g_cons=None, g_cons_args=()):
        self.V = vcache  # Reference to a vertex cache
        self.nfev = 0
        self.g_cons = g_cons
        self.g_cons_args = g_cons_args
        self.gpool = set()  # A set of tuples to process for feasibility

    def feasibility_check(self):
        if self.g_cons is not None:
            for g, args in zip(self.g_cons, self.g_cons_args):
                if g(self.x_a, *args) < 0.0:
                    self.f = np.inf
                    self.feasible = False
                    break


class ScalarFieldCache(FieldCache):
    """
    Cache for field computations, can be used to associate a scalar field with
    the geometry. Triangulation of non-linear geometries is approximated by
    cutting the infeasible points defined by the constraints.

    #TODO: Implement
    """
    def __init__(self, vcache, g_cons=None, g_cons_args=(), field=None,
                 field_args=()):
        super().__init__(vcache, g_cons=None, g_cons_args=())

        self.field = field
        self.field_args = field_args
        self.fpool = set()  # A set of tuples to process for scalar function
        self.sfc_lock = False  # True if self.fpool is non-Empty

    def process_field_pool(self):
        """
        Compute the
        :return:
        """

        #TODO: NOTE THAT WE CAN STORE THE VALUE OF THE FIELD IN THE CACHE AS
        #      AS THE .f ATTRIBUTES OF THE VERTEX

    def compute(self):
        """
        Compute the function values for vertices
        :return:
        """
        for v in self.V:
            print(f'v')