# %% Vertex group classes
class VertexGroup(object):
    def __init__(self, p_gen=0, p_hgr=0):
        self.p_gen = p_gen  # parent generation
        self.p_hgr = p_hgr  # parent homology group rank
        self.hg_n = None
        self.hg_d = None

        # Maybe add parent homology group rank total history
        # This is the sum off all previously split cells
        # cumulatively throughout its entire history
        self.C = []

    def __add__(self, v):
        #self.C.append(v)
        self.add_vertex(v)

    def __call__(self):
        return self.C

    def add_vertex(self, V):
        if V not in self.C:
            self.C.append(V)

    def homology_group_rank(self):
        """
        Returns the homology group order of the current cell
        """
        if self.hg_n is None:
            self.hg_n = sum(1 for v in self.C if v.minimiser())

        return self.hg_n

    def homology_group_differential(self):
        """
        Returns the difference between the current homology group of the
        cell and it's parent group
        """
        if self.hg_d is None:
            self.hgd = self.hg_n - self.p_hgr

        return self.hgd

    def polytopial_sperner_lemma(self):
        """
        Returns the number of stationary points theoretically contained in the
        cell based information currently known about the cell
        """
        pass

    def print_out(self):
        """
        Print the current cell to console
        """
        for v in self():
            v.print_out()

class Subgroup(VertexGroup):
    """
    Contains a subgroup of vertices
    """
    def __init__(self, p_gen=0, p_hgr=0, origin=None, supremum=None):
        super(Subgroup, self).__init__(p_gen, p_hgr)


class Cell(VertexGroup):
    """
    Contains a cell that is symmetric to the initial hypercube triangulation
    """

    def __init__(self, p_gen, p_hgr, origin, supremum, generation_cycle=0):
        super(Cell, self).__init__(p_gen, p_hgr)

        self.origin = origin
        self.supremum = supremum
        self.centroid = None  # (Not always used)
        self.generation_cycle = generation_cycle
        # TODO: self.bounds


class Simplex(VertexGroup):
    """
    Contains a simplex that is symmetric to the initial symmetry constrained
    hypersimplex triangulation
    """

    def __init__(self, p_gen, p_hgr, generation_cycle, dim):
        super(Simplex, self).__init__(p_gen, p_hgr)

        self.generation_cycle = (generation_cycle + 1) % (dim - 1)
