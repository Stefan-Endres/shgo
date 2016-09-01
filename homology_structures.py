import tables
from matplotlib import pyplot
import numpy

import numpy as np
import tables

# Generate some data
if False:
    x = np.random.random((100,100,100))

    # Store "x" in a chunked array...
    f = tables.openFile('test.hdf', 'w')
    atom = tables.Atom.from_dtype(x.dtype)
    ds = f.createCArray(f.root, 'somename', atom, x.shape)
    ds[:] = x
    f.close()

    print(ds)

#TODO: We will the high level objects used in the class below to pytable
# objects later


class HomologyComplex():
    """
    A class relating the vertexes of a hypercomplex of hypersimplices without
    physical storage by exploiting the symmetry of the simplicial powerset.
    """

    def __init__(self, dim):
        import numpy
        self.dim = dim
        self.V = self.hyper_cube()
        #self.V  = stretch_hyper_cube(self.V )
        print(self.V)


        #for k in range(self.dim):
        #    self.k_plex = []

    def hyper_cube(self):
        """
        Generate the n_dimensional hypercube on extrema

        Returns
        -------

        """
        #TODO: Use permentation group algorithm to generate vertices
        self.V = numpy.ones([self.dim**2, self.dim])
        self.V = numpy.tril(self.V, k=-1)
        self.V = numpy.triu(self.V, k=-self.dim)#self.dim**2 + 1)
        return self.V


    def stretch_hyper_cube(self, V):
        """
        Stretch the outer vertices of a hypercube over the defined
        search space

        # TODO: Also add constraints.

        Parameters
        ----------
        V

        Returns
        -------

        """
        pass

if __name__ == '__main__':
    #build_complex = HomologyComplex(2)
    build_complex = HomologyComplex(2)
    build_complex = HomologyComplex(3)
    build_complex = HomologyComplex(4)
    build_complex = HomologyComplex(6)