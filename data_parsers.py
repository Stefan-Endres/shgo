import go_funcs.go_funcs_L
import numpy

"""
Contains methods that allow the interfacing of data files
to benchmarking methods. Each method should take as input
a path to a data file that is to be parsed, and should
return an object that has the type of a child of the
Benchmark class.
"""


def LJ_parser(path):
    """
    Parameters
    ----------
    path : string
        path to file containing data for a LJ benchmark
        The name of the file must be a number N indicating the number
        of particles in the LJ cluster.
        The format of the file is N lines. The i-th line contains 3 floats
        corresponding to the x, y and z spacial coordinates of the
        i-th particle. Arranging the particles according to these coordinates
        should yield the globally minimum energy..
    
    Returns
    -------
    A go_funcs.go_funcs_L.LennardJonesN object.
    """

    f = open(path, 'r')
    lines = f.readlines()
    atoms = numpy.array([ numpy.array([float(num) for num in line.split()]) for line in lines ])
    
    p = [len(atoms)]

    fglob = None
    global_optimum = [[coord for atom in atoms for coord in atom]]
    
    epsilon = [[]]
    sigma = [[]]
    
    tol = 1e-5
    bounds = [ ( atoms.min() - tol, atoms.max() + tol ) ] * len(atoms) * 3
    
    return go_funcs.go_funcs_L.LennardJonesN(p, fglob, global_optimum, epsilon, sigma, bounds, path)


def BLJ_parser(path):
    """
    Parameters
    ----------
    path : string
        path to file containing data for a BLJ benchmark
        The name of the file must be a number N indicating the number
        of particles in the LJ cluster.
        The format of the file is as follows:
        Line 1 contains a single number N being the total number of
        particles in the system.
        Line 2 contains a string of the form:
        Energy of minimum  1=       <value1> first found at step        1 (NA,SIG_BB)   <value2> <value3>
            Where <value1> is the fglob, <value2> is the number of the
            first particle type, and <value3> is the value of
            sigmaBB/sigmaAA.
        The following N lines contain information about each particle
        in the system. With the first <value2> particles being of type A.
        
        It is assumed that \epsilon_{AA} = \epsilon_{BB} = \epsilon_{AB}
        and that \sigma_{AB} = (\sigma_{AA} + \sigma_{BB}) / 2
    
    Returns
    -------
    A go_funcs.go_funcs_L.LennardJonesN object.
    """

    f = open(path, 'r')
    lines = f.readlines()
    
    N = int(lines[0].split()[0])
    fglob, p_A, sigma_BB = [float(value) for value, i in zip( lines[1].split(), range(9999) ) if i in [4,11,12] ]
    
    atoms = numpy.array([ numpy.array([float(num) for num in line.split()[1:] ]) for line in lines[2:] ])
    
    p = [int(p_A), N - int(p_A)]
    global_optimum = [[coord for atom in atoms for coord in atom]]
    epsilon = [[]]

    sigma_AA = 1.0
    sigma_AB = (sigma_AA + sigma_BB)/2
    sigma = [ [sigma_AA, sigma_AB], [sigma_AB, sigma_BB] ]

    tol = 1e-5
    bounds = [ ( atoms.min() - tol, atoms.max() + tol ) ] * len(atoms) * 3
    
    return go_funcs.go_funcs_L.LennardJonesN(p, fglob, global_optimum, epsilon, sigma, bounds, path)

def TIP4P_parser(path):
    """
    Parameters
    ----------
    path : string
        path to file containing data for a TIP4P benchmark
        The name of the file must be of the form "TIP4P-N.xyz"
        with N indicating the number of water molecules in
        the TIP4P cluster. The format of the file is as follows:
        Line 1 contains a single number N being the total number of
        atoms in the system.
        Line 2 contains a string of the form:
        <value1>   Energy =     <value2>  kJ/mol
            Where <value1> is a number with an unknown meanign,
            and <value2> is the energy of the cluster. 
        The following N lines contain information about each atom
        in the system. Every three lines describe one water molecule.
        The first atom being the O atom.
    
    Returns
    -------
    A go_funcs.go_funcs_T.TIP4P object.
    """
    f = open(path, 'r')
    lines = f.readlines()

    atoms = numpy.array([ numpy.array([float(num) for num in line.split()[1:] ]) for line in lines[2:] ])
    
    W = int( len(atoms) / 3)

    global_optimum = [[coord for atom in atoms for coord in atom]]

    tol = 1e-5
    bounds = [ ( atoms.min() - tol, atoms.max() + tol ) ] * 3 
    bounds+= [(0.0, numpy.pi), (0.0, 2 * numpy.pi), (0.0, 2 * numpy.pi)]
    bounds*= W
    
    return go_funcs.go_funcs_T.TIP4P(W, None, global_optimum, True, bounds, path)

def LJ38_start_bounds_parser(path):
    """
    Parameters
    ----------
    path : string
        path to file containing data for a LJ starting structure
        The name of the file must be of the form "cluster_<X>.xyz"
        with X being a 4 digit integer. The format of the file is as follows: 
        Line 1 and Line 2 contain fairly useless information.
        The next 38 lines each contain information about a particle
        in the system.
    
    Returns
    -------
    bounds : [(float, float)]
        A list of bounds for each dimension.
        len(bounds) should be equal to 38 * 3
    """
    f = open(path, 'r')
    lines = f.readlines()
    atoms = [ numpy.array([float(coord) for coord in line.split()[1:] ]) for line in lines[2:]]
    return list( zip( numpy.minimum.reduce(atoms), numpy.maximum.reduce(atoms) ) ) * 38

if __name__ == "__main__":
    BLJ, LJ, TIP = BLJ_parser("Data/BLJ_5-100/1.3/5"), LJ_parser("Data/LJ_3-150/3"), TIP4P_parser("Data/TIP4P_2-21/TIP4P-2.xyz")
    LJ_bounds = LJ38_start_bounds_parser("Data/lj38_starting_clusters/cluster_0000.xyz")
