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

    tol = 1e-5
    
    p = [len(atoms)]
    fglob = None
    global_optimum = [[coord for atom in atoms for coord in atom]]
    epsilon = [[]]
    sigma = [[]]
    bounds = [ ( min(atoms[:,i]) - tol, max(atoms[:,i]) + tol ) for i in range(3) ] * len(atoms)
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
    tol = 1e-5
    
    p = [int(p_A), N - int(p_A)]
    global_optimum = [[coord for atom in atoms for coord in atom]]
    epsilon = [[]]

    sigma_AA = 1.0
    sigma_AB = (sigma_AA + sigma_BB)/2
    sigma = [ [sigma_AA, sigma_AB], [sigma_AB, sigma_BB] ]

    bounds = [ ( min(atoms[:,i]) - tol, max(atoms[:,i]) + tol ) for i in range(3) ] * len(atoms)
    return go_funcs.go_funcs_L.LennardJonesN(p, fglob, global_optimum, epsilon, sigma, bounds, path)
