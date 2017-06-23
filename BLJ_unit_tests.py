import os, glob
import go_funcs.go_funcs_L

def BLJtest(fpath):
    f = open(fpath, 'r')
    lines = f.readlines()
    fglob, k_A, k_B = [float(num) for num in lines[0].split()]
    epsilon = [float(num) for num in lines[1].split()]
    sigma = [float(num) for num in lines[2].split()]
    x = [ float(num) for line in lines[3:] for num in line.split()[1:]]

    
    BLJ2 = go_funcs.go_funcs_L.LennardJones02(k_A, k_B, epsilon, sigma, fglob, x)
    assert BLJ2.success(x), "BLJtest: Objective fucntion did not return correct result for file {0}".format(fpath)
   
#Tests were retrieved from http://www-wales.ch.cam.ac.uk/CCD.html 

if __name__ == "__main__":
    for filename in glob.glob(os.path.join('BLJ_unit_tests', 'BLJ*.txt')):
        BLJtest(filename)
        print("Test {0}: \tPassed".format(filename) )
    print("All tests passed")
