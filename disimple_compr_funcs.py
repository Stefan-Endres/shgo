"""
Note that all functions should be of the form:

minimize f(x) subject to

g_i(x) >= 0,  i = 1,...,m
h_j(x)  = 0,  j = 1,...,p

NOTE:

    bunnag and s### expected minima values were calculated

"""

from _shgo import *
import numpy

class TestFunction(object):
    def __init__(self, bounds, expected_x, expected_fun=None):
        self.bounds = bounds
        self.expected_x = expected_x
        self.expected_fun = expected_fun
        self.options = {}
        self.options['f_min'] = expected_fun


class Horst1(TestFunction):
    def f(self, x):
        return -x[0]**2 - 4*x[1]**2 + 4*x[0]*x[1] + 2*x[0] + 4*x[1]

    def g1(x):
        return 1 -(-4*x[0] + 2*x[1])

    def g2(x):
        return 4 -(x[0] + x[1])

    def g3(x):
        return 1 -(x[0] - 4*x[1])

    g = (g1, g2, g3)

    x0 = [1.5, 1.0]

horst1 = Horst1(bounds=[(0, 3), (0, 2)],
              expected_fun= -1.0625,
              expected_x=[0.75, 2.0])

class Horst2(TestFunction):
    def f(self, x):
        return -x[0]**2 - x[1]**(3/2.0)

    def g1(x):
        return 4 -(x[0] + 2*x[1])
    def g2(x):
        return 1 -(x[0] - 2*x[1])
    def g3(x):
        return 1 -(-x[0] + x[1])

    g = (g1, g2, g3)

    x0 = [1.0, 1.0]

horst2 = Horst2(bounds=[(0, 2.5), (0, 2.0)],
              expected_fun=-6.8995,
              expected_x=[2.5, 0.75])

class Horst3(TestFunction):
    def f(self, x):
        return -x[0]**2 + 4/3.0 * x[0] + numpy.log(1 + x[1]) - 4/9.0

    def g1(x):
        return 1 - (-2*x[0] +x[1])
    def g2(x):
        return 3/2.0 - (x[0] + x[1])
    def g3(x):
        return 1 - (x[0] + 1/10.0 * x[1])
    g = (g1, g2, g3)

    x0 = [1.0, 1.0]

horst3 = Horst3(bounds=[(0, 1.5),]*2,
              expected_fun=-4/9.0,
              expected_x=[0.0, 0.0])

class Horst4(TestFunction):
    def f(self, x):
        return - (numpy.abs(x[0] + 0.5*x[1] + 2/3.0 * x[2]))**(3/2.0)

    def g1(x):
        return 6 - (x[0] + x[1] + 2*x[2])

    def g2(x):
        return 2 - (x[0] + 0.5*x[1])

    def g3(x):
        return -1 - (-x[1] - 2*x[2])

    g = (g1, g2, g3)

    x0 = [0.0, 0.0, 0.0]

horst4 = Horst4(bounds=[(0, 2.0), (0, 2.0), (0, 2.0)],
              expected_fun=-6.0858,
              expected_x=[2.0, 0.0, 2.0])

class Horst5(TestFunction):
    def f(self, x):
        return - (numpy.abs(x[0] + 0.5*x[1] + 2/3.0 * x[2]))**(3/2.0) - x[0]**2

    def g1(x):
        return 2 - (x[0] + x[1] + x[2])

    def g2(x):
        return 1 - (x[0] + x[1] - (1/4.0)*x[2])

    def g3(x):
        return -1 - (-2*x[0] - 2*x[1] + x[2])

    g = (g1, g2, g3)

    x0 = [1.0, 1.0, 1.5]

horst5 = Horst5(bounds=[(0, 2), (0, 2), (0, 3)],
                expected_fun=-3.722,
                expected_x=[1.2, 0.0, 0.8]
                )

class Horst6(TestFunction):
    def f(self, x):
        x = numpy.atleast_2d(x).T
        Q = numpy.array([[ 0.992934, -0.640117, 0.337286],
                         [-0.640117, -0.814622, 0.960807],
                         [ 0.337286,  0.960807, 0.500874]])

        q = numpy.array([[-0.992372],
                         [-0.046466],
                         [ 0.891766]])

        xT_Q = numpy.dot(x.T, Q)
        return numpy.dot(xT_Q, x) + numpy.dot(q.T, x)

    def g1(x):
        return  2.865062 - ( 0.488509*x[0] + 0.063565*x[1] + 0.945686*x[2])
    def g2(x):
        return -1.491608 - (-0.578592*x[0] - 0.324014*x[1] - 0.501754*x[2])
    def g3(x):
        return  0.519588 - (-0.719203*x[0] + 0.099562*x[1] + 0.445225*x[2])
    def g4(x):
        return  1.584087 - (-0.346896*x[0] + 0.637939*x[1] - 0.257623*x[2])
    def g5(x):
        # Paulavius textbook:
        #return  2.198036 - (-0.202821*x[0] + 0.647361*x[1] + 0.920135*x[2])
        # https://link.springer.com/content/pdf/10.1007/BF00429750.pdf
        return  2.198036 - (+0.202821*x[0] + 0.647361*x[1] + 0.920135*x[2])
    def g6(x):
        return -1.301853 - (-0.983091*x[0] - 0.886420*x[1] - 0.802444*x[2])
    def g7(x):
        return -0.738290 - (-0.305441*x[0] - 0.180123*x[1] - 0.515399*x[2])

    g = (g1,g2,g3,g4,g5,g6,g6,g7)

    x0 = [1.0,]*7

horst6 = Horst6(bounds=[(0, 10),]*3,
              expected_fun=-31.5285,
              expected_x=[5.210677, 5.027908, 0.0])

class Horst7(TestFunction):
    def f(self, x):
        return -(x[0] + (1/2.0) * x[2] - 2)**2 - (numpy.abs(x[0] + 0.5*x[1] + 2/3.0 * x[2]))**(3/2.0)

    def g1(x):
        return 1 - (-x[0] - x[1] + (1/2.0)*x[2])
    def g2(x):
        return 6 - (x[0] + 2*x[1])
    def g3(x):
        return - 1 + (2*x[0] + 4*x[1] + 2*x[2])
        #return -(- 1 + (2*x[0] + 4*x[1] + 2*x[2]))

    g = (g1,g2,g3)

    x0 = [1.0, 1.0]

horst7 = Horst7(bounds=[(0, 6),(0, 1), (0, 3)],
              #expected_fun=-44.859,  # LITERATURE VALUE
              expected_fun=-52.87741699796952,
              #expected_x=[6.0, 0.0, 2.0],  # LITERATURE VALUE
              expected_x=[ 6.,  0.,  3.]
                )

class Hs021(TestFunction):
    def f(self, x):
        return x[0]**2/100 + x[1]**2 - 100

    def g1(x):
        return 10*x[0] - x[1] - 10

    g = (g1,)

    x0 = [-1, -1]

hs021 = Hs021(bounds=[(2, 50), (-50, 50)],
              expected_fun=-99.96,
              expected_x=[2.0, 0.0])

class Hs024(TestFunction):
    def f(self, x):
        return (((x[0] - 3)**2 - 9.0) * x[1]**3) / (27.0*numpy.sqrt(3))

    def g1(x):
        return (x[0]/numpy.sqrt(3) - x[1])

    def g2(x):
        return (x[0] + numpy.sqrt(3)*x[1])

    def g3(x):
        return -x[0] - numpy.sqrt(3)*x[1] + 6

    g = (g1, g2, g3)

    x0 = [1.0, 0.5]

hs024 = Hs024(bounds=[(0, 5),]*2,  #TODO: (0, None) ?
              expected_fun=-1.0,
              expected_x=[3.0, numpy.sqrt(3)])

class Hs035(TestFunction):
    def f(self, x):
        return 9 - 8*x[0] - 6*x[1] - 4*x[2] + 2*x[0]**2 + 2*x[1]**2 + x[2]**2+ 2*x[0]*x[1] + 2*x[0]*x[2]

    def g1(x):
        return -(x[0] + x[1] + 2*x[2] - 3)

    g = (g1,)

    x0 = [0.5, 0.5, 0.5]

hs035 = Hs035(bounds=[(0, 5),]*3,  #TODO: (0, None) ?
              expected_fun=1/9.0,
              expected_x=[4/3.0, 7/9.0, 4/9.0])

class Hs036(TestFunction):
    def f(self, x):
        return -x[0]*x[1]*x[2]

    def g1(x):
        #return (x[0] + 2*x[1] + 2*x[2] - 72)
        return -x[0] - 2*x[1] - 2*x[2] + 72

    g = (g1,)

    x0 = [10.0, 10.0, 10.0]

hs036 = Hs036(bounds=[(0, 20),(0, 11),(0, 42)],
              expected_fun=-3300, # LITERATURE VALUE
              #expected_fun=-9240.0,
              expected_x=[20, 11, 15] # LITERATURE VALUE
              #expected_x=[ 20.,  11.,  42.]
              )

class Hs037(TestFunction):
    def f(self, x):
        return -x[0]*x[1]*x[2]

    def g1(x):
        return -(x[0] + 2*x[1] + 2*x[2] - 72)

    def g2(x):
        return x[0] + 2*x[1] + 2*x[2]

    g = (g1, g2)

    x0 = [10, 10, 10]

hs037 = Hs037(bounds=[(0, 42),]*3,
              expected_fun=-3456,
              expected_x=[24, 12, 12])

class Hs038(TestFunction):
    # Colville function in SciPy suite
    def f(self, x):
        return (100.0 * (x[0] - x[1] ** 2) ** 2
                + (1 - x[0]) ** 2 + (1 - x[2]) ** 2
                + 90 * (x[3] - x[2] ** 2) ** 2
                + 10.1 * ((x[1] - 1) ** 2 + (x[3] - 1) ** 2)
                + 19.8 * (x[1] - 1) * (x[3] - 1))

    def g1(x):
        return None

    g = None

    x0 = [0.0, 0.0, 0.0, 0.0]

hs038 = Hs038(bounds=[(-10, 10),]*4,
              expected_fun=0.0,
              expected_x=[1, 1, 1, 1])

class Hs044(TestFunction):
    def f(self, x):
        return x[0] - x[1] - x[2] - x[0]*x[2] + x[0]*x[3] + x[1]*x[2] - x[1]*x[3]

    def g1(x):
        return -(x[0] + 2*x[1] - 8.0)

    def g2(x):
        return -( 4*x[0] + x[1] - 12.0)

    def g3(x):
        return -(3*x[0] + 4*x[1] - 12.0)

    def g4(x):
        return -(2*x[2] + x[3] - 8.0)

    def g5(x):
        return -(x[2] + 2*x[3] - 8.0)

    def g6(x):
        return -(x[2] + x[3] - 5.0)

    g = (g1, g2, g3, g4, g5, g6)

    x0 = [0.0, 0.0, 0.0, 0.0]

hs044 = Hs044(bounds=[(0.0, 5.0),]*4,  #TODO: (0, None) ?
              expected_fun=-15.0,
              expected_x=[0.0, 3.0, 0.0, 4.0])

class Hs076(TestFunction):
    def f(self, x):
        return x[0]**2 + 0.5*x[1]**2 + x[2]**2 + 0.5*x[3]**2 - x[0]*x[2] + x[2]*x[3]- x[0] - 3*x[1] + x[2] - x[3]

    def g1(x):
        return -(x[0] + 2*x[1] + x[2] + x[3] - 5)

    def g2(x):
        return -(3*x[0] + x[1] + 2*x[2] - x[3] - 4)

    def g3(x):
        return (x[1] + 4*x[2] - 1.5)

    g = (g1, g2, g3)

    x0 = [0.5, 0.5, 0.5, 0.5]

hs076 = Hs076(bounds=[(0, 5),]*4,
              expected_fun=-4.681818181,
              expected_x=[0.2727273, 2.09090, 0.26e-10, 0.5454545])

class S224(TestFunction):
    def f(self, x):
        return 2*x[0]**2 + x[1]**2 - 48*x[0] - 40*x[1]

    def g1(x):
        return x[0] + 3*x[1]

    def g2(x):
        return 18 - x[0] - 3*x[1]

    def g3(x):
        return x[0] + x[1]

    def g4(x):
        return 8 - x[0] - x[1]

    g = (g1,g2,g3,g4)

    x0 = [0.1, 0.1]

s224 = S224(bounds=[(0, 6),]*2,
              expected_fun=-304.0,
              expected_x=[4.0, 4.0])

class S231(TestFunction):
    def f(self, x):
        return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

    def g1(x):
        return x[0]/3 + x[1] + 0.1

    def g2(x):
        return -1*x[0]/3 + x[1] + 0.1

    g = (g1,g2)

    x0 = [-1.2, 1]

s231 = S231(bounds=[(-2, 2),]*2,
                expected_fun=0.0,
                expected_x=[1.0, 1.0])

class S232(TestFunction):
    def f(self, x):
        return -1*(9-(x[0] - 3)**2)*(x[1]**3/(27*numpy.sqrt(3)))

    def g1(x):
        return (x[0] / numpy.sqrt(3) - x[1])

    def g2(x):
        return x[0]/numpy.sqrt(3) - x[1]

    def g3(x):
        return 6 - x[0] - numpy.sqrt(3) * x[1]

    g = (g1, g2, g3)

    x0 = [1.0, 0.5]

s232 = S232(bounds=[(0, 5), ] * 2,  # TODO: (0, None) ?
              expected_fun=-1.0,
              expected_x=[3.0, numpy.sqrt(3)])

class S250(TestFunction):
    def f(self, x):
        return -x[0] * x[1] * x[2]

    def g1(x):
        # return (x[0] + 2 * x[1] + 2 * x[2] - 72)
        return -x[0] - 2 * x[1] - 2 * x[2] + 72

    g = (g1,)

    x0 = [10.0, 10.0, 10.0]

s250 = S250(bounds=[(0, 20), (0, 11), (0, 42)],
              expected_fun=-3300,
              expected_x=[20, 11, 15])

class S251(TestFunction):
    def f(self, x):
        return -x[0] * x[1] * x[2]

    def g1(x):
        return -(x[0] + 2 * x[1] + 2 * x[2] - 72)

    def g2(x):
        return x[0] + 2 * x[1] + 2 * x[2]

    g = (g1, g2)

    x0 = [10, 10, 10]

s251 = S251(bounds=[(0, 42),]*3,
              expected_fun=-3456,
              expected_x=[24, 12, 12])


class Bunnag1(TestFunction):
    def f(self, x):
        return 9-8*x[0]-6*x[1]-4*x[2]+2*x[0]**2+2*x[1]**2+x[2]**2+2*x[0]*x[1]+2*x[0]*x[2]

    def g1(x):
       return -(x[0]+x[1]+2*x[2] - 3)

    g = (g1,)

bunnag1 = Bunnag1(bounds=[(0, 3),]*3,
                  # Try to find lower
                  expected_fun=1/9.0,  # Exact analytical solution
                  expected_x=[1.33333327,  0.77777775,  0.44444449])


class Bunnag1(TestFunction):
    def f(self, x):
        return 9-8*x[0]-6*x[1]-4*x[2]+2*x[0]**2+2*x[1]**2+x[2]**2+2*x[0]*x[1]+2*x[0]*x[2]

    def g1(x):
       return -(x[0]+x[1]+2*x[2] - 3)

    g = (g1,)

bunnag1 = Bunnag1(bounds=[(0, 3),]*3,
                  # Try to find lower
                  expected_fun=1/9.0,  # Exact analytical solution
                  expected_x=[1.33333327,  0.77777775,  0.44444449])


class Bunnag2(TestFunction):
    def f(self, x):
        return x[0]**0.6 + 2*x[1]**0.6 - 2*x[1] + 2*x[2] - x[3]

    def g1(x):
       return -(x[0]+2*x[2] - 4)

    def g2(x):
        return -(-3*x[0]+x[3] - 1)

    g = (g1, g2)

bunnag2 = Bunnag2(bounds=[(0, 4),]*4,
                  #expected_fun=-2.07,
                  expected_fun=-6.4052065800118605,  # shgo solution
                  #expected_x=[4/3.0, 4, 0, 0]
                  expected_x=[1.,  4.,  0.,  4.]  # shgo solution
                  )


    # Converts the interger indices of a .mod file string to a Python
    # compatible function"
def str_conv(in_str):
    out_str = ''
    int_char = False
    for char in in_str:
        if (char is not '[') and (not int_char):
            if char is ';':
                continue
            else:
                if char == '^':
                    out_str += '**'
                    continue
                out_str += char
        elif char is '[':
            out_str += char
            int_char = True
        elif int_char:
            int_char_str = int(char) - 1
            int_char_str = str(int_char_str)
            out_str += int_char_str
            int_char = False
    print(out_str)
    return out_str

def str_conv_cons(in_str):
    out_str = '-('
    int_char = False
    for char in in_str:
        if (char is not '[') and (not int_char):
            if char is ';':
                continue
            else:
                if char == '^':
                    out_str += '**'
                    continue
                out_str += char
        elif char is '[':
            out_str += char
            int_char = True
        elif int_char:
            int_char_str = int(char) - 1
            int_char_str = str(int_char_str)
            out_str += int_char_str
            int_char = False
    print(out_str)
    return out_str

def sanity_test(test, tol=1e-5):
    # Test if a minima is at expected solution and feasible
    print("=" * 30)
    print(test.__class__.__name__)
    print("=" * 6)
    good = True
    sol = test.f(test.expected_x)
    diff = test.expected_fun - sol
    print('f* - f(x*) = {}'.format(diff))
    if abs(diff) <= tol:
        print("GOOD")
    else:
        print("BAD")
        good = False
    if test.g is not None:
        for i, gcons in enumerate(test.g):
            con = gcons(test.expected_x)
            print('cons{} = {}'.format(i, con))
            if con < 0:
                print("UNFEASIBLE BOUNDS")
                good = False

    for i, x_b in enumerate(test.bounds):
        if test.expected_x[i] < x_b[0]:
            good = False
            print('x* = {} is out of lower bounds {}'.format(test.expected_x, x_b[0]))
        if test.expected_x[i] > x_b[1]:
            good = False
            print(f'x* = {} is out of upper bounds {}'.format(test.expected_x, x_b[1]))
    print("="*2)
    print(test.__class__.__name__)
    if good:
        print("ALL GOOD")
    else:
        print("BAD "*10)
    #print(f'f* - f(x*) = {expected_fun - sol}')

if __name__ == '__main__':
    #str_conv('-1*x[1]*x[2]*x[3];')
    #str_conv('x[1]/sqrt(3) - x[2] >= 0;')
    #str_conv('6 - x[1] - sqrt(3)*x[2] >= 0')

    sanity_test(horst6, tol=1e-3)

    if 0:  # HS and s### set
        sanity_test(hs021)
        sanity_test(hs024)
        sanity_test(hs035)
        sanity_test(hs036)
        sanity_test(hs037)
        sanity_test(hs038)
        sanity_test(hs044)
        sanity_test(hs076)
        sanity_test(s224)
        sanity_test(s231)
        sanity_test(s250)
        sanity_test(s251)

    if 0:  # Horst problem set
        sanity_test(horst1)
        sanity_test(horst2,tol=1e-4)
        sanity_test(horst3)
        sanity_test(horst4)
        sanity_test(horst5, tol=1e-4)
        sanity_test(horst6)
        sanity_test(horst7, tol=1e-3)

    if 0: # bunnag problem set
        sanity_test(bunnag1)
        sanity_test(bunnag2)

    test = horst6
    test = bunnag2
    if 0:
        for iter in range(1,7):
            res = shgo(test.f, test.bounds, g_cons=test.g, iter=iter)
            print('iterations = {}'.format(iter))
            #print(res)
            print('fun = {}'.format(res.fun))
            print('x = {}'.format(res.x))


    if 0:  # bunnag
        print(bunnag1.f([3,2,3.36]))
        print(bunnag1.g[0]([3,2,3.36]))
        print(shgo(bunnag1.f, bunnag1.bounds, g_cons=bunnag1.g, iter=1))
        #print(shgo(bunnag1.f, bunnag1.bounds, g_cons=bunnag1.g, iter=3))
        #print(shgo(bunnag1.f, bunnag1.bounds, g_cons=bunnag1.g, iter=5))

        #res = shgo(bunnag1.f, bunnag1.bounds, g_cons=bunnag1.g, iter=7)
        res = shgo(bunnag2.f, bunnag2.bounds, g_cons=bunnag2.g, iter=1)
        print(res)
        print(bunnag2.g[0](res.x))
        print(bunnag2.g[1](res.x))
        print(bunnag2.f([4/3.0, 4, 0, 0]))
        print(bunnag2.f(res.x))

        def cons(x):
            print(x[0]+2*x[2]<=4)
            print(-3*x[0]+x[3]<=1)
            return

        cons(res.x)

        print(bunnag2.f([2,4,0,4]))
        print(bunnag2.f([4/3.0, 4.0, 0.0, 0.0]))

    #print(shgo(bunnag1.f, bunnag1.bounds, g_cons=bunnag1.g, iter=1))
    #print(shgo(bunnag1.f, bunnag1.bounds, g_cons=bunnag1.g, iter=2))