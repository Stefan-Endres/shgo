"""
  Licensing:
    This code is distributed under the GNU LGPL license.

  Authors:
    Original FORTRAN77 version of i4_sobol by Bennett Fox.
    MATLAB version by John Burkardt.
    PYTHON version by Corrado Chisari

    Original Python version of is_prime by Corrado Chisari

    Original MATLAB versions of other functions by John Burkardt.
    PYTHON versions by Corrado Chisari

    Original code is available from http://people.sc.fsu.edu/~jburkardt/py_src/sobol/sobol.html
"""

from __future__ import division
import numpy as np
from scipy.stats import norm

__all__ = ['i4_bit_hi1', 'i4_bit_lo0', 'i4_sobol_generate',
           'i4_sobol', 'i4_uniform', 'prime_ge', 'is_prime']


def i4_bit_hi1(n):
    """
    i4_bit_hi1 returns the position of the high 1 bit base 2 in an integer.

    Example:
      +------+-------------+-----
      |    N |      Binary | BIT
      +------|-------------+-----
      |    0 |           0 |   0
      |    1 |           1 |   1
      |    2 |          10 |   2
      |    3 |          11 |   2
      |    4 |         100 |   3
      |    5 |         101 |   3
      |    6 |         110 |   3
      |    7 |         111 |   3
      |    8 |        1000 |   4
      |    9 |        1001 |   4
      |   10 |        1010 |   4
      |   11 |        1011 |   4
      |   12 |        1100 |   4
      |   13 |        1101 |   4
      |   14 |        1110 |   4
      |   15 |        1111 |   4
      |   16 |       10000 |   5
      |   17 |       10001 |   5
      | 1023 |  1111111111 |  10
      | 1024 | 10000000000 |  11
      | 1025 | 10000000001 |  11

    Parameters:
      Input, integer N, the integer to be measured.
      N should be nonnegative.  If N is nonpositive, the value will always be 0.

      Output, integer BIT, the number of bits base 2.
    """
    i = np.floor(n)
    bit = 0
    while i > 0:
        bit += 1
        i //= 2
    return bit


def i4_bit_lo0(n):
    """
    I4_BIT_LO0 returns the position of the low 0 bit base 2 in an integer.

    Example:
      +------+------------+----
      |    N |     Binary | BIT
      +------+------------+----
      |    0 |          0 |   1
      |    1 |          1 |   2
      |    2 |         10 |   1
      |    3 |         11 |   3
      |    4 |        100 |   1
      |    5 |        101 |   2
      |    6 |        110 |   1
      |    7 |        111 |   4
      |    8 |       1000 |   1
      |    9 |       1001 |   2
      |   10 |       1010 |   1
      |   11 |       1011 |   3
      |   12 |       1100 |   1
      |   13 |       1101 |   2
      |   14 |       1110 |   1
      |   15 |       1111 |   5
      |   16 |      10000 |   1
      |   17 |      10001 |   2
      | 1023 | 1111111111 |   1
      | 1024 | 0000000000 |   1
      | 1025 | 0000000001 |   1

    Parameters:
      Input, integer N, the integer to be measured.
      N should be nonnegative.

      Output, integer BIT, the position of the low 1 bit.
    """
    bit = 1
    i = np.floor(n)
    while i != 2 * (i // 2):
        bit += 1
        i //= 2
    return bit


def i4_sobol_generate(dim_num, n, skip=1):
    """
    i4_sobol_generate generates a Sobol dataset.

    Parameters:
      Input, integer dim_num, the spatial dimension.
      Input, integer N, the number of points to generate.
      Input, integer SKIP, the number of initial points to skip.

      Output, real R(M,N), the points.
    """
    r = np.full((n, dim_num), np.nan)
    for j in range(n):
        seed = j + skip
        r[j, 0:dim_num], next_seed = i4_sobol(dim_num, seed)

    return r


def i4_sobol_generate_std_normal(dim_num, n, skip=1):
    """
    Generates multivariate standard normal quasi-random variables.

    Parameters:
      Input, integer dim_num, the spatial dimension.
      Input, integer n, the number of points to generate.
      Input, integer SKIP, the number of initial points to skip.

      Output, real np array of shape (n, dim_num).
    """

    sobols = i4_sobol_generate(dim_num, n, skip)

    normals = norm.ppf(sobols)

    return normals


def i4_sobol(dim_num, seed):
    """
    i4_sobol generates a new quasirandom Sobol vector with each call.

    Discussion:
      The routine adapts the ideas of Antonov and Saleev.

    Reference:
      Antonov, Saleev,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 19, 1980, pages 252 - 256.

      Paul Bratley, Bennett Fox,
      Algorithm 659:
      Implementing Sobol's Quasirandom Sequence Generator,
      ACM Transactions on Mathematical Software,
      Volume 14, Number 1, pages 88-100, 1988.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Ilya Sobol,
      USSR Computational Mathematics and Mathematical Physics,
      Volume 16, pages 236-242, 1977.

      Ilya Sobol, Levitan,
      The Production of Points Uniformly Distributed in a Multidimensional
      Cube (in Russian),
      Preprint IPM Akad. Nauk SSSR,
      Number 40, Moscow 1976.

    Parameters:
      Input, integer DIM_NUM, the number of spatial dimensions.
      DIM_NUM must satisfy 1 <= DIM_NUM <= 40.

      Input/output, integer SEED, the "seed" for the sequence.
      This is essentially the index in the sequence of the quasirandom
      value to be generated.  On output, SEED has been set to the
      appropriate next value, usually simply SEED+1.
      If SEED is less than 0 on input, it is treated as though it were 0.
      An input value of 0 requests the first (0-th) element of the sequence.

      Output, real QUASI(DIM_NUM), the next quasirandom vector.
    """
    global atmost
    global dim_max
    global dim_num_save
    global initialized
    global lastq
    global log_max
    global maxcol
    global poly
    global recipd
    global seed_save
    global v

    if 'initialized' not in list(globals().keys()):
        initialized = 0
        dim_num_save = -1

    if not initialized or dim_num != dim_num_save:
        initialized = 1
        dim_max = 40
        dim_num_save = -1
        log_max = 30
        seed_save = -1

        #  Initialize (part of) V.
        v = np.zeros((dim_max, log_max))
        v[0:40, 0] = np.transpose([
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

        v[2:40, 1] = np.transpose([
                  1, 3, 1, 3, 1, 3, 3, 1,
            3, 1, 3, 1, 3, 1, 1, 3, 1, 3,
            1, 3, 1, 3, 3, 1, 3, 1, 3, 1,
            3, 1, 1, 3, 1, 3, 1, 3, 1, 3])

        v[3:40, 2] = np.transpose([
                     7, 5, 1, 3, 3, 7, 5,
            5, 7, 7, 1, 3, 3, 7, 5, 1, 1,
            5, 3, 3, 1, 7, 5, 1, 3, 3, 7,
            5, 1, 1, 5, 7, 7, 5, 1, 3, 3])

        v[5:40, 3] = np.transpose([
                                1, 7,  9,  13, 11,
            1, 3,  7,  9,  5,  13, 13, 11, 3,  15,
            5, 3,  15, 7,  9,  13, 9,  1,  11, 7,
            5, 15, 1,  15, 11, 5,  3,  1,  7,  9])

        v[7:40, 4] = np.transpose([
                                        9,  3,  27,
            15, 29, 21, 23, 19, 11, 25, 7,  13, 17,
            1,  25, 29, 3,  31, 11, 5,  23, 27, 19,
            21, 5,  1,  17, 13, 7,  15, 9,  31, 9])

        v[13:40, 5] = np.transpose([
                        37, 33, 7,  5,  11, 39, 63,
            27, 17, 15, 23, 29, 3,  21, 13, 31, 25,
            9,  49, 33, 19, 29, 11, 19, 27, 15, 25])

        v[19:40, 6] = np.transpose([
                                                   13,
            33, 115, 41, 79, 17, 29,  119, 75, 73, 105,
            7,  59,  65, 21, 3,  113, 61,  89, 45, 107])

        v[37:40, 7] = np.transpose([
            7, 23, 39])

        #  Set POLY.
        poly = [
            1,   3,   7,   11,  13,  19,  25,  37,  59,  47,
            61,  55,  41,  67,  97,  91,  109, 103, 115, 131,
            193, 137, 145, 143, 241, 157, 185, 167, 229, 171,
            213, 191, 253, 203, 211, 239, 247, 285, 369, 299]

        atmost = 2 ** log_max - 1

        #  Find the number of bits in ATMOST.
        maxcol = i4_bit_hi1(atmost)

        #  Initialize row 1 of V.
        v[0, 0:maxcol] = 1


    #  Things to do only if the dimension changed.
    if dim_num != dim_num_save:

        #  Check parameters.
        if dim_num < 1 or dim_max < dim_num:
            print('I4_SOBOL - Fatal error!')
            print('  The spatial dimension DIM_NUM should satisfy:')
            print('    1 <= DIM_NUM <= %d' % dim_max)
            print('  But this input value is DIM_NUM = %d' % dim_num)
            return

        dim_num_save = dim_num

        #  Initialize the remaining rows of V.
        for i in range(2, dim_num + 1):

            #  The bits of the integer POLY(I) gives the form of polynomial I.
            #  Find the degree of polynomial I from binary encoding.
            j = poly[i - 1]
            m = 0
            j //= 2
            while j > 0:
                j //= 2
                m += 1

            #  Expand this bit pattern to separate components of the logical array INCLUD.
            j = poly[i - 1]
            includ = np.zeros(m)
            for k in range(m, 0, -1):
                j2 = j // 2
                includ[k - 1] = (j != 2 * j2)
                j = j2

            #  Calculate the remaining elements of row I as explained
            #  in Bratley and Fox, section 2.
            for j in range(m + 1, maxcol + 1):
                newv = v[i - 1, j - m - 1]
                l = 1
                for k in range(1, m + 1):
                    l *= 2
                    if includ[k - 1]:
                        newv = np.bitwise_xor(
                            int(newv), int(l * v[i - 1, j - k - 1]))
                v[i - 1, j - 1] = newv

        #  Multiply columns of V by appropriate power of 2.
        l = 1
        for j in range(maxcol - 1, 0, -1):
            l *= 2
            v[0:dim_num, j - 1] = v[0:dim_num, j - 1] * l

        #  RECIPD is 1/(common denominator of the elements in V).
        recipd = 1.0 / (2 * l)
        lastq = np.zeros(dim_num)

    seed = int(np.floor(seed))

    if seed < 0:
        seed = 0

    l = 1
    if seed == 0:
        lastq = np.zeros(dim_num)

    elif seed == seed_save + 1:

        #  Find the position of the right-hand zero in SEED.
        l = i4_bit_lo0(seed)

    elif seed <= seed_save:

        seed_save = 0
        lastq = np.zeros(dim_num)

        for seed_temp in range(int(seed_save), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(
                    int(lastq[i - 1]), int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)

    elif seed_save + 1 < seed:

        for seed_temp in range(int(seed_save + 1), int(seed)):
            l = i4_bit_lo0(seed_temp)
            for i in range(1, dim_num + 1):
                lastq[i - 1] = np.bitwise_xor(
                    int(lastq[i - 1]), int(v[i - 1, l - 1]))

        l = i4_bit_lo0(seed)

    #  Check that the user is not calling too many times!
    if maxcol < l:
        print('I4_SOBOL - Fatal error!')
        print('  Too many calls!')
        print('  MAXCOL = %d\n' % maxcol)
        print('  L =      %d\n' % l)
        return

    #  Calculate the new components of QUASI.
    quasi = np.zeros(dim_num)
    for i in range(1, dim_num + 1):
        quasi[i - 1] = lastq[i - 1] * recipd
        lastq[i - 1] = np.bitwise_xor(
            int(lastq[i - 1]), int(v[i - 1, l - 1]))

    seed_save = seed
    seed += 1

    return [quasi, seed]


def i4_uniform(a, b, seed):
    """
    i4_uniform returns a scaled pseudorandom I4.

    Discussion:
      The pseudorandom number will be scaled to be uniformly distributed
      between A and B.

    Reference:
      Paul Bratley, Bennett Fox, Linus Schrage,
      A Guide to Simulation,
      Springer Verlag, pages 201-202, 1983.

      Pierre L'Ecuyer,
      Random Number Generation,
      in Handbook of Simulation,
      edited by Jerry Banks,
      Wiley Interscience, page 95, 1998.

      Bennett Fox,
      Algorithm 647:
      Implementation and Relative Efficiency of Quasirandom
      Sequence Generators,
      ACM Transactions on Mathematical Software,
      Volume 12, Number 4, pages 362-376, 1986.

      Peter Lewis, Allen Goodman, James Miller
      A Pseudo-Random Number Generator for the System/360,
      IBM Systems Journal,
      Volume 8, pages 136-143, 1969.

    Parameters:
      Input, integer A, B, the minimum and maximum acceptable values.
      Input, integer SEED, a seed for the random number generator.

      Output, integer C, the randomly chosen integer.
      Output, integer SEED, the updated seed.
    """
    if seed == 0:
        print('I4_UNIFORM - Fatal error!')
        print('  Input SEED = 0!')

    seed = np.floor(seed)
    a = round(a)
    b = round(b)

    seed = np.mod(seed, 2147483647)

    if seed < 0:
        seed += 2147483647

    k = seed // 127773

    seed = 16807 * (seed - k * 127773) - k * 2836

    if seed < 0:
        seed += 2147483647

    r = seed * 4.656612875E-10

    #  Scale R to lie between A-0.5 and B+0.5.
    r = (1.0 - r) * (min(a, b) - 0.5) + r * (max(a, b) + 0.5)

    #  Use rounding to convert R to an integer between A and B.
    value = round(r)

    value = max(value, min(a, b))
    value = min(value, max(a, b))

    c = value

    return [int(c), int(seed)]


def prime_ge(n):
    """
    PRIME_GE returns the smallest prime greater than or equal to N.

    Example:
      +-----+---------
      |   N | PRIME_GE
      +-----+---------
      | -10 |        2
      |   1 |        2
      |   2 |        2
      |   3 |        3
      |   4 |        5
      |   5 |        5
      |   6 |        7
      |   7 |        7
      |   8 |       11
      |   9 |       11
      |  10 |       11

    Parameters:
      Input, integer N, the number to be bounded.

      Output, integer P, the smallest prime number that is greater
      than or equal to N.
    """
    p = max(np.ceil(n), 2)
    while not is_prime(p):
        p += 1

    return p


def is_prime(n):
    """
    is_prime returns True if N is a prime number, False otherwise

    Parameters:
       Input, integer N, the number to be checked.

       Output, boolean value, True or False
    """
    if n != int(n) or n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    # All primes >3 are of the form 6n+1 or 6n+5 (6n, 6n+2, 6n+4 are 2-divisible, 6n+3 is 3-divisible)
    p = 5
    root = int(np.ceil(np.sqrt(n)))
    while p <= root:
        if n % p == 0 or n % (p + 2) == 0:
            return False
        p += 6
    return True
