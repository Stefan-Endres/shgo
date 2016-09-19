class Point:
    def __init__(self, position, neighbours=None):
        self.position = position
        if neighbours is not None:
            self.neighbours = neighbours
        else:
            self.neighbours = []
        self.fval = None

p1 = Point([0, 0])
p2 = Point([0, 1])
p3 = Point([1, 0])
p4 = Point([1, 1])
p1.neighbours = [p2, p3, p4]

v = [p1, p2, p3, p4]

import time

class FunctionCache:
    def __init__(self, f):
        self.f = f
        self.cache = {}
    def __call__(self, x):
        if x in self.cache:
            return self.cache[x]
        else:
            value = self.f(x)
            self.cache[x] = value
            return value

def objective_fun(x):
    time.sleep(1)
    return sum(x)

print("blah")

def f2(blac):
    a = objective_fun
    objective_fun = 2

print("after")

f2(2)

f_cached = FunctionCache(objective_fun)

print('eval1')
print(f_cached((1, 2, 3)))
print('eval2')
print(f_cached((1, 2, 3)))
print(f_cached.cache)


