import collections, time, functools

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    """
    Arrow used in the plotting of 3D vecotrs

    ex.
    a = Arrow3D([0, 1], [0, 1], [0, 1], mutation_scale=20,
            lw=1, arrowstyle="-|>", color="k")
    ax.add_artist(a)
    """
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


# Note to avoid using external packages such as functools32 we use this code
# only using the standard library
def lru_cache(maxsize=255, timeout=None):
    """
    Thanks to ilialuk @ https://stackoverflow.com/users/2121105/ilialuk for
    this code snippet. Modifications by S. Endres
    """

    class LruCacheClass(object):
        def __init__(self, input_func, max_size, timeout):
            self._input_func = input_func
            self._max_size = max_size
            self._timeout = timeout

            # This will store the cache for this function,
            # format - {caller1 : [OrderedDict1, last_refresh_time1],
            #  caller2 : [OrderedDict2, last_refresh_time2]}.
            #   In case of an instance method - the caller is the instance,
            # in case called from a regular function - the caller is None.
            self._caches_dict = {}

        def cache_clear(self, caller=None):
            # Remove the cache for the caller, only if exists:
            if caller in self._caches_dict:
                del self._caches_dict[caller]
                self._caches_dict[caller] = [collections.OrderedDict(),
                                             time.time()]

        def __get__(self, obj, objtype):
            """ Called for instance methods """
            return_func = functools.partial(self._cache_wrapper, obj)
            return_func.cache_clear = functools.partial(self.cache_clear,
                                                        obj)
            # Return the wrapped function and wraps it to maintain the
            # docstring and the name of the original function:
            return functools.wraps(self._input_func)(return_func)

        def __call__(self, *args, **kwargs):
            """ Called for regular functions """
            return self._cache_wrapper(None, *args, **kwargs)

        # Set the cache_clear function in the __call__ operator:
        __call__.cache_clear = cache_clear

        def _cache_wrapper(self, caller, *args, **kwargs):
            # Create a unique key including the types (in order to
            # differentiate between 1 and '1'):
            kwargs_key = "".join(map(
                lambda x: str(x) + str(type(kwargs[x])) + str(kwargs[x]),
                sorted(kwargs)))
            key = "".join(
                map(lambda x: str(type(x)) + str(x), args)) + kwargs_key

            # Check if caller exists, if not create one:
            if caller not in self._caches_dict:
                self._caches_dict[caller] = [collections.OrderedDict(),
                                             time.time()]
            else:
                # Validate in case the refresh time has passed:
                if self._timeout is not None:
                    if (time.time() - self._caches_dict[caller][1]
                            > self._timeout):
                        self.cache_clear(caller)

            # Check if the key exists, if so - return it:
            cur_caller_cache_dict = self._caches_dict[caller][0]
            if key in cur_caller_cache_dict:
                return cur_caller_cache_dict[key]

            # Validate we didn't exceed the max_size:
            if len(cur_caller_cache_dict) >= self._max_size:
                # Delete the first item in the dict:
                try:
                    cur_caller_cache_dict.popitem(False)
                except KeyError:
                    pass
            # Call the function and store the data in the cache (call it
            # with the caller in case it's an instance function)
            if caller is not None:
                args = (caller,) + args
            cur_caller_cache_dict[key] = self._input_func(*args, **kwargs)

            return cur_caller_cache_dict[key]

    # Return the decorator wrapping the class (also wraps the instance to
    # maintain the docstring and the name of the original function):
    return (lambda input_func: functools.wraps(input_func)(
        LruCacheClass(input_func, maxsize, timeout)))

