#%%
from auto_test import auto_test
from functools import lru_cache,cache

#%% 
@auto_test()
@cache
def fib(n):
    if n < 2:
        return n
    return fib(n - 2) + fib(n - 1)
# %%
fib(5)
