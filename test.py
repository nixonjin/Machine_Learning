#%%
import timeit

TIMES = 10000

SETUP = """
symbols = '$¢£¥€¤'
def non_ascii(c):
    return c > 127
"""

def clock(label, cmd):
    res = timeit.repeat(cmd, setup=SETUP, number=TIMES)
    print(label, *('{:.3f}'.format(x) for x in res))

clock('listcomp        :', '[ord(s) for s in symbols if ord(s) > 127]')
clock('listcomp + func :', '[ord(s) for s in symbols if non_ascii(ord(s))]')
clock('filter + lambda :', 'list(filter(lambda c: c > 127, map(ord, symbols)))')
clock('filter + func   :', 'list(filter(non_ascii, map(ord, symbols)))')
# %%
import matplotlib.pyplot as plt

x = [0, 0, 0.5, 0.5, 0.5, 1, 1]
y = [0, 0.25, 0.25, 0.5, 0.75, 0.75, 1]

plt.plot(x,y,c='orange')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()
# %%
import matplotlib.pyplot as plt

x = [0, 0, 1, 1, 1, 2, 2]
y = [0, 1, 1, 2, 3, 3, 4]
plt.plot(x,y,c='green')
plt.xlabel("FPN")
plt.ylabel("TPN")
plt.show()
# %%
from array import array
from random import random
a = array('d',(random() for i in range(10)))
# %%
a
# %%
