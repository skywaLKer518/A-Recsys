

val = range(20)
starts = [0, 1, 3, 9,  20]
lengths = [1, 2, 6, 11]


user = [1, 0, 2]

print 'val', val
print ''

'''
way two
'''
import numpy as np

a_value = np.zeros((100,), dtype=float)
a_i1 = np.zeros((100, ), dtype=int)
a_i2 = np.zeros((100, ), dtype=int)

value = []
i1 = []
i2 = []


def modify(u):
  s = starts[u]
  l = lengths[u]
  return val[s:s+l]

def modify2(u):
  l = lengths[u]
  return [0] * l

vv =  [val[starts[u]:starts[u]+lengths[u]] for u in user]
Ls =  [lengths[u] for u in user]

i1 =  [Ls[i] * [i] for i in range(len(Ls))]
i2 =  [range(Ls[i]) for i in range(len(Ls))]

import itertools
print list(itertools.chain.from_iterable(vv))
print list(itertools.chain.from_iterable(i1))
print list(itertools.chain.from_iterable(i2))
# print map(modify2, user)

print '\n\n'

'''
way one
'''

value = []
i1 = []
i2 = []

c = 0
for u in user:
  s = starts[u]
  l = lengths[u]
  v = val[s:s+l]
  value.extend(v)
  i1.extend([c] * l)
  i2.extend(range(l))
  c += 1

print 'value', value
print 'i1', i1
print 'i2', i2










