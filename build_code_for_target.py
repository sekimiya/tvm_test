from __future__ import absolute_import, print_function

import tvm
import numpy as np

n = tvm.var('n')
m = tvm.var('m')

A = tvm.placeholder((m, n), name='A')
B = tvm.placeholder((m, n), name='B')
C = tvm.compute((m, n), lambda i, j: A[i, j] * B[i, j], name='C')

s = tvm.create_schedule([C.op])

print("tvm.lower(s, [A, B, C], simple_mode=True)")
print(tvm.lower(s, [A, B, C], simple_mode=True))

f = tvm.lower(s, [A, B, C], name="test_add")

m = tvm.build(f, target="llvm")

print(m)
