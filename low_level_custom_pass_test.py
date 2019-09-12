from __future__ import absolute_import, print_function
import tvm
import numpy as np

#loopを定義
n = tvm.const(128, "int32")
a = tvm.placeholder((n, ), name="a")
b = tvm.placeholder((n, ), name="b")
c = tvm.compute((n, ), lambda i: a[i] + b[i], name='c')

sch = tvm.create_schedule(c.op)
ir  = tvm.lower(sch, [a, b, c], simple_mode=True)

#単純なloopのirをprint
print("ir")
print(ir)


loops = []
def find_width8(op):
    """ Find all the 'For' nodes whose extent can be divided by 8. """
    if isinstance(op, tvm.stmt.For):
        if isinstance(op.extent, tvm.expr.IntImm):
            if op.extent.value % 8 == 0:
                print(op.extent.value)
                loops.append(op)

def vectorize8(op):
    """ Split can vectorize the loops found in `find_width8`. """
    if op in loops:
        extent = op.extent.value
        name = op.loop_var.name
        lo, li = tvm.var(name + '.outer'), tvm.var(name + '.inner')
        body = tvm.ir_pass.Substitute(op.body, {op.loop_var: lo * 8 + li})
        body = tvm.make.For(li, 0, 8, tvm.stmt.For.Vectorized, 0, body)
        body = tvm.make.For(lo, 0, extent // 8, tvm.stmt.For.Serial, 0, body)
        return body
    return None

def vectorize(stmt):
    global loops
    # find_width8(self)
    tvm.ir_pass.PostOrderVisit(stmt, find_width8)
    # tvm.ir_pass.PostOrderVisit(stmt, find_width8(self))

    if not loops:
        return stmt
    stmt = tvm.ir_pass.IRTransform(stmt, None, vectorize8, ['For'])

    return stmt

def test_loop_transform(stmt):
    global loops
    
    tvm.ir_pass.PostOrderVisit(stmt, find_width8)

print("vectorize(ir)")
print(vectorize(ir))

print("codegen")
with tvm.build_config(add_lower_pass=[(1, vectorize)]) as cfg:
    print(tvm.lower(sch, [a, b, c], simple_mode=True))
