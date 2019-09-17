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

# find_width8で探索したloopを格納する
loops = []

test_loop = []

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
    # 変形対象のstmtをfind_width8()に定義にしたがって、IRを探索して取得 
    tvm.ir_pass.PostOrderVisit(stmt, find_width8)

    # tvm.ir_pass.PostOrderVisit(stmt, find_width8(self))
    print ("stmt")
    print (stmt)
    if not loops:
        return stmt
    # vectorize8()にしたがってstmtを変形
    stmt = tvm.ir_pass.IRTransform(stmt, None, vectorize8, ['For'])
    print ("after tansform stmet")
    print (stmt)

    return stmt

def find(op):
    """ Find all the 'For' nodes whose extent can be divided by 8. """
    if isinstance(op, tvm.stmt.For):
        print ("op")
        print (op)
        if isinstance(op.extent, tvm.expr.IntImm):
            loops.append(op)

def test_tranceform(op):
    """ Split can vectorize the loops found in `find`. """
    if op in loops:
        extent = op.extent.value
        name = op.loop_var.name
        lo, li = tvm.var(name + '.outer'), tvm.var(name + '.inner')
        body = tvm.ir_pass.Substitute(op.body, {op.loop_var: lo * 8 + li})
        body = tvm.make.For(li, 0, 8, tvm.stmt.For.Vectorized, 0, body)
        body = tvm.make.For(lo, 0, extent // 8, tvm.stmt.For.Serial, 0, body)
        return body
    return None

def test(stmt):
    global loops
    # find(self)
    # 変形対象のstmtをfind()に定義にしたがって、IRを探索して取得 
    tvm.ir_pass.PostOrderVisit(stmt, find)

    # tvm.ir_pass.PostOrderVisit(stmt, find_width8(self))
    print ("stmt")
    print (stmt)
    if not loops:
        return stmt
    # test_tranceform()にしたがってstmtを変形
    stmt = tvm.ir_pass.IRTransform(stmt, None, test_tranceform, ['For'])
    print ("after tansform stmet")
    print (stmt)

    return stmt


print("test(ir)")
print(test(ir))

print("vectorize(ir)")
print(vectorize(ir))


print("codegen")
with tvm.build_config(add_lower_pass=[(1, vectorize)]) as cfg:
    print(tvm.lower(sch, [a, b, c], simple_mode=True))

print("codegen")
with tvm.build_config(add_lower_pass=[(1, test)]) as cfg:
    print(tvm.lower(sch, [a, b, c], simple_mode=True))