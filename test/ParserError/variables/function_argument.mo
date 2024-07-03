// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown variable identifier argument.

function foo
    input Integer[:] x;
    output Integer[:,:] y;
algorithm
    y := diagonal(x);
end foo;

function bar
    input Integer[:] argument1;
    output Integer[:,:] y;
algorithm
    y := foo(argument);
end bar;

