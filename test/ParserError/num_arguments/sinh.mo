// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion when converting function sinh.

function foo
    input Real x;
    output Real y;
algorithm
    y := sinh(x, x);
end foo;
