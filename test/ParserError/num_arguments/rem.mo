// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: rem: expected 2 argument(s) but got 1.

function foo
    input Integer x;
    input Integer y;
    output Integer z;
algorithm
    z := rem(x);
end foo;
