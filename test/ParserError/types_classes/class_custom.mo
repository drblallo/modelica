// RUN: not marco -mc1 %s --omc-bypass -emit-mlir -o - 2>&1 | FileCheck %s

// CHECK: Error in AST to MLIR conversion. Unknown type or class identifier RECRD at line 10, column 11. Did you mean RECORD?

record RECORD
    Real x;
end RECORD;

function Foo
    input RECRD r;
    output Real x;
algorithm
    x := r.x;
end Foo;
