// RUN: marco %s --omc-bypass --emit-modelica-dialect | FileCheck %s

// CHECK-LABEL: @foo
// CHECK: modelica.transpose
// CHECK-SAME: !modelica.array<?x?x!modelica.int> -> !modelica.array<?x?x!modelica.int>

function foo
    input Integer[:,:] x;
    output Integer[:,:] y;

algorithm
    y := transpose(x);
end foo;