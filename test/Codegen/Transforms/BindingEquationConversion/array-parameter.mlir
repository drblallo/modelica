// RUN: modelica-opt %s --split-input-file --convert-binding-equations | FileCheck %s

// Binding equation for an array parameter.

// CHECK-LABEL: @Test
// CHECK:       bmodelica.start @x {
// CHECK-NEXT:      %[[el:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-NEXT:      %[[value:.*]] = bmodelica.array_broadcast %[[el]]
// CHECK-NEXT:      bmodelica.yield %[[value]]
// CHECK-NEXT:  } {each = false, fixed = true}

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int, parameter>

    bmodelica.binding_equation @x {
      %0 = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
      %1 = bmodelica.array_broadcast %0: !bmodelica.int -> !bmodelica.array<3x!bmodelica.int>
      bmodelica.yield %1 : !bmodelica.array<3x!bmodelica.int>
    }
}
