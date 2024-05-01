// RUN: modelica-opt %s --split-input-file --insert-explicit-initial-equations | FileCheck %s

// Scalar variable with fixed start value.

// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [] {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[value]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.initial_model {
// CHECK-NEXT:      bmodelica.equation_instance %[[t0]]
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<!bmodelica.int>

    bmodelica.start @x {
        %0 = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
        bmodelica.yield %0 : !bmodelica.int
    } {each = false, fixed = true}
}

// -----

// Array variable with fixed start scalar value.

// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[x_load:.*]] = bmodelica.load %[[x]][%[[i0]]]
// CHECK-DAG:       %[[value:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
// CHECK-DAG:       %[[rhs:.*]] = bmodelica.equation_side %[[value]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.initial_model {
// CHECK-NEXT:      bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.start @x {
        %0 = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
        bmodelica.yield %0 : !bmodelica.int
    } {each = true, fixed = true}
}

// -----

// Array variable with fixed start array value

// CHECK:       %[[t0:.*]] = bmodelica.equation_template inductions = [%[[i0:.*]]] {
// CHECK-DAG:       %[[x:.*]] = bmodelica.variable_get @x
// CHECK-DAG:       %[[value_0:.*]] = bmodelica.constant #bmodelica.int<0>
// CHECK-DAG:       %[[value_1:.*]] = bmodelica.constant #bmodelica.int<1>
// CHECK-DAG:       %[[value_2:.*]] = bmodelica.constant #bmodelica.int<2>
// CHECK-DAG:       %[[array:.*]] = bmodelica.array_from_elements %[[value_0]], %[[value_1]], %[[value_2]]
// CHECK-DAG:       %[[x_load:.*]] = bmodelica.load %[[x]][%[[i0]]]
// CHECK-DAG:       %[[array_load:.*]] = bmodelica.load %[[array]][%[[i0]]]
// CHECK-NEXT:      %[[lhs:.*]] = bmodelica.equation_side %[[x_load]]
// CHECK-NEXT:      %[[rhs:.*]] = bmodelica.equation_side %[[array_load]]
// CHECK:           bmodelica.equation_sides %[[lhs]], %[[rhs]]
// CHECK-NEXT:  }
// CHECK:       bmodelica.initial_model {
// CHECK-NEXT:      bmodelica.equation_instance %[[t0]] {indices = #modeling<multidim_range [0,2]>}
// CHECK-NEXT:  }

bmodelica.model @Test {
    bmodelica.variable @x : !bmodelica.variable<3x!bmodelica.int>

    bmodelica.start @x {
        %0 = bmodelica.constant #bmodelica.int<0> : !bmodelica.int
        %1 = bmodelica.constant #bmodelica.int<1> : !bmodelica.int
        %2 = bmodelica.constant #bmodelica.int<2> : !bmodelica.int
        %3 = bmodelica.array_from_elements %0, %1, %2 : !bmodelica.int, !bmodelica.int, !bmodelica.int -> !bmodelica.array<3x!bmodelica.int>
        bmodelica.yield %3 : !bmodelica.array<3x!bmodelica.int>
    } {each = false, fixed = true}
}
