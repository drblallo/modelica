// RUN: modelica-opt %s --split-input-file --match | FileCheck %s

// l + fl = 0
// fl = 0
// h + fh = 0
// fh = 0
// for i in 0:4
//   fl + f[i] + x[i] = 0
// for i in 0:4
//   fh + f[i] + y[i] = 0
// for i in 0:4
//   f[i] = 0

bmodelica.model @Test {
    bmodelica.variable @l : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @h : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @fl : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @fh : !bmodelica.variable<!bmodelica.real>
    bmodelica.variable @x : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @y : !bmodelica.variable<5x!bmodelica.real>
    bmodelica.variable @f : !bmodelica.variable<5x!bmodelica.real>

    // l + fl = 0
    // CHECK-DAG: %[[t0:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t0"}
    %t0 = bmodelica.equation_template inductions = [] attributes {id = "t0"} {
        %0 = bmodelica.variable_get @l : !bmodelica.real
        %1 = bmodelica.variable_get @fl : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica.real<0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // fl = 0
    // CHECK-DAG: %[[t1:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t1"}
    %t1 = bmodelica.equation_template inductions = [] attributes {id = "t1"} {
        %0 = bmodelica.variable_get @fl : !bmodelica.real
        %1 = bmodelica.constant #bmodelica.real<0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // h + fh = 0
    // CHECK-DAG: %[[t2:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t2"}
    %t2 = bmodelica.equation_template inductions = [] attributes {id = "t2"} {
        %0 = bmodelica.variable_get @h : !bmodelica.real
        %1 = bmodelica.variable_get @fh : !bmodelica.real
        %2 = bmodelica.add %0, %1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %3 = bmodelica.constant #bmodelica.real<0.0>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        %5 = bmodelica.equation_side %3 : tuple<!bmodelica.real>
        bmodelica.equation_sides %4, %5 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // fh = 0
    // CHECK-DAG: %[[t3:.*]] = bmodelica.equation_template inductions = [] attributes {id = "t3"}
    %t3 = bmodelica.equation_template inductions = [] attributes {id = "t3"} {
        %0 = bmodelica.variable_get @fh : !bmodelica.real
        %1 = bmodelica.constant #bmodelica.real<0.0>
        %2 = bmodelica.equation_side %0 : tuple<!bmodelica.real>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        bmodelica.equation_sides %2, %3 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // fl + f[i] + x[i] = 0
    // CHECK-DAG: %[[t4:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t4"}
    %t4 = bmodelica.equation_template inductions = [%i0] attributes {id = "t4"} {
        %0 = bmodelica.variable_get @fl : !bmodelica.real
        %1 = bmodelica.variable_get @f : !bmodelica.array<5x!bmodelica.real>
        %2 = bmodelica.variable_get @x : !bmodelica.array<5x!bmodelica.real>
        %3 = bmodelica.load %1[%i0] : !bmodelica.array<5x!bmodelica.real>
        %4 = bmodelica.load %2[%i0] : !bmodelica.array<5x!bmodelica.real>
        %5 = bmodelica.add %0, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.add %5, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %7 = bmodelica.constant #bmodelica.real<0.0>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // fh + f[i] + y[i] = 0
    // CHECK-DAG: %[[t5:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t5"}
    %t5 = bmodelica.equation_template inductions = [%i0] attributes {id = "t5"} {
        %0 = bmodelica.variable_get @fh : !bmodelica.real
        %1 = bmodelica.variable_get @f : !bmodelica.array<5x!bmodelica.real>
        %2 = bmodelica.variable_get @y : !bmodelica.array<5x!bmodelica.real>
        %3 = bmodelica.load %1[%i0] : !bmodelica.array<5x!bmodelica.real>
        %4 = bmodelica.load %2[%i0] : !bmodelica.array<5x!bmodelica.real>
        %5 = bmodelica.add %0, %3 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %6 = bmodelica.add %5, %4 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
        %7 = bmodelica.constant #bmodelica.real<0.0>
        %8 = bmodelica.equation_side %6 : tuple<!bmodelica.real>
        %9 = bmodelica.equation_side %7 : tuple<!bmodelica.real>
        bmodelica.equation_sides %8, %9 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    // f[i] = 0
    // CHECK-DAG: %[[t6:.*]] = bmodelica.equation_template inductions = [%{{.*}}] attributes {id = "t6"}
    %t6 = bmodelica.equation_template inductions = [%i0] attributes {id = "t6"} {
        %0 = bmodelica.variable_get @f : !bmodelica.array<5x!bmodelica.real>
        %1 = bmodelica.load %0[%i0] : !bmodelica.array<5x!bmodelica.real>
        %2 = bmodelica.constant #bmodelica.real<0.0>
        %3 = bmodelica.equation_side %1 : tuple<!bmodelica.real>
        %4 = bmodelica.equation_side %2 : tuple<!bmodelica.real>
        bmodelica.equation_sides %3, %4 : tuple<!bmodelica.real>, tuple<!bmodelica.real>
    }

    bmodelica.main_model {
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t0]] {path = #bmodelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t1]] {path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t2]] {path = #bmodelica<equation_path [L, 0, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t3]] {path = #bmodelica<equation_path [L, 0]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t4]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0, 1]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t5]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0, 1]>}
        // CHECK-DAG: bmodelica.matched_equation_instance %[[t6]] {indices = #modeling<multidim_range [0,4]>, path = #bmodelica<equation_path [L, 0]>}
        bmodelica.equation_instance %t0 : !bmodelica.equation
        bmodelica.equation_instance %t1 : !bmodelica.equation
        bmodelica.equation_instance %t2 : !bmodelica.equation
        bmodelica.equation_instance %t3 : !bmodelica.equation
        bmodelica.equation_instance %t4 {indices = #modeling<multidim_range [0,4]>} : !bmodelica.equation
        bmodelica.equation_instance %t5 {indices = #modeling<multidim_range [0,4]>} : !bmodelica.equation
        bmodelica.equation_instance %t6 {indices = #modeling<multidim_range [0,4]>} : !bmodelica.equation
    }
}
