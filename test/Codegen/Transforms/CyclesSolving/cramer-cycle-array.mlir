// RUN: modelica-opt %s --split-input-file --pass-pipeline="solve-cycles{model-name=Test process-ic-model=false debug-view=true}" | FileCheck %s

// CHECK: [[VAL:%[0-9]+]] = modelica.constant #modelica.real<2.500000e+00> : !modelica.real
// CHECK: [[LHS:%[0-9]+]] = modelica.equation_side %2 : tuple<!modelica.real>
// CHECK: [[RHS:%[0-9]+]] = modelica.equation_side [[VAL]] : tuple<!modelica.real>
// CHECK: modelica.equation_sides [[LHS]], [[RHS]] : tuple<!modelica.real>, tuple<!modelica.real>

// CHECK: [[VAL:%[0-9]+]] = modelica.constant #modelica.real<-1.000000e+00> : !modelica.real
// CHECK: [[LHS:%[0-9]+]] = modelica.equation_side %1 : tuple<!modelica.real>
// CHECK: [[RHS:%[0-9]+]] = modelica.equation_side [[VAL]] : tuple<!modelica.real>
// CHECK: modelica.equation_sides [[LHS]], [[RHS]] : tuple<!modelica.real>, tuple<!modelica.real>

// CHECK: [[VAL:%[0-9]+]] = modelica.constant #modelica.real<-5.000000e-01> : !modelica.real
// CHECK: [[LHS:%[0-9]+]] = modelica.equation_side %arg2 : tuple<!modelica.array<!modelica.real>>
// CHECK: [[RHS:%[0-9]+]] = modelica.equation_side [[VAL]] : tuple<!modelica.real>
// CHECK: modelica.equation_sides [[LHS]], [[RHS]] : tuple<!modelica.array<!modelica.real>>, tuple<!modelica.real>

modelica.model @Model attributes {derivatives = []} {
  %0 = modelica.member_create @x : !modelica.member<4x!modelica.real>
  modelica.yield %0 : !modelica.member<4x!modelica.real>
} body {
^bb0(%arg0: !modelica.array<4x!modelica.real>):
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 0 : index, 0 : index]}]} {
    %0 = modelica.constant #modelica.int<1> : !modelica.int
    %1 = modelica.constant #modelica.int<1> : !modelica.int
    %2 = modelica.constant -1 : index
    %3 = modelica.constant #modelica.int<0> : !modelica.int
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    %6 = modelica.constant #modelica.int<4> : !modelica.int
    %7 = modelica.subscription %arg0[%3] : !modelica.array<4x!modelica.real>
    %8 = modelica.subscription %arg0[%1] : !modelica.array<4x!modelica.real>
    %9 = modelica.load %8[] : !modelica.array<!modelica.real>
    %10 = modelica.load %7[] : !modelica.array<!modelica.real>
    %11 = modelica.add %10, %9 : (!modelica.real, !modelica.real) -> !modelica.real
    %12 = modelica.subscription %arg0[%4] : !modelica.array<4x!modelica.real>
    %13 = modelica.load %12[] : !modelica.array<!modelica.real>
    %14 = modelica.add %11, %13 : (!modelica.real, !modelica.real) -> !modelica.real
    %15 = modelica.subscription %arg0[%5] : !modelica.array<4x!modelica.real>
    %16 = modelica.load %15[] : !modelica.array<!modelica.real>
    %17 = modelica.add %14, %16 : (!modelica.real, !modelica.real) -> !modelica.real
    %18 = modelica.equation_side %17 : tuple<!modelica.real>
    %19 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %18, %19 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 0 : index, 1 : index]}]} {
    %0 = modelica.constant #modelica.int<2> : !modelica.int
    %1 = modelica.constant #modelica.int<1> : !modelica.int
    %2 = modelica.constant -1 : index
    %3 = modelica.constant #modelica.int<0> : !modelica.int
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    %6 = modelica.constant #modelica.int<4> : !modelica.int
    %7 = modelica.subscription %arg0[%3] : !modelica.array<4x!modelica.real>
    %8 = modelica.subscription %arg0[%1] : !modelica.array<4x!modelica.real>
    %9 = modelica.load %8[] : !modelica.array<!modelica.real>
    %10 = modelica.load %7[] : !modelica.array<!modelica.real>
    %11 = modelica.add %10, %9 : (!modelica.real, !modelica.real) -> !modelica.real
    %12 = modelica.subscription %arg0[%4] : !modelica.array<4x!modelica.real>
    %13 = modelica.load %12[] : !modelica.array<!modelica.real>
    %14 = modelica.add %11, %13 : (!modelica.real, !modelica.real) -> !modelica.real
    %15 = modelica.subscription %arg0[%5] : !modelica.array<4x!modelica.real>
    %16 = modelica.load %15[] : !modelica.array<!modelica.real>
    %17 = modelica.sub %14, %16 : (!modelica.real, !modelica.real) -> !modelica.real
    %18 = modelica.equation_side %17 : tuple<!modelica.real>
    %19 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %18, %19 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 0 : index, 1 : index]}]} {
    %0 = modelica.constant #modelica.int<3> : !modelica.int
    %1 = modelica.constant #modelica.int<1> : !modelica.int
    %2 = modelica.constant -1 : index
    %3 = modelica.constant #modelica.int<0> : !modelica.int
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    %6 = modelica.constant #modelica.int<4> : !modelica.int
    %7 = modelica.subscription %arg0[%3] : !modelica.array<4x!modelica.real>
    %8 = modelica.subscription %arg0[%1] : !modelica.array<4x!modelica.real>
    %9 = modelica.load %8[] : !modelica.array<!modelica.real>
    %10 = modelica.load %7[] : !modelica.array<!modelica.real>
    %11 = modelica.add %10, %9 : (!modelica.real, !modelica.real) -> !modelica.real
    %12 = modelica.subscription %arg0[%4] : !modelica.array<4x!modelica.real>
    %13 = modelica.load %12[] : !modelica.array<!modelica.real>
    %14 = modelica.sub %11, %13 : (!modelica.real, !modelica.real) -> !modelica.real
    %15 = modelica.subscription %arg0[%5] : !modelica.array<4x!modelica.real>
    %16 = modelica.load %15[] : !modelica.array<!modelica.real>
    %17 = modelica.add %14, %16 : (!modelica.real, !modelica.real) -> !modelica.real
    %18 = modelica.equation_side %17 : tuple<!modelica.real>
    %19 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %18, %19 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.equation attributes {match = [{indices = [[[0, 0]]], path = ["L", 1 : index]}]} {
    %0 = modelica.constant #modelica.int<4> : !modelica.int
    %1 = modelica.constant #modelica.int<1> : !modelica.int
    %2 = modelica.constant -1 : index
    %3 = modelica.constant #modelica.int<0> : !modelica.int
    %4 = modelica.constant #modelica.int<2> : !modelica.int
    %5 = modelica.constant #modelica.int<3> : !modelica.int
    %6 = modelica.constant #modelica.int<4> : !modelica.int
    %7 = modelica.subscription %arg0[%3] : !modelica.array<4x!modelica.real>
    %8 = modelica.subscription %arg0[%1] : !modelica.array<4x!modelica.real>
    %9 = modelica.load %8[] : !modelica.array<!modelica.real>
    %10 = modelica.load %7[] : !modelica.array<!modelica.real>
    %11 = modelica.sub %10, %9 : (!modelica.real, !modelica.real) -> !modelica.real
    %12 = modelica.subscription %arg0[%4] : !modelica.array<4x!modelica.real>
    %13 = modelica.load %12[] : !modelica.array<!modelica.real>
    %14 = modelica.add %11, %13 : (!modelica.real, !modelica.real) -> !modelica.real
    %15 = modelica.subscription %arg0[%5] : !modelica.array<4x!modelica.real>
    %16 = modelica.load %15[] : !modelica.array<!modelica.real>
    %17 = modelica.add %14, %16 : (!modelica.real, !modelica.real) -> !modelica.real
    %18 = modelica.equation_side %17 : tuple<!modelica.real>
    %19 = modelica.equation_side %0 : tuple<!modelica.int>
    modelica.equation_sides %18, %19 : tuple<!modelica.real>, tuple<!modelica.int>
  }
  modelica.start (%arg0 : !modelica.array<4x!modelica.real>) {each = true, fixed = false} {
    %0 = modelica.constant #modelica.real<0.000000e+00> : !modelica.real
    modelica.yield %0 : !modelica.real
  }
}

