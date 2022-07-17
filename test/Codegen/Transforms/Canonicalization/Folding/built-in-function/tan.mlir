// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.0>
    %result = modelica.tan %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.577350268391
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.523598775>
    %result = modelica.tan %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = modelica.constant #modelica.real
// CHECK-SAME: 0.999999999205
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.real) {
    %x = modelica.constant #modelica.real<0.785398163>
    %result = modelica.tan %x : !modelica.real -> !modelica.real
    return %result : !modelica.real
}
