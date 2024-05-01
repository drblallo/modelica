// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.0>
    %result = bmodelica.tanh %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.761594155955
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<1.0>
    %result = bmodelica.tanh %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}
