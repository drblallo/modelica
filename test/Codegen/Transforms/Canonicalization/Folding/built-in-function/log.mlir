// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.000000e+00
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<1.0>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 0.999999999831
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<2.718281828>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: 2.000000000009
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<7.389056099>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:.*]] = bmodelica.constant #bmodelica.real
// CHECK-SAME: -1.000000003184
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!bmodelica.real) {
    %x = bmodelica.constant #bmodelica.real<0.36787944>
    %result = bmodelica.log %x : !bmodelica.real -> !bmodelica.real
    return %result : !bmodelica.real
}
