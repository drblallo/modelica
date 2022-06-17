// RUN: modelica-opt %s --split-input-file --canonicalize | FileCheck %s

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<false>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.bool<false>
    %y = modelica.constant #modelica.bool<false>
    %result = modelica.or %x, %y : (!modelica.bool, !modelica.bool) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.bool<false>
    %y = modelica.constant #modelica.bool<true>
    %result = modelica.or %x, %y : (!modelica.bool, !modelica.bool) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.bool<true>
    %y = modelica.constant #modelica.bool<false>
    %result = modelica.or %x, %y : (!modelica.bool, !modelica.bool) -> !modelica.bool
    return %result : !modelica.bool
}

// -----

// CHECK-LABEL: @test
// CHECK-NEXT: %[[VALUE:[a-zA-Z0-9]*]] = modelica.constant #modelica.bool<true>
// CHECK-NEXT: return %[[VALUE]]

func.func @test() -> (!modelica.bool) {
    %x = modelica.constant #modelica.bool<true>
    %y = modelica.constant #modelica.bool<true>
    %result = modelica.or %x, %y : (!modelica.bool, !modelica.bool) -> !modelica.bool
    return %result : !modelica.bool
}
