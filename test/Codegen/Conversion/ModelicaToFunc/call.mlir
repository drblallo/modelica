// RUN: modelica-opt %s --split-input-file --convert-modelica-to-func | FileCheck %s

bmodelica.raw_function @foo(%arg0: !bmodelica.bool) -> !bmodelica.bool {
    bmodelica.raw_return %arg0 : !bmodelica.bool
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !bmodelica.bool) -> !bmodelica.bool
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (i1) -> i1
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i1 to !bmodelica.bool
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !bmodelica.bool) -> !bmodelica.bool {
    %0 = bmodelica.call @foo(%arg0) : (!bmodelica.bool) -> !bmodelica.bool
    return %0 : !bmodelica.bool
}

// -----

bmodelica.raw_function @foo(%arg0: !bmodelica.int) -> !bmodelica.int {
    bmodelica.raw_return %arg0 : !bmodelica.int
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (i64) -> i64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.call @foo(%arg0) : (!bmodelica.int) -> !bmodelica.int
    return %0 : !bmodelica.int
}

// -----

bmodelica.raw_function @foo(%arg0: !bmodelica.real) -> !bmodelica.real {
    bmodelica.raw_return %arg0 : !bmodelica.real
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (f64) -> f64
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.call @foo(%arg0) : (!bmodelica.real) -> !bmodelica.real
    return %0 : !bmodelica.real
}

// -----

bmodelica.raw_function @foo(%arg0: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.array<5x3x!bmodelica.int> {
    bmodelica.raw_return %arg0 : !bmodelica.array<5x3x!bmodelica.int>
}

// CHECK-LABEL: @test
// CHECK-SAME:  (%[[arg0:.*]]: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.array<5x3x!bmodelica.int>
// CHECK:       %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.array<5x3x!bmodelica.int> to memref<5x3xi64>
// CHECK:       %[[result:.*]] = call @foo(%[[arg0_casted]]) : (memref<5x3xi64>) -> memref<5x3xi64>
// CHECK:       %[[result_casted:.*]] = builtin.unrealized_conversion_cast %[[result]] : memref<5x3xi64> to !bmodelica.array<5x3x!bmodelica.int>
// CHECK:       return %[[result_casted]]

func.func @test(%arg0: !bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.array<5x3x!bmodelica.int> {
    %0 = bmodelica.call @foo(%arg0) : (!bmodelica.array<5x3x!bmodelica.int>) -> !bmodelica.array<5x3x!bmodelica.int>
    return %0 : !bmodelica.array<5x3x!bmodelica.int>
}
