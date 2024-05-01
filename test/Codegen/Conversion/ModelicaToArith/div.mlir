// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith --cse | FileCheck %s

// Integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.int
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[result:.*]] = arith.divsi %[[arg0_casted]], %[[arg1_casted]] : i64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : i64 to !bmodelica.int
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.int, %arg1 : !bmodelica.int) -> !bmodelica.int {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.int, !bmodelica.int) -> !bmodelica.int
    func.return %0 : !bmodelica.int
}

// -----

// Real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0_casted]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.real, %arg1 : !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.real, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK: %[[arg0_f64:.*]] = arith.sitofp %[[arg0_casted]] : i64 to f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0_f64]], %[[arg1_casted]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.int, %arg1 : !bmodelica.real) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.int, !bmodelica.real) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real, %[[arg1:.*]]: !bmodelica.int) -> !bmodelica.real
// CHECK-DAG: %[[arg0_casted:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.int to i64
// CHECK: %[[arg1_f64:.*]] = arith.sitofp %[[arg1_casted]] : i64 to f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0_casted]], %[[arg1_f64]] : f64
// CHECK: %[[result_casted:.*]] =  builtin.unrealized_conversion_cast %[[result]] : f64 to !bmodelica.real
// CHECK: return %[[result_casted]]

func.func @foo(%arg0 : !bmodelica.real, %arg1 : !bmodelica.int) -> !bmodelica.real {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.real, !bmodelica.int) -> !bmodelica.real
    func.return %0 : !bmodelica.real
}

// -----

// Array and scalar operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.array<3x?x!bmodelica.real>, %[[arg1:.*]]: !bmodelica.real) -> !bmodelica.array<3x?x!bmodelica.real>
// CHECK-DAG:   %[[arg1_casted:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !bmodelica.real to f64
// CHECK-DAG:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG:   %[[c1:.*]] = arith.constant 1 : index
// CHECK-DAG:   %[[arg0_dim0:.*]] = bmodelica.dim %[[arg0]], %[[c0]]
// CHECK-DAG:   %[[arg0_dim1:.*]] = bmodelica.dim %[[arg0]], %[[c1]]
// CHECK-DAG:   %[[result:.*]] = bmodelica.alloc %[[arg0_dim1]] : <3x?x!bmodelica.real>
// CHECK:       scf.for %[[index_0:.*]] = %[[c0]] to %[[arg0_dim0]] step %[[c1]] {
// CHECK:           scf.for %[[index_1:.*]] = %[[c0]] to %[[arg0_dim1]] step %[[c1]] {
// CHECK:               %[[array_value:.*]] = bmodelica.load %[[arg0]][%[[index_0]], %[[index_1]]]
// CHECK:               %[[array_value_casted:.*]] = builtin.unrealized_conversion_cast %[[array_value]] : !bmodelica.real to f64
// CHECK:               %[[div:.*]] = arith.divf %[[array_value_casted]], %[[arg1_casted]]
// CHECK:               %[[div_casted:.*]] = builtin.unrealized_conversion_cast %[[div]] : f64 to !bmodelica.real
// CHECK:               bmodelica.store %[[result]][%[[index_0]], %[[index_1]]], %[[div_casted]]
// CHECK:           }
// CHECK:       }
// CHECK:       return %[[result]]

func.func @foo(%arg0 : !bmodelica.array<3x?x!bmodelica.real>, %arg1 : !bmodelica.real) -> !bmodelica.array<3x?x!bmodelica.real> {
    %0 = bmodelica.div %arg0, %arg1 : (!bmodelica.array<3x?x!bmodelica.real>, !bmodelica.real) -> !bmodelica.array<3x?x!bmodelica.real>
    func.return %0 : !bmodelica.array<3x?x!bmodelica.real>
}

// -----

// MLIR index operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index, %[[arg1:.*]]: index) -> index
// CHECK: %[[result:.*]] = arith.divsi %[[arg0]], %[[arg1]] : index
// CHECK: return %[[result]] : index

func.func @foo(%arg0 : index, %arg1 : index) -> index {
    %0 = bmodelica.div %arg0, %arg1 : (index, index) -> index
    func.return %0 : index
}

// -----

// MLIR integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64, %[[arg1:.*]]: i64) -> i64
// CHECK: %[[result:.*]] = arith.divsi %[[arg0]], %[[arg1]] : i64
// CHECK: return %[[result]] : i64

func.func @foo(%arg0 : i64, %arg1 : i64) -> i64 {
    %0 = bmodelica.div %arg0, %arg1 : (i64, i64) -> i64
    func.return %0 : i64
}

// -----

// MLIR float operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = arith.divf %[[arg0]], %[[arg1]] : f64
// CHECK: return %[[result]] : f64

func.func @foo(%arg0 : f64, %arg1 : f64) -> f64 {
    %0 = bmodelica.div %arg0, %arg1 : (f64, f64) -> f64
    func.return %0 : f64
}
