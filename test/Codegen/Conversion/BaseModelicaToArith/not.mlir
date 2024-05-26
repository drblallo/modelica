// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-arith --cse | FileCheck %s

// Boolean operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.bool) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.bool to i1
// CHECK-DAG: %[[zero:.*]] = arith.constant false
// CHECK: %[[cmp:.*]] = arith.cmpi eq, %[[x]], %[[zero]] : i1
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.bool) -> !bmodelica.bool {
    %0 = bmodelica.not %arg0 : !bmodelica.bool -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Integer operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.int) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.int to i64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[cmp:.*]] = arith.cmpi eq, %[[x]], %[[zero]] : i64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.int) -> !bmodelica.bool {
    %0 = bmodelica.not %arg0 : !bmodelica.int -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// Real operand.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !bmodelica.real) -> !bmodelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !bmodelica.real to f64
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[cmp:.*]] = arith.cmpf oeq, %[[x]], %[[zero]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !bmodelica.bool
// CHECK: return %[[result]] : !bmodelica.bool

func.func @foo(%arg0 : !bmodelica.real) -> !bmodelica.bool {
    %0 = bmodelica.not %arg0 : !bmodelica.real -> !bmodelica.bool
    func.return %0 : !bmodelica.bool
}

// -----

// MLIR index operands.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: index) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : index
// CHECK: %[[result:.*]] = arith.cmpi eq, %[[arg0]], %[[zero]] : index
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : index) -> i1 {
    %0 = bmodelica.not %arg0 : index -> i1
    func.return %0 : i1
}

// -----

// MLIR integer operands.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: i64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0 : i64
// CHECK: %[[result:.*]] = arith.cmpi eq, %[[arg0]], %[[zero]] : i64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : i64) -> i1 {
    %0 = bmodelica.not %arg0 : i64 -> i1
    func.return %0 : i1
}

// -----

// MLIR float operands.

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64) -> i1
// CHECK-DAG: %[[zero:.*]] = arith.constant 0.000000e+00 : f64
// CHECK: %[[result:.*]] = arith.cmpf oeq, %[[arg0]], %[[zero]] : f64
// CHECK: return %[[result]] : i1

func.func @foo(%arg0 : f64) -> i1 {
    %0 = bmodelica.not %arg0 : f64 -> i1
    func.return %0 : i1
}
