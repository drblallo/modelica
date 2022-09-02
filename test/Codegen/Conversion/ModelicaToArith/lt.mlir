// RUN: modelica-opt %s --split-input-file --convert-modelica-to-arith | FileCheck %s

// Integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.int to i64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.int to i64
// CHECK: %[[cmp:.*]] = arith.cmpi slt, %[[x]], %[[y]] : i64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.lt %arg0, %arg1 : (!modelica.int, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf olt, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.lt %arg0, %arg1 : (!modelica.real, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Integer and real operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.int, %[[arg1:.*]]: !modelica.real) -> !modelica.bool
// CHECK-DAG: %[[arg0_casted:.*]] = modelica.cast %[[arg0]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0_casted]] : !modelica.real to f64
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1]] : !modelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf olt, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.int, %arg1 : !modelica.real) -> !modelica.bool {
    %0 = modelica.lt %arg0, %arg1 : (!modelica.int, !modelica.real) -> !modelica.bool
    func.return %0 : !modelica.bool
}

// -----

// Real and integer operands

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: !modelica.real, %[[arg1:.*]]: !modelica.int) -> !modelica.bool
// CHECK-DAG: %[[arg1_casted:.*]] = modelica.cast %[[arg1]] : !modelica.int -> !modelica.real
// CHECK-DAG: %[[y:.*]] = builtin.unrealized_conversion_cast %[[arg1_casted]] : !modelica.real to f64
// CHECK-DAG: %[[x:.*]] = builtin.unrealized_conversion_cast %[[arg0]] : !modelica.real to f64
// CHECK: %[[cmp:.*]] = arith.cmpf olt, %[[x]], %[[y]] : f64
// CHECK: %[[result:.*]] =  builtin.unrealized_conversion_cast %[[cmp]] : i1 to !modelica.bool
// CHECK: return %[[result]] : !modelica.bool

func.func @foo(%arg0 : !modelica.real, %arg1 : !modelica.int) -> !modelica.bool {
    %0 = modelica.lt %arg0, %arg1 : (!modelica.real, !modelica.int) -> !modelica.bool
    func.return %0 : !modelica.bool
}
