// RUN: modelica-opt %s --split-input-file --convert-bmodelica-to-runtime-call | FileCheck %s

// CHECK: runtime.function private @_Mdiv_f64_f64_f64(f64, f64) -> f64

// CHECK-LABEL: @foo
// CHECK-SAME: (%[[arg0:.*]]: f64, %[[arg1:.*]]: f64) -> f64
// CHECK: %[[result:.*]] = runtime.call @_Mdiv_f64_f64_f64(%[[arg0]], %[[arg1]])
// CHECK: return %[[result]]

func.func @foo(%arg0: f64, %arg1: f64) -> f64 {
    %0 = bmodelica.div_trunc %arg0, %arg1 : (f64, f64) -> f64
    func.return %0 : f64
}
