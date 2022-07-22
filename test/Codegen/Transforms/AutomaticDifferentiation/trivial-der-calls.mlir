// RUN: modelica-opt %s                             \
// RUN:     --auto-diff                             \
// RUN:     --convert-modelica-to-cf                \
// RUN:     --convert-modelica-to-arith             \
// RUN:     --convert-modelica-to-llvm              \
// RUN:     --convert-scf-to-cf                     \
// RUN:     --convert-func-to-llvm                  \
// RUN:     --convert-cf-to-llvm                    \
// RUN:     --reconcile-unrealized-casts            \
// RUN: | mlir-cpu-runner                           \
// RUN:     -e main -entry-point-result=void -O0    \
// RUN:     -shared-libs=%runtime_lib               \
// RUN: | FileCheck %s

// CHECK: -1.979985e+00

modelica.function @sin_der : (!modelica.real, !modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @der_x : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create @y : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.sin %3 : !modelica.real -> !modelica.real
    %5 = modelica.der %4 : !modelica.real -> !modelica.real
    modelica.member_store %2, %5 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_sin() -> () {
    %x = modelica.constant #modelica.real<3.0>
    %der_x = modelica.constant #modelica.real<2.0>
    %result = modelica.call @sin_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

// CHECK: 2.822400e-01

modelica.function @cos_der : (!modelica.real, !modelica.real) -> (!modelica.real) {
    %0 = modelica.member_create @x : !modelica.member<!modelica.real, input>
    %1 = modelica.member_create @der_x : !modelica.member<!modelica.real, input>
    %2 = modelica.member_create @y : !modelica.member<!modelica.real, output>
    %3 = modelica.member_load %0 : !modelica.member<!modelica.real, input> -> !modelica.real
    %4 = modelica.cos %3 : !modelica.real -> !modelica.real
    %5 = modelica.der %4 : !modelica.real -> !modelica.real
    modelica.member_store %2, %5 : !modelica.member<!modelica.real, output>, !modelica.real
}

func.func @test_cos() -> () {
    %x = modelica.constant #modelica.real<3.0>
    %der_x = modelica.constant #modelica.real<2.0>
    %result = modelica.call @cos_der(%x, %der_x) : (!modelica.real, !modelica.real) -> (!modelica.real)
    modelica.print %result : !modelica.real
    return
}

func.func @main() -> () {
    call @test_sin() : () -> ()
    call @test_cos() : () -> ()
    return
}
