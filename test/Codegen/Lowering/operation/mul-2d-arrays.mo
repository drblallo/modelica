// RUN: marco -mc1 %s --omc-bypass -emit-mlir -o - | FileCheck %s

// CHECK-LABEL: @Integers
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x5x!bmodelica.int>, tensor<5x4x!bmodelica.int>) -> tensor<3x4x!bmodelica.int>

function Integers
    input Integer[3,5] x;
    input Integer[5,4] y;
    output Integer[3,4] z;
algorithm
    z := x * y;
end Integers;

// CHECK-LABEL: @Reals
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x5x!bmodelica.real>, tensor<5x4x!bmodelica.real>) -> tensor<3x4x!bmodelica.real>

function Reals
    input Real[3,5] x;
    input Real[5,4] y;
    output Real[3,4] z;
algorithm
    z := x * y;
end Reals;

// CHECK-LABEL: @IntegerReal
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x5x!bmodelica.int>, tensor<5x4x!bmodelica.real>) -> tensor<3x4x!bmodelica.real>

function IntegerReal
    input Integer[3,5] x;
    input Real[5,4] y;
    output Real[3,4] z;
algorithm
    z := x * y;
end IntegerReal;

// CHECK-LABEL: @RealInteger
// CHECK: bmodelica.mul %{{.*}}, %{{.*}} : (tensor<3x5x!bmodelica.real>, tensor<5x4x!bmodelica.int>) -> tensor<3x4x!bmodelica.real>

function RealInteger
    input Real[3,5] x;
    input Integer[5,4] y;
    output Real[3,4] z;
algorithm
    z := x * y;
end RealInteger;
