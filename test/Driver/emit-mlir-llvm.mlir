// RUN: marco -emit-mlir-llvm --omc-bypass -o - %s | FileCheck %s

// CHECK: llvm.mlir.global internal constant @modelName

modelica.model @M {

}
