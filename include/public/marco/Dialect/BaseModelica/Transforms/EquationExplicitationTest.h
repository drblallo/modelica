#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPLICITATIONTEST_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPLICITATIONTEST_H

#include "mlir/Pass/Pass.h"
#include <string>

namespace mlir::bmodelica
{
#define GEN_PASS_DECL_EQUATIONEXPLICITATIONTESTPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createEquationExplicitationTestPass();
}

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONEXPLICITATIONTEST_H
