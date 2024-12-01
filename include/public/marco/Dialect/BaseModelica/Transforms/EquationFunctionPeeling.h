#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONFUNCTIONPEELING_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONFUNCTIONPEELING_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_EQUATIONFUNCTIONPEELINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createEquationFunctionPeelingPass();

std::unique_ptr<mlir::Pass> createEquationFunctionPeelingPass(
    const EquationFunctionPeelingPassOptions &options);
} // namespace mlir::bmodelica

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_EQUATIONFUNCTIONPEELING_H
