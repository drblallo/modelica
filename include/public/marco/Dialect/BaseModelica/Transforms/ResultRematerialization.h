#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RESULTREMATERIALIZATION_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RESULTREMATERIALIZATION_H

#include "mlir/Pass/Pass.h"

namespace mlir::bmodelica {
#define GEN_PASS_DECL_RESULTREMATERIALIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"

std::unique_ptr<mlir::Pass> createResultRematerializationPass();
} // namespace mlir::bmodelica


#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_RESULTREMATERIALIZATION_H
