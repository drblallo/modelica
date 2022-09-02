#ifndef MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H
#define MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H

#include "mlir/Pass/Pass.h"

namespace mlir::modelica
{
#define GEN_PASS_DECL_NEGATEOPDISTRIBUTIONPASS
#define GEN_PASS_DECL_MULOPDISTRIBUTIONPASS
#define GEN_PASS_DECL_DIVOPDISTRIBUTIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"

  std::unique_ptr<mlir::Pass> createNegateOpDistributionPass();

  std::unique_ptr<mlir::Pass> createMulOpDistributionPass();

  std::unique_ptr<mlir::Pass> createDivOpDistributionPass();
}

#endif // MARCO_CODEGEN_TRANSFORMS_OPDISTRIBUTION_H
