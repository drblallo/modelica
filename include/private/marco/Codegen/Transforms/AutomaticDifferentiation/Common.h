#ifndef MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_PASS_H
#define MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_PASS_H

#include "mlir/IR/BlockAndValueMapping.h"

namespace marco::codegen
{
  /// Compose the full derivative member name according to the derivative order.
  /// If the order is 1, then it is omitted.
  ///
  /// @param variableName  base variable name
  /// @param order 				 derivative order
  /// @return derived variable name
  std::string getFullDerVariableName(llvm::StringRef baseName, unsigned int order);

  /// Given a full derivative variable name of order n, compose the name of the
  /// n + 1 variable order.
  ///
  /// @param currentName 	   current variable name
  /// @param requestedOrder  desired derivative order
  /// @return next order derived variable name
  std::string getNextFullDerVariableName(llvm::StringRef currentName, unsigned int requestedOrder);

  void mapFullDerivatives(mlir::BlockAndValueMapping& mapping, llvm::ArrayRef<mlir::Value> members);
}

#endif // MARCO_CODEGEN_TRANSFORMS_AUTOMATIC_DIFFERENTIATION_PASS_H
