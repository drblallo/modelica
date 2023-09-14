#include "marco/Codegen/Analysis/VariableAccessAnalysis.h"

using namespace ::mlir::modelica;

namespace mlir::modelica
{
  VariableAccessAnalysis::VariableAccessAnalysis(EquationTemplateOp op)
      : equationTemplate(op),
        initialized(false),
        valid(true)
  {
  }

  mlir::LogicalResult VariableAccessAnalysis::initialize(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    if (initialized) {
      return mlir::success();
    }

    if (mlir::failed(loadAccesses(symbolTableCollection))) {
      initialized = false;
      return mlir::failure();
    }

    initialized = true;
    return mlir::success();
  }

  void VariableAccessAnalysis::invalidate()
  {
    valid = false;
  }

  bool VariableAccessAnalysis::isInvalidated(
      const mlir::detail::PreservedAnalyses& pa) const
  {
    return !initialized || !valid;
  }

  std::optional<llvm::ArrayRef<VariableAccess>>
  VariableAccessAnalysis::getAccesses(
      EquationInstanceOp instanceOp,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    assert(initialized && "Variable access analysis not initialized");

    if (!valid) {
      if (mlir::failed(loadAccesses(symbolTableCollection))) {
        return std::nullopt;
      }
    }

    uint64_t elementIndex = instanceOp.getViewElementIndex().value_or(0);

    if (auto it = accesses.find(elementIndex); it != accesses.end()) {
      return {it->getSecond()};
    }

    return std::nullopt;
  }

  std::optional<llvm::ArrayRef<VariableAccess>>
  VariableAccessAnalysis::getAccesses(
      MatchedEquationInstanceOp instanceOp,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    assert(initialized && "Variable access analysis not initialized");

    if (!valid) {
      if (mlir::failed(loadAccesses(symbolTableCollection))) {
        return std::nullopt;
      }
    }

    uint64_t elementIndex = instanceOp.getViewElementIndex();

    if (auto it = accesses.find(elementIndex); it != accesses.end()) {
      return {it->getSecond()};
    }

    return std::nullopt;
  }

  std::optional<llvm::ArrayRef<VariableAccess>>
  VariableAccessAnalysis::getAccesses(
      ScheduledEquationInstanceOp instanceOp,
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    assert(initialized && "Variable access analysis not initialized");

    if (!valid) {
      if (mlir::failed(loadAccesses(symbolTableCollection))) {
        return std::nullopt;
      }
    }

    uint64_t elementIndex = instanceOp.getViewElementIndex();

    if (auto it = accesses.find(elementIndex); it != accesses.end()) {
      return {it->getSecond()};
    }

    return std::nullopt;
  }

  mlir::LogicalResult VariableAccessAnalysis::loadAccesses(
      mlir::SymbolTableCollection& symbolTableCollection)
  {
    mlir::Block* bodyBlock = equationTemplate.getBody();

    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(bodyBlock->getTerminator());

    assert(equationSidesOp.getLhsValues().size() ==
           equationSidesOp.getRhsValues().size());

    size_t numOfSideElements = equationSidesOp.getLhsValues().size();

    for (size_t i = 0; i < numOfSideElements; ++i) {
      accesses[i].clear();

      if (mlir::failed(equationTemplate.getAccesses(
              accesses[i], symbolTableCollection, i))) {
        return mlir::failure();
      }
    }

    valid = true;
    return mlir::success();
  }
}
