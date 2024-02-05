#include "marco/Codegen/Transforms/Modeling/EquationBridge.h"

using namespace ::mlir::modelica;
using namespace ::mlir::modelica::bridge;

namespace mlir::modelica::bridge
{
  EquationBridge::EquationBridge(
      EquationInstanceOp op,
      mlir::SymbolTableCollection& symbolTable,
      VariableAccessAnalysis& accessAnalysis,
      llvm::DenseMap<mlir::SymbolRefAttr, VariableBridge*>& variablesMap)
      : op(op),
        symbolTable(&symbolTable),
        accessAnalysis(&accessAnalysis),
        variablesMap(&variablesMap)
  {
  }
}

namespace marco::modeling::matching
{
  EquationTraits<EquationBridge*>::Id
  EquationTraits<EquationBridge*>::getId(const Equation* equation)
  {
    return (*equation)->op.getOperation();
  }

  size_t EquationTraits<EquationBridge*>::getNumOfIterationVars(
      const Equation* equation)
  {
    auto numOfInductions = static_cast<uint64_t>(
        (*equation)->op.getInductionVariables().size());

    if (numOfInductions == 0) {
      // Scalar equation.
      return 1;
    }

    return static_cast<size_t>(numOfInductions);
  }

  IndexSet EquationTraits<EquationBridge*>::getIterationRanges(
      const Equation* equation)
  {
    IndexSet iterationSpace = (*equation)->op.getIterationSpace();

    if (iterationSpace.empty()) {
      // Scalar equation.
      iterationSpace += MultidimensionalRange(Range(0, 1));
    }

    return iterationSpace;
  }

  std::vector<Access<
      EquationTraits<EquationBridge*>::VariableType,
      EquationTraits<EquationBridge*>::AccessProperty>>
  EquationTraits<EquationBridge*>::getAccesses(const Equation* equation)
  {
    std::vector<Access<VariableType, AccessProperty>> accesses;

    auto cachedAccesses = (*equation)->accessAnalysis->getAccesses(
        (*equation)->op, *(*equation)->symbolTable);

    if (cachedAccesses) {
      for (auto& access : *cachedAccesses) {
        auto accessFunction = getAccessFunction(
            (*equation)->op.getContext(), access);

        auto variableIt =
            (*(*equation)->variablesMap).find(access.getVariable());

        if (variableIt != (*(*equation)->variablesMap).end()) {
          accesses.emplace_back(
              variableIt->getSecond(),
              std::move(accessFunction),
              access.getPath());
        }
      }
    }

    return accesses;
  }

  std::unique_ptr<AccessFunction>
  EquationTraits<EquationBridge*>::getAccessFunction(
      mlir::MLIRContext* context,
      const mlir::modelica::VariableAccess& access)
  {
    const AccessFunction& accessFunction = access.getAccessFunction();

    if (accessFunction.getNumOfResults() == 0) {
      // Access to scalar variable.
      return AccessFunction::build(mlir::AffineMap::get(
          accessFunction.getNumOfDims(), 0,
          mlir::getAffineConstantExpr(0, context)));
    }

    return accessFunction.clone();
  }
}
