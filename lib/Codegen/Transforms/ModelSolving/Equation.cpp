#include "marco/Codegen/Transforms/ModelSolving/Equation.h"
#include "marco/Codegen/Transforms/ModelSolving/EquationImpl.h"
#include "marco/Codegen/Transforms/ModelSolving/LoopEquation.h"
#include "marco/Codegen/Transforms/ModelSolving/ScalarEquation.h"
#include "marco/Codegen/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/FoldUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ThreadPool.h"
#include <mutex>
#include <numeric>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

static long getIntFromAttribute(mlir::Attribute attribute)
{
  if (auto indexAttr = attribute.dyn_cast<mlir::IntegerAttr>()) {
    return indexAttr.getInt();
  }

  if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
    return booleanAttr.getValue() ? 1 : 0;
  }

  if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
    return integerAttr.getValue().getSExtValue();
  }

  if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
    return realAttr.getValue().convertToDouble();
  }

  llvm_unreachable("Unknown attribute type");
  return 0;
}

static mlir::Attribute getIntegerAttribute(mlir::OpBuilder& builder, mlir::Type type, int value)
{
  if (type.isa<BooleanType>()) {
    return BooleanAttr::get(type, value > 0);
  }

  if (type.isa<IntegerType>()) {
    return IntegerAttr::get(type, value);
  }

  if (type.isa<RealType>()) {
    return RealAttr::get(type, value);
  }

  return builder.getIndexAttr(value);
}

static void foldValue(EquationInterface equationOp, mlir::Value value)
{
  mlir::OperationFolder helper(value.getContext());
  std::stack<mlir::Operation*> visitStack;
  llvm::SmallVector<mlir::Operation*, 3> ops;
  llvm::DenseSet<mlir::Operation*> processed;

  if (auto definingOp = value.getDefiningOp()) {
    visitStack.push(definingOp);
  }

  while (!visitStack.empty()) {
    auto op = visitStack.top();
    visitStack.pop();

    ops.push_back(op);

    for (const auto& operand : op->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        visitStack.push(definingOp);
      }
    }
  }

  llvm::SmallVector<mlir::Operation*, 3> constants;

  for (mlir::Operation* op : llvm::reverse(ops)) {
    if (processed.contains(op)) {
      continue;
    }

    processed.insert(op);

    if (mlir::failed(helper.tryToFold(op, [&](mlir::Operation* constant) {
          constants.push_back(constant);
        }))) {
      break;
    }
  }

  for (auto* op : llvm::reverse(constants)) {
    op->moveBefore(equationOp.bodyBlock(), equationOp.bodyBlock()->begin());
  }
}

static bool isZeroAttr(mlir::Attribute attribute)
{
  if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
    return !booleanAttr.getValue();
  }

  if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
    return integerAttr.getValue() == 0;
  }

  if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
    return realAttr.getValue().isZero();
  }

  return attribute.cast<mlir::IntegerAttr>().getValue() == 0;
}

/// Check if an equation has explicit or implicit induction variables.
///
/// @param equation  equation
/// @return true if the equation is surrounded by explicit loops or defines implicit ones
static bool hasInductionVariables(EquationInterface equation)
{
  auto hasExplicitLoops = [&]() -> bool {
    return equation->getParentOfType<ForEquationOp>() != nullptr;
  };

  auto hasImplicitLoops = [&]() -> bool {
    auto terminator = mlir::cast<EquationSidesOp>(equation.bodyBlock()->getTerminator());

    return llvm::any_of(terminator.getLhsValues(), [](mlir::Value value) {
      return value.getType().isa<ArrayType>();
    });
  };

  return hasExplicitLoops() || hasImplicitLoops();
}

static std::pair<mlir::Value, std::vector<mlir::Value>> collectSubscriptionIndexes(mlir::Value value)
{
  std::vector<mlir::Value> indexes;
  mlir::Operation* op = value.getDefiningOp();

  while (op != nullptr && mlir::isa<LoadOp, SubscriptionOp>(op)) {
    if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
      auto loadIndexes = loadOp.getIndices();

      for (size_t i = 0, e = loadIndexes.size(); i < e; ++i) {
        indexes.push_back(loadIndexes[e - i - 1]);
      }

      value = loadOp.getArray();
      op = value.getDefiningOp();
    } else {
      auto subscriptionOp = mlir::cast<SubscriptionOp>(op);
      auto subscriptionIndexes = subscriptionOp.getIndices();

      for (size_t i = 0, e = subscriptionIndexes.size(); i < e; ++i) {
        indexes.push_back(subscriptionIndexes[e - i - 1]);
      }

      value = subscriptionOp.getSource();
      op = value.getDefiningOp();
    }
  }

  std::reverse(indexes.begin(), indexes.end());
  return std::make_pair(value, std::move(indexes));
}

static mlir::LogicalResult removeSubtractions(mlir::OpBuilder& builder, mlir::Operation* root)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation* op = root;

  if (op == nullptr) {
    return mlir::success();
  }

  if (!mlir::isa<SubscriptionOp>(op) && !mlir::isa<LoadOp>(op)) {
    for (auto operand : op->getOperands()) {
      if (auto res = removeSubtractions(builder, operand.getDefiningOp()); mlir::failed(res)) {
        return res;
      }
    }
  }

  if (auto subOp = mlir::dyn_cast<SubOp>(op)) {
    builder.setInsertionPoint(subOp);
    mlir::Value rhs = subOp.getRhs();
    mlir::Value negatedRhs = builder.create<NegateOp>(rhs.getLoc(), rhs.getType(), rhs);
    auto addOp = builder.create<AddOp>(subOp->getLoc(), subOp.getResult().getType(), subOp.getLhs(), negatedRhs);
    subOp->replaceAllUsesWith(addOp.getOperation());
    subOp->erase();
  }

  return mlir::success();
}

static mlir::LogicalResult distributeMulAndDivOps(mlir::OpBuilder& builder, mlir::Operation* root)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation* op = root;

  if (op == nullptr) {
    return mlir::success();
  }

  for (auto operand : op->getOperands()) {
    if (auto res = distributeMulAndDivOps(builder, operand.getDefiningOp()); mlir::failed(res)) {
      return res;
    }
  }

  if (auto distributableOp = mlir::dyn_cast<DistributableOpInterface>(op)) {
    if (!mlir::isa<NegateOp>(op)) {
      builder.setInsertionPoint(distributableOp);
      mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

      if (result != op) {
        op->replaceAllUsesWith(result);
        op->erase();
      }
    }
  }

  return mlir::success();
}

static mlir::LogicalResult pushNegateOps(mlir::OpBuilder& builder, mlir::Operation* root)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  mlir::Operation* op = root;

  if (op == nullptr) {
    return mlir::success();
  }

  for (auto operand : op->getOperands()) {
    if (auto res = pushNegateOps(builder, operand.getDefiningOp()); mlir::failed(res)) {
      return res;
    }
  }

  if (auto distributableOp = mlir::dyn_cast<NegateOp>(op)) {
    builder.setInsertionPoint(distributableOp);
    mlir::Operation* result = distributableOp.distribute(builder).getDefiningOp();

    if (result != op) {
      op->replaceAllUsesWith(result);
      op->erase();
    }
  }

  return mlir::success();
}

static mlir::LogicalResult collectSummedValues(std::vector<mlir::Value>& result, mlir::Value root)
{
  if (auto definingOp = root.getDefiningOp()) {
    if (auto addOp = mlir::dyn_cast<AddOp>(definingOp)) {
      if (auto res = collectSummedValues(result, addOp.getLhs()); mlir::failed(res)) {
        return res;
      }

      if (auto res = collectSummedValues(result, addOp.getRhs()); mlir::failed(res)) {
        return res;
      }

      return mlir::success();
    }
  }

  result.push_back(root);
  return mlir::success();
}

namespace marco::codegen
{
  std::unique_ptr<Equation> Equation::build(mlir::modelica::EquationInterface equation, Variables variables)
  {
    if (hasInductionVariables(equation)) {
      return std::make_unique<LoopEquation>(std::move(equation), std::move(variables));
    }

    return std::make_unique<ScalarEquation>(std::move(equation), std::move(variables));
  }

  Equation::~Equation() = default;

  void Equation::dumpIR() const
  {
    dumpIR(llvm::dbgs());
  }

  llvm::Optional<Variable*> Equation::findVariable(llvm::StringRef name) const
  {
    return getVariables().findVariable(name);
  }

  void Equation::searchAccesses(
      std::vector<Access>& accesses,
      mlir::Value value,
      EquationPath path) const
  {
    std::vector<DimensionAccess> dimensionAccesses;
    searchAccesses(accesses, value, dimensionAccesses, std::move(path));
  }

  void Equation::searchAccesses(
      std::vector<Access>& accesses,
      mlir::Value value,
      std::vector<DimensionAccess>& dimensionAccesses,
      EquationPath path) const
  {
    if (mlir::Operation* definingOp = value.getDefiningOp(); definingOp != nullptr) {
      searchAccesses(accesses, definingOp, dimensionAccesses, std::move(path));
    }
  }

  void Equation::searchAccesses(
      std::vector<Access>& accesses,
      mlir::Operation* op,
      std::vector<DimensionAccess>& dimensionAccesses,
      EquationPath path) const
  {
    auto processIndexesFn = [&](mlir::ValueRange indexes) {
      for (size_t i = 0, e = indexes.size(); i < e; ++i) {
        mlir::Value index = indexes[e - 1 - i];
        auto evaluatedAccess = evaluateDimensionAccess(index);
        dimensionAccesses.push_back(resolveDimensionAccess(evaluatedAccess));
      }
    };

    if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      resolveAccess(accesses, variableGetOp.getMember(), dimensionAccesses, path);
    } else if (auto loadOp = mlir::dyn_cast<LoadOp>(op)) {
      processIndexesFn(loadOp.getIndices());
      searchAccesses(accesses, loadOp.getArray(), dimensionAccesses, std::move(path));
    } else if (auto subscriptionOp = mlir::dyn_cast<SubscriptionOp>(op)) {
      processIndexesFn(subscriptionOp.getIndices());
      searchAccesses(accesses, subscriptionOp.getSource(), dimensionAccesses, std::move(path));
    } else {
      for (size_t i = 0, e = op->getNumOperands(); i < e; ++i) {
        EquationPath::Guard guard(path);
        path.append(i);
        searchAccesses(accesses, op->getOperand(i), path);
      }
    }
  }

  void Equation::resolveAccess(
      std::vector<Access>& accesses,
      llvm::StringRef variableName,
      std::vector<DimensionAccess>& dimensionsAccesses,
      EquationPath path) const
  {
    llvm::Optional<Variable*> variable = findVariable(variableName);

    if (variable.has_value()) {
      std::vector<DimensionAccess> reverted(dimensionsAccesses.rbegin(), dimensionsAccesses.rend());
      MemberType variableType = (*variable)->getDefiningOp().getMemberType();

      if (variableType.isScalar()) {
        // Scalar variables are masked as arrays with just one element.
        // Thus, an access to a scalar variable is masked as an access to that unique element.

        assert(dimensionsAccesses.empty());
        reverted.push_back(DimensionAccess::constant(0));
        accesses.emplace_back(*variable, AccessFunction(reverted), std::move(path));
      } else {
        if (variableType.getShape().size() == dimensionsAccesses.size()) {
          accesses.emplace_back(*variable, AccessFunction(reverted), std::move(path));
        } else {
          // If the variable is not subscribed enough times, then the remaining indices must be
          // added in their full ranges.

          std::vector<Range> additionalRanges;
          auto shape = variableType.getShape();

          for (size_t i = dimensionsAccesses.size(); i < shape.size(); ++i) {
            additionalRanges.push_back(modeling::Range(0, shape[i]));
          }

          MultidimensionalRange additionalMultidimensionalRange(additionalRanges);

          for (const auto& indices : additionalMultidimensionalRange) {
            std::vector<DimensionAccess> completeDimensionsAccesses(reverted.begin(), reverted.end());

            for (const auto& index : indices) {
              completeDimensionsAccesses.push_back(DimensionAccess::constant(index));
            }

            accesses.emplace_back(*variable, AccessFunction(completeDimensionsAccesses), std::move(path));
          }
        }
      }
    }
  }

  std::pair<mlir::Value, long> Equation::evaluateDimensionAccess(mlir::Value value) const
  {
    if (value.isa<mlir::BlockArgument>()) {
      return std::make_pair(value, 0);
    }

    mlir::Operation* op = value.getDefiningOp();
    assert((mlir::isa<ConstantOp>(op) || mlir::isa<AddOp>(op) || mlir::isa<SubOp>(op)) && "Invalid access pattern");

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return std::make_pair(nullptr, getIntFromAttribute(constantOp.getValue()));
    }

    if (auto addOp = mlir::dyn_cast<AddOp>(op)) {
      auto first = evaluateDimensionAccess(addOp.getLhs());
      auto second = evaluateDimensionAccess(addOp.getRhs());

      assert(first.first == nullptr || second.first == nullptr);
      mlir::Value induction = first.first != nullptr ? first.first : second.first;
      return std::make_pair(induction, first.second + second.second);
    }

    auto subOp = mlir::dyn_cast<SubOp>(op);

    auto first = evaluateDimensionAccess(subOp.getLhs());
    auto second = evaluateDimensionAccess(subOp.getRhs());

    assert(first.first == nullptr || second.first == nullptr);
    mlir::Value induction = first.first != nullptr ? first.first : second.first;
    return std::make_pair(induction, first.second - second.second);
  }

  TemporaryEquationGuard::TemporaryEquationGuard(Equation& equation) : equation(&equation)
  {
  }

  TemporaryEquationGuard::~TemporaryEquationGuard()
  {
    equation->eraseIR();
  }

  BaseEquation::BaseEquation(mlir::modelica::EquationInterface equation, Variables variables)
      : equationOp(equation.getOperation()),
        variables(std::move(variables))
  {
    assert(getTerminator().getLhsValues().size() == 1);
    assert(getTerminator().getRhsValues().size() == 1);
  }

  mlir::modelica::EquationInterface BaseEquation::getOperation() const
  {
    return mlir::cast<EquationInterface>(equationOp);
  }

  Variables BaseEquation::getVariables() const
  {
    return variables;
  }

  void BaseEquation::setVariables(Variables value)
  {
    this->variables = std::move(value);
  }

  mlir::Value BaseEquation::getValueAtPath(const EquationPath& path) const
  {
    auto side = path.getEquationSide();
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    mlir::Value value = side == EquationPath::LEFT ? terminator.getLhsValues()[0] : terminator.getRhsValues()[0];

    for (auto index : path) {
      mlir::Operation* op = value.getDefiningOp();
      assert(op != nullptr && "Invalid equation path");
      value = op->getOperand(index);
    }

    return value;
  }

  void BaseEquation::traversePath(
      const EquationPath& path,
      std::function<bool(mlir::Value)> traverseFn) const
  {
    auto side = path.getEquationSide();
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    mlir::Value value = side == EquationPath::LEFT ? terminator.getLhsValues()[0] : terminator.getRhsValues()[0];

    if (!traverseFn(value)) {
      return;
    }

    for (auto index : path) {
      mlir::Operation* op = value.getDefiningOp();
      assert(op != nullptr && "Invalid equation path");
      value = op->getOperand(index);

      if (!traverseFn(value)) {
        return;
      }
    }
  }

  mlir::LogicalResult BaseEquation::explicitate(
      mlir::OpBuilder& builder, const IndexSet& equationIndices, const EquationPath& path)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    // Get all the paths that lead to accesses with the same accessed variable
    // and function.
    auto requestedAccess = getAccessAtPath(path);
    std::vector<Access> accesses;
    auto allAccesses = getAccesses();

    for (const auto& access : getAccesses()) {
      if (requestedAccess.getVariable() != access.getVariable()) {
        continue;
      }

      auto requestedIndices = requestedAccess.getAccessFunction().map(equationIndices);
      auto currentIndices = access.getAccessFunction().map(equationIndices);

      if (requestedIndices != currentIndices && requestedIndices.overlaps(currentIndices)) {
        return mlir::failure();
      }

      assert(requestedIndices == currentIndices || !requestedIndices.overlaps(currentIndices));

      if (requestedIndices == currentIndices) {
        accesses.push_back(access);
      }
    }

    assert(!accesses.empty());

    // If there is only one access, then it is sufficient to follow the path
    // and invert the operations.

    if (accesses.size() == 1) {
      auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());

      auto lhsOp = terminator.getLhs().getDefiningOp();
      auto rhsOp = terminator.getRhs().getDefiningOp();

      builder.setInsertionPoint(lhsOp);

      if (rhsOp->isBeforeInBlock(lhsOp)) {
        builder.setInsertionPoint(rhsOp);
      }

      for (auto index : path) {
        if (auto res = explicitate(builder, index, path.getEquationSide()); mlir::failed(res)) {
          return res;
        }
      }

      if (path.getEquationSide() == EquationPath::RIGHT) {
        builder.setInsertionPointAfter(terminator);
        builder.create<EquationSidesOp>(terminator->getLoc(), terminator.getRhs(), terminator.getLhs());
        terminator->erase();
      }
    } else {
      // If there are multiple accesses, then we must group all of them and
      // extract the common multiplying factor.

      if (auto res = groupLeftHandSide(builder, equationIndices, requestedAccess); mlir::failed(res)) {
        return res;
      }
    }

    return mlir::success();
  }

  std::unique_ptr<Equation> BaseEquation::cloneIRAndExplicitate(
      mlir::OpBuilder& builder, const IndexSet& equationIndices, const EquationPath& path) const
  {
    EquationInterface clonedOp = cloneIR();
    auto result = Equation::build(clonedOp, getVariables());

    if (auto res = result->explicitate(builder, equationIndices, path); mlir::failed(res)) {
      result->eraseIR();
      return nullptr;
    }

    return result;
  }

  mlir::LogicalResult BaseEquation::replaceInto(
      mlir::OpBuilder& builder,
      const IndexSet& equationIndices,
      Equation& destination,
      const ::marco::modeling::AccessFunction& destinationAccessFunction,
      const EquationPath& destinationPath) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    mlir::Value valueToBeReplaced = destination.getValueAtPath(destinationPath);

    if (valueToBeReplaced.getUsers().empty()) {
      // Substitution is useless.
      // Just a safety check, normally not happening.
      return mlir::failure();
    }

    // Determine where the cloned operations will be placed, that is the first point
    // within the IR where the value to be replaced is used.
    mlir::Operation* insertionPoint = destination.getOperation().bodyBlock()->getTerminator();

    for (const auto& user : valueToBeReplaced.getUsers()) {
      if (user->isBeforeInBlock(insertionPoint)) {
        insertionPoint = user;
      }
    }

    builder.setInsertionPoint(insertionPoint);

    // Map of the source equation values to the destination ones
    mlir::BlockAndValueMapping mapping;

    /*
    // First map the variables to themselves, so that direct accesses can be replaced in case of implicit loops
    for (const auto& variable : getVariables()) {
      auto variableValue = variable->getValue();
      mapping.map(variableValue, variableValue);
    }
     */

    // Determine the access transformation to be applied to each induction variable usage.
    // For example, given the following equations:
    //   destination: x[i0, i1] = 1 - y[i1 + 3, i0 - 2]
    //   source:      y[i1 + 5, i0 - 1] = 3 - x[i0 + 1, i1 + 2] + z[i1 + 3, i0] + i1
    // In order to correctly insert the x[i0 + 1, i1 + 2] source access (and other ones,
    // if existing) into the destination  equation, some operations have to be performed.
    // First, the write access of the access must be inverted:
    //   ([i1 + 5, i0 - 1]) ^ (-1) = [i1 + 1, i0 - 5]
    // The destination access function is then composed with such inverted access:
    //   [i1 + 3, i0 - 2] * [i1 + 1, i0 - 5] = [i0 - 1, i1 - 2]
    // And finally it is combined with the access to be moved into the destination:
    //   [i0 - 1, i1 - 2] * [i0 + 1, i1 + 2] = [i0, i1]
    // In the same way, z[i1, i0] becomes z[i1 + 1, i0 - 1] and i1 becomes [i1 - 2].
    auto sourceAccess = getAccessAtPath(EquationPath::LEFT);
    const auto& sourceAccessFunction = sourceAccess.getAccessFunction();

    if (sourceAccessFunction.isInvertible()) {
      auto combinedAccess = destinationAccessFunction.combine(sourceAccess.getAccessFunction().inverse());

      if (auto res = mapInductionVariables(builder, mapping, destination, combinedAccess); mlir::failed(res)) {
        return res;
      }
    } else {
      // If the access function is not invertible, it may still be possible to move the
      // equation body. In fact, if all the induction variables not appearing in the
      // write access do iterate on a single value (i.e. [n,n+1)), then those constants
      // ('n', in the previous example), can be used to replace the induction variables
      // usages.
      // For example, given the equation "x[10, i1] = ..." , with i1 belonging to [5,6),
      // then i1 can be replaced everywhere within the equation with the constant value
      // 5. Then, if we consider just the [i1] access of 'x', the reduced access
      // function can be now inverted and combined with the destination access, as
      // in the previous case.
      // Note that this always happens in case of scalar variables, as they are accessed
      // by means of a dummy access to their first element, as if they were arrays.

      llvm::SmallVector<bool, 3> usedInductions(getNumOfIterationVars(), false);
      llvm::SmallVector<DimensionAccess, 3> reducedSourceAccesses;
      llvm::SmallVector<DimensionAccess, 3> reducedDestinationAccesses;

      for (size_t i = 0, e = sourceAccessFunction.size(); i < e; ++i) {
        if (!sourceAccessFunction[i].isConstantAccess()) {
          usedInductions[sourceAccessFunction[i].getInductionVariableIndex()] = true;
          reducedSourceAccesses.push_back(sourceAccessFunction[i]);
          reducedDestinationAccesses.push_back(destinationAccessFunction[i]);
        }
      }

      for (const auto& usage : llvm::enumerate(usedInductions)) {
        if (!usage.value()) {
          // If the induction variable is not used, then ensure that it iterates
          // on just one value and thus can be replaced with a constant value.

          if (equationIndices.minContainingRange()[usage.index()].size() != 1) {
            getOperation().emitError("The write access is not invertible");
            return mlir::failure();
          }
        }
      }

      AccessFunction reducedSourceAccessFunction(reducedSourceAccesses);
      AccessFunction reducedDestinationAccessFunction(reducedDestinationAccesses);

      // Before inverting the reduced access function, we need to remap the induction variable indices.
      // Access functions like [i3 - 1][i2] are not in fact invertible, as it expects to operate on 4 induction
      // variables. We first convert it to [i0 - 1][i1], keeping track of the mappings, and then invert it.

      llvm::SmallVector<DimensionAccess, 3> remappedReducedSourceAccesses;
      std::set<size_t> remappedSourceInductions;
      llvm::SmallVector<size_t, 3> sourceDimensionMapping(reducedSourceAccesses.size(), 0);

      for (const auto& dimensionAccess : llvm::enumerate(reducedSourceAccesses)) {
        assert(!dimensionAccess.value().isConstantAccess());
        auto inductionIndex = dimensionAccess.value().getInductionVariableIndex();
        remappedSourceInductions.insert(inductionIndex);
        sourceDimensionMapping[dimensionAccess.index()] = inductionIndex;
        remappedReducedSourceAccesses.push_back(DimensionAccess::relative(dimensionAccess.index(), dimensionAccess.value().getOffset()));
      }

      // The reduced access function is now invertible.
      // Invert the function and combine it with the destination access.

      AccessFunction remappedReducedSourceAccessFunction(remappedReducedSourceAccesses);
      auto combinedReducedAccess = reducedDestinationAccessFunction.combine(remappedReducedSourceAccessFunction.inverse());

      // Then, revert the mappings done previously.
      llvm::SmallVector<DimensionAccess, 3> transformationAccesses;
      size_t usedInductionIndex = 0;

      for (size_t i = 0, e = getNumOfIterationVars(); i < e; ++i) {
        if (usedInductions[i]) {
          transformationAccesses.push_back(combinedReducedAccess[usedInductionIndex++]);

        } else {
          assert(equationIndices.isSingleMultidimensionalRange());
          const auto& range = equationIndices.minContainingRange()[i];
          assert(range.size() == 1);
          transformationAccesses.push_back(DimensionAccess::constant(range.getBegin()));
        }
      }

      AccessFunction transformation(transformationAccesses);

      if (auto res = mapInductionVariables(builder, mapping, destination, transformation); mlir::failed(res)) {
        return res;
      }
    }

    // Obtain the value to be used for the replacement
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    mlir::Value replacement = terminator.getRhsValues()[0];

    // The operations to be cloned, in reverse order
    std::vector<mlir::Operation*> toBeCloned;

    // Perform a depth-first traversal of the tree to determine which operations must
    // be cloned and in which order.
    std::stack<mlir::Operation*> cloneStack;

    if (auto op = replacement.getDefiningOp(); op != nullptr) {
      cloneStack.push(op);
    }

    while (!cloneStack.empty()) {
      auto op = cloneStack.top();
      cloneStack.pop();

      toBeCloned.push_back(op);

      for (const auto& operand : op->getOperands()) {
        if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
          cloneStack.push(operandOp);
        }
      }
    }

    // Clone the operations
    for (auto it = toBeCloned.rbegin(); it != toBeCloned.rend(); ++it) {
      mlir::Operation* op = *it;
      builder.clone(*op, mapping);
    }

    // Replace the original value with the one obtained through the cloned operations
    mlir::Value mappedReplacement = mapping.lookup(replacement);

    // Add the missing subscriptions, if any.
    // This is required when the source equation has implicit loops.
    if (auto mappedReplacementArrayType = mappedReplacement.getType().dyn_cast<ArrayType>()) {
      size_t expectedRank = 0;

      if (auto originalArrayType = valueToBeReplaced.getType().dyn_cast<ArrayType>()) {
        expectedRank = originalArrayType.getRank();
      }

      if (static_cast<size_t>(mappedReplacementArrayType.getRank()) > expectedRank) {
        auto originalIndexes = collectSubscriptionIndexes(valueToBeReplaced);
        size_t rankDifference = mappedReplacementArrayType.getRank() - expectedRank;
        std::vector<mlir::Value> additionalIndexes;

        for (size_t i = originalIndexes.second.size() - rankDifference; i < originalIndexes.second.size(); ++i) {
          additionalIndexes.push_back(originalIndexes.second[i]);
        }

        mlir::Value subscription = builder.create<SubscriptionOp>(
            mappedReplacement.getLoc(), mappedReplacement, additionalIndexes);

        mappedReplacement = builder.create<LoadOp>(mappedReplacement.getLoc(), subscription);
        mapping.map(replacement, mappedReplacement);
      }
    }

    valueToBeReplaced.replaceAllUsesWith(mappedReplacement);

    // Erase the replaced operations, which are now useless
    std::stack<mlir::Operation*> eraseStack;

    if (auto op = valueToBeReplaced.getDefiningOp(); op != nullptr) {
      eraseStack.push(op);
    }

    while (!eraseStack.empty()) {
      auto op = eraseStack.top();
      eraseStack.pop();

      if (op->getUsers().empty()) {
        for (const auto& operand : op->getOperands()) {
          if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
            eraseStack.push(operandOp);
          }
        }

        op->erase();
      }
    }

    return mlir::success();
  }

  EquationSidesOp BaseEquation::getTerminator() const
  {
    return mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
  }

  mlir::LogicalResult BaseEquation::explicitate(
      mlir::OpBuilder& builder, size_t argumentIndex, EquationPath::EquationSide side)
  {
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());
    assert(terminator.getLhsValues().size() == 1);
    assert(terminator.getRhsValues().size() == 1);

    mlir::Value toExplicitate = side == EquationPath::LEFT ? terminator.getLhsValues()[0] : terminator.getRhsValues()[0];
    mlir::Value otherExp = side == EquationPath::RIGHT ? terminator.getLhsValues()[0] : terminator.getRhsValues()[0];

    mlir::Operation* op = toExplicitate.getDefiningOp();

    if (!op->hasTrait<InvertibleOpInterface::Trait>()) {
      op->emitError("Operation is not invertible");
      return mlir::failure();
    }

    return mlir::cast<InvertibleOpInterface>(op).invert(builder, argumentIndex, otherExp);
  }

  mlir::LogicalResult BaseEquation::groupLeftHandSide(
      mlir::OpBuilder& builder,
      const ::marco::modeling::IndexSet& equationIndices,
      const Access& access)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto lhs = getValueAtPath(access.getPath());

    // Determine whether the access to be grouped is inside both the equation's sides or just one of them.
    // When the requested access is found, also check that the path goes through linear operations. If not,
    // explicitation is not possible.
    bool lhsHasAccess = false;
    bool rhsHasAccess = false;

    for (const auto& acc : getAccesses()) {
      if (acc.getVariable() != access.getVariable()) {
        continue;
      }

      auto requestedIndices = access.getAccessFunction().map(equationIndices);
      auto currentIndices = acc.getAccessFunction().map(equationIndices);

      assert(requestedIndices == currentIndices || !requestedIndices.overlaps(currentIndices));

      if (requestedIndices == currentIndices) {
        lhsHasAccess |= acc.getPath().getEquationSide() == EquationPath::LEFT;
        rhsHasAccess |= acc.getPath().getEquationSide() == EquationPath::RIGHT;
      }
    }

    // Convert the expression to a sum of values.
    auto convertToSumsFn = [&](std::function<mlir::Value()> root) -> mlir::LogicalResult {
      if (auto res = removeSubtractions(builder, root().getDefiningOp()); mlir::failed(res)) {
        return res;
      }

      if (auto res = distributeMulAndDivOps(builder, root().getDefiningOp()); mlir::failed(res)) {
        return res;
      }

      if (auto res = pushNegateOps(builder, root().getDefiningOp()); mlir::failed(res)) {
        return res;
      }

      return mlir::success();
    };

    std::vector<mlir::Value> lhsSummedValues;
    std::vector<mlir::Value> rhsSummedValues;

    if (lhsHasAccess) {
      auto rootFn = [&]() -> mlir::Value {
        return getTerminator().getLhsValues()[0];
      };

      if (auto res = convertToSumsFn(rootFn); mlir::failed(res)) {
        return res;
      }

      if (auto res = collectSummedValues(lhsSummedValues, rootFn()); mlir::failed(res)) {
        return res;
      }
    }

    if (rhsHasAccess) {
      auto rootFn = [&]() -> mlir::Value {
        return getTerminator().getRhsValues()[0];
      };

      if (auto res = convertToSumsFn(rootFn); mlir::failed(res)) {
        return res;
      }

      if (auto res = collectSummedValues(rhsSummedValues, rootFn()); mlir::failed(res)) {
        return res;
      }
    }

    auto containsAccessFn = [&](mlir::Value value, const Access& access, EquationPath::EquationSide side) -> bool {
      EquationPath path(side);
      std::vector<Access> accesses;
      searchAccesses(accesses, value, path);

      bool result = llvm::any_of(accesses, [&](const Access& acc) {
        if (acc.getVariable() != access.getVariable()) {
          return false;
        }

        auto requestedIndices = access.getAccessFunction().map(equationIndices);
        auto currentIndices = acc.getAccessFunction().map(equationIndices);

        assert(requestedIndices == currentIndices || !requestedIndices.overlaps(currentIndices));
        return requestedIndices == currentIndices;
      });

      return result;
    };

    auto groupFactorsFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      mlir::Value result = builder.create<ConstantOp>(getOperation()->getLoc(), RealAttr::get(builder.getContext(), 0));

      for (auto it = beginIt; it != endIt; ++it) {
        mlir::Value value = *it;

        auto factor = getMultiplyingFactor(
            builder, equationIndices, value,
            access.getVariable()->getDefiningOp().getSymName(),
            IndexSet(access.getAccessFunction().map(equationIndices)));

        if (!factor.second || factor.first > 1) {
          return nullptr;
        }

        result = builder.create<AddOp>(
            value.getLoc(),
            getMostGenericType(result.getType(), value.getType()),
            result, factor.second);
      }

      return result;
    };

    auto groupRemainingFn = [&](auto beginIt, auto endIt) -> mlir::Value {
      return std::accumulate(
          beginIt, endIt,
          builder.create<ConstantOp>(getOperation()->getLoc(), RealAttr::get(builder.getContext(), 0)).getResult(),
          [&](mlir::Value acc, mlir::Value value) -> mlir::Value {
            return builder.create<AddOp>(value.getLoc(), getMostGenericType(acc.getType(), value.getType()), acc, value);
          });
    };

    builder.setInsertionPoint(getTerminator());

    if (lhsHasAccess && rhsHasAccess) {
      auto leftPos = llvm::partition(lhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::LEFT);
      });

      auto rightPos = llvm::partition(rhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::RIGHT);
      });

      mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);
      mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

      if (lhsFactor == nullptr || rhsFactor == nullptr) {
        return mlir::failure();
      }

      mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());
      mlir::Value rhsRemaining = groupRemainingFn(rightPos, rhsSummedValues.end());

      auto terminator = getTerminator();
      auto loc = terminator->getLoc();

      auto rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(rhsRemaining.getType(), lhsRemaining.getType()), rhsRemaining, lhsRemaining),
          builder.create<SubOp>(loc, getMostGenericType(lhsFactor.getType(), rhsFactor.getType()), lhsFactor, rhsFactor));

      // Check if we are dividing by zero
      foldValue(getOperation(), rhs.getRhs());

      if (auto divisorOp = mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
        if (isZeroAttr(divisorOp.getValue())) {
          return mlir::failure();
        }
      }

      auto lhsOp = builder.create<EquationSideOp>(loc, lhs);
      auto oldLhsOp = terminator.getLhs().getDefiningOp();
      oldLhsOp->replaceAllUsesWith(lhsOp);
      oldLhsOp->erase();

      auto rhsOp = builder.create<EquationSideOp>(loc, rhs.getResult());
      auto oldRhsOp = terminator.getRhs().getDefiningOp();
      oldRhsOp->replaceAllUsesWith(rhsOp);
      oldRhsOp->erase();

      return mlir::success();
    }

    if (lhsHasAccess) {
      auto leftPos = llvm::partition(lhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::LEFT);
      });

      mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);

      if (lhsFactor == nullptr) {
        return mlir::failure();
      }

      mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());

      auto terminator = getTerminator();
      auto loc = terminator->getLoc();

      auto rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(terminator.getRhsValues()[0].getType(), lhsRemaining.getType()), terminator.getRhsValues()[0], lhsRemaining),
          lhsFactor);

      // Check if we are dividing by zero
      foldValue(getOperation(), rhs.getRhs());

      if (auto divisorOp = mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
        if (isZeroAttr(divisorOp.getValue())) {
          return mlir::failure();
        }
      }

      auto lhsOp = builder.create<EquationSideOp>(loc, lhs);
      auto oldLhsOp = terminator.getLhs().getDefiningOp();
      oldLhsOp->replaceAllUsesWith(lhsOp);
      oldLhsOp->erase();

      auto rhsOp = builder.create<EquationSideOp>(loc, rhs.getResult());
      auto oldRhsOp = terminator.getRhs().getDefiningOp();
      oldRhsOp->replaceAllUsesWith(rhsOp);
      oldRhsOp->erase();

      return mlir::success();
    }

    if (rhsHasAccess) {
      auto rightPos = llvm::partition(rhsSummedValues, [&](const auto& value) {
        return containsAccessFn(value, access, EquationPath::RIGHT);
      });

      mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

      if (rhsFactor == nullptr) {
        return mlir::failure();
      }

      mlir::Value rhsRemaining = groupRemainingFn(rightPos, rhsSummedValues.end());

      auto terminator = getTerminator();
      auto loc = terminator->getLoc();

      auto rhs = builder.create<DivOp>(
          loc, lhs.getType(),
          builder.create<SubOp>(loc, getMostGenericType(terminator.getLhsValues()[0].getType(), rhsRemaining.getType()), terminator.getLhsValues()[0], rhsRemaining),
          rhsFactor);

      // Check if we are dividing by zero
      foldValue(getOperation(), rhs.getRhs());

      if (auto divisorOp = mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
        if (isZeroAttr(divisorOp.getValue())) {
          return mlir::failure();
        }
      }

      auto lhsOp = builder.create<EquationSideOp>(loc, lhs);
      auto oldLhsOp = terminator.getLhs().getDefiningOp();
      oldLhsOp->replaceAllUsesWith(lhsOp);
      oldLhsOp->erase();

      auto rhsOp = builder.create<EquationSideOp>(loc, rhs.getResult());
      auto oldRhsOp = terminator.getRhs().getDefiningOp();
      oldRhsOp->replaceAllUsesWith(rhsOp);
      oldRhsOp->erase();

      return mlir::success();
    }

    llvm_unreachable("Access not found");
    return mlir::failure();
  }

  std::pair<unsigned int, mlir::Value> BaseEquation::getMultiplyingFactor(
      mlir::OpBuilder& builder,
      const IndexSet& equationIndices,
      mlir::Value value,
      llvm::StringRef variable,
      const IndexSet& variableIndices) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);

    if (getVariables().isReferenceAccess(value)) {
      std::vector<Access> accesses;
      EquationPath path(EquationPath::LEFT);
      searchAccesses(accesses, value, path);

      if (accesses.size() != 1) {
        return std::make_pair(1, nullptr);
      }

      assert(accesses.size() == 1);

      if (accesses[0].getVariable()->getDefiningOp().getSymName() == variable &&
          variableIndices == accesses[0].getAccessFunction().map(equationIndices)) {
        mlir::Value one = builder.create<ConstantOp>(value.getLoc(), getIntegerAttribute(builder, value.getType(), 1));
        return std::make_pair(1, one);
      }
    }

    mlir::Operation* op = value.getDefiningOp();

    if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
      return std::make_pair(0, constantOp.getResult());
    }

    if (auto negateOp = mlir::dyn_cast<NegateOp>(op)) {
      auto operand = getMultiplyingFactor(builder, equationIndices, negateOp.getOperand(), variable, variableIndices);

      if (!operand.second) {
        return std::make_pair(operand.first, nullptr);
      }

      mlir::Value result = builder.create<NegateOp>(
          negateOp.getLoc(), negateOp.getResult().getType(), operand.second);

      return std::make_pair(operand.first, result);
    }

    if (auto mulOp = mlir::dyn_cast<MulOp>(op)) {
      auto lhs = getMultiplyingFactor(builder, equationIndices, mulOp.getLhs(), variable, variableIndices);
      auto rhs = getMultiplyingFactor(builder, equationIndices, mulOp.getRhs(), variable, variableIndices);

      if (!lhs.second || !rhs.second) {
        return std::make_pair(0, nullptr);
      }

      mlir::Value result = builder.create<MulOp>(
          mulOp.getLoc(), mulOp.getResult().getType(), lhs.second, rhs.second);

      return std::make_pair(lhs.first + rhs.first, result);
    }

    auto hasAccessToVar = [&](mlir::Value value) -> bool {
      // Dummy path. Not used, but required by the infrastructure.
      EquationPath path(EquationPath::LEFT);

      std::vector<Access> accesses;
      searchAccesses(accesses, value, path);

      bool hasAccess = llvm::any_of(accesses, [&](const auto& access) {
        return access.getVariable()->getDefiningOp().getSymName() == variable &&
            variableIndices == access.getAccessFunction().map(equationIndices);
      });

      if (hasAccess) {
        return true;
      }

      return false;
    };

    if (auto divOp = mlir::dyn_cast<DivOp>(op)) {
      auto dividend = getMultiplyingFactor(builder, equationIndices, divOp.getLhs(), variable, variableIndices);

      if (!dividend.second) {
        return dividend;
      }

      // Check that the right-hand side value has no access to the variable of interest
      if (hasAccessToVar(divOp.getRhs())) {
        return std::make_pair(dividend.first, nullptr);
      }

      mlir::Value result = builder.create<DivOp>(
          divOp.getLoc(), divOp.getResult().getType(), dividend.second, divOp.getRhs());

      return std::make_pair(dividend.first, result);
    }

    // Check that the value is not the result of an operation using the variable of interest.
    // If it has such access, then we are not able to extract the multiplying factor.
    if (hasAccessToVar(value)) {
      return std::make_pair(1, nullptr);
    }

    return std::make_pair(0, value);
  }

  void BaseEquation::createIterationLoops(
      mlir::OpBuilder& builder,
      mlir::Location loc,
      mlir::ValueRange beginIndices,
      mlir::ValueRange endIndices,
      mlir::ValueRange steps,
      marco::modeling::scheduling::Direction iterationDirection,
      std::function<void(mlir::OpBuilder&, mlir::ValueRange)> bodyBuilder) const
  {
    std::vector<mlir::Value> inductionVariables;

    assert(beginIndices.size() == endIndices.size());
    assert(beginIndices.size() == steps.size());

    assert(iterationDirection == modeling::scheduling::Direction::Forward ||
           iterationDirection == modeling::scheduling::Direction::Backward);

    auto conditionFn = [&](mlir::Value index, mlir::Value end) -> mlir::Value {
      if (iterationDirection == modeling::scheduling::Direction::Backward) {
        return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::sgt, index, end).getResult();
      }

      return builder.create<mlir::arith::CmpIOp>(loc, mlir::arith::CmpIPredicate::slt, index, end).getResult();
    };

    auto updateFn = [&](mlir::Value index, mlir::Value step) -> mlir::Value {
      if (iterationDirection == modeling::scheduling::Direction::Backward) {
        return builder.create<mlir::arith::SubIOp>(loc, index, step).getResult();
      }

      return builder.create<mlir::arith::AddIOp>(loc, index, step).getResult();
    };

    mlir::Operation* firstLoop = nullptr;

    for (size_t i = 0; i < steps.size(); ++i) {
      auto whileOp = builder.create<mlir::scf::WhileOp>(loc, builder.getIndexType(), beginIndices[i]);

      if (i == 0) {
        firstLoop = whileOp.getOperation();
      }

      // Check the condition.
      // A naive check can consist in the equality comparison. However, in order to be future-proof with
      // respect to steps greater than one, we need to check if the current value is beyond the end boundary.
      // This in turn requires to know the iteration direction.
      mlir::Block* beforeBlock = builder.createBlock(&whileOp.getBefore(), {}, builder.getIndexType(), loc);
      builder.setInsertionPointToStart(beforeBlock);
      mlir::Value condition = conditionFn(whileOp.getBefore().getArgument(0), endIndices[i]);
      builder.create<mlir::scf::ConditionOp>(loc, condition, whileOp.getBefore().getArgument(0));

      // Execute the loop body
      mlir::Block* afterBlock = builder.createBlock(&whileOp.getAfter(), {}, builder.getIndexType(), loc);
      mlir::Value inductionVariable = afterBlock->getArgument(0);
      inductionVariables.push_back(inductionVariable);
      builder.setInsertionPointToStart(afterBlock);

      // Update the induction variable
      mlir::Value nextValue = updateFn(inductionVariable, steps[i]);
      builder.create<mlir::scf::YieldOp>(loc, nextValue);
      builder.setInsertionPoint(nextValue.getDefiningOp());
    }

    bodyBuilder(builder, inductionVariables);

    if (firstLoop != nullptr) {
      builder.setInsertionPointAfter(firstLoop);
    }
  }

  mlir::func::FuncOp BaseEquation::createTemplateFunction(
      llvm::ThreadPool& threadPool,
      mlir::OpBuilder& builder,
      llvm::StringRef functionName,
      ::marco::modeling::scheduling::Direction iterationDirection,
      const mlir::SymbolTable& symbolTable,
      llvm::SmallVectorImpl<VariableOp>& usedVariables) const
  {
    /*
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto equationInt = getOperation();
    mlir::Location loc = equationInt.getLoc();

    auto moduleOp = equationInt->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(moduleOp.getBody());

    // Determine the accessed variables.
    llvm::DenseSet<VariableOp> accessedVariablesSet;

    for (const auto& access : getAccesses()) {
      accessedVariablesSet.insert(access.getVariable()->getDefiningOp());
    }

    // Determine the argument types of the function.
    llvm::SmallVector<VariableOp> accessedVariables(
        accessedVariablesSet.begin(), accessedVariablesSet.end());

    llvm::sort(accessedVariables, [](VariableOp first, VariableOp second) {
      return first.getSymName().compare(second.getSymName());
    });

    llvm::SmallVector<mlir::Type> argTypes;

    for (VariableOp variableOp : accessedVariables) {
      argTypes.push_back(variableOp.getMemberType().toArrayType());
    }

    // For each iteration variable we need to specify three value: the lower
    // bound, the upper bound and the iteration step.
    size_t numOfIterationVars = getNumOfIterationVars();
    argTypes.append(3 * numOfIterationVars, builder.getIndexType());

    // Create the function.
    auto functionOp = builder.create<FunctionOp>(
        loc, functionName, builder.getFunctionType(argTypes, llvm::None));

    mlir::Block* entryBlock = builder.createBlock(&functionOp.getBody());
    builder.setInsertionPointToStart(entryBlock);

    // Declare the variables.
    mlir::BlockAndValueMapping mapping;

    for (VariableOp variableOp : accessedVariables) {
      auto clonedVariableOp = mlir::cast<VariableOp>(
          builder.clone(*variableOp.getOperation(), mapping));
    }

    llvm::SmallVector<VariableOp> lowerBoundVariables;
    llvm::SmallVector<VariableOp> upperBoundVariables;
    llvm::SmallVector<VariableOp> stepVariables;

    for (size_t i = 0; i < numOfIterationVars; ++i) {
      auto variableType = MemberType::get(
          llvm::None, builder.getIndexType(),
          VariabilityProperty::none,
          IOProperty::input);

      lowerBoundVariables.push_back(builder.create<VariableOp>(
          loc, variableType, llvm::None));

      upperBoundVariables.push_back(builder.create<VariableOp>(
          loc, variableType, llvm::None));

      stepVariables.push_back(builder.create<VariableOp>(
          loc, variableType, llvm::None));
    }

    // Create the body of the function.
    auto algorithmOp = builder.create<AlgorithmOp>(loc);

    mlir::Block* algorithmBody =
        builder.createBlock(&algorithmOp.getBodyRegion());

    builder.setInsertionPointToStart(algorithmBody);
     */

    auto equation = getOperation();

    auto loc = getOperation()->getLoc();
    mlir::OpBuilder::InsertionGuard guard(builder);

    auto modelOp = equation->getParentOfType<ModelOp>();
    auto module = modelOp->getParentOfType<mlir::ModuleOp>();
    builder.setInsertionPointToEnd(module.getBody());

    // Determine the variables that are used by the equation.
    auto accesses = getAccesses();
    size_t numOfAccesses = accesses.size();

    llvm::DenseSet<VariableOp> usedVariablesSet;
    std::mutex accesesMutex;

    llvm::ThreadPoolTaskGroup tasks(threadPool);
    unsigned int numOfThreads = threadPool.getThreadCount();
    size_t chunkSize = (numOfAccesses + numOfThreads - 1) / numOfThreads;

    auto accessMapFn = [&](size_t from, size_t to) {
      for (size_t i = from; i < to; ++i) {
        const auto& access = accesses[i];
        VariableOp variableOp = access.getVariable()->getDefiningOp();

        std::lock_guard<std::mutex> lockGuard(accesesMutex);
        usedVariablesSet.insert(variableOp);
      }
    };

    for (unsigned int i = 0, e = threadPool.getThreadCount(); i < e; ++i) {
      size_t from = std::min(numOfAccesses, i * chunkSize);
      size_t to = std::min(numOfAccesses, (i + 1) * chunkSize);

      if (from < to) {
        tasks.async(accessMapFn, from, to);
      }
    }

    tasks.wait();

    usedVariables.insert(usedVariables.end(), usedVariablesSet.begin(), usedVariablesSet.end());

    llvm::sort(usedVariables, [](VariableOp first, VariableOp second) {
      return first.getSymName().compare(second.getSymName());
    });

    // Determine the arguments of the function.
    llvm::SmallVector<mlir::Type, 6> argsTypes;

    // Add the variables to the function signature
    for (VariableOp variableOp : usedVariables) {
      argsTypes.push_back(variableOp.getMemberType().toArrayType());
    }

    // For each iteration variable we need to specify three value: the lower
    // bound, the upper bound and the iteration step.
    argsTypes.append(3 * getNumOfIterationVars(), builder.getIndexType());

    // Create the "template" function and its entry block
    auto functionType = builder.getFunctionType(argsTypes, llvm::None);
    auto function = builder.create<mlir::func::FuncOp>(loc, functionName, functionType);

    function->setAttr(
        "llvm.linkage",
        mlir::LLVM::LinkageAttr::get(
            builder.getContext(), mlir::LLVM::Linkage::Internal));

    auto* entryBlock = function.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    mlir::BlockAndValueMapping mapping;
    llvm::StringMap<mlir::Value> variablesMap;

    for (size_t i = 0, e = usedVariables.size(); i < e; ++i) {
      VariableOp from = usedVariables[i];
      mlir::Value to = function.getArgument(i);
      variablesMap[from.getSymName()] = to;
    }

    // Create the iteration loops
    llvm::SmallVector<mlir::Value, 3> lowerBounds;
    llvm::SmallVector<mlir::Value, 3> upperBounds;
    llvm::SmallVector<mlir::Value, 3> steps;

    size_t numOfVariables = usedVariablesSet.size();

    for (size_t i = 0, e = getNumOfIterationVars(); i < e; ++i) {
      lowerBounds.push_back(function.getArgument(numOfVariables + 0 + i * 3));
      upperBounds.push_back(function.getArgument(numOfVariables + 1 + i * 3));
      steps.push_back(function.getArgument(numOfVariables + 2 + i * 3));
    }

    // Delegate the body creation to the actual equation implementation
    if (auto res = createTemplateFunctionBody(
        builder, mapping, lowerBounds, upperBounds, steps, variablesMap, iterationDirection); mlir::failed(res)) {
      return nullptr;
    }

    builder.setInsertionPointToEnd(&function.getBody().back());
    builder.create<mlir::func::ReturnOp>(loc);
    return function;
  }
}
