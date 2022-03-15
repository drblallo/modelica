#include "marco/Codegen/Transforms/Model/IDA.h"
#include "marco/Dialect/IDA/IDADialect.h"
#include "mlir/IR/AffineMap.h"
#include <queue>

using namespace ::marco;
using namespace ::marco::codegen;
using namespace ::marco::codegen::modelica;

namespace marco::codegen
{
  IDASolver::IDASolver() = default;

  bool IDASolver::isEnabled() const
  {
    // TODO should depend on the compiler's flags
    return true;
  }

  mlir::Type IDASolver::getSolverInstanceType(mlir::MLIRContext* context) const
  {
    return mlir::ida::InstanceType::get(context);
  }

  bool IDASolver::hasVariable(mlir::Value variable) const
  {
    return llvm::find(variables, variable) != variables.end();
  }

  void IDASolver::addVariable(mlir::Value variable)
  {
    assert(variable.isa<mlir::BlockArgument>());

    if (!hasVariable(variable)) {
      variables.push_back(variable);
    }
  }

  bool IDASolver::hasEquation(ScheduledEquation* equation) const
  {
    return llvm::find(equations, equation) != equations.end();
  }

  void IDASolver::addEquation(ScheduledEquation* equation)
  {
    equations.emplace(equation);
  }

  mlir::LogicalResult IDASolver::init(mlir::OpBuilder& builder, mlir::FuncOp initFunction)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&initFunction.getBody().front());

    // Create the IDA instance.
    // To do so, we need to first compute the total number of scalar variables that IDA
    // has to manage. Such number is equal to the number of scalar equations.
    size_t numberOfScalarEquations = 0;

    for (const auto& equation : equations) {
      numberOfScalarEquations += equation->getIterationRanges().flatSize();
    }

    idaInstance = builder.create<mlir::ida::CreateOp>(
        initFunction.getLoc(), builder.getI64IntegerAttr(numberOfScalarEquations));

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processVariables(mlir::OpBuilder& builder, mlir::FuncOp initFunction, const mlir::BlockAndValueMapping& derivatives)
  {
    mlir::BlockAndValueMapping inverseDerivatives = derivatives.getInverse();
    auto terminator = mlir::cast<YieldOp>(initFunction.getBody().back().getTerminator());

    // Change the ownership of the variables managed to IDA

    for (const auto& variable : variables) {
      mlir::Value value = terminator.values()[variable.cast<mlir::BlockArgument>().getArgNumber()];
      auto memberCreateOp = value.getDefiningOp<MemberCreateOp>();
      builder.setInsertionPoint(memberCreateOp);

      auto memberType = memberCreateOp.resultType().cast<MemberType>();
      auto dimensions = memberType.toArrayType().getShape();

      if (derivatives.contains(variable)) {
        // State variable
        auto idaVariable = builder.create<mlir::ida::AddVariableOp>(
            memberCreateOp.getLoc(), idaInstance,
            builder.getI64ArrayAttr(dimensions),
            builder.getBoolAttr(true));

        mappedVariables.map(variable, idaVariable);
        mlir::Value derivative = derivatives.lookup(variable);
        mappedVariables.map(derivative, idaVariable);

      } else if (!inverseDerivatives.contains(variable)) {
        // Algebraic variable
        auto idaVariable = builder.create<mlir::ida::AddVariableOp>(
            memberCreateOp.getLoc(), idaInstance,
            builder.getI64ArrayAttr(dimensions),
            builder.getBoolAttr(false));

        mappedVariables.map(variable, idaVariable);
      }
    }

    for (const auto& variable : variables) {
      mlir::Value value = terminator.values()[variable.cast<mlir::BlockArgument>().getArgNumber()];
      auto memberCreateOp = value.getDefiningOp<MemberCreateOp>();
      builder.setInsertionPoint(memberCreateOp);
      auto memberType = memberCreateOp.resultType().cast<MemberType>();
      mlir::Value idaVariable = mappedVariables.lookup(variable);

      if (inverseDerivatives.contains(variable)) {
        auto castedVariable = builder.create<mlir::ida::GetDerivativeOp>(
            memberCreateOp.getLoc(),
            memberType.toArrayType(),
            idaInstance, idaVariable);

        memberCreateOp->replaceAllUsesWith(castedVariable);
      } else {
        auto castedVariable = builder.create<mlir::ida::GetVariableOp>(
            memberCreateOp.getLoc(),
            memberType.toArrayType(),
            idaInstance, idaVariable);

        memberCreateOp->replaceAllUsesWith(castedVariable);
      }

      memberCreateOp.erase();
    }

    return mlir::success();
  }

  mlir::LogicalResult IDASolver::processEquations(
      mlir::OpBuilder& builder,
      const Model<ScheduledEquationsBlock>& model,
      mlir::FuncOp initFunction,
      mlir::TypeRange variableTypesWithoutTime,
      const mlir::BlockAndValueMapping& derivatives)
  {
    // Substitute the accesses to non-IDA variables with the equations writing in such variables
    std::vector<std::unique_ptr<ScheduledEquation>> independentEquations;
    std::multimap<unsigned int, std::pair<modeling::MultidimensionalRange, ScheduledEquation*>> writesMap;

    for (const auto& equationsBlock : model.getScheduledBlocks()) {
      for (const auto& equation : *equationsBlock) {
        if (equations.find(equation.get()) == equations.end()) {
          const auto& write = equation->getWrite();
          auto varPosition = write.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
          auto writtenIndices = write.getAccessFunction().map(equation->getIterationRanges());
          writesMap.emplace(varPosition, std::make_pair(writtenIndices, equation.get()));
        }
      }
    }

    std::queue<std::unique_ptr<ScheduledEquation>> processedEquations;

    for (const auto& equation : equations) {
      auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

      auto matchedClone = std::make_unique<MatchedEquation>(
          std::move(clone), equation->getIterationRanges(), equation->getWrite().getPath());

      auto scheduledClone = std::make_unique<ScheduledEquation>(
          std::move(matchedClone), equation->getIterationRanges(), equation->getSchedulingDirection());

      processedEquations.push(std::move(scheduledClone));
    }

    while (!processedEquations.empty()) {
      auto& equation = processedEquations.front();
      bool atLeastOneAccessReplaced = false;

      for (const auto& access : equation->getReads()) {
        auto readIndices = access.getAccessFunction().map(equation->getIterationRanges());
        auto varPosition = access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
        auto writingEquations = llvm::make_range(writesMap.equal_range(varPosition));

        for (const auto& entry : writingEquations) {
          ScheduledEquation* writingEquation = entry.second.second;
          auto writtenVariableIndices = entry.second.first;

          if (!writtenVariableIndices.overlaps(readIndices)) {
            continue;
          }

          atLeastOneAccessReplaced = true;

          auto clone = Equation::build(equation->cloneIR(), equation->getVariables());

          auto explicitWritingEquation = writingEquation->cloneIRAndExplicitate(builder);
          TemporaryEquationGuard guard(*explicitWritingEquation);
          auto res = explicitWritingEquation->replaceInto(builder, *clone, access.getAccessFunction(), access.getPath());
          assert(mlir::succeeded(res));

          // Add the equation with the replaced access
          auto readAccessIndices = access.getAccessFunction().inverseMap(
              modeling::IndexSet(writtenVariableIndices),
              modeling::IndexSet(equation->getIterationRanges()));

          auto newEquationIndices = readAccessIndices.intersect(equation->getIterationRanges());

          for (const auto& range : newEquationIndices) {
            auto matchedEquation = std::make_unique<MatchedEquation>(
                clone->clone(), range, equation->getWrite().getPath());

            auto scheduledEquation = std::make_unique<ScheduledEquation>(
                std::move(matchedEquation), range, equation->getSchedulingDirection());

            processedEquations.push(std::move(scheduledEquation));
          }
        }
      }

      if (atLeastOneAccessReplaced) {
        equation->eraseIR();
      } else {
        independentEquations.push_back(std::move(equation));
      }

      processedEquations.pop();
    }

    // Set inside IDA the information about the equations
    auto terminator = mlir::cast<YieldOp>(initFunction.getBody().back().getTerminator());
    builder.setInsertionPoint(terminator);

    size_t residualFunctionsCounter = 0;
    size_t jacobianFunctionsCounter = 0;

    for (const auto& equation : independentEquations) {
      equation->dumpIR();
      auto ranges = equation->getIterationRanges();
      std::vector<mlir::Attribute> rangesAttr;

      for (size_t i = 0; i < ranges.rank(); ++i) {
        rangesAttr.push_back(builder.getI64ArrayAttr({ ranges[i].getBegin(), ranges[i].getEnd() }));
      }

      auto idaEquation = builder.create<mlir::ida::AddEquationOp>(
          equation->getOperation().getLoc(),
          idaInstance,
          builder.getArrayAttr(rangesAttr));

      for (const auto& access : equation->getAccesses()) {
        mlir::Value idaVariable = mappedVariables.lookup(access.getVariable()->getValue());
        const auto& accessFunction = access.getAccessFunction();

        std::vector<mlir::AffineExpr> expressions;

        for (const auto& dimensionAccess : accessFunction) {
          if (dimensionAccess.isConstantAccess()) {
            expressions.push_back(mlir::getAffineConstantExpr(dimensionAccess.getPosition(), builder.getContext()));
          } else {
            auto baseAccess = mlir::getAffineDimExpr(dimensionAccess.getInductionVariableIndex(), builder.getContext());
            auto withOffset = baseAccess + dimensionAccess.getOffset();
            expressions.push_back(withOffset);
          }
        }

        builder.create<mlir::ida::AddVariableAccessOp>(
            equation->getOperation().getLoc(),
            idaInstance, idaEquation, idaVariable,
            mlir::AffineMap::get(accessFunction.size(), 0, expressions, builder.getContext()));

        auto residualFunctionName = "residualFunction" + std::to_string(residualFunctionsCounter++);
        auto jacobianFunctionName = "jacobianFunction" + std::to_string(jacobianFunctionsCounter++);

        {
          // Create and populate the residual function
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(equation->getOperation()->getParentOfType<mlir::ModuleOp>().getBody());

          auto residualFunction = builder.create<mlir::ida::ResidualFunctionOp>(
              equation->getOperation().getLoc(), residualFunctionName, RealType::get(builder.getContext()), variableTypesWithoutTime, equation->getNumOfIterationVars());

          /*
          mlir::BlockAndValueMapping mapping;

          for (const auto& originalAccess : equation->getAccesses()) {
            auto access.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
          }
           */
        }

        {
          mlir::OpBuilder::InsertionGuard g(builder);
          builder.setInsertionPointToStart(equation->getOperation()->getParentOfType<mlir::ModuleOp>().getBody());
          auto jacobianFunction = builder.create<mlir::ida::JacobianFunctionOp>(equation->getOperation().getLoc(), jacobianFunctionName);
        }

        builder.create<mlir::ida::AddResidualOp>(equation->getOperation().getLoc(), idaInstance, idaEquation, residualFunctionName);
        builder.create<mlir::ida::AddJacobianOp>(equation->getOperation().getLoc(), idaInstance, idaEquation, jacobianFunctionName);
      }
    }
  }
}
