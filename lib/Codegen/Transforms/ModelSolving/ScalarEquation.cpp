#include "marco/Codegen/Transforms/ModelSolving/ScalarEquation.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace ::marco::modeling;
using namespace ::mlir::modelica;

namespace marco::codegen
{
  ScalarEquation::ScalarEquation(EquationOp equation, Variables variables)
      : BaseEquation(equation, variables)
  {
    // Check that the equation is not enclosed in a loop
    assert(equation->getParentOfType<ForEquationOp>() == nullptr);

    // Check that all the values are scalars
    [[maybe_unused]] auto isScalarFn = [](mlir::Value value) {
      auto type = value.getType();
      return type.isa<BooleanType>() || type.isa<IntegerType>() || type.isa<RealType>() || type.isa<mlir::IndexType>();
    };

    assert(llvm::all_of(getTerminator().lhsValues(), isScalarFn));
    assert(llvm::all_of(getTerminator().rhsValues(), isScalarFn));
  }

  std::unique_ptr<Equation> ScalarEquation::clone() const
  {
    return std::make_unique<ScalarEquation>(*this);
  }

  EquationOp ScalarEquation::cloneIR() const
  {
    EquationOp equationOp = getOperation();
    mlir::OpBuilder builder(equationOp);
    builder.setInsertionPointAfter(equationOp);
    return mlir::cast<EquationOp>(builder.clone(*equationOp.getOperation()));
  }

  void ScalarEquation::eraseIR()
  {
    getOperation().erase();
  }

  void ScalarEquation::dumpIR(llvm::raw_ostream& os) const
  {
    getOperation()->print(os);
  }

  size_t ScalarEquation::getNumOfIterationVars() const
  {
    return 1;
  }

  MultidimensionalRange ScalarEquation::getIterationRanges() const
  {
    return MultidimensionalRange(Point(0));
  }

  std::vector<Access> ScalarEquation::getAccesses() const
  {
    std::vector<Access> accesses;
    auto terminator = mlir::cast<EquationSidesOp>(getOperation().bodyBlock()->getTerminator());

    auto processFn = [&](mlir::Value value, EquationPath path) {
      searchAccesses(accesses, value, std::move(path));
    };

    processFn(terminator.lhsValues()[0], EquationPath(EquationPath::LEFT));
    processFn(terminator.rhsValues()[0], EquationPath(EquationPath::RIGHT));

    return accesses;
  }

  DimensionAccess ScalarEquation::resolveDimensionAccess(std::pair<mlir::Value, long> access) const
  {
    assert(access.first == nullptr);
    return DimensionAccess::constant(access.second);
  }

  Access ScalarEquation::getAccessAtPath(const EquationPath& path) const
  {
    std::vector<Access> accesses;

    mlir::Value access = getValueAtPath(path);
    searchAccesses(accesses, access, path);

    assert(accesses.size() == 1);
    return accesses[0];
  }

  std::vector<mlir::Value> ScalarEquation::getInductionVariables() const
  {
    return {};
  }

  mlir::LogicalResult ScalarEquation::mapInductionVariables(
      mlir::OpBuilder& builder,
      mlir::BlockAndValueMapping& mapping,
      Equation& destination,
      const ::marco::modeling::AccessFunction& transformation) const
  {
    // Nothing to be mapped
    return mlir::success();
  }

  mlir::LogicalResult ScalarEquation::createTemplateFunctionBody(
      mlir::OpBuilder& builder,
      mlir::BlockAndValueMapping& mapping,
      mlir::ValueRange beginIndexes,
      mlir::ValueRange endIndexes,
      mlir::ValueRange steps,
      ::marco::modeling::scheduling::Direction iterationDirection) const
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    auto equationOp = getOperation();
    auto loc = equationOp.getLoc();

    for (auto& op : equationOp.bodyBlock()->getOperations()) {
      if (auto terminator = mlir::dyn_cast<EquationSidesOp>(op)) {
        // Convert the equality into an assignment
        for (auto [lhs, rhs] : llvm::zip(terminator.lhsValues(), terminator.rhsValues())) {
          mlir::Value mappedLhs = mapping.lookup(lhs);
          mlir::Value mappedRhs = mapping.lookup(rhs);

          if (auto mappedLhsArrayType = mappedLhs.getType().dyn_cast<ArrayType>()) {
            assert(mappedLhsArrayType.getRank() != 0);

            createIterationLoops(
                builder, loc, beginIndexes, endIndexes, steps, iterationDirection,
                [&](mlir::OpBuilder& nestedBuilder, mlir::ValueRange indices) {
                  assert(mappedLhs.getType().cast<ArrayType>().getRank() == indices.size());
                  mlir::Value rhsValue = nestedBuilder.create<LoadOp>(loc, mappedRhs, indices);
                  rhsValue = nestedBuilder.create<CastOp>(loc, mappedLhsArrayType.getElementType(), rhsValue);
                  nestedBuilder.create<StoreOp>(loc, rhsValue, mappedLhs, indices);
                });
          } else {
            auto loadOp = mlir::cast<LoadOp>(mappedLhs.getDefiningOp());
            mappedRhs = builder.create<CastOp>(loc, mappedLhs.getType(), mappedRhs);
            builder.create<StoreOp>(loc, mappedRhs, loadOp.array(), loadOp.indices());
          }
        }
      } else if (mlir::isa<EquationSideOp>(op)) {
        // Ignore equation sides
        continue;
      } else {
        // Clone all the other operations
        builder.clone(op, mapping);
      }
    }

    return mlir::success();
  }
}
