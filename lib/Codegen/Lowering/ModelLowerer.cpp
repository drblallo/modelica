#include "marco/Codegen/Lowering/ModelLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

static IOProperty getIOProperty(const Member& member) {
  if (member.isInput()) {
    return IOProperty::input;
  }

  if (member.isOutput()) {
    return IOProperty::output;
  }

  return IOProperty::none;
}

namespace marco::codegen::lowering
{
  ModelLowerer::ModelLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  std::vector<mlir::Operation*> ModelLowerer::lower(const Model& model)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    Lowerer::SymbolScope varScope(symbolTable());

    auto location = loc(model.getLocation());

    llvm::SmallVector<mlir::Type, 3> variableTypes;
    llvm::SmallVector<mlir::Location, 3> variableLocations;

    // Determine the type of each variable
    llvm::SmallVector<mlir::Attribute, 3> variableNames;

    for (const auto& member : model.getMembers()) {
      mlir::Type type = lower(member->getType());

      if (!type.isa<ArrayType>()) {
        type = ArrayType::get(builder().getContext(), type, llvm::None);
      }

      variableTypes.push_back(type);

      mlir::StringAttr nameAttribute = builder().getStringAttr(member->getName());
      variableNames.push_back(nameAttribute);

      variableLocations.push_back(loc(member->getLocation()));
    }

    // Create the model operation and its blocks
    auto modelOp = builder().create<ModelOp>(location);

    mlir::Block* initBlock = builder().createBlock(&modelOp.getInitRegion());
    mlir::Block* equationsBlock = builder().createBlock(&modelOp.getEquationsRegion(), {}, variableTypes, variableLocations);
    mlir::Block* initialEquationsBlock = builder().createBlock(&modelOp.getInitialEquationsRegion(), {}, variableTypes, variableLocations);

    {
      // Simulation variables
      builder().setInsertionPointToStart(initBlock);
      llvm::SmallVector<mlir::Value, 3> vars;

      for (const auto& member : model.getMembers()) {
        lower(*member);
        vars.push_back(symbolTable().lookup(member->getName()).getReference());
      }

      builder().create<YieldOp>(location, vars);
    }

    {
      // Equations
      builder().setInsertionPointToStart(equationsBlock);
      symbolTable().insert("time", Reference::time(&builder()));

      for (const auto& member : llvm::enumerate(model.getMembers())) {
        symbolTable().insert(
            member.value()->getName(),
            Reference::memory(&builder(), modelOp.getEquationsRegion().getArgument(member.index())));
      }

      // Members with an assigned value are conceptually the same as equations performing that assignment.
      for (const auto& member : model.getMembers()) {
        if (!member->isParameter() && member->hasModification()) {
          if (const auto* modification = member->getModification(); modification->hasExpression()) {
            createMemberEquation(*member, *modification->getExpression());
            builder().setInsertionPointToEnd(equationsBlock);
          }
        }
      }

      // Create the equations
      for (const auto& block : model.getEquationsBlocks()) {
        for (const auto& equation : block->getEquations()) {
          lower(*equation);
        }

        for (const auto& forEquation : block->getForEquations()) {
          lower(*forEquation);
        }
      }
    }

    {
      // Initial equations
      builder().setInsertionPointToStart(initialEquationsBlock);
      symbolTable().insert("time", Reference::time(&builder()));

      for (const auto& member : llvm::enumerate(model.getMembers())) {
        symbolTable().insert(
            member.value()->getName(),
            Reference::memory(&builder(), modelOp.getInitialEquationsRegion().getArgument(member.index())));
      }

      // Create the initial equations from the 'start' attributes
      for (const auto& member : model.getMembers()) {
        // TODO: check also for fixed = true, but only when OMC will propagate it after the flattening stage

        if (member->hasStartProperty() && member->getFixedProperty()) {
          auto startProperty = member->getStartProperty();
          createMemberEquation(*member, *startProperty.value);
          builder().setInsertionPointToEnd(initialEquationsBlock);
        }
      }

      // Create the equations
      for (const auto& block : model.getInitialEquationsBlocks()) {
        for (const auto& equation : block->getEquations()) {
          lower(*equation);
        }

        for (const auto& forEquation : block->getForEquations()) {
          lower(*forEquation);
        }
      }
    }

    // Add the model operation to the list of top-level operations
    result.push_back(modelOp);

    // Process the inner classes
    builder().setInsertionPointAfter(modelOp);

    for (const auto& innerClass : model.getInnerClasses()) {
      for (auto& loweredInnerClass : lower(*innerClass)) {
        result.push_back(std::move(loweredInnerClass));
      }
    }

    return result;
  }

  void ModelLowerer::lower(const Member& member)
  {
    auto location = loc(member.getLocation());

    const auto& frontendType = member.getType();
    mlir::Type type = lower(frontendType);
    auto memberType = MemberType::wrap(type, member.isParameter(), getIOProperty(member));

    mlir::Value memberOp = builder().create<MemberCreateOp>(
        location, member.getName(), memberType, llvm::None);

    symbolTable().insert(member.getName(), Reference::member(&builder(), memberOp));

    Reference reference = symbolTable().lookup(member.getName());

    if (member.hasStartProperty() && !member.getFixedProperty()) {
      auto startProperty = member.getStartProperty();
      auto values = lower(*startProperty.value);
      assert(values.size() == 1);

      if (startProperty.each) {
        assert(type.isa<ArrayType>());
        builder().create<ArrayFillOp>(location, *reference, *values[0]);
      } else {
        reference.set(*values[0]);
      }
    } else if (member.isParameter() && member.hasExpression()) {
      mlir::Value value = *lower(*member.getExpression())[0];

      if (type.isa<ArrayType>()) {
        if (value.getType().isa<ArrayType>()) {
          reference.set(value);
        } else {
          builder().create<ArrayFillOp>(location, *reference, value);
        }
      } else {
        reference.set(value);
      }
    } else {
      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        mlir::Value zero = builder().create<ConstantOp>(location, getZeroAttr(arrayType.getElementType()));
        builder().create<ArrayFillOp>(location, *reference, zero);
      } else {
        mlir::Value zero = builder().create<ConstantOp>(location, getZeroAttr(type));
        reference.set(zero);
      }
    }
  }

  void ModelLowerer::createEquations(const ast::Model& model)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());

  }

  void ModelLowerer::createInitialEquations(const ast::Model& model)
  {

  }

  void ModelLowerer::createMemberEquation(
      const ast::Member& member, const ast::Expression& expression)
  {
    mlir::OpBuilder::InsertionGuard guard(builder());
    auto location = loc(expression.getLocation());

    auto memberType = lower(member.getType());
    auto expressionType = lower(expression.getType());

    std::vector<mlir::Value> inductionVariables;

    if (auto memberArrayType = memberType.dyn_cast<ArrayType>()) {
      unsigned int expressionRank = 0;

      if (auto expressionArrayType = expressionType.dyn_cast<ArrayType>()) {
        expressionRank = expressionArrayType.getRank();
      }

      auto memberRank = memberArrayType.getRank();
      assert(expressionRank == 0 || expressionRank == memberRank);

      for (unsigned int i = 0; i < memberRank - expressionRank; ++i) {
        auto forEquationOp = builder().create<ForEquationOp>(location, 0, memberArrayType.getShape()[i] - 1);
        inductionVariables.push_back(forEquationOp.induction());
        builder().setInsertionPointToStart(forEquationOp.bodyBlock());
      }
    }

    auto equationOp = builder().create<EquationOp>(location);
    assert(equationOp.getBodyRegion().empty());
    mlir::Block* equationBodyBlock = builder().createBlock(&equationOp.getBodyRegion());
    builder().setInsertionPointToStart(equationBodyBlock);

    // Left-hand side
    mlir::Value lhsValue = *symbolTable().lookup(member.getName());

    if (!inductionVariables.empty()) {
      lhsValue = builder().create<LoadOp>(location, lhsValue, inductionVariables);
    }

    // Right-hand side
    auto rhs = lower(expression);
    assert(rhs.size() == 1);
    mlir::Value rhsValue = *rhs[0];

    // Create the assignment
    mlir::Value lhsTuple = builder().create<EquationSideOp>(location, lhsValue);
    mlir::Value rhsTuple = builder().create<EquationSideOp>(location, rhsValue);
    builder().create<EquationSidesOp>(location, lhsTuple, rhsTuple);
  }
}
