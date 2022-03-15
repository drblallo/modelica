#include "marco/Codegen/Lowering/StandardFunctionLowerer.h"

using namespace ::marco;
using namespace ::marco::ast;
using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace marco::codegen::lowering
{
  StandardFunctionLowerer::StandardFunctionLowerer(LoweringContext* context, BridgeInterface* bridge)
      : Lowerer(context, bridge)
  {
  }

  std::vector<mlir::Operation*> StandardFunctionLowerer::lower(const StandardFunction& function)
  {
    std::vector<mlir::Operation*> result;

    mlir::OpBuilder::InsertionGuard guard(builder());
    Lowerer::SymbolScope varScope(symbolTable());

    auto location = loc(function.getLocation());

    // Input variables
    llvm::SmallVector<llvm::StringRef, 3> argNames;
    llvm::SmallVector<mlir::Type, 3> argTypes;

    for (const auto& member : function.getArgs()) {
      argNames.emplace_back(member->getName());

      mlir::Type type = lower(member->getType(), ArrayAllocationScope::unknown);

      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        type = arrayType.toUnknownAllocationScope();
      }

      argTypes.emplace_back(type);
    }

    // Output variables
    llvm::SmallVector<llvm::StringRef, 3> returnNames;
    llvm::SmallVector<mlir::Type, 3> returnTypes;
    auto outputMembers = function.getResults();

    for (const auto& member : outputMembers) {
      mlir::Type type = lower(member->getType(), ArrayAllocationScope::heap);

      if (auto arrayType = type.dyn_cast<ArrayType>()) {
        type = arrayType.toAllocationScope(ArrayAllocationScope::heap);
      }

      returnNames.emplace_back(member->getName());
      returnTypes.emplace_back(type);
    }

    // Create the function
    auto functionType = builder().getFunctionType(argTypes, returnTypes);
    auto functionOp = builder().create<FunctionOp>(location, function.getName(), functionType);

    // Process the annotations
    if (function.hasAnnotation()) {
      const auto* annotation = function.getAnnotation();

      // Inline attribute
      functionOp->setAttr("inline", builder().getBoolAttr(function.getAnnotation()->getInlineProperty()));

      // Inverse functions attribute
      auto inverseFunctionAnnotation = annotation->getInverseFunctionAnnotation();
      InverseFunctionsMap map;

      // Create a map of the function members indexes for faster retrieval
      llvm::StringMap<unsigned int> indexes;

      for (const auto& name : llvm::enumerate(argNames)) {
        indexes[name.value()] = name.index();
      }

      for (const auto& name : llvm::enumerate(returnNames)) {
        indexes[name.value()] = argNames.size() + name.index();
      }

      mlir::StorageUniquer::StorageAllocator allocator;

      // Iterate over the input arguments and for each invertible one
      // add the function to the inverse map.
      for (const auto& arg : argNames) {
        if (!inverseFunctionAnnotation.isInvertible(arg)) {
          continue;
        }

        auto inverseArgs = inverseFunctionAnnotation.getInverseArgs(arg);
        llvm::SmallVector<unsigned int, 3> permutation;

        for (const auto& inverseArg : inverseArgs) {
          assert(indexes.find(inverseArg) != indexes.end());
          permutation.push_back(indexes[inverseArg]);
        }

        map[indexes[arg]] = std::make_pair(
            inverseFunctionAnnotation.getInverseFunction(arg),
            allocator.copyInto(llvm::ArrayRef<unsigned int>(permutation)));
      }

      if (!map.empty()) {
        auto inverseFunctionAttribute = InverseFunctionsAttr::get(builder().getContext(), map);
        functionOp->setAttr("inverse", inverseFunctionAttribute);
      }

      if (annotation->hasDerivativeAnnotation()) {
        auto derivativeAnnotation = annotation->getDerivativeAnnotation();
        auto derivativeAttribute = DerivativeAttr::get(builder().getContext(), derivativeAnnotation.getName(), derivativeAnnotation.getOrder());
        functionOp->setAttr("derivative", derivativeAttribute);
      }
    }

    // Start the body of the function
    mlir::Block* entryBlock = functionOp.addEntryBlock();
    builder().setInsertionPointToStart(entryBlock);

    // Declare all the function arguments in the symbol table
    for (const auto& [name, value] : llvm::zip(argNames, entryBlock->getArguments())) {
      symbolTable().insert(name, Reference::ssa(&builder(), value));
    }

    // Initialize members
    for (const auto& member : function.getMembers()) {
      lower(*member);
    }

    // Emit the body of the function
    const auto& algorithm = function.getAlgorithms()[0];

    for (const auto& statement : *algorithm) {
      lower(*statement);
    }

    builder().create<FunctionTerminatorOp>(location);

    result.push_back(functionOp);
    return result;
  }

  void StandardFunctionLowerer::lower(const Member& member)
  {
    auto location = loc(member.getLocation());

    // Input values are supposed to be read-only by the Modelica standard,
    // thus they don't need to be copied for local modifications.

    if (member.isInput()) {
      return;
    }

    const auto& frontendType = member.getType();
    mlir::Type type = lower(frontendType, member.isOutput() ? ArrayAllocationScope::heap : ArrayAllocationScope::stack);

    llvm::SmallVector<mlir::Value, 3> dynamicDimensions;
    llvm::SmallVector<long, 3> shape;

    if (auto arrayType = type.dyn_cast<ArrayType>()) {
      for (auto dimension : arrayType.getShape())
        shape.push_back(dimension);

      auto expressionsCount = llvm::count_if(
          member.getType().getDimensions(),
          [](const auto& dimension) {
            return dimension.hasExpression();
          });

      // If all the dynamic dimensions have an expression to determine their
      // values, then the member can be instantiated from the beginning.

      bool initialized = expressionsCount == arrayType.getDynamicDimensionsCount();

      if (initialized) {
        for (const auto& dimension : member.getType().getDimensions()) {
          if (dimension.hasExpression()) {
            mlir::Value size = *lower(*dimension.getExpression())[0];
            size = builder().create<CastOp>(location, builder().getIndexType(), size);
            dynamicDimensions.push_back(size);
          }
        }
      }
    }

    auto memberType = MemberType::wrap(type, MemberAllocationScope::stack);

    mlir::Value var = builder().create<MemberCreateOp>(location, member.getName(), memberType, dynamicDimensions, false);
    symbolTable().insert(member.getName(), Reference::member(&builder(), var));

    if (member.hasInitializer()) {
      // If the member has an initializer expression, lower and assign it as
      // if it was a regular assignment statement.

      Reference memory = symbolTable().lookup(member.getName());
      mlir::Value value = *lower(*member.getInitializer())[0];
      memory.set(value);
    }
  }
}
