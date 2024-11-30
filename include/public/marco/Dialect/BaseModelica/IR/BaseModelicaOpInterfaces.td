#ifndef MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAOPINTERFACES_TD
#define MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAOPINTERFACES_TD

include "marco/Dialect/BaseModelica/IR/BaseModelica.td"
include "mlir/IR/OpBase.td"

//===---------------------------------------------------------------------===//
// Base Modelica op interface definitions
//===---------------------------------------------------------------------===//

class BaseModelica_OpInterface<string name>
    : OpInterface<name>
{
    let cppNamespace = "::mlir::bmodelica";
}

def ClassInterface : BaseModelica_OpInterface<"ClassInterface">
{
    let description = [{
        Interface for a Base Modelica class.
    }];
}

def VectorizableOpInterface : BaseModelica_OpInterface<"VectorizableOpInterface">
{
    let description = [{
        Interface for a vectorizable operation.
    }];

    let methods = [
        InterfaceMethod<
            "Get the arguments of the operation.",
            "::mlir::ValueRange", "getArgs",
            (ins), "",
            [{
                return $_op->getOperands();
            }]
        >,
        InterfaceMethod<
            "Get the expected rank of an argument in case of a scalar usage of the operation.",
            "unsigned int", "getArgExpectedRank",
            (ins "unsigned int":$argIndex,
                 "::mlir::SymbolTableCollection&":$symbolTableCollection),
            "",
            [{
                return 0;
            }]
        >,
        InterfaceMethod<
            "Convert the vectorized operation into a scalar one.",
            "::mlir::LogicalResult", "scalarize",
            (ins "::mlir::OpBuilder&":$builder,
                 "::mlir::ValueRange":$args,
                 "::mlir::TypeRange":$resultTypes,
                 "::llvm::SmallVectorImpl<::mlir::Value>&":$results),
            "",
            [{
              llvm_unreachable("'scalarize' not implemented");
            }]
        >,
    ];
}

def EquationInstanceInterface
    : BaseModelica_OpInterface<"EquationInstanceInterface">
{
    let methods = [
        InterfaceMethod<
            "Get the iteration indices, if any",
            "std::optional<MultidimensionalRangeAttr>", "getIndices">,
        InterfaceMethod<
            [{"Get the indices visited by the induction variables.
              An empty iteration space is returned if there is no iteration
              variable."}],
            "::marco::modeling::IndexSet", "getIterationSpace">
    ];
}

def EquationExpressionOpInterface
    : BaseModelica_OpInterface<"EquationExpressionOpInterface">
{
    let description = [{
        This interface is used to represent an operation that can be used to
        represent an expression within an equation.
    }];

    let methods = [
        InterfaceMethod<
            "Print inline representation of the expression.",
            "void", "printExpression",
            (ins "::llvm::raw_ostream&":$os,
                 "const ::llvm::DenseMap<::mlir::Value, int64_t>&":$inductions)>,
        InterfaceMethod<
            "Check if two expressions are equivalent. Induction accesses are treated as equal, and are returned in [pairedInductions]",
            "bool", "isEquivalent",
            (ins "::mlir::Operation*":$other,
                 "::llvm::SmallVectorImpl<std::pair<::mlir::BlockArgument, ::mlir::BlockArgument>>&":$pairedInductions,
                 "::mlir::SymbolTableCollection&":$symbolTableCollection), "", [{
                // Safely assume that the two expressions are different.
                return false;
            }]>,
        InterfaceMethod<
            "Check if two expressions are equivalent. Induction to the same variable are treated as equal.",
            "bool", "isEquivalentDefault",
            (ins "::mlir::Operation*":$other,
                 "::mlir::SymbolTableCollection&":$symbolTableCollection), "", [{
                 ::llvm::SmallVector<std::pair<mlir::BlockArgument, mlir::BlockArgument>> pairedInductions;
                 auto opInterface = mlir::cast<EquationExpressionOpInterface>($_op.getOperation());
                 bool equivalent = opInterface.isEquivalent(other, pairedInductions, symbolTableCollection);
                 return equivalent && llvm::all_of(pairedInductions, [&](auto accessPair) {
                     return accessPair.first == accessPair.second;
                 });
            }]>,
        InterfaceMethod<
            "Get the number of elements.",
            "uint64_t", "getNumOfExpressionElements",
            (ins), "", [{
                return $_op.getOperation()->getNumOperands();
            }]>,
        InterfaceMethod<
            "Get the element at a given position.",
            "::mlir::Value", "getExpressionElement",
            (ins "uint64_t":$position),
            "", [{
                return $_op.getOperation()->getOperand(position);
            }]>,
        InterfaceMethod<
            "Get the additional induction variables introduced by the operation.",
            "::llvm::SmallVector<mlir::Value>", "getAdditionalInductions",
            (ins), "", [{
                return {};
            }]>,
        InterfaceMethod<
            "Map additional induction variables.",
            "::mlir::LogicalResult", "mapAdditionalInductions",
            (ins "AdditionalInductions&":$additionalInductions),
            "", [{
                return mlir::success();
            }]>,
        InterfaceMethod<
            "Get the accesses of this expression",
            "::mlir::LogicalResult", "getEquationAccesses",
            (ins "::llvm::SmallVectorImpl<VariableAccess>&":$accesses,
                 "::mlir::SymbolTableCollection&":$symbolTable,
                 "::llvm::DenseMap<::mlir::Value, unsigned int>&":$explicitInductionsPositionMap,
                 "AdditionalInductions&":$additionalInductions,
                 "::llvm::SmallVectorImpl<std::unique_ptr<DimensionAccess>>&":$dimensionAccesses,
                 "EquationPath":$path),
            "", [{
                auto opInterface = mlir::cast<EquationExpressionOpInterface>($_op.getOperation());
                uint64_t numOfChildren = opInterface.getNumOfExpressionElements();

                if (mlir::failed(opInterface.mapAdditionalInductions(additionalInductions))) {
                    return mlir::failure();
                }

                llvm::SmallVector<std::unique_ptr<DimensionAccess>> childDimensionAccesses;

                for (uint64_t i = 0; i < numOfChildren; ++i) {
                    childDimensionAccesses.clear();
                    mlir::Value child = opInterface.getExpressionElement(i);
                    auto childOp = child.getDefiningOp();

                    if (!childOp) {
                        continue;
                    }

                    auto childExpressionInt =
                        mlir::dyn_cast<EquationExpressionOpInterface>(childOp);

                    if (!childExpressionInt) {
                        return mlir::failure();
                    }

                    if (mlir::failed(childExpressionInt.getEquationAccesses(
                            accesses, symbolTable,
                            explicitInductionsPositionMap,
                            additionalInductions,
                            childDimensionAccesses,
                            path + i))) {
                       return mlir::failure();
                    }
                }

                return mlir::success();
             }]>
    ];
}

def InvertibleOpInterface : BaseModelica_OpInterface<"InvertibleOpInterface">
{
    let description = [{
        This interface is used to abstract an operation that can be propagated
        down into the operations tree (i.e. towards the leaf values).
    }];

    let methods = [
        InterfaceMethod<
            "Invert the operation with respect to one of its arguments.",
            "::mlir::Value", "inverse",
            (ins "::mlir::OpBuilder&":$builder,
                 "unsigned int":$argumentIndex,
                 "::mlir::ValueRange":$currentResult)
        >
    ];
}

def DistributableOpInterface : BaseModelica_OpInterface<"DistributableOpInterface">
{
    let description = [{
        Interface for an distributable operation.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute the operation among its arguments, if possible. The returned value is the one that will replace the previous operation result",
            "::mlir::LogicalResult", "distribute",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder)
        >
    ];
}

def NegateOpDistributionInterface : BaseModelica_OpInterface<"NegateOpDistributionInterface">
{
    let description = [{
        Interface to describe an operation that can accept the NegateOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a negate operation among the arguments. The returned value is the result after propagation",
            "::mlir::LogicalResult", "distributeNegateOp",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder)
        >
    ];
}

def MulOpDistributionInterface : BaseModelica_OpInterface<"MulOpDistributionInterface">
{
    let description = [{
        Interface to describe an operation that can accept the MulOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a multiplication operation among the arguments. The returned value is the result after propagation",
            "::mlir::LogicalResult", "distributeMulOp",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder,
                 "::mlir::Value":$factor)
        >
    ];
}

def DivOpDistributionInterface : BaseModelica_OpInterface<"DivOpDistributionInterface">
{
    let description = [{
        Interface to describe an operation that can accept the DivOp to be distributed among its operands.
    }];

    let methods = [
        InterfaceMethod<
            "Distribute a division operation among the arguments. The returned value is the result after propagation",
            "::mlir::LogicalResult", "distributeDivOp",
            (ins "llvm::SmallVectorImpl<::mlir::Value>&":$results,
                 "::mlir::OpBuilder&":$builder,
                 "::mlir::Value":$divisor)
        >
    ];
}

def DerivableOpInterface : BaseModelica_OpInterface<"DerivableOpInterface">
{
    let description = [{
        Interface to describe an operation that can be traversed while
        performing the automatic differentiation.
    }];

    let methods = [
        InterfaceMethod<
            "Compute the partial derivative of the operation.",
            "::mlir::LogicalResult", "createPartialDerivative",
            (ins "::mlir::OpBuilder&":$builder,
                 "ad::forward::State&":$state)
        >,
        InterfaceMethod<
            "Compute the derivative of the operation with respect to time.",
            "::mlir::LogicalResult", "createTimeDerivative",
            (ins "::mlir::OpBuilder&":$builder,
                 "ad::forward::State&":$state,
                 "bool":$deriveDependencies)
        >
    ];
}

def AffineLikeOpInterface
    : BaseModelica_OpInterface<"AffineLikeOpInterface">
{
    let description = [{
        This interface is used to represent an operation that can be represent
        in the affine domain.
    }];

    let methods = [
        InterfaceMethod<
            "Get the affine expression.",
            "std::optional<mlir::AffineExpr>", "getAffineExpression",
            (ins "const ::llvm::DenseMap<::mlir::Value, int64_t>&":$inductionsPosMap)>
    ];
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAOPINTERFACES_TD
