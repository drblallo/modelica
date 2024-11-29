#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/ScopeExit.h"
#include <cmath>

using namespace ::mlir::bmodelica;

//===---------------------------------------------------------------------===//
// BaseModelica Dialect
//===---------------------------------------------------------------------===//

namespace mlir::bmodelica {
void BaseModelicaDialect::registerOperations() {
  addOperations<
#define GET_OP_LIST
#include "marco/Dialect/BaseModelica/IR/BaseModelicaOps.cpp.inc"
      >();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// BaseModelica Operations
//===---------------------------------------------------------------------===//

static bool parseWrittenVars(mlir::OpAsmParser &parser, VariablesList &prop) {
  if (parser.parseKeyword("writtenVariables") || parser.parseEqual()) {
    return true;
  }

  if (mlir::failed(parse(parser, prop))) {
    return true;
  }

  return false;
}

static void printWrittenVars(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                             const VariablesList &prop) {
  printer << "writtenVariables = ";
  print(printer, prop);
}

static bool parseReadVars(mlir::OpAsmParser &parser, VariablesList &prop) {
  if (parser.parseKeyword("readVariables") || parser.parseEqual()) {
    return true;
  }

  if (mlir::failed(parse(parser, prop))) {
    return true;
  }

  return false;
}

static void printReadVars(mlir::OpAsmPrinter &printer, mlir::Operation *op,
                          const VariablesList &prop) {
  printer << "readVariables = ";
  print(printer, prop);
}

static bool parseModelDerivativesMap(mlir::OpAsmParser &parser,
                                     DerivativesMap &prop) {
  if (mlir::succeeded(parser.parseOptionalKeyword("der"))) {
    if (parser.parseEqual()) {
      return true;
    }

    if (mlir::failed(parse(parser, prop))) {
      return true;
    }
  }

  return false;
}

static void printModelDerivativesMap(mlir::OpAsmPrinter &printer,
                                     mlir::Operation *op,
                                     const DerivativesMap &prop) {
  if (!prop.empty()) {
    printer << "der = ";
    print(printer, prop);
  }
}

static bool parseAbstractEquationWrittenVars(mlir::OpAsmParser &parser,
                                             VariablesList &prop) {
  return parseWrittenVars(parser, prop);
}

static void printAbstractEquationWrittenVars(mlir::OpAsmPrinter &printer,
                                             mlir::Operation *op,
                                             const VariablesList &prop) {
  return printWrittenVars(printer, op, prop);
}

static bool parseAbstractEquationReadVars(mlir::OpAsmParser &parser,
                                          VariablesList &prop) {
  return parseReadVars(parser, prop);
}

static void printAbstractEquationReadVars(mlir::OpAsmPrinter &printer,
                                          mlir::Operation *op,
                                          const VariablesList &prop) {
  return printReadVars(printer, op, prop);
}

static bool parseScheduleBlockWrittenVars(mlir::OpAsmParser &parser,
                                          VariablesList &prop) {
  return parseWrittenVars(parser, prop);
}

static void printScheduleBlockWrittenVars(mlir::OpAsmPrinter &printer,
                                          mlir::Operation *op,
                                          const VariablesList &prop) {
  return printWrittenVars(printer, op, prop);
}

static bool parseScheduleBlockReadVars(mlir::OpAsmParser &parser,
                                       VariablesList &prop) {
  return parseReadVars(parser, prop);
}

static void printScheduleBlockReadVars(mlir::OpAsmPrinter &printer,
                                       mlir::Operation *op,
                                       const VariablesList &prop) {
  return printReadVars(printer, op, prop);
}

#define GET_OP_CLASSES
#include "marco/Dialect/BaseModelica/IR/BaseModelicaOps.cpp.inc"

namespace {
template <typename T>
std::optional<T> getScalarAttributeValue(mlir::Attribute attribute) {
  if (isScalarIntegerLike(attribute)) {
    return static_cast<T>(getScalarIntegerLikeValue(attribute));
  } else if (isScalarFloatLike(attribute)) {
    return static_cast<T>(getScalarFloatLikeValue(attribute));
  } else {
    return std::nullopt;
  }
}

template <typename T>
bool getScalarAttributesValues(llvm::ArrayRef<mlir::Attribute> attributes,
                               llvm::SmallVectorImpl<T> &result) {
  for (mlir::Attribute attribute : attributes) {
    if (auto value = getScalarAttributeValue<T>(attribute)) {
      result.push_back(*value);
    } else {
      return false;
    }
  }

  return true;
}
} // namespace

static mlir::LogicalResult
cleanEquationTemplates(mlir::RewriterBase &rewriter,
                       llvm::ArrayRef<EquationTemplateOp> templateOps) {
  mlir::RewritePatternSet patterns(rewriter.getContext());

  for (mlir::RegisteredOperationName registeredOp :
       rewriter.getContext()->getRegisteredOperations()) {
    registeredOp.getCanonicalizationPatterns(patterns, rewriter.getContext());
  }

  mlir::FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  mlir::GreedyRewriteConfig config;

  mlir::OpBuilder::Listener *listener = rewriter.getListener();
  mlir::RewriterBase::ForwardingListener forwardingListener(listener);

  if (listener != nullptr) {
    config.listener = &forwardingListener;
  }

  for (EquationTemplateOp templateOp : templateOps) {
    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            templateOp, frozenPatterns, config))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

//===---------------------------------------------------------------------===//
// Iteration space operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// RangeOp

namespace mlir::bmodelica {
mlir::LogicalResult RangeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lowerBoundType = adaptor.getLowerBound().getType();
  mlir::Type upperBoundType = adaptor.getUpperBound().getType();
  mlir::Type stepType = adaptor.getStep().getType();

  if (isScalar(lowerBoundType) && isScalar(upperBoundType) &&
      isScalar(stepType)) {
    mlir::Type resultType =
        getMostGenericScalarType(lowerBoundType, upperBoundType);

    resultType = getMostGenericScalarType(resultType, stepType);
    returnTypes.push_back(RangeType::get(context, resultType));
    return mlir::success();
  }

  return mlir::failure();
}

bool RangeOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult RangeOp::fold(FoldAdaptor adaptor) {
  auto lowerBound = adaptor.getLowerBound();
  auto upperBound = adaptor.getUpperBound();
  auto step = adaptor.getStep();

  if (!lowerBound || !upperBound || !step) {
    return {};
  }

  if (isScalarIntegerLike(lowerBound) && isScalarIntegerLike(upperBound) &&
      isScalarIntegerLike(step)) {
    int64_t lowerBoundValue = getScalarIntegerLikeValue(lowerBound);
    int64_t upperBoundValue = getScalarIntegerLikeValue(upperBound);
    int64_t stepValue = getScalarIntegerLikeValue(step);

    return IntegerRangeAttr::get(getContext(), getResult().getType(),
                                 lowerBoundValue, upperBoundValue, stepValue);
  }

  if ((isScalarIntegerLike(lowerBound) || isScalarFloatLike(lowerBound)) &&
      (isScalarIntegerLike(upperBound) || isScalarFloatLike(upperBound)) &&
      (isScalarIntegerLike(step) || isScalarFloatLike(step))) {
    double lowerBoundValue;
    double upperBoundValue;
    double stepValue;

    if (isScalarIntegerLike(lowerBound)) {
      lowerBoundValue =
          static_cast<double>(getScalarIntegerLikeValue(lowerBound));
    } else {
      lowerBoundValue = getScalarFloatLikeValue(lowerBound);
    }

    if (isScalarIntegerLike(upperBound)) {
      upperBoundValue =
          static_cast<double>(getScalarIntegerLikeValue(upperBound));
    } else {
      upperBoundValue = getScalarFloatLikeValue(upperBound);
    }

    if (isScalarIntegerLike(step)) {
      stepValue = static_cast<double>(getScalarIntegerLikeValue(step));
    } else {
      stepValue = getScalarFloatLikeValue(step);
    }

    return RealRangeAttr::get(
        getContext(), getResult().getType(), llvm::APFloat(lowerBoundValue),
        llvm::APFloat(upperBoundValue), llvm::APFloat(stepValue));
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RangeBeginOp

namespace mlir::bmodelica {
mlir::OpFoldResult RangeBeginOp::fold(FoldAdaptor adaptor) {
  auto range = adaptor.getRange();

  if (!range) {
    return {};
  }

  if (auto intRange = range.dyn_cast<IntegerRangeAttr>()) {
    mlir::Type inductionType =
        intRange.getType().cast<RangeType>().getInductionType();

    if (inductionType.isa<mlir::IndexType>()) {
      return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()),
                                    intRange.getLowerBound());
    } else {
      return IntegerAttr::get(getContext(), intRange.getLowerBound());
    }
  }

  if (auto realRange = range.dyn_cast<RealRangeAttr>()) {
    return RealAttr::get(getContext(),
                         realRange.getLowerBound().convertToDouble());
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RangeEndOp

namespace mlir::bmodelica {
mlir::OpFoldResult RangeEndOp::fold(FoldAdaptor adaptor) {
  auto range = adaptor.getRange();

  if (!range) {
    return {};
  }

  if (auto intRange = range.dyn_cast<IntegerRangeAttr>()) {
    mlir::Type inductionType =
        intRange.getType().cast<RangeType>().getInductionType();

    if (inductionType.isa<mlir::IndexType>()) {
      return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()),
                                    intRange.getUpperBound());
    } else {
      return IntegerAttr::get(getContext(), intRange.getUpperBound());
    }
  }

  if (auto realRange = range.dyn_cast<RealRangeAttr>()) {
    return RealAttr::get(getContext(),
                         realRange.getUpperBound().convertToDouble());
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RangeStepOp

namespace mlir::bmodelica {
mlir::OpFoldResult RangeStepOp::fold(FoldAdaptor adaptor) {
  auto range = adaptor.getRange();

  if (!range) {
    return {};
  }

  if (auto intRange = range.dyn_cast<IntegerRangeAttr>()) {
    mlir::Type inductionType =
        intRange.getType().cast<RangeType>().getInductionType();

    if (inductionType.isa<mlir::IndexType>()) {
      return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()),
                                    intRange.getStep());
    } else {
      return IntegerAttr::get(getContext(), intRange.getStep());
    }
  }

  if (auto realRange = range.dyn_cast<RealRangeAttr>()) {
    return RealAttr::get(getContext(), realRange.getStep().convertToDouble());
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RangeSizeOp

namespace mlir::bmodelica {
mlir::OpFoldResult RangeSizeOp::fold(FoldAdaptor adaptor) {
  auto range = adaptor.getRange();

  if (!range) {
    return {};
  }

  if (auto intRange = range.dyn_cast<IntegerRangeAttr>()) {
    int64_t beginValue = intRange.getLowerBound();
    int64_t endValue = intRange.getUpperBound();
    int64_t step = intRange.getStep();
    int64_t result = 1 + (endValue - beginValue) / step;

    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), result);
  }

  if (auto realRange = range.dyn_cast<RealRangeAttr>()) {
    double beginValue = realRange.getLowerBound().convertToDouble();
    double endValue = realRange.getUpperBound().convertToDouble();
    double step = realRange.getStep().convertToDouble();
    double result = 1 + (endValue - beginValue) / step;

    return mlir::IntegerAttr::get(mlir::IndexType::get(getContext()),
                                  static_cast<int64_t>(result));
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Tensor operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// TensorFromElementsOp

namespace mlir::bmodelica {
mlir::LogicalResult TensorFromElementsOp::verify() {
  if (!getResult().getType().hasStaticShape()) {
    return emitOpError("the shape must be fixed");
  }

  int64_t tensorFlatSize = getResult().getType().getNumElements();
  size_t numOfValues = getValues().size();

  if (tensorFlatSize != static_cast<int64_t>(numOfValues)) {
    return emitOpError("incorrect number of values (expected " +
                       std::to_string(tensorFlatSize) + ", got " +
                       std::to_string(numOfValues) + ")");
  }

  return mlir::success();
}

mlir::OpFoldResult TensorFromElementsOp::fold(FoldAdaptor adaptor) {
  if (llvm::all_of(adaptor.getOperands(),
                   [](mlir::Attribute attr) { return attr != nullptr; })) {
    mlir::TensorType tensorType = getResult().getType();

    if (!tensorType.hasStaticShape()) {
      return {};
    }

    mlir::Type elementType = tensorType.getElementType();

    if (elementType.isa<BooleanType>()) {
      llvm::SmallVector<bool> casted;

      if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
        return {};
      }

      return DenseBooleanElementsAttr::get(tensorType, casted);
    }

    if (elementType.isa<IntegerType>()) {
      llvm::SmallVector<int64_t> casted;

      if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
        return {};
      }

      return DenseIntegerElementsAttr::get(tensorType, casted);
    }

    if (elementType.isa<RealType>()) {
      llvm::SmallVector<double> casted;

      if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
        return {};
      }

      return DenseRealElementsAttr::get(tensorType, casted);
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// TensorViewOp

namespace {
struct InferTensorViewResultTypePattern
    : public mlir::OpRewritePattern<TensorViewOp> {
  using mlir::OpRewritePattern<TensorViewOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TensorViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::TensorType inferredResultType = TensorViewOp::inferResultType(
        op.getSource().getType(), op.getSubscriptions());

    if (inferredResultType != op.getResult().getType()) {
      auto newOp =
          rewriter.create<TensorViewOp>(op.getLoc(), inferredResultType,
                                        op.getSource(), op.getSubscriptions());

      newOp->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct MergeTensorViewsPattern : public mlir::OpRewritePattern<TensorViewOp> {
  using mlir::OpRewritePattern<TensorViewOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TensorViewOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto viewOp = op.getSource().getDefiningOp<TensorViewOp>();

    if (!viewOp) {
      return mlir::failure();
    }

    llvm::SmallVector<TensorViewOp> viewOps;

    while (viewOp) {
      viewOps.push_back(viewOp);
      viewOp = viewOp.getSource().getDefiningOp<TensorViewOp>();
    }

    assert(!viewOps.empty());
    mlir::Value source = viewOps.back().getSource();
    llvm::SmallVector<mlir::Value, 3> subscriptions;

    while (!viewOps.empty()) {
      TensorViewOp current = viewOps.pop_back_val();
      subscriptions.append(current.getSubscriptions().begin(),
                           current.getSubscriptions().end());
    }

    subscriptions.append(op.getSubscriptions().begin(),
                         op.getSubscriptions().end());

    rewriter.replaceOpWithNewOp<TensorViewOp>(op, source, subscriptions);
    return mlir::success();
  }
};
} // namespace

namespace mlir::bmodelica {
void TensorViewOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                         mlir::Value source, mlir::ValueRange subscriptions) {
  mlir::TensorType resultType =
      inferResultType(source.getType().cast<mlir::TensorType>(), subscriptions);

  build(builder, state, resultType, source, subscriptions);
}

mlir::LogicalResult TensorViewOp::verify() {
  mlir::TensorType sourceType = getSource().getType();
  mlir::TensorType resultType = getResult().getType();

  mlir::TensorType expectedResultType =
      inferResultType(sourceType, getSubscriptions());

  if (resultType.getRank() != expectedResultType.getRank()) {
    return emitOpError() << "incompatible result rank";
  }

  for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
    int64_t actualDimSize = resultType.getDimSize(i);
    int64_t expectedDimSize = expectedResultType.getDimSize(i);

    if (actualDimSize != mlir::ShapedType::kDynamic &&
        actualDimSize != expectedDimSize) {
      return emitOpError() << "incompatible size for dimension " << i
                           << " (expected " << expectedDimSize << ", got "
                           << actualDimSize << ")";
    }
  }

  return mlir::success();
}

void TensorViewOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<InferTensorViewResultTypePattern, MergeTensorViewsPattern>(
      context);
}

mlir::TensorType TensorViewOp::inferResultType(mlir::TensorType source,
                                               mlir::ValueRange indices) {
  llvm::SmallVector<int64_t> shape;
  size_t numOfSubscriptions = indices.size();

  for (size_t i = 0; i < numOfSubscriptions; ++i) {
    mlir::Value index = indices[i];

    if (index.getType().isa<RangeType>()) {
      int64_t dimension = mlir::ShapedType::kDynamic;

      if (auto constantOp = index.getDefiningOp<ConstantOp>()) {
        auto indexAttr = constantOp.getValue();

        if (auto rangeAttr = mlir::dyn_cast<RangeAttrInterface>(indexAttr)) {
          dimension = rangeAttr.getNumOfElements();
        }
      }

      shape.push_back(dimension);
    }
  }

  for (int64_t dimension : source.getShape().drop_front(numOfSubscriptions)) {
    shape.push_back(dimension);
  }

  return source.clone(shape);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// TensorExtractOp

namespace {
struct MergeTensorViewIntoTensorExtractPattern
    : public mlir::OpRewritePattern<TensorExtractOp> {
  using mlir::OpRewritePattern<TensorExtractOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(TensorExtractOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto viewOp = op.getTensor().getDefiningOp<TensorViewOp>();

    if (!viewOp) {
      return mlir::failure();
    }

    llvm::SmallVector<TensorViewOp> viewOps;

    while (viewOp) {
      viewOps.push_back(viewOp);
      viewOp = viewOp.getSource().getDefiningOp<TensorViewOp>();
    }

    assert(!viewOps.empty());
    mlir::Value source = viewOps.back().getSource();
    llvm::SmallVector<mlir::Value, 3> subscriptions;

    while (!viewOps.empty()) {
      TensorViewOp current = viewOps.pop_back_val();
      subscriptions.append(current.getSubscriptions().begin(),
                           current.getSubscriptions().end());
    }

    subscriptions.append(op.getIndices().begin(), op.getIndices().end());
    rewriter.replaceOpWithNewOp<TensorExtractOp>(op, source, subscriptions);
    return mlir::success();
  }
};
} // namespace

namespace mlir::bmodelica {
mlir::ParseResult TensorExtractOp::parse(mlir::OpAsmParser &parser,
                                         mlir::OperationState &result) {
  auto loc = parser.getCurrentLocation();
  mlir::OpAsmParser::UnresolvedOperand array;
  mlir::Type tensorType;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
  llvm::SmallVector<mlir::Type, 3> indicesTypes;

  if (parser.parseOperand(array) ||
      parser.parseOperandList(indices, mlir::OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(tensorType) ||
      parser.resolveOperand(array, tensorType, result.operands)) {
    return mlir::failure();
  }

  indicesTypes.resize(indices.size(),
                      mlir::IndexType::get(result.getContext()));

  size_t i = 0;

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(indicesTypes[i++])) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperands(indices, indicesTypes, loc, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(tensorType.cast<mlir::ShapedType>().getElementType());
  return mlir::success();
}

void TensorExtractOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getTensor() << "[" << getIndices() << "]";
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getTensor().getType();

  if (!llvm::all_of(getIndices(), [](mlir::Value index) {
        return index.getType().isa<mlir::IndexType>();
      })) {
    for (mlir::Value index : getIndices()) {
      printer << ", " << index.getType();
    }
  }
}

mlir::LogicalResult TensorExtractOp::verify() {
  size_t indicesAmount = getIndices().size();
  int64_t rank = getTensor().getType().getRank();

  if (rank != static_cast<int64_t>(indicesAmount)) {
    return emitOpError() << "incorrect number of indices (expected " << rank
                         << ", got " << indicesAmount << ")";
  }

  for (size_t i = 0; i < indicesAmount; ++i) {
    if (auto constantOp = getIndices()[i].getDefiningOp<ConstantOp>()) {
      if (auto index =
              getScalarAttributeValue<int64_t>(constantOp.getValue())) {
        if (*index < 0) {
          return emitOpError() << "invalid index (" << *index << ")";
        }

        if (int64_t dimSize = getTensor().getType().getDimSize(i);
            *index >= dimSize) {
          return emitOpError() << "out of bounds access (index = " << *index
                               << ", dimension = " << dimSize << ")";
        }
      }
    }
  }

  return mlir::success();
}

void TensorExtractOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<MergeTensorViewIntoTensorExtractPattern>(context);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Array operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// AllocaOp

namespace mlir::bmodelica {
mlir::LogicalResult AllocaOp::verify() {
  int64_t dynamicDimensionsAmount = getArrayType().getNumDynamicDims();
  size_t valuesAmount = getDynamicSizes().size();

  if (dynamicDimensionsAmount != static_cast<int64_t>(valuesAmount)) {
    return emitOpError(
        "incorrect number of values for dynamic dimensions (expected " +
        std::to_string(dynamicDimensionsAmount) + ", got " +
        std::to_string(valuesAmount) + ")");
  }

  return mlir::success();
}

void AllocaOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  if (auto arrayType = getResult().getType().dyn_cast<ArrayType>()) {
    effects.emplace_back(
        mlir::MemoryEffects::Allocate::get(), getResult(),
        mlir::SideEffects::AutomaticAllocationScopeResource::get());
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// AllocOp

namespace mlir::bmodelica {
mlir::LogicalResult AllocOp::verify() {
  int64_t dynamicDimensionsAmount = getArrayType().getNumDynamicDims();
  size_t valuesAmount = getDynamicSizes().size();

  if (dynamicDimensionsAmount != static_cast<int64_t>(valuesAmount)) {
    return emitOpError(
        "incorrect number of values for dynamic dimensions (expected " +
        std::to_string(dynamicDimensionsAmount) + ", got " +
        std::to_string(valuesAmount) + ")");
  }

  return mlir::success();
}

void AllocOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  if (auto arrayType = getResult().getType().dyn_cast<ArrayType>()) {
    effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(),
                         mlir::SideEffects::DefaultResource::get());
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ArrayFromElementsOp

namespace mlir::bmodelica {
mlir::LogicalResult ArrayFromElementsOp::verify() {
  if (!getArrayType().hasStaticShape()) {
    return emitOpError("the shape must be fixed");
  }

  int64_t arrayFlatSize = getArrayType().getNumElements();
  size_t numOfValues = getValues().size();

  if (arrayFlatSize != static_cast<int64_t>(numOfValues)) {
    return emitOpError("incorrect number of values (expected " +
                       std::to_string(arrayFlatSize) + ", got " +
                       std::to_string(numOfValues) + ")");
  }

  return mlir::success();
}

mlir::OpFoldResult ArrayFromElementsOp::fold(FoldAdaptor adaptor) {
  if (llvm::all_of(adaptor.getOperands(),
                   [](mlir::Attribute attr) { return attr != nullptr; })) {
    ArrayType arrayType = getArrayType();

    if (!arrayType.hasStaticShape()) {
      return {};
    }

    mlir::Type elementType = arrayType.getElementType();

    if (elementType.isa<BooleanType>()) {
      llvm::SmallVector<bool> casted;

      if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
        return {};
      }

      return DenseBooleanElementsAttr::get(arrayType, casted);
    }

    if (elementType.isa<IntegerType>()) {
      llvm::SmallVector<int64_t> casted;

      if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
        return {};
      }

      return DenseIntegerElementsAttr::get(arrayType, casted);
    }

    if (elementType.isa<RealType>()) {
      llvm::SmallVector<double> casted;

      if (!getScalarAttributesValues(adaptor.getOperands(), casted)) {
        return {};
      }

      return DenseRealElementsAttr::get(arrayType, casted);
    }
  }

  return {};
}

void ArrayFromElementsOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(),
                       mlir::SideEffects::DefaultResource::get());

  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(),
                       mlir::SideEffects::DefaultResource::get());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ArrayBroadcastOp

namespace mlir::bmodelica {
void ArrayBroadcastOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(),
                       mlir::SideEffects::DefaultResource::get());

  effects.emplace_back(mlir::MemoryEffects::Write::get(), getResult(),
                       mlir::SideEffects::DefaultResource::get());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// FreeOp

namespace mlir::bmodelica {
void FreeOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                            mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Free::get(), getArray(),
                       mlir::SideEffects::DefaultResource::get());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// DimOp

namespace {
struct DimOpStaticDimensionPattern : public mlir::OpRewritePattern<DimOp> {
  using mlir::OpRewritePattern<DimOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DimOp op, mlir::PatternRewriter &rewriter) const override {
    auto constantOp = op.getDimension().getDefiningOp<ConstantOp>();

    if (!constantOp) {
      return mlir::failure();
    }

    ArrayType arrayType = op.getArray().getType();

    int64_t dimSize = arrayType.getDimSize(
        constantOp.getValue().cast<mlir::IntegerAttr>().getInt());

    if (dimSize == ArrayType::kDynamic) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<ConstantOp>(op, rewriter.getIndexAttr(dimSize));

    return mlir::success();
  }
};
} // namespace

namespace mlir::bmodelica {
void DimOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *context) {
  patterns.add<DimOpStaticDimensionPattern>(context);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// LoadOp

namespace {
struct MergeSubscriptionsIntoLoadPattern
    : public mlir::OpRewritePattern<LoadOp> {
  using mlir::OpRewritePattern<LoadOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LoadOp op, mlir::PatternRewriter &rewriter) const override {
    auto subscriptionOp = op.getArray().getDefiningOp<SubscriptionOp>();

    if (!subscriptionOp) {
      return mlir::failure();
    }

    llvm::SmallVector<SubscriptionOp> subscriptionOps;

    while (subscriptionOp) {
      subscriptionOps.push_back(subscriptionOp);

      subscriptionOp =
          subscriptionOp.getSource().getDefiningOp<SubscriptionOp>();
    }

    assert(!subscriptionOps.empty());
    mlir::Value source = subscriptionOps.back().getSource();
    llvm::SmallVector<mlir::Value, 3> indices;

    while (!subscriptionOps.empty()) {
      SubscriptionOp current = subscriptionOps.pop_back_val();
      indices.append(current.getIndices().begin(), current.getIndices().end());
    }

    indices.append(op.getIndices().begin(), op.getIndices().end());
    rewriter.replaceOpWithNewOp<LoadOp>(op, source, indices);
    return mlir::success();
  }
};
} // namespace

namespace mlir::bmodelica {
mlir::ParseResult LoadOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  auto loc = parser.getCurrentLocation();
  mlir::OpAsmParser::UnresolvedOperand array;
  mlir::Type arrayType;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
  llvm::SmallVector<mlir::Type, 3> indicesTypes;

  if (parser.parseOperand(array) ||
      parser.parseOperandList(indices, mlir::OpAsmParser::Delimiter::Square) ||
      parser.parseColonType(arrayType) ||
      parser.resolveOperand(array, arrayType, result.operands)) {
    return mlir::failure();
  }

  indicesTypes.resize(indices.size(),
                      mlir::IndexType::get(result.getContext()));

  size_t i = 0;

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(indicesTypes[i++])) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperands(indices, indicesTypes, loc, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(arrayType.cast<mlir::ShapedType>().getElementType());
  return mlir::success();
}

void LoadOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getArray() << "[" << getIndices() << "]";
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getArray().getType();

  if (!llvm::all_of(getIndices(), [](mlir::Value index) {
        return index.getType().isa<mlir::IndexType>();
      })) {
    for (mlir::Value index : getIndices()) {
      printer << ", " << index.getType();
    }
  }
}

mlir::LogicalResult LoadOp::verify() {
  size_t indicesAmount = getIndices().size();
  int64_t rank = getArrayType().getRank();

  if (rank != static_cast<int64_t>(indicesAmount)) {
    return emitOpError() << "incorrect number of indices (expected " << rank
                         << ", got " << indicesAmount << ")";
  }

  for (size_t i = 0; i < indicesAmount; ++i) {
    if (auto constantOp = getIndices()[i].getDefiningOp<ConstantOp>()) {
      if (auto index =
              getScalarAttributeValue<int64_t>(constantOp.getValue())) {
        if (*index < 0) {
          return emitOpError() << "invalid index (" << *index << ")";
        }

        if (int64_t dimSize = getArrayType().getDimSize(i); *index >= dimSize) {
          return emitOpError() << "out of bounds access (index = " << *index
                               << ", dimension = " << dimSize << ")";
        }
      }
    }
  }

  return mlir::success();
}

void LoadOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *context) {
  patterns.add<MergeSubscriptionsIntoLoadPattern>(context);
}

void LoadOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                            mlir::MemoryEffects::Effect>> &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), getArray(),
                       mlir::SideEffects::DefaultResource::get());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// StoreOp

namespace {
struct StoreOpMergeSubscriptionPattern
    : public mlir::OpRewritePattern<StoreOp> {
  using mlir::OpRewritePattern<StoreOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(StoreOp op, mlir::PatternRewriter &rewriter) const override {
    auto subscriptionOp = op.getArray().getDefiningOp<SubscriptionOp>();

    if (!subscriptionOp) {
      return mlir::failure();
    }

    if (llvm::any_of(
            subscriptionOp.getIndices().getTypes(),
            [](mlir::Type type) { return mlir::isa<RangeType>(type); })) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::Value> subscripts;

    auto subscriptionIndices = subscriptionOp.getIndices();
    subscripts.append(subscriptionIndices.begin(), subscriptionIndices.end());

    auto storeIndices = op.getIndices();
    subscripts.append(storeIndices.begin(), storeIndices.end());

    rewriter.replaceOpWithNewOp<StoreOp>(
        op, op.getValue(), subscriptionOp.getSource(), subscripts);

    return mlir::success();
  };
};
} // namespace

namespace mlir::bmodelica {
mlir::ParseResult StoreOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  auto loc = parser.getCurrentLocation();
  mlir::OpAsmParser::UnresolvedOperand array;
  mlir::Type arrayType;
  mlir::OpAsmParser::UnresolvedOperand value;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
  llvm::SmallVector<mlir::Type, 3> indicesTypes;

  if (parser.parseOperand(array) ||
      parser.parseOperandList(indices, mlir::OpAsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseOperand(value) ||
      parser.parseColonType(arrayType) ||
      parser.resolveOperand(value,
                            arrayType.cast<mlir::ShapedType>().getElementType(),
                            result.operands) ||
      parser.resolveOperand(array, arrayType, result.operands)) {
    return mlir::failure();
  }

  indicesTypes.resize(indices.size(),
                      mlir::IndexType::get(result.getContext()));

  size_t i = 0;

  while (mlir::succeeded(parser.parseOptionalComma())) {
    if (parser.parseType(indicesTypes[i++])) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperands(indices, indicesTypes, loc, result.operands)) {
    return mlir::failure();
  }

  return mlir::success();
}

void StoreOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getArray() << "[" << getIndices() << "]"
          << ", " << getValue();

  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : " << getArray().getType();

  if (!llvm::all_of(getIndices(), [](mlir::Value index) {
        return index.getType().isa<mlir::IndexType>();
      })) {
    for (mlir::Value index : getIndices()) {
      printer << ", " << index.getType();
    }
  }
}

mlir::LogicalResult StoreOp::verify() {
  size_t indicesAmount = getIndices().size();
  int64_t rank = getArrayType().getRank();

  if (rank != static_cast<int64_t>(indicesAmount)) {
    return emitOpError() << "incorrect number of indices (expected " << rank
                         << ", got " << indicesAmount << ")";
  }

  for (size_t i = 0; i < indicesAmount; ++i) {
    if (auto constantOp = getIndices()[i].getDefiningOp<ConstantOp>()) {
      if (auto index = getScalarAttributeValue<int64_t>(
              mlir::cast<mlir::Attribute>(constantOp.getValue()))) {
        if (*index < 0) {
          return emitOpError() << "invalid index (" << *index << ")";
        }

        if (int64_t dimSize = getArrayType().getDimSize(i); *index >= dimSize) {
          return emitOpError() << "out of bounds access (index = " << *index
                               << ", dimension = " << dimSize << ")";
        }
      }
    }
  }

  return mlir::success();
}

void StoreOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *context) {
  patterns.add<StoreOpMergeSubscriptionPattern>(context);
}

void StoreOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getArray(),
                       mlir::SideEffects::DefaultResource::get());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SubscriptionOp

namespace {
struct InferSubscriptionResultTypePattern
    : public mlir::OpRewritePattern<SubscriptionOp> {
  using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SubscriptionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    ArrayType inferredResultType = SubscriptionOp::inferResultType(
        op.getSource().getType(), op.getIndices());

    if (inferredResultType != op.getResultArrayType()) {
      auto newOp = rewriter.create<SubscriptionOp>(
          op.getLoc(), inferredResultType, op.getSource(), op.getIndices());

      newOp->setAttrs(op->getAttrs());
      rewriter.replaceOp(op, newOp);
      return mlir::success();
    }

    return mlir::failure();
  }
};

struct MergeSubscriptionsPattern
    : public mlir::OpRewritePattern<SubscriptionOp> {
  using mlir::OpRewritePattern<SubscriptionOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SubscriptionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto subscriptionOp = op.getSource().getDefiningOp<SubscriptionOp>();

    if (!subscriptionOp) {
      return mlir::failure();
    }

    llvm::SmallVector<SubscriptionOp> subscriptionOps;

    while (subscriptionOp) {
      subscriptionOps.push_back(subscriptionOp);

      subscriptionOp =
          subscriptionOp.getSource().getDefiningOp<SubscriptionOp>();
    }

    assert(!subscriptionOps.empty());
    mlir::Value source = subscriptionOps.back().getSource();
    llvm::SmallVector<mlir::Value, 3> indices;

    while (!subscriptionOps.empty()) {
      SubscriptionOp current = subscriptionOps.pop_back_val();
      indices.append(current.getIndices().begin(), current.getIndices().end());
    }

    indices.append(op.getIndices().begin(), op.getIndices().end());
    rewriter.replaceOpWithNewOp<SubscriptionOp>(op, source, indices);
    return mlir::success();
  }
};
} // namespace

namespace mlir::bmodelica {
void SubscriptionOp::build(mlir::OpBuilder &builder,
                           mlir::OperationState &state, mlir::Value source,
                           mlir::ValueRange indices) {
  build(builder, state,
        inferResultType(source.getType().cast<ArrayType>(), indices), source,
        indices);
}

mlir::LogicalResult SubscriptionOp::verify() {
  ArrayType sourceType = getSourceArrayType();
  ArrayType resultType = getResultArrayType();
  ArrayType expectedResultType = inferResultType(sourceType, getIndices());

  if (resultType.getRank() != expectedResultType.getRank()) {
    return emitOpError() << "incompatible result rank";
  }

  for (int64_t i = 0, e = resultType.getRank(); i < e; ++i) {
    int64_t actualDimSize = resultType.getDimSize(i);
    int64_t expectedDimSize = expectedResultType.getDimSize(i);

    if (actualDimSize != ArrayType::kDynamic &&
        actualDimSize != expectedDimSize) {
      return emitOpError() << "incompatible size for dimension " << i
                           << " (expected " << expectedDimSize << ", got "
                           << actualDimSize << ")";
    }
  }

  return mlir::success();
}

void SubscriptionOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<InferSubscriptionResultTypePattern, MergeSubscriptionsPattern>(
      context);
}

ArrayType SubscriptionOp::inferResultType(ArrayType source,
                                          mlir::ValueRange indices) {
  llvm::SmallVector<int64_t> shape;
  size_t numOfSubscriptions = indices.size();

  for (size_t i = 0; i < numOfSubscriptions; ++i) {
    mlir::Value index = indices[i];

    if (index.getType().isa<RangeType>()) {
      int64_t dimension = ArrayType::kDynamic;

      if (auto constantOp = index.getDefiningOp<ConstantOp>()) {
        auto indexAttr = constantOp.getValue();

        if (auto rangeAttr = mlir::dyn_cast<RangeAttrInterface>(indexAttr)) {
          dimension = rangeAttr.getNumOfElements();
        }
      }

      shape.push_back(dimension);
    }
  }

  for (int64_t dimension : source.getShape().drop_front(numOfSubscriptions)) {
    shape.push_back(dimension);
  }

  return source.withShape(shape);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ArrayFillOp

namespace mlir::bmodelica {
void ArrayFillOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getArray(),
                       mlir::SideEffects::DefaultResource::get());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ArrayCopyOp

namespace mlir::bmodelica {
void ArrayCopyOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), getSource(),
                       mlir::SideEffects::DefaultResource::get());

  effects.emplace_back(mlir::MemoryEffects::Write::get(), getDestination(),
                       mlir::SideEffects::DefaultResource::get());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Variable operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// VariableOp

namespace mlir::bmodelica {
void VariableOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name, VariableType variableType) {
  llvm::SmallVector<mlir::Attribute, 3> constraints(
      variableType.getNumDynamicDims(),
      builder.getStringAttr(kDimensionConstraintUnbounded));

  build(builder, state, name, variableType, builder.getArrayAttr(constraints));
}

mlir::ParseResult VariableOp::parse(mlir::OpAsmParser &parser,
                                    mlir::OperationState &result) {
  auto &builder = parser.getBuilder();

  // Variable name.
  mlir::StringAttr nameAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return mlir::failure();
  }

  // Attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  // Variable type.
  mlir::Type type;

  if (parser.parseColonType(type)) {
    return mlir::failure();
  }

  result.attributes.append(getTypeAttrName(result.name),
                           mlir::TypeAttr::get(type));

  // Dimensions constraints.
  llvm::SmallVector<llvm::StringRef> dimensionsConstraints;

  if (mlir::succeeded(parser.parseOptionalLSquare())) {
    do {
      if (mlir::succeeded(
              parser.parseOptionalKeyword(kDimensionConstraintUnbounded))) {
        dimensionsConstraints.push_back(kDimensionConstraintUnbounded);
      } else {
        if (parser.parseKeyword(kDimensionConstraintFixed)) {
          return mlir::failure();
        }

        dimensionsConstraints.push_back(kDimensionConstraintFixed);
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return mlir::failure();
    }
  }

  result.attributes.append(getDimensionsConstraintsAttrName(result.name),
                           builder.getStrArrayAttr(dimensionsConstraints));

  // Region for the dimensions constraints.
  mlir::Region *constraintsRegion = result.addRegion();

  mlir::OptionalParseResult constraintsRegionParseResult =
      parser.parseOptionalRegion(*constraintsRegion);

  if (constraintsRegionParseResult.has_value() &&
      failed(*constraintsRegionParseResult)) {
    return mlir::failure();
  }

  return mlir::success();
}

void VariableOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());

  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());
  elidedAttrs.push_back(getTypeAttrName());
  elidedAttrs.push_back(getDimensionsConstraintsAttrName());

  printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

  printer << " : " << getType();

  auto dimConstraints =
      getDimensionsConstraints().getAsRange<mlir::StringAttr>();

  if (llvm::any_of(dimConstraints, [](mlir::StringAttr constraint) {
        return constraint == kDimensionConstraintFixed;
      })) {
    printer << " [";

    for (const auto &constraint : llvm::enumerate(dimConstraints)) {
      if (constraint.index() != 0) {
        printer << ", ";
      }

      printer << constraint.value().getValue();
    }

    printer << "] ";
  }

  if (mlir::Region &region = getConstraintsRegion(); !region.empty()) {
    printer.printRegion(region);
  }
}

mlir::LogicalResult VariableOp::verify() {
  // Verify the semantics for fixed dimensions constraints.
  size_t numOfFixedDims = getNumOfFixedDimensions();
  mlir::Region &constraintsRegion = getConstraintsRegion();
  size_t numOfConstraints = 0;

  if (!constraintsRegion.empty()) {
    auto yieldOp =
        mlir::cast<YieldOp>(constraintsRegion.back().getTerminator());

    numOfConstraints = yieldOp.getValues().size();
  }

  if (numOfFixedDims != numOfConstraints) {
    return emitOpError(
        "not enough constraints for dynamic dimension constraints have been "
        "provided (expected " +
        std::to_string(numOfFixedDims) + ", got " +
        std::to_string(numOfConstraints) + ")");
  }

  if (!constraintsRegion.empty()) {
    auto yieldOp =
        mlir::cast<YieldOp>(constraintsRegion.back().getTerminator());

    // Check that the amount of values is the same of the fixed dimensions.
    if (yieldOp.getValues().size() != getNumOfFixedDimensions()) {
      return mlir::failure();
    }

    // Check that all the dimensions have 'index' type.
    if (llvm::any_of(yieldOp.getValues(), [](mlir::Value value) {
          return !value.getType().isa<mlir::IndexType>();
        })) {
      return emitOpError(
          "constraints for dynamic dimensions must have 'index' type");
    }
  }

  return mlir::success();
}

VariableType VariableOp::getVariableType() {
  return getType().cast<VariableType>();
}

bool VariableOp::isInput() { return getVariableType().isInput(); }

bool VariableOp::isOutput() { return getVariableType().isOutput(); }

bool VariableOp::isDiscrete() { return getVariableType().isDiscrete(); }

bool VariableOp::isParameter() { return getVariableType().isParameter(); }

bool VariableOp::isConstant() { return getVariableType().isConstant(); }

bool VariableOp::isReadOnly() { return getVariableType().isReadOnly(); }

size_t VariableOp::getNumOfUnboundedDimensions() {
  return llvm::count_if(
      getDimensionsConstraints().getAsRange<mlir::StringAttr>(),
      [](mlir::StringAttr dimensionConstraint) {
        return dimensionConstraint.getValue() == kDimensionConstraintUnbounded;
      });
}

size_t VariableOp::getNumOfFixedDimensions() {
  return llvm::count_if(
      getDimensionsConstraints().getAsRange<mlir::StringAttr>(),
      [](mlir::StringAttr dimensionConstraint) {
        return dimensionConstraint.getValue() == kDimensionConstraintFixed;
      });
}

IndexSet VariableOp::getIndices() {
  VariableType variableType = getVariableType();

  if (variableType.isScalar()) {
    return {};
  }

  llvm::SmallVector<Range> ranges;

  for (int64_t dimension : variableType.getShape()) {
    ranges.push_back(Range(0, dimension));
  }

  return IndexSet(MultidimensionalRange(ranges));
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// VariableGetOp

namespace mlir::bmodelica {
void VariableGetOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          VariableOp variableOp) {
  auto variableType = variableOp.getVariableType();
  auto variableName = variableOp.getSymName();
  build(builder, state, variableType.unwrap(), variableName);
}

mlir::LogicalResult VariableGetOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  auto parentClass = getOperation()->getParentOfType<ClassInterface>();

  if (!parentClass) {
    return emitOpError() << "the operation must be used inside a class";
  }

  mlir::Operation *symbol =
      symbolTableCollection.lookupSymbolIn(parentClass, getVariableAttr());

  if (!symbol) {
    return emitOpError() << "variable " << getVariable()
                         << " has not been declared";
  }

  if (!mlir::isa<VariableOp>(symbol)) {
    return emitOpError() << "symbol " << getVariable() << " is not a variable";
  }

  auto variableOp = mlir::cast<VariableOp>(symbol);
  mlir::Type unwrappedType = variableOp.getVariableType().unwrap();

  if (unwrappedType != getResult().getType()) {
    return emitOpError() << "result type does not match the variable type";
  }

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// VariableSetOp

namespace mlir::bmodelica {
void VariableSetOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          VariableOp variableOp, mlir::Value value) {
  auto variableName = variableOp.getSymName();
  build(builder, state, variableName, std::nullopt, value);
}

void VariableSetOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          VariableOp variableOp, mlir::ValueRange indices,
                          mlir::Value value) {
  auto variableName = variableOp.getSymName();
  build(builder, state, variableName, indices, value);
}

mlir::ParseResult VariableSetOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  auto loc = parser.getCurrentLocation();

  mlir::StringAttr variableAttr;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
  mlir::OpAsmParser::UnresolvedOperand value;
  llvm::SmallVector<mlir::Type, 3> types;

  if (parser.parseSymbolName(variableAttr)) {
    return mlir::failure();
  }

  if (variableAttr) {
    result.getOrAddProperties<VariableSetOp::Properties>().variable =
        variableAttr;
  }

  if (mlir::succeeded(parser.parseOptionalLSquare())) {
    do {
      if (parser.parseOperand(indices.emplace_back())) {
        return mlir::failure();
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return mlir::failure();
    }
  }

  if (parser.parseComma() || parser.parseOperand(value) ||
      parser.parseColonTypeList(types)) {
    return mlir::failure();
  }

  if (types.size() != indices.size() + 1) {
    return mlir::failure();
  }

  if (!indices.empty()) {
    if (parser.resolveOperands(indices, llvm::ArrayRef(types).drop_back(), loc,
                               result.operands)) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperand(value, types.back(), result.operands)) {
    return mlir::failure();
  }

  return mlir::success();
}

void VariableSetOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getVariable());

  if (auto indices = getIndices(); !indices.empty()) {
    printer << "[" << indices << "]";
  }

  printer << ", " << getValue() << " : ";

  if (auto indices = getIndices(); !indices.empty()) {
    printer << indices.getTypes() << ", ";
  }

  printer << getValue().getType();
}

mlir::LogicalResult VariableSetOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  auto parentClass = getOperation()->getParentOfType<ClassInterface>();

  if (!parentClass) {
    return emitOpError("the operation must be used inside a class");
  }

  mlir::Operation *symbol =
      symbolTableCollection.lookupSymbolIn(parentClass, getVariableAttr());

  if (!symbol) {
    return emitOpError("variable " + getVariable() + " has not been declared");
  }

  auto variableOp = mlir::dyn_cast<VariableOp>(symbol);

  if (!variableOp) {
    return emitOpError("symbol " + getVariable() + " is not a variable");
  }

  if (variableOp.isInput()) {
    return emitOpError("can't set a value for an input variable");
  }

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// VariableComponentSetOp

namespace mlir::bmodelica {
mlir::ParseResult VariableComponentSetOp::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  auto loc = parser.getCurrentLocation();

  llvm::SmallVector<mlir::Attribute> path;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> subscripts;
  llvm::SmallVector<int64_t, 3> subscriptsAmounts;
  mlir::OpAsmParser::UnresolvedOperand value;
  llvm::SmallVector<mlir::Type, 3> types;

  do {
    mlir::StringAttr component;

    if (parser.parseSymbolName(component)) {
      return mlir::failure();
    }

    path.push_back(mlir::FlatSymbolRefAttr::get(component));

    llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3>
        componentSubscripts;

    if (mlir::succeeded(parser.parseOptionalLSquare())) {
      do {
        if (parser.parseOperand(componentSubscripts.emplace_back())) {
          return mlir::failure();
        }
      } while (mlir::succeeded(parser.parseOptionalComma()));

      if (parser.parseRSquare()) {
        return mlir::failure();
      }
    }

    subscriptsAmounts.push_back(
        static_cast<int64_t>(componentSubscripts.size()));

    subscripts.append(componentSubscripts);
  } while (mlir::succeeded(parser.parseOptionalColon()) &&
           mlir::succeeded(parser.parseOptionalColon()));

  result.getOrAddProperties<VariableComponentSetOp::Properties>().path =
      parser.getBuilder().getArrayAttr(path);

  result.getOrAddProperties<VariableComponentSetOp::Properties>()
      .subscriptionsAmounts =
      parser.getBuilder().getI64ArrayAttr(subscriptsAmounts);

  if (parser.parseComma() || parser.parseOperand(value) ||
      parser.parseColonTypeList(types)) {
    return mlir::failure();
  }

  if (!subscripts.empty()) {
    if (parser.resolveOperands(subscripts, llvm::ArrayRef(types).drop_back(),
                               loc, result.operands)) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperand(value, types.back(), result.operands)) {
    return mlir::failure();
  }

  return mlir::success();
}

void VariableComponentSetOp::print(mlir::OpAsmPrinter &printer) {
  size_t pathLength = getPath().size();
  printer << " ";

  for (size_t component = 0; component < pathLength; ++component) {
    if (component != 0) {
      printer << "::";
    }

    printer << getPath()[component];

    if (auto subscripts = getComponentSubscripts(component);
        !subscripts.empty()) {
      printer << "[";
      llvm::interleaveComma(subscripts, printer);
      printer << "]";
    }
  }

  printer << ", " << getValue() << " : ";

  if (auto subscripts = getSubscriptions(); !subscripts.empty()) {
    for (mlir::Value subscript : subscripts) {
      printer << subscript.getType() << ", ";
    }
  }

  printer << getValue().getType();
}

mlir::ValueRange
VariableComponentSetOp::getComponentSubscripts(size_t componentIndex) {
  auto subscripts = getSubscriptions();

  if (subscripts.empty()) {
    return std::nullopt;
  }

  auto numOfSubscripts = getSubscriptionsAmounts()[componentIndex]
                             .cast<mlir::IntegerAttr>()
                             .getInt();

  if (numOfSubscripts == 0) {
    return std::nullopt;
  }

  size_t beginPos = 0;

  for (size_t i = 0; i < componentIndex; ++i) {
    beginPos += getSubscriptionsAmounts()[i].cast<mlir::IntegerAttr>().getInt();
  }

  return subscripts.slice(beginPos, numOfSubscripts);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ComponentGetOp

namespace mlir::bmodelica {
mlir::LogicalResult ComponentGetOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::Type variableType = getVariable().getType();
  mlir::Type recordType = variableType;

  if (auto tensorType = recordType.dyn_cast<mlir::TensorType>()) {
    recordType = tensorType.getElementType();
  }

  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  auto recordOp =
      mlir::dyn_cast<RecordOp>(recordType.cast<RecordType>().getRecordOp(
          symbolTableCollection, moduleOp));

  if (!recordOp) {
    return emitOpError() << "Can't resolve record type";
  }

  VariableOp componentVariable = nullptr;

  for (auto variable : recordOp.getVariables()) {
    if (variable.getSymName() == getComponentName()) {
      componentVariable = variable;
      break;
    }
  }

  if (!componentVariable) {
    return emitOpError() << "Can't resolve record component";
  }

  llvm::SmallVector<int64_t> expectedResultShape;

  if (auto variableShapedType = variableType.dyn_cast<mlir::ShapedType>()) {
    auto variableShape = variableShapedType.getShape();
    expectedResultShape.append(variableShape.begin(), variableShape.end());
  }

  if (auto componentShapedType =
          componentVariable.getType().dyn_cast<mlir::ShapedType>()) {
    auto componentShape = componentShapedType.getShape();
    expectedResultShape.append(componentShape.begin(), componentShape.end());
  }

  mlir::Type expectedResultType = componentVariable.getVariableType().unwrap();

  if (!expectedResultShape.empty()) {
    if (auto expectedResultShapedType =
            mlir::dyn_cast<mlir::ShapedType>(expectedResultType)) {
      expectedResultType = expectedResultShapedType.clone(expectedResultShape);
    } else {
      expectedResultType =
          mlir::RankedTensorType::get(expectedResultShape, expectedResultType);
    }
  }

  mlir::Type resultType = getResult().getType();

  if (resultType != expectedResultType) {
    return emitOpError() << "Incompatible result type. Expected "
                         << expectedResultType << ", got " << resultType;
  }

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// GlobalVariableOp

namespace mlir::bmodelica {
void GlobalVariableOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, mlir::StringAttr name,
                             mlir::TypeAttr type) {
  build(builder, state, name, type, nullptr);
}

void GlobalVariableOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state, llvm::StringRef name,
                             mlir::Type type) {
  build(builder, state, name, type, nullptr);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// GlobalVariableGetOp

namespace mlir::bmodelica {
void GlobalVariableGetOp::build(mlir::OpBuilder &builder,
                                mlir::OperationState &state,
                                GlobalVariableOp globalVariableOp) {
  auto type = globalVariableOp.getType();
  auto name = globalVariableOp.getSymName();
  build(builder, state, type, name);
}

mlir::LogicalResult GlobalVariableGetOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  mlir::Operation *symbol =
      symbolTableCollection.lookupSymbolIn(moduleOp, getVariableAttr());

  if (!symbol) {
    return emitOpError() << "global variable " << getVariable()
                         << " has not been declared";
  }

  if (!mlir::isa<GlobalVariableOp>(symbol)) {
    return emitOpError() << "symbol " << getVariable()
                         << " is not a global variable";
  }

  auto globalVariableOp = mlir::cast<GlobalVariableOp>(symbol);

  if (globalVariableOp.getType() != getResult().getType()) {
    return emitOpError()
           << "result type does not match the global variable type";
  }

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// QualifiedVariableGetOp

namespace mlir::bmodelica {
void QualifiedVariableGetOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state,
                                   VariableOp variableOp) {
  auto variableType = variableOp.getVariableType();
  auto qualifiedRef = getSymbolRefFromRoot(variableOp);
  build(builder, state, variableType.unwrap(), qualifiedRef);
}

mlir::LogicalResult QualifiedVariableGetOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  // TODO
  return mlir::success();
}

VariableOp QualifiedVariableGetOp::getVariableOp() {
  mlir::SymbolTableCollection symbolTableCollection;
  return getVariableOp(symbolTableCollection);
}

VariableOp QualifiedVariableGetOp::getVariableOp(
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::ModuleOp moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  mlir::Operation *variable =
      resolveSymbol(moduleOp, symbolTableCollection, getVariable());

  return mlir::dyn_cast<VariableOp>(variable);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// QualifiedVariableSetOp

namespace mlir::bmodelica {
void QualifiedVariableSetOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state,
                                   VariableOp variableOp, mlir::Value value) {
  build(builder, state, variableOp, std::nullopt, value);
}

void QualifiedVariableSetOp::build(mlir::OpBuilder &builder,
                                   mlir::OperationState &state,
                                   VariableOp variableOp,
                                   mlir::ValueRange indices,
                                   mlir::Value value) {
  auto qualifiedRef = getSymbolRefFromRoot(variableOp);
  build(builder, state, qualifiedRef, indices, value);
}

mlir::ParseResult QualifiedVariableSetOp::parse(mlir::OpAsmParser &parser,
                                                mlir::OperationState &result) {
  auto loc = parser.getCurrentLocation();

  mlir::SymbolRefAttr variableAttr;
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 3> indices;
  mlir::OpAsmParser::UnresolvedOperand value;
  llvm::SmallVector<mlir::Type, 3> types;

  if (parser.parseCustomAttributeWithFallback(
          variableAttr, parser.getBuilder().getType<mlir::NoneType>())) {
    return mlir::failure();
  }

  if (variableAttr) {
    result.getOrAddProperties<QualifiedVariableSetOp::Properties>().variable =
        variableAttr;
  }

  if (mlir::succeeded(parser.parseOptionalLSquare())) {
    do {
      if (parser.parseOperand(indices.emplace_back())) {
        return mlir::failure();
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return mlir::failure();
    }
  }

  if (parser.parseComma() || parser.parseOperand(value) ||
      parser.parseColonTypeList(types)) {
    return mlir::failure();
  }

  if (types.size() != indices.size() + 1) {
    return mlir::failure();
  }

  if (!indices.empty()) {
    if (parser.resolveOperands(indices, llvm::ArrayRef(types).drop_back(), loc,
                               result.operands)) {
      return mlir::failure();
    }
  }

  if (parser.resolveOperand(value, types.back(), result.operands)) {
    return mlir::failure();
  }

  return mlir::success();
}

void QualifiedVariableSetOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getVariable();

  if (auto indices = getIndices(); !indices.empty()) {
    printer << "[" << indices << "]";
  }

  printer << ", " << getValue() << " : ";

  if (auto indices = getIndices(); !indices.empty()) {
    printer << indices.getTypes() << ", ";
  }

  printer << getValue().getType();
}

mlir::LogicalResult QualifiedVariableSetOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  // TODO
  return mlir::success();
}

VariableOp QualifiedVariableSetOp::getVariableOp() {
  mlir::SymbolTableCollection symbolTableCollection;
  return getVariableOp(symbolTableCollection);
}

VariableOp QualifiedVariableSetOp::getVariableOp(
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::ModuleOp moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  mlir::Operation *variable =
      resolveSymbol(moduleOp, symbolTableCollection, getVariable());

  return mlir::dyn_cast<VariableOp>(variable);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// ConstantOp

namespace mlir::bmodelica {
mlir::OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) {
  return getValue().cast<mlir::Attribute>();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// NegateOp

namespace mlir::bmodelica {
mlir::LogicalResult NegateOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type operandType = adaptor.getOperand().getType();

  if (isScalar(operandType)) {
    returnTypes.push_back(operandType);
    return mlir::success();
  }

  if (auto shapedType = operandType.dyn_cast<mlir::ShapedType>()) {
    returnTypes.push_back(shapedType);
    return mlir::success();
  }

  return mlir::failure();
}

bool NegateOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult NegateOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      return getAttr(resultType, -1 * getScalarIntegerLikeValue(operand));
    }

    if (isScalarFloatLike(operand)) {
      return getAttr(resultType, -1 * getScalarFloatLikeValue(operand));
    }
  }

  return {};
}

mlir::LogicalResult
NegateOp::distribute(llvm::SmallVectorImpl<mlir::Value> &results,
                     mlir::OpBuilder &builder) {
  mlir::Value operand = getOperand();
  mlir::Operation *operandOp = operand.getDefiningOp();

  if (operandOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(operandOp)) {
      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(results, builder))) {
        return mlir::success();
      }
    }
  }

  // The operation can't be propagated because the child doesn't know how to
  // distribute the negation to its children.
  results.push_back(getResult());
  return mlir::failure();
}

mlir::LogicalResult
NegateOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                             mlir::OpBuilder &builder) {
  mlir::Value operand = getOperand();
  bool operandDistributed = false;
  mlir::Operation *operandOp = operand.getDefiningOp();

  if (operandOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(operandOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        operand = childResults[0];
        operandDistributed = true;
      }
    }
  }

  if (!operandDistributed) {
    auto newOperandOp = builder.create<NegateOp>(getLoc(), operand);
    operand = newOperandOp.getResult();
  }

  auto resultOp = builder.create<NegateOp>(getLoc(), operand);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
NegateOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                          mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value operand = getOperand();
  bool operandDistributed = false;
  mlir::Operation *operandOp = operand.getDefiningOp();

  if (operandOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(operandOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        operand = childResults[0];
        operandDistributed = true;
      }
    }
  }

  if (!operandDistributed) {
    auto newOperandOp = builder.create<MulOp>(getLoc(), operand, value);
    operand = newOperandOp.getResult();
  }

  auto resultOp = builder.create<NegateOp>(getLoc(), operand);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
NegateOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                          mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value operand = getOperand();
  bool operandDistributed = false;
  mlir::Operation *operandOp = operand.getDefiningOp();

  if (operandOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(operandOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        operand = childResults[0];
        operandDistributed = true;
      }
    }
  }

  if (!operandDistributed) {
    auto newOperandOp = builder.create<DivOp>(getLoc(), operand, value);
    operand = newOperandOp.getResult();
  }

  auto resultOp = builder.create<NegateOp>(getLoc(), operand);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// AddOp

namespace {
struct AddOpRangeOrderingPattern : public mlir::OpRewritePattern<AddOp> {
  using mlir::OpRewritePattern<AddOp>::OpRewritePattern;

  mlir::LogicalResult match(AddOp op) const override {
    mlir::Value lhs = op.getLhs();
    mlir::Value rhs = op.getRhs();

    return mlir::LogicalResult::success(!lhs.getType().isa<RangeType>() &&
                                        rhs.getType().isa<RangeType>());
  }

  void rewrite(AddOp op, mlir::PatternRewriter &rewriter) const override {
    // Swap the operands.
    rewriter.replaceOpWithNewOp<AddOp>(op, op.getResult().getType(),
                                       op.getRhs(), op.getLhs());
  }
};
} // namespace

namespace mlir::bmodelica {
mlir::LogicalResult AddOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    mlir::Type resultElementType = getMostGenericScalarType(
        lhsShapedType.getElementType(), rhsShapedType.getElementType());

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, resultElementType));

    return mlir::success();
  }

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsRangeType = lhsType.dyn_cast<RangeType>();
  auto rhsRangeType = rhsType.dyn_cast<RangeType>();

  if (isScalar(lhsType) && rhsRangeType) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsType, rhsRangeType.getInductionType());

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  if (lhsRangeType && isScalar(rhsType)) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsRangeType.getInductionType(), rhsType);

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  return mlir::failure();
}

bool AddOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, lhsValue + rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue + rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue + rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue + rhsValue);
    }
  }

  if (auto lhsRange = lhs.dyn_cast<IntegerRangeAttr>();
      lhsRange && isScalar(rhs)) {
    if (isScalarIntegerLike(rhs)) {
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      int64_t lowerBound = lhsRange.getLowerBound() + rhsValue;
      int64_t upperBound = lhsRange.getUpperBound() + rhsValue;
      int64_t step = lhsRange.getStep();

      return IntegerRangeAttr::get(getContext(), lhsRange.getType(), lowerBound,
                                   upperBound, step);
    }

    if (isScalarFloatLike(rhs)) {
      double rhsValue = getScalarFloatLikeValue(rhs);

      double lowerBound =
          static_cast<double>(lhsRange.getLowerBound()) + rhsValue;

      double upperBound =
          static_cast<double>(lhsRange.getUpperBound()) + rhsValue;

      auto step = static_cast<double>(lhsRange.getStep());

      return RealRangeAttr::get(getContext(), lowerBound, upperBound, step);
    }
  }

  if (auto lhsRange = lhs.dyn_cast<RealRangeAttr>();
      lhsRange && isScalar(rhs)) {
    if (isScalarIntegerLike(rhs)) {
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));

      double lowerBound = lhsRange.getLowerBound().convertToDouble() + rhsValue;

      double upperBound = lhsRange.getUpperBound().convertToDouble() + rhsValue;

      double step = lhsRange.getStep().convertToDouble();

      return RealRangeAttr::get(lhsRange.getType(), lowerBound, upperBound,
                                step);
    }

    if (isScalarFloatLike(rhs)) {
      double rhsValue = getScalarFloatLikeValue(rhs);

      double lowerBound = lhsRange.getLowerBound().convertToDouble() + rhsValue;

      double upperBound = lhsRange.getUpperBound().convertToDouble() + rhsValue;

      double step = lhsRange.getStep().convertToDouble();

      return RealRangeAttr::get(getContext(), lowerBound, upperBound, step);
    }
  }

  return {};
}

void AddOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *context) {
  patterns.add<AddOpRangeOrderingPattern>(context);
}

mlir::LogicalResult
AddOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                          mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<NegateOp>(lhs.getLoc(), lhs);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<NegateOp>(rhs.getLoc(), rhs);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<AddOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
AddOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<MulOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<MulOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<AddOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
AddOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<DivOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<DivOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<AddOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// AddEWOp

namespace mlir::bmodelica {
mlir::LogicalResult AddEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (isScalar(lhsType) && rhsShapedType) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsType, rhsShapedType.getElementType());

    returnTypes.push_back(mlir::RankedTensorType::get(rhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && isScalar(rhsType)) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsShapedType.getElementType(), rhsType);

    returnTypes.push_back(mlir::RankedTensorType::get(lhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    mlir::Type resultElementType = getMostGenericScalarType(
        lhsShapedType.getElementType(), rhsShapedType.getElementType());

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, resultElementType));

    return mlir::success();
  }

  return mlir::failure();
}

bool AddEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult AddEWOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, lhsValue + rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue + rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue + rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue + rhsValue);
    }
  }

  return {};
}

mlir::LogicalResult
AddEWOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                            mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto negDistributionOp =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionOp.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionOp =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionOp.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<NegateOp>(lhs.getLoc(), lhs);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<NegateOp>(rhs.getLoc(), rhs);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<AddEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
AddEWOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<MulOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<MulOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<AddEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
AddEWOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<DivOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<DivOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<AddEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SubOp

namespace mlir::bmodelica {
mlir::LogicalResult SubOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    mlir::Type resultElementType = getMostGenericScalarType(
        lhsShapedType.getElementType(), rhsShapedType.getElementType());

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, resultElementType));

    return mlir::success();
  }

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsRangeType = lhsType.dyn_cast<RangeType>();
  auto rhsRangeType = rhsType.dyn_cast<RangeType>();

  if (isScalar(lhsType) && rhsRangeType) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsType, rhsRangeType.getInductionType());

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  if (lhsRangeType && isScalar(rhsType)) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsRangeType.getInductionType(), rhsType);

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  return mlir::failure();
}

bool SubOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, lhsValue - rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue - rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue - rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue - rhsValue);
    }
  }

  if (auto lhsRange = lhs.dyn_cast<IntegerRangeAttr>();
      lhsRange && isScalar(rhs)) {
    if (isScalarIntegerLike(rhs)) {
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      int64_t lowerBound = lhsRange.getLowerBound() - rhsValue;
      int64_t upperBound = lhsRange.getUpperBound() - rhsValue;
      int64_t step = lhsRange.getStep();

      return IntegerRangeAttr::get(getContext(), lhsRange.getType(), lowerBound,
                                   upperBound, step);
    }

    if (isScalarFloatLike(rhs)) {
      double rhsValue = getScalarFloatLikeValue(rhs);

      double lowerBound =
          static_cast<double>(lhsRange.getLowerBound()) - rhsValue;

      double upperBound =
          static_cast<double>(lhsRange.getUpperBound()) - rhsValue;

      auto step = static_cast<double>(lhsRange.getStep());

      return RealRangeAttr::get(getContext(), lowerBound, upperBound, step);
    }
  }

  if (auto lhsRange = lhs.dyn_cast<RealRangeAttr>();
      lhsRange && isScalar(rhs)) {
    if (isScalarIntegerLike(rhs)) {
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));

      double lowerBound = lhsRange.getLowerBound().convertToDouble() - rhsValue;

      double upperBound = lhsRange.getUpperBound().convertToDouble() - rhsValue;

      double step = lhsRange.getStep().convertToDouble();

      return RealRangeAttr::get(lhsRange.getType(), lowerBound, upperBound,
                                step);
    }

    if (isScalarFloatLike(rhs)) {
      double rhsValue = getScalarFloatLikeValue(rhs);

      double lowerBound = lhsRange.getLowerBound().convertToDouble() - rhsValue;

      double upperBound = lhsRange.getUpperBound().convertToDouble() - rhsValue;

      double step = lhsRange.getStep().convertToDouble();

      return RealRangeAttr::get(getContext(), lowerBound, upperBound, step);
    }
  }

  return {};
}

mlir::LogicalResult
SubOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                          mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<NegateOp>(lhs.getLoc(), lhs);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<NegateOp>(rhs.getLoc(), rhs);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<SubOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
SubOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<MulOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<MulOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<SubOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
SubOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<DivOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<DivOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<SubOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SubEWOp

namespace mlir::bmodelica {
mlir::LogicalResult SubEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (isScalar(lhsType) && rhsShapedType) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsType, rhsShapedType.getElementType());

    returnTypes.push_back(mlir::RankedTensorType::get(rhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && isScalar(rhsType)) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsShapedType.getElementType(), rhsType);

    returnTypes.push_back(mlir::RankedTensorType::get(lhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    mlir::Type resultElementType = getMostGenericScalarType(
        lhsShapedType.getElementType(), rhsShapedType.getElementType());

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, resultElementType));

    return mlir::success();
  }

  return mlir::failure();
}

bool SubEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult SubEWOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, lhsValue - rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue - rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue - rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue - rhsValue);
    }
  }

  return {};
}

mlir::LogicalResult
SubEWOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                            mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<NegateOp>(lhs.getLoc(), lhs);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<NegateOp>(rhs.getLoc(), rhs);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<SubEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
SubEWOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<MulOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<MulOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<SubEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
SubEWOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  bool lhsDistributed = false;
  bool rhsDistributed = false;

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];
        lhsDistributed = true;
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];
        rhsDistributed = true;
      }
    }
  }

  if (!lhsDistributed) {
    auto newLhsOp = builder.create<DivOp>(lhs.getLoc(), lhs, value);
    lhs = newLhsOp.getResult();
  }

  if (!rhsDistributed) {
    auto newRhsOp = builder.create<DivOp>(rhs.getLoc(), rhs, value);
    rhs = newRhsOp.getResult();
  }

  auto resultOp = builder.create<SubEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// MulOp

namespace mlir::bmodelica {
mlir::LogicalResult MulOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (isScalar(lhsType) && rhsShapedType) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsType, rhsShapedType.getElementType());

    returnTypes.push_back(mlir::RankedTensorType::get(rhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && rhsShapedType) {
    mlir::Type resultElementType = getMostGenericScalarType(
        lhsShapedType.getElementType(), rhsShapedType.getElementType());

    if (lhsShapedType.getRank() == 1 && rhsShapedType.getRank() == 1) {
      returnTypes.push_back(resultElementType);
      return mlir::success();
    }

    if (lhsShapedType.getRank() == 1 && rhsShapedType.getRank() == 2) {
      returnTypes.push_back(mlir::RankedTensorType::get(
          rhsShapedType.getShape()[1], resultElementType));

      return mlir::success();
    }

    if (lhsShapedType.getRank() == 2 && rhsShapedType.getRank() == 1) {
      returnTypes.push_back(mlir::RankedTensorType::get(
          lhsShapedType.getShape()[0], resultElementType));

      return mlir::success();
    }

    if (lhsShapedType.getRank() == 2 && rhsShapedType.getRank() == 2) {
      llvm::SmallVector<int64_t, 2> shape;
      shape.push_back(lhsShapedType.getShape()[0]);
      shape.push_back(rhsShapedType.getShape()[1]);

      returnTypes.push_back(
          mlir::RankedTensorType::get(shape, resultElementType));

      return mlir::success();
    }
  }

  auto lhsRangeType = lhsType.dyn_cast<RangeType>();
  auto rhsRangeType = rhsType.dyn_cast<RangeType>();

  if (isScalar(lhsType) && rhsRangeType) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsType, rhsRangeType.getInductionType());

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  if (lhsRangeType && isScalar(rhsType)) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsRangeType.getInductionType(), rhsType);

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  return mlir::failure();
}

bool MulOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  auto resultType = getResult().getType();

  if (lhs && isScalar(lhs) && getScalarAttributeValue<double>(lhs) == 0) {
    if (!resultType.isa<mlir::ShapedType>()) {
      return getAttr(resultType, static_cast<int64_t>(0));
    }
  }

  if (rhs && isScalar(rhs) && getScalarAttributeValue<double>(rhs) == 0) {
    if (!resultType.isa<mlir::ShapedType>()) {
      return getAttr(resultType, static_cast<int64_t>(0));
    }
  }

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, lhsValue * rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue * rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue * rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue * rhsValue);
    }
  }

  return {};
}

mlir::LogicalResult
MulOp::distribute(llvm::SmallVectorImpl<mlir::Value> &results,
                  mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      mlir::Value toDistribute = rhs;
      results.clear();

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(results, builder,
                                                             toDistribute))) {
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      mlir::Value toDistribute = lhs;
      results.clear();

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(results, builder,
                                                             toDistribute))) {
        return mlir::success();
      }
    }
  }

  // The operation can't be propagated because none of the children
  // know how to distribute the multiplication to their children.
  results.push_back(getResult());
  return mlir::failure();
}

mlir::LogicalResult
MulOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                          mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
MulOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
MulOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<MulOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

bool MulOp::isScalarProduct() {
  return !getLhs().getType().isa<mlir::TensorType>() &&
         getRhs().getType().isa<mlir::TensorType>();
}

bool MulOp::isCrossProduct() {
  auto lhsTensorType = getLhs().getType().dyn_cast<mlir::TensorType>();
  auto rhsTensorType = getRhs().getType().dyn_cast<mlir::TensorType>();

  return lhsTensorType && rhsTensorType && lhsTensorType.getRank() == 1 &&
         rhsTensorType.getRank() == 1;
}

bool MulOp::isVectorMatrixProduct() {
  auto lhsTensorType = getLhs().getType().dyn_cast<mlir::TensorType>();
  auto rhsTensorType = getRhs().getType().dyn_cast<mlir::TensorType>();

  if (!lhsTensorType || !rhsTensorType) {
    return false;
  }

  return lhsTensorType.getRank() == 1 && rhsTensorType.getRank() == 2;
}

bool MulOp::isMatrixVectorProduct() {
  auto lhsTensorType = getLhs().getType().dyn_cast<mlir::TensorType>();
  auto rhsTensorType = getRhs().getType().dyn_cast<mlir::TensorType>();

  if (!lhsTensorType || !rhsTensorType) {
    return false;
  }

  return lhsTensorType.getRank() == 2 && rhsTensorType.getRank() == 1;
}

bool MulOp::isMatrixProduct() {
  auto lhsTensorType = getLhs().getType().dyn_cast<mlir::TensorType>();
  auto rhsTensorType = getRhs().getType().dyn_cast<mlir::TensorType>();

  if (!lhsTensorType || !rhsTensorType) {
    return false;
  }

  return lhsTensorType.getRank() == 2 && rhsTensorType.getRank() == 2;
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// MulEWOp

namespace mlir::bmodelica {
mlir::LogicalResult MulEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (isScalar(lhsType) && rhsShapedType) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsType, rhsShapedType.getElementType());

    returnTypes.push_back(mlir::RankedTensorType::get(rhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && isScalar(rhsType)) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsShapedType.getElementType(), rhsType);

    returnTypes.push_back(mlir::RankedTensorType::get(lhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    mlir::Type resultElementType = getMostGenericScalarType(
        lhsShapedType.getElementType(), rhsShapedType.getElementType());

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, resultElementType));

    return mlir::success();
  }

  return mlir::failure();
}

bool MulEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult MulEWOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, lhsValue * rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue * rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue * rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue * rhsValue);
    }
  }

  return {};
}

mlir::LogicalResult
MulEWOp::distribute(llvm::SmallVectorImpl<mlir::Value> &results,
                    mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      mlir::Value toDistribute = rhs;
      results.clear();

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(results, builder,
                                                             toDistribute))) {
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      mlir::Value toDistribute = lhs;
      results.clear();

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(results, builder,
                                                             toDistribute))) {
        return mlir::success();
      }
    }
  }

  // The operation can't be propagated because none of the children
  // know how to distribute the multiplication to their children.
  results.push_back(getResult());
  return mlir::failure();
}

mlir::LogicalResult
MulEWOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                            mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
MulEWOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
MulEWOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<MulEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// DivOp

namespace mlir::bmodelica {
mlir::LogicalResult DivOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();

  if (lhsShapedType && isScalar(rhsType)) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsShapedType.getElementType(), rhsType);

    returnTypes.push_back(mlir::RankedTensorType::get(lhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  auto lhsRangeType = lhsType.dyn_cast<RangeType>();
  auto rhsRangeType = rhsType.dyn_cast<RangeType>();

  if (isScalar(lhsType) && rhsRangeType) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsType, rhsRangeType.getInductionType());

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  if (lhsRangeType && isScalar(rhsType)) {
    mlir::Type inductionType =
        getMostGenericScalarType(lhsRangeType.getInductionType(), rhsType);

    returnTypes.push_back(RangeType::get(context, inductionType));
    return mlir::success();
  }

  return mlir::failure();
}

bool DivOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue / rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue / rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue / rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue / rhsValue);
    }
  }

  return {};
}

mlir::LogicalResult
DivOp::distribute(llvm::SmallVectorImpl<mlir::Value> &results,
                  mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Operation *lhsOp = lhs.getDefiningOp();

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      mlir::Value toDistribute = getRhs();

      if (mlir::succeeded(divDistributionInt.distributeDivOp(results, builder,
                                                             toDistribute))) {
        return mlir::success();
      }
    }
  }

  // The operation can't be propagated because the dividend does not know
  // how to distribute the division to their children.
  results.push_back(getResult());
  return mlir::success();
}

mlir::LogicalResult
DivOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                          mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
DivOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
DivOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                       mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhs.getDefiningOp())) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhs.getDefiningOp())) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<DivOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// DivEWOp

namespace mlir::bmodelica {
mlir::LogicalResult DivEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(getMostGenericScalarType(lhsType, rhsType));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (isScalar(lhsType) && rhsShapedType) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsType, rhsShapedType.getElementType());

    returnTypes.push_back(mlir::RankedTensorType::get(rhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && isScalar(rhsType)) {
    mlir::Type resultElementType =
        getMostGenericScalarType(lhsShapedType.getElementType(), rhsType);

    returnTypes.push_back(mlir::RankedTensorType::get(lhsShapedType.getShape(),
                                                      resultElementType));

    return mlir::success();
  }

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    mlir::Type resultElementType = getMostGenericScalarType(
        lhsShapedType.getElementType(), rhsShapedType.getElementType());

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, resultElementType));

    return mlir::success();
  }

  return mlir::failure();
}

bool DivEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult DivEWOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue / rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue / rhsValue);
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, lhsValue / rhsValue);
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, lhsValue / rhsValue);
    }
  }

  return {};
}

mlir::LogicalResult
DivEWOp::distribute(llvm::SmallVectorImpl<mlir::Value> &results,
                    mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Operation *lhsOp = lhs.getDefiningOp();

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      mlir::Value toDistribute = getRhs();

      if (mlir::succeeded(divDistributionInt.distributeDivOp(results, builder,
                                                             toDistribute))) {
        return mlir::success();
      }
    }
  }

  // The operation can't be propagated because the dividend does not know
  // how to distribute the division to their children.
  results.push_back(getResult());
  return mlir::failure();
}

mlir::LogicalResult
DivEWOp::distributeNegateOp(llvm::SmallVectorImpl<mlir::Value> &results,
                            mlir::OpBuilder &builder) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto negDistributionInt =
            mlir::dyn_cast<NegateOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(
              negDistributionInt.distributeNegateOp(childResults, builder)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<NegateOp>(getLoc(), lhs);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
DivEWOp::distributeMulOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<MulOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}

mlir::LogicalResult
DivEWOp::distributeDivOp(llvm::SmallVectorImpl<mlir::Value> &results,
                         mlir::OpBuilder &builder, mlir::Value value) {
  mlir::Value lhs = getLhs();
  mlir::Value rhs = getRhs();

  mlir::Operation *lhsOp = lhs.getDefiningOp();
  mlir::Operation *rhsOp = rhs.getDefiningOp();

  if (lhsOp) {
    if (auto divDistributionInt =
            mlir::dyn_cast<DivOpDistributionInterface>(lhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(divDistributionInt.distributeDivOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        lhs = childResults[0];

        auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  if (rhsOp) {
    if (auto mulDistributionInt =
            mlir::dyn_cast<MulOpDistributionInterface>(rhsOp)) {
      llvm::SmallVector<mlir::Value, 1> childResults;

      if (mlir::succeeded(mulDistributionInt.distributeMulOp(childResults,
                                                             builder, value)) &&
          childResults.size() == 1) {
        rhs = childResults[0];

        auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
        results.push_back(resultOp.getResult());
        return mlir::success();
      }
    }
  }

  auto lhsNewOp = builder.create<DivOp>(getLoc(), lhs, value);
  lhs = lhsNewOp.getResult();

  auto resultOp = builder.create<DivEWOp>(getLoc(), lhs, rhs);
  results.push_back(resultOp.getResult());

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// PowOp

namespace mlir::bmodelica {
mlir::LogicalResult PowOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type baseType = adaptor.getBase().getType();
  mlir::Type exponentType = adaptor.getExponent().getType();

  if (isScalar(baseType)) {
    if (exponentType.isa<RealType, mlir::FloatType>()) {
      returnTypes.push_back(exponentType);
      return mlir::success();
    }

    returnTypes.push_back(baseType);
    return mlir::success();
  }

  if (auto baseShapedType = baseType.dyn_cast<mlir::ShapedType>()) {
    if (!isScalar(exponentType)) {
      return mlir::failure();
    }

    returnTypes.push_back(baseShapedType);
    return mlir::success();
  }

  return mlir::failure();
}

bool PowOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult PowOp::fold(FoldAdaptor adaptor) {
  auto base = adaptor.getBase();
  auto exponent = adaptor.getExponent();

  if (!base || !exponent) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(base) && isScalar(exponent)) {
    if (isScalarIntegerLike(base) && isScalarIntegerLike(exponent)) {
      auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));

      auto exponentValue =
          static_cast<double>(getScalarIntegerLikeValue(exponent));

      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }

    if (isScalarFloatLike(base) && isScalarFloatLike(exponent)) {
      double baseValue = getScalarFloatLikeValue(base);
      double exponentValue = getScalarFloatLikeValue(exponent);
      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }

    if (isScalarIntegerLike(base) && isScalarFloatLike(exponent)) {
      auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));
      double exponentValue = getScalarFloatLikeValue(exponent);
      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }

    if (isScalarFloatLike(base) && isScalarIntegerLike(exponent)) {
      double baseValue = getScalarFloatLikeValue(base);

      auto exponentValue =
          static_cast<double>(getScalarIntegerLikeValue(exponent));

      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// PowEWOp

namespace mlir::bmodelica {
mlir::LogicalResult PowEWOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type baseType = adaptor.getBase().getType();
  mlir::Type exponentType = adaptor.getExponent().getType();

  auto inferResultType = [](mlir::Type baseType,
                            mlir::Type exponentType) -> mlir::Type {
    if (exponentType.isa<RealType, mlir::FloatType>()) {
      return exponentType;
    }

    return baseType;
  };

  if (isScalar(baseType) && isScalar(exponentType)) {
    returnTypes.push_back(inferResultType(baseType, exponentType));
    return mlir::success();
  }

  auto baseShapedType = baseType.dyn_cast<mlir::ShapedType>();
  auto exponentShapedType = exponentType.dyn_cast<mlir::ShapedType>();

  if (isScalar(baseType) && exponentShapedType) {
    returnTypes.push_back(mlir::RankedTensorType::get(
        exponentShapedType.getShape(),
        inferResultType(baseType, exponentShapedType.getElementType())));

    return mlir::success();
  }

  if (baseShapedType && isScalar(exponentType)) {
    returnTypes.push_back(mlir::RankedTensorType::get(
        baseShapedType.getShape(),
        inferResultType(baseShapedType.getElementType(), exponentType)));

    return mlir::success();
  }

  if (baseShapedType && exponentShapedType) {
    if (baseShapedType.getRank() != exponentShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = baseShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = baseShapedType.getDimSize(dim);
      int64_t rhsDimSize = exponentShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    mlir::Type resultElementType = inferResultType(
        baseShapedType.getElementType(), exponentShapedType.getElementType());

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, resultElementType));

    return mlir::success();
  }

  return mlir::failure();
}

bool PowEWOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult PowEWOp::fold(FoldAdaptor adaptor) {
  auto base = adaptor.getBase();
  auto exponent = adaptor.getExponent();

  if (!base || !exponent) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(base) && isScalar(exponent)) {
    if (isScalarIntegerLike(base) && isScalarIntegerLike(exponent)) {
      auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));

      auto exponentValue =
          static_cast<double>(getScalarIntegerLikeValue(exponent));

      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }

    if (isScalarFloatLike(base) && isScalarFloatLike(exponent)) {
      double baseValue = getScalarFloatLikeValue(base);
      double exponentValue = getScalarFloatLikeValue(exponent);
      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }

    if (isScalarIntegerLike(base) && isScalarFloatLike(exponent)) {
      auto baseValue = static_cast<double>(getScalarIntegerLikeValue(base));
      double exponentValue = getScalarFloatLikeValue(exponent);
      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }

    if (isScalarFloatLike(base) && isScalarIntegerLike(exponent)) {
      double baseValue = getScalarFloatLikeValue(base);

      auto exponentValue =
          static_cast<double>(getScalarIntegerLikeValue(exponent));

      return getAttr(resultType, std::pow(baseValue, exponentValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Comparison operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// EqOp

namespace mlir::bmodelica {
mlir::LogicalResult EqOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(BooleanType::get(context));
  return mlir::success();
}

bool EqOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult EqOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, static_cast<int64_t>(lhsValue == rhsValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// NotEqOp

namespace mlir::bmodelica {
mlir::LogicalResult NotEqOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(BooleanType::get(context));
  return mlir::success();
}

bool NotEqOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                      mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult NotEqOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, static_cast<int64_t>(lhsValue != rhsValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// GtOp

namespace mlir::bmodelica {
mlir::LogicalResult GtOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(BooleanType::get(context));
  return mlir::success();
}

bool GtOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult GtOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, static_cast<int64_t>(lhsValue > rhsValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// GteOp

namespace mlir::bmodelica {
mlir::LogicalResult GteOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(BooleanType::get(context));
  return mlir::success();
}

bool GteOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult GteOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, static_cast<int64_t>(lhsValue >= rhsValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// LtOp

namespace mlir::bmodelica {
mlir::LogicalResult LtOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(BooleanType::get(context));
  return mlir::success();
}

bool LtOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult LtOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, static_cast<int64_t>(lhsValue < rhsValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// LteOp

namespace mlir::bmodelica {
mlir::LogicalResult LteOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  returnTypes.push_back(BooleanType::get(context));
  return mlir::success();
}

bool LteOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult LteOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      auto lhsValue = static_cast<double>(getScalarIntegerLikeValue(lhs));
      double rhsValue = getScalarFloatLikeValue(rhs);
      return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      auto rhsValue = static_cast<double>(getScalarIntegerLikeValue(rhs));
      return getAttr(resultType, static_cast<int64_t>(lhsValue <= rhsValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Logic operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// NotOp

namespace mlir::bmodelica {
mlir::LogicalResult NotOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type operandType = adaptor.getOperand().getType();

  if (isScalar(operandType)) {
    returnTypes.push_back(BooleanType::get(context));
    return mlir::success();
  }

  if (auto shapedType = operandType.dyn_cast<mlir::ShapedType>()) {
    returnTypes.push_back(mlir::RankedTensorType::get(
        shapedType.getShape(), BooleanType::get(context)));

    return mlir::success();
  }

  return mlir::failure();
}

bool NotOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult NotOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      int64_t operandValue = getScalarIntegerLikeValue(operand);
      return getAttr(resultType, static_cast<int64_t>(operandValue == 0));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, static_cast<int64_t>(operandValue == 0));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// AndOp

namespace mlir::bmodelica {
mlir::LogicalResult AndOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(BooleanType::get(context));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, BooleanType::get(context)));

    return mlir::success();
  }

  return mlir::failure();
}

bool AndOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 && rhsValue != 0));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// OrOp

namespace mlir::bmodelica {
mlir::LogicalResult OrOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type lhsType = adaptor.getLhs().getType();
  mlir::Type rhsType = adaptor.getRhs().getType();

  if (isScalar(lhsType) && isScalar(rhsType)) {
    returnTypes.push_back(BooleanType::get(context));
    return mlir::success();
  }

  auto lhsShapedType = lhsType.dyn_cast<mlir::ShapedType>();
  auto rhsShapedType = rhsType.dyn_cast<mlir::ShapedType>();

  if (lhsShapedType && rhsShapedType) {
    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return mlir::failure();
    }

    int64_t rank = lhsShapedType.getRank();
    llvm::SmallVector<int64_t, 10> shape;

    for (int64_t dim = 0; dim < rank; ++dim) {
      int64_t lhsDimSize = lhsShapedType.getDimSize(dim);
      int64_t rhsDimSize = rhsShapedType.getDimSize(dim);

      if (lhsDimSize != mlir::ShapedType::kDynamic &&
          rhsDimSize != mlir::ShapedType::kDynamic &&
          lhsDimSize != rhsDimSize) {
        return mlir::failure();
      }

      if (lhsDimSize != mlir::ShapedType::kDynamic) {
        shape.push_back(lhsDimSize);
      } else {
        shape.push_back(rhsDimSize);
      }
    }

    returnTypes.push_back(
        mlir::RankedTensorType::get(shape, BooleanType::get(context)));

    return mlir::success();
  }

  return mlir::failure();
}

bool OrOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  auto lhs = adaptor.getLhs();
  auto rhs = adaptor.getRhs();

  if (!lhs || !rhs) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(lhs) && isScalar(rhs)) {
    if (isScalarIntegerLike(lhs) && isScalarIntegerLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
    }

    if (isScalarFloatLike(lhs) && isScalarFloatLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
    }

    if (isScalarIntegerLike(lhs) && isScalarFloatLike(rhs)) {
      int64_t lhsValue = getScalarIntegerLikeValue(lhs);
      double rhsValue = getScalarFloatLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
    }

    if (isScalarFloatLike(lhs) && isScalarIntegerLike(rhs)) {
      double lhsValue = getScalarFloatLikeValue(lhs);
      int64_t rhsValue = getScalarIntegerLikeValue(rhs);

      return getAttr(resultType,
                     static_cast<int64_t>(lhsValue != 0 || rhsValue != 0));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SelectOp

namespace mlir::bmodelica {
mlir::LogicalResult SelectOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);

  if (adaptor.getTrueValues().size() != adaptor.getFalseValues().size()) {
    return mlir::failure();
  }

  for (auto [trueValue, falseValue] :
       llvm::zip(adaptor.getTrueValues(), adaptor.getFalseValues())) {
    mlir::Type trueValueType = trueValue.getType();
    mlir::Type falseValueType = falseValue.getType();

    if (trueValueType == falseValueType) {
      returnTypes.push_back(trueValueType);
    } else if (isScalar(trueValueType) && isScalar(falseValueType)) {
      returnTypes.push_back(
          getMostGenericScalarType(trueValueType, falseValueType));
    } else {
      auto trueValueShapedType = trueValueType.dyn_cast<mlir::ShapedType>();

      auto falseValueShapedType = falseValueType.dyn_cast<mlir::ShapedType>();

      if (trueValueShapedType && falseValueShapedType) {
        if (trueValueShapedType.getRank() != falseValueShapedType.getRank()) {
          return mlir::failure();
        }

        int64_t rank = trueValueShapedType.getRank();
        llvm::SmallVector<int64_t, 10> shape;

        for (int64_t dim = 0; dim < rank; ++dim) {
          int64_t lhsDimSize = trueValueShapedType.getDimSize(dim);
          int64_t rhsDimSize = falseValueShapedType.getDimSize(dim);

          if (lhsDimSize != mlir::ShapedType::kDynamic &&
              rhsDimSize != mlir::ShapedType::kDynamic &&
              lhsDimSize != rhsDimSize) {
            return mlir::failure();
          }

          if (lhsDimSize != mlir::ShapedType::kDynamic) {
            shape.push_back(lhsDimSize);
          } else {
            shape.push_back(rhsDimSize);
          }
        }

        mlir::Type resultElementType =
            getMostGenericScalarType(trueValueShapedType.getElementType(),
                                     falseValueShapedType.getElementType());

        returnTypes.push_back(
            mlir::RankedTensorType::get(shape, resultElementType));
      } else {
        return mlir::failure();
      }
    }
  }

  return mlir::success();
}

bool SelectOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                       mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::ParseResult SelectOp::parse(mlir::OpAsmParser &parser,
                                  mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand condition;
  mlir::Type conditionType;

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> trueValues;
  llvm::SmallVector<mlir::Type, 1> trueValuesTypes;

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> falseValues;
  llvm::SmallVector<mlir::Type, 1> falseValuesTypes;

  llvm::SmallVector<mlir::Type, 1> resultTypes;

  if (parser.parseLParen() || parser.parseOperand(condition) ||
      parser.parseColonType(conditionType) || parser.parseRParen() ||
      parser.resolveOperand(condition, conditionType, result.operands)) {
    return mlir::failure();
  }

  if (parser.parseComma()) {
    return mlir::failure();
  }

  auto trueValuesLoc = parser.getCurrentLocation();

  if (parser.parseLParen() || parser.parseOperandList(trueValues) ||
      parser.parseColonTypeList(trueValuesTypes) || parser.parseRParen() ||
      parser.resolveOperands(trueValues, trueValuesTypes, trueValuesLoc,
                             result.operands)) {
    return mlir::failure();
  }

  if (parser.parseComma()) {
    return mlir::failure();
  }

  auto falseValuesLoc = parser.getCurrentLocation();

  if (parser.parseLParen() || parser.parseOperandList(falseValues) ||
      parser.parseColonTypeList(falseValuesTypes) || parser.parseRParen() ||
      parser.resolveOperands(falseValues, falseValuesTypes, falseValuesLoc,
                             result.operands)) {
    return mlir::failure();
  }

  if (parser.parseArrowTypeList(resultTypes)) {
    return mlir::failure();
  }

  result.addTypes(resultTypes);
  return mlir::success();
}

void SelectOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer << "(" << getCondition() << " : " << getCondition().getType() << ")";

  printer << ", ";

  printer << "(" << getTrueValues() << " : " << getTrueValues().getTypes()
          << ")";

  printer << ", ";

  printer << "(" << getFalseValues() << " : " << getFalseValues().getTypes()
          << ")";

  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " -> ";

  auto resultTypes = getResultTypes();

  if (resultTypes.size() == 1) {
    printer << resultTypes;
  } else {
    printer << "(" << resultTypes << ")";
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Built-in operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// AbsOp

namespace mlir::bmodelica {
mlir::OpFoldResult AbsOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      int64_t operandValue = getScalarIntegerLikeValue(operand);
      return getAttr(resultType, std::abs(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::abs(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// AcosOp

namespace mlir::bmodelica {
mlir::OpFoldResult AcosOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::acos(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::acos(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// AsinOp

namespace mlir::bmodelica {
mlir::OpFoldResult AsinOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::asin(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::asin(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// AtanOp

namespace mlir::bmodelica {
mlir::OpFoldResult AtanOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::atan(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::atan(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// CeilOp

namespace mlir::bmodelica {
mlir::OpFoldResult CeilOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      int64_t operandValue = getScalarIntegerLikeValue(operand);
      return getAttr(resultType, operandValue);
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::ceil(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// CosOp

namespace mlir::bmodelica {
mlir::OpFoldResult CosOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::cos(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::cos(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// CoshOp

namespace mlir::bmodelica {
mlir::OpFoldResult CoshOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::cosh(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::cosh(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// DivTruncOp

namespace mlir::bmodelica {
mlir::OpFoldResult DivTruncOp::fold(FoldAdaptor adaptor) {
  auto x = adaptor.getX();
  auto y = adaptor.getY();

  if (!x || !y) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(x) && isScalar(y)) {
    if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
      auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
      auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));
      return getAttr(resultType, xValue / yValue);
    }

    if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
      double xValue = getScalarFloatLikeValue(x);
      double yValue = getScalarFloatLikeValue(y);
      return getAttr(resultType, std::trunc(xValue / yValue));
    }

    if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
      auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
      double yValue = getScalarFloatLikeValue(y);
      return getAttr(resultType, std::trunc(xValue / yValue));
    }

    if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
      double xValue = getScalarFloatLikeValue(x);
      auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));
      return getAttr(resultType, std::trunc(xValue / yValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ExpOp

namespace mlir::bmodelica {
mlir::OpFoldResult ExpOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getExponent();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::exp(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::exp(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// FloorOp

namespace mlir::bmodelica {
mlir::OpFoldResult FloorOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      int64_t operandValue = getScalarIntegerLikeValue(operand);
      return getAttr(resultType, operandValue);
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::floor(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// IntegerOp

namespace mlir::bmodelica {
mlir::OpFoldResult IntegerOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      int64_t operandValue = getScalarIntegerLikeValue(operand);
      return getAttr(resultType, operandValue);
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::floor(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// LogOp

namespace mlir::bmodelica {
mlir::OpFoldResult LogOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::log(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::log(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Log10Op

namespace mlir::bmodelica {
mlir::OpFoldResult Log10Op::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::log10(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::log10(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// MaxOp

namespace mlir::bmodelica {
mlir::LogicalResult MaxOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type firstType = adaptor.getFirst().getType();

  if (adaptor.getSecond()) {
    mlir::Type secondType = adaptor.getSecond().getType();

    if (isScalar(firstType) && isScalar(secondType)) {
      returnTypes.push_back(getMostGenericScalarType(firstType, secondType));
      return mlir::success();
    }

    return mlir::failure();
  }

  if (auto firstShapedType = firstType.dyn_cast<mlir::ShapedType>()) {
    returnTypes.push_back(firstShapedType.getElementType());
    return mlir::success();
  }

  return mlir::failure();
}

bool MaxOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::ParseResult MaxOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand first;
  mlir::Type firstType;

  mlir::OpAsmParser::UnresolvedOperand second;
  mlir::Type secondType;

  size_t numOperands = 1;

  if (parser.parseOperand(first)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    numOperands = 2;

    if (parser.parseOperand(second)) {
      return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  if (parser.parseColon()) {
    return mlir::failure();
  }

  if (numOperands == 1) {
    if (parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands)) {
      return mlir::failure();
    }
  } else {
    if (parser.parseLParen() || parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands) ||
        parser.parseComma() || parser.parseType(secondType) ||
        parser.resolveOperand(second, secondType, result.operands) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  mlir::Type resultType;

  if (parser.parseArrow() || parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);

  return mlir::success();
}

void MaxOp::print(mlir::OpAsmPrinter &printer) {
  printer << getFirst();

  if (getOperation()->getNumOperands() == 2) {
    printer << ", " << getSecond();
  }

  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : ";

  if (getOperation()->getNumOperands() == 1) {
    printer << getFirst().getType();
  } else {
    printer << "(" << getFirst().getType() << ", " << getSecond().getType()
            << ")";
  }

  printer << " -> " << getResult().getType();
}

mlir::OpFoldResult MaxOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getOperands().size() == 2) {
    auto first = adaptor.getFirst();
    auto second = adaptor.getSecond();

    if (!first || !second) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(first) && isScalar(second)) {
      if (isScalarIntegerLike(first) && isScalarIntegerLike(second)) {
        int64_t firstValue = getScalarIntegerLikeValue(first);
        int64_t secondValue = getScalarIntegerLikeValue(second);
        return getAttr(resultType, std::max(firstValue, secondValue));
      }

      if (isScalarFloatLike(first) && isScalarFloatLike(second)) {
        double firstValue = getScalarFloatLikeValue(first);
        double secondValue = getScalarFloatLikeValue(second);
        return getAttr(resultType, std::max(firstValue, secondValue));
      }

      if (isScalarIntegerLike(first) && isScalarFloatLike(second)) {
        auto firstValue = static_cast<double>(getScalarIntegerLikeValue(first));

        double secondValue = getScalarFloatLikeValue(second);

        if (firstValue >= secondValue) {
          return getAttr(resultType, firstValue);
        } else {
          return getAttr(resultType, secondValue);
        }
      }

      if (isScalarFloatLike(first) && isScalarIntegerLike(second)) {
        double firstValue = getScalarFloatLikeValue(first);

        auto secondValue =
            static_cast<double>(getScalarIntegerLikeValue(second));

        if (firstValue >= secondValue) {
          return getAttr(resultType, firstValue);
        } else {
          return getAttr(resultType, secondValue);
        }
      }
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// MinOp

namespace mlir::bmodelica {
mlir::LogicalResult MinOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);
  mlir::Type firstType = adaptor.getFirst().getType();

  if (adaptor.getSecond()) {
    mlir::Type secondType = adaptor.getSecond().getType();

    if (isScalar(firstType) && isScalar(secondType)) {
      returnTypes.push_back(getMostGenericScalarType(firstType, secondType));
      return mlir::success();
    }

    return mlir::failure();
  }

  if (auto firstShapedType = firstType.dyn_cast<mlir::ShapedType>()) {
    returnTypes.push_back(firstShapedType.getElementType());
    return mlir::success();
  }

  return mlir::failure();
}

bool MinOp::isCompatibleReturnTypes(mlir::TypeRange lhs, mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}

mlir::ParseResult MinOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand first;
  mlir::Type firstType;

  mlir::OpAsmParser::UnresolvedOperand second;
  mlir::Type secondType;

  size_t numOperands = 1;

  if (parser.parseOperand(first)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    numOperands = 2;

    if (parser.parseOperand(second)) {
      return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  if (parser.parseColon()) {
    return mlir::failure();
  }

  if (numOperands == 1) {
    if (parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands)) {
      return mlir::failure();
    }
  } else {
    if (parser.parseLParen() || parser.parseType(firstType) ||
        parser.resolveOperand(first, firstType, result.operands) ||
        parser.parseComma() || parser.parseType(secondType) ||
        parser.resolveOperand(second, secondType, result.operands) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  mlir::Type resultType;

  if (parser.parseArrow() || parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);

  return mlir::success();
}

void MinOp::print(mlir::OpAsmPrinter &printer) {
  printer << getFirst();

  if (getOperation()->getNumOperands() == 2) {
    printer << ", " << getSecond();
  }

  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : ";

  if (getOperation()->getNumOperands() == 1) {
    printer << getFirst().getType();
  } else {
    printer << "(" << getFirst().getType() << ", " << getSecond().getType()
            << ")";
  }

  printer << " -> " << getResult().getType();
}

mlir::OpFoldResult MinOp::fold(FoldAdaptor adaptor) {
  if (adaptor.getOperands().size() == 2) {
    auto first = adaptor.getFirst();
    auto second = adaptor.getSecond();

    if (!first || !second) {
      return {};
    }

    auto resultType = getResult().getType();

    if (isScalar(first) && isScalar(second)) {
      if (isScalarIntegerLike(first) && isScalarIntegerLike(second)) {
        int64_t firstValue = getScalarIntegerLikeValue(first);
        int64_t secondValue = getScalarIntegerLikeValue(second);
        return getAttr(resultType, std::min(firstValue, secondValue));
      }

      if (isScalarFloatLike(first) && isScalarFloatLike(second)) {
        double firstValue = getScalarFloatLikeValue(first);
        double secondValue = getScalarFloatLikeValue(second);
        return getAttr(resultType, std::min(firstValue, secondValue));
      }

      if (isScalarIntegerLike(first) && isScalarFloatLike(second)) {
        auto firstValue = static_cast<double>(getScalarIntegerLikeValue(first));

        double secondValue = getScalarFloatLikeValue(second);

        if (firstValue <= secondValue) {
          return getAttr(resultType, firstValue);
        } else {
          return getAttr(resultType, secondValue);
        }
      }

      if (isScalarFloatLike(first) && isScalarIntegerLike(second)) {
        double firstValue = getScalarFloatLikeValue(first);

        auto secondValue =
            static_cast<double>(getScalarIntegerLikeValue(second));

        if (firstValue <= secondValue) {
          return getAttr(resultType, firstValue);
        } else {
          return getAttr(resultType, secondValue);
        }
      }
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ModOp

namespace mlir::bmodelica {
mlir::OpFoldResult ModOp::fold(FoldAdaptor adaptor) {
  auto x = adaptor.getX();
  auto y = adaptor.getY();

  if (!x || !y) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(x) && isScalar(y)) {
    if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
      auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
      auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));

      return getAttr(resultType, xValue - std::floor(xValue / yValue) * yValue);
    }

    if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
      double xValue = getScalarFloatLikeValue(x);
      double yValue = getScalarFloatLikeValue(y);

      return getAttr(resultType, xValue - std::floor(xValue / yValue) * yValue);
    }

    if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
      auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
      double yValue = getScalarFloatLikeValue(y);

      return getAttr(resultType, xValue - std::floor(xValue / yValue) * yValue);
    }

    if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
      double xValue = getScalarFloatLikeValue(x);
      auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));

      return getAttr(resultType, xValue - std::floor(xValue / yValue) * yValue);
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RemOp

namespace mlir::bmodelica {
mlir::OpFoldResult RemOp::fold(FoldAdaptor adaptor) {
  auto x = adaptor.getX();
  auto y = adaptor.getY();

  if (!x || !y) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(x) && isScalar(y)) {
    if (isScalarIntegerLike(x) && isScalarIntegerLike(y)) {
      int64_t xValue = getScalarIntegerLikeValue(x);
      int64_t yValue = getScalarIntegerLikeValue(y);
      return getAttr(resultType, xValue % yValue);
    }

    if (isScalarFloatLike(x) && isScalarFloatLike(y)) {
      double xValue = getScalarFloatLikeValue(x);
      double yValue = getScalarFloatLikeValue(y);
      return getAttr(resultType, std::fmod(xValue, yValue));
    }

    if (isScalarIntegerLike(x) && isScalarFloatLike(y)) {
      auto xValue = static_cast<double>(getScalarIntegerLikeValue(x));
      double yValue = getScalarFloatLikeValue(y);
      return getAttr(resultType, std::fmod(xValue, yValue));
    }

    if (isScalarFloatLike(x) && isScalarIntegerLike(y)) {
      double xValue = getScalarFloatLikeValue(x);
      auto yValue = static_cast<double>(getScalarIntegerLikeValue(y));
      return getAttr(resultType, std::fmod(xValue, yValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SignOp

namespace mlir::bmodelica {
mlir::OpFoldResult SignOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      int64_t value = getScalarIntegerLikeValue(operand);

      if (value == 0) {
        return getAttr(resultType, static_cast<int64_t>(0));
      } else if (value > 0) {
        return getAttr(resultType, static_cast<int64_t>(1));
      } else {
        return getAttr(resultType, static_cast<int64_t>(-1));
      }
    }

    if (isScalarFloatLike(operand)) {
      double value = getScalarFloatLikeValue(operand);

      if (value == 0) {
        return getAttr(resultType, static_cast<int64_t>(0));
      } else if (value > 0) {
        return getAttr(resultType, static_cast<int64_t>(1));
      } else {
        return getAttr(resultType, static_cast<int64_t>(-1));
      }
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SinOp

namespace mlir::bmodelica {
mlir::OpFoldResult SinOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::sin(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::sin(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SinhOp

namespace mlir::bmodelica {
mlir::OpFoldResult SinhOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::sinh(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::sinh(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SizeOp

namespace mlir::bmodelica {
mlir::ParseResult SizeOp::parse(mlir::OpAsmParser &parser,
                                mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand array;
  mlir::Type tensorType;

  mlir::OpAsmParser::UnresolvedOperand dimension;
  mlir::Type dimensionType;

  size_t numOperands = 1;

  if (parser.parseOperand(array)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalComma())) {
    numOperands = 2;

    if (parser.parseOperand(dimension)) {
      return mlir::failure();
    }
  }

  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  if (parser.parseColon()) {
    return mlir::failure();
  }

  if (numOperands == 1) {
    if (parser.parseType(tensorType) ||
        parser.resolveOperand(array, tensorType, result.operands)) {
      return mlir::failure();
    }
  } else {
    if (parser.parseLParen() || parser.parseType(tensorType) ||
        parser.resolveOperand(array, tensorType, result.operands) ||
        parser.parseComma() || parser.parseType(dimensionType) ||
        parser.resolveOperand(dimension, dimensionType, result.operands) ||
        parser.parseRParen()) {
      return mlir::failure();
    }
  }

  mlir::Type resultType;

  if (parser.parseArrow() || parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);

  return mlir::success();
}

void SizeOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << getArray();

  if (getOperation()->getNumOperands() == 2) {
    printer << ", " << getDimension();
  }

  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " : ";

  if (getOperation()->getNumOperands() == 1) {
    printer << getArray().getType();
  } else {
    printer << "(" << getArray().getType() << ", " << getDimension().getType()
            << ")";
  }

  printer << " -> " << getResult().getType();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SqrtOp

namespace mlir::bmodelica {
mlir::OpFoldResult SqrtOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::sqrt(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::sqrt(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// TanOp

namespace mlir::bmodelica {
mlir::OpFoldResult TanOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::tan(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::tan(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// TanhOp

namespace mlir::bmodelica {
mlir::OpFoldResult TanhOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getOperand();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      auto operandValue =
          static_cast<double>(getScalarIntegerLikeValue(operand));

      return getAttr(resultType, std::tanh(operandValue));
    }

    if (isScalarFloatLike(operand)) {
      double operandValue = getScalarFloatLikeValue(operand);
      return getAttr(resultType, std::tanh(operandValue));
    }
  }

  return {};
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// TransposeOp

namespace mlir::bmodelica {
mlir::LogicalResult TransposeOp::inferReturnTypes(
    mlir::MLIRContext *context, std::optional<mlir::Location> location,
    mlir::ValueRange operands, mlir::DictionaryAttr attributes,
    mlir::OpaqueProperties properties, mlir::RegionRange regions,
    llvm::SmallVectorImpl<mlir::Type> &returnTypes) {
  Adaptor adaptor(operands, attributes, properties, regions);

  auto matrixShapedType =
      adaptor.getMatrix().getType().dyn_cast<mlir::ShapedType>();

  if (!matrixShapedType || matrixShapedType.getRank() != 2) {
    return mlir::failure();
  }

  llvm::SmallVector<int64_t, 2> shape;
  shape.push_back(matrixShapedType.getDimSize(1));
  shape.push_back(matrixShapedType.getDimSize(0));

  returnTypes.push_back(mlir::cast<mlir::Type>(matrixShapedType.clone(shape)));

  return mlir::success();
}

bool TransposeOp::isCompatibleReturnTypes(mlir::TypeRange lhs,
                                          mlir::TypeRange rhs) {
  if (lhs.size() != rhs.size()) {
    return false;
  }

  for (auto [lhsType, rhsType] : llvm::zip(lhs, rhs)) {
    if (!areTypesCompatible(lhsType, rhsType)) {
      return false;
    }
  }

  return true;
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ReductionOp

namespace mlir::bmodelica {
mlir::ParseResult ReductionOp::parse(mlir::OpAsmParser &parser,
                                     mlir::OperationState &result) {
  auto loc = parser.getCurrentLocation();
  std::string action;

  if (parser.parseString(&action) || parser.parseComma()) {
    return mlir::failure();
  }

  result.addAttribute(getActionAttrName(result.name),
                      parser.getBuilder().getStringAttr(action));

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> iterables;
  llvm::SmallVector<mlir::Type> iterablesTypes;

  llvm::SmallVector<mlir::OpAsmParser::Argument> inductions;

  mlir::Region *expressionRegion = result.addRegion();
  mlir::Type resultType;

  if (parser.parseKeyword("iterables") || parser.parseEqual() ||
      parser.parseOperandList(iterables, mlir::AsmParser::Delimiter::Square) ||
      parser.parseComma() || parser.parseKeyword("inductions") ||
      parser.parseEqual() ||
      parser.parseArgumentList(inductions, mlir::AsmParser::Delimiter::Square,
                               true) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*expressionRegion, inductions) ||
      parser.parseColon() || parser.parseLParen() ||
      parser.parseTypeList(iterablesTypes) || parser.parseRParen() ||
      parser.resolveOperands(iterables, iterablesTypes, loc, result.operands) ||
      parser.parseArrow() || parser.parseType(resultType)) {
    return mlir::failure();
  }

  result.addTypes(resultType);

  return mlir::success();
}

void ReductionOp::print(mlir::OpAsmPrinter &printer) {
  printer << " \"" << getAction() << "\", iterables = [" << getIterables()
          << "], inductions = [";

  for (size_t i = 0, e = getInductions().size(); i < e; ++i) {
    if (i != 0) {
      printer << ", ";
    }

    printer.printRegionArgument(getInductions()[i]);
  }

  printer << "] ";

  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(getActionAttrName().getValue());

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elidedAttrs);

  printer.printRegion(getExpressionRegion(), false);
  printer << " : ";

  auto iterables = getIterables();
  printer << "(";

  for (size_t i = 0, e = iterables.size(); i < e; ++i) {
    if (i != 0) {
      printer << ", ";
    }

    printer << iterables[i].getType();
  }

  printer << ") -> ";

  printer << getResult().getType();
}

mlir::Block *ReductionOp::createExpressionBlock(mlir::OpBuilder &builder) {
  mlir::OpBuilder::InsertionGuard guard(builder);

  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Location> argLocs;

  for (mlir::Value iterable : getIterables()) {
    auto iterableType = iterable.getType().cast<IterableTypeInterface>();
    argTypes.push_back(iterableType.getInductionType());
    argLocs.push_back(builder.getUnknownLoc());
  }

  return builder.createBlock(&getExpressionRegion(), {}, argTypes, argLocs);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Modeling operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// PackageOp

namespace mlir::bmodelica {
void PackageOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                      llvm::StringRef name) {
  state.addRegion()->emplaceBlock();

  state.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

mlir::ParseResult PackageOp::parse(mlir::OpAsmParser &parser,
                                   mlir::OperationState &result) {
  mlir::StringAttr nameAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  if (bodyRegion->empty()) {
    bodyRegion->emplaceBlock();
  }

  return mlir::success();
}

void PackageOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  printer << " ";

  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elidedAttrs);

  printer.printRegion(getBodyRegion());
}

mlir::Block *PackageOp::bodyBlock() {
  assert(getBodyRegion().hasOneBlock());
  return &getBodyRegion().front();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ModelOp

namespace {
struct InitialModelMergePattern : public mlir::OpRewritePattern<ModelOp> {
  using mlir::OpRewritePattern<ModelOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ModelOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<InitialOp> initialOps;

    for (InitialOp initialOp : op.getOps<InitialOp>()) {
      initialOps.push_back(initialOp);
    }

    if (initialOps.size() <= 1) {
      return mlir::failure();
    }

    for (size_t i = 1, e = initialOps.size(); i < e; ++i) {
      rewriter.mergeBlocks(initialOps[i].getBody(), initialOps[0].getBody());

      rewriter.eraseOp(initialOps[i]);
    }

    return mlir::success();
  }
};

struct MainModelMergePattern : public mlir::OpRewritePattern<ModelOp> {
  using mlir::OpRewritePattern<ModelOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ModelOp op, mlir::PatternRewriter &rewriter) const override {
    llvm::SmallVector<DynamicOp> dynamicOps;

    for (DynamicOp dynamicOp : op.getOps<DynamicOp>()) {
      dynamicOps.push_back(dynamicOp);
    }

    if (dynamicOps.size() <= 1) {
      return mlir::failure();
    }

    for (size_t i = 1, e = dynamicOps.size(); i < e; ++i) {
      rewriter.mergeBlocks(dynamicOps[i].getBody(), dynamicOps[0].getBody());

      rewriter.eraseOp(dynamicOps[i]);
    }

    return mlir::success();
  }
};
} // namespace

namespace mlir::bmodelica {
void ModelOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *context) {
  patterns.add<InitialModelMergePattern, MainModelMergePattern>(context);
}

mlir::RegionKind ModelOp::getRegionKind(unsigned index) {
  return mlir::RegionKind::Graph;
}

void ModelOp::getCleaningPatterns(mlir::RewritePatternSet &patterns,
                                  mlir::MLIRContext *context) {
  getCanonicalizationPatterns(patterns, context);
  EquationTemplateOp::getCanonicalizationPatterns(patterns, context);
  InitialOp::getCanonicalizationPatterns(patterns, context);
  DynamicOp::getCanonicalizationPatterns(patterns, context);
  SCCOp::getCanonicalizationPatterns(patterns, context);
}

void ModelOp::collectVariables(llvm::SmallVectorImpl<VariableOp> &variables) {
  for (VariableOp variableOp : getVariables()) {
    variables.push_back(variableOp);
  }
}

void ModelOp::collectInitialAlgorithms(
    llvm::SmallVectorImpl<AlgorithmOp> &algorithms) {
  for (InitialOp initialOp : getOps<InitialOp>()) {
    initialOp.collectAlgorithms(algorithms);
  }
}

void ModelOp::collectMainAlgorithms(
    llvm::SmallVectorImpl<AlgorithmOp> &algorithms) {
  for (DynamicOp dynamicOp : getOps<DynamicOp>()) {
    dynamicOp.collectAlgorithms(algorithms);
  }
}

void ModelOp::collectInitialSCCs(llvm::SmallVectorImpl<SCCOp> &SCCs) {
  for (InitialOp initialOp : getOps<InitialOp>()) {
    initialOp.collectSCCs(SCCs);
  }
}

void ModelOp::collectMainSCCs(llvm::SmallVectorImpl<SCCOp> &SCCs) {
  for (DynamicOp dynamicOp : getOps<DynamicOp>()) {
    dynamicOp.collectSCCs(SCCs);
  }
}

void ModelOp::collectSCCGroups(
    llvm::SmallVectorImpl<SCCGroupOp> &initialSCCGroups,
    llvm::SmallVectorImpl<SCCGroupOp> &SCCGroups) {
  for (SCCGroupOp op : getOps<SCCGroupOp>()) {
    SCCGroups.push_back(op);
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// OperatorRecordOp

namespace mlir::bmodelica {
void OperatorRecordOp::build(mlir::OpBuilder &builder,
                             mlir::OperationState &state,
                             llvm::StringRef name) {
  state.addRegion()->emplaceBlock();

  state.attributes.push_back(builder.getNamedAttr(
      mlir::SymbolTable::getSymbolAttrName(), builder.getStringAttr(name)));
}

mlir::ParseResult OperatorRecordOp::parse(mlir::OpAsmParser &parser,
                                          mlir::OperationState &result) {
  mlir::StringAttr nameAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes)) {
    return mlir::failure();
  }

  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes) ||
      parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  if (bodyRegion->empty()) {
    bodyRegion->emplaceBlock();
  }

  return mlir::success();
}

void OperatorRecordOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());
  printer << " ";

  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           elidedAttrs);

  printer.printRegion(getBodyRegion());
}

mlir::Block *OperatorRecordOp::bodyBlock() {
  assert(getBodyRegion().hasOneBlock());
  return &getBodyRegion().front();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// StartOp

namespace mlir::bmodelica {
VariableOp StartOp::getVariableOp(mlir::SymbolTableCollection &symbolTable) {
  auto cls = getOperation()->getParentOfType<ClassInterface>();
  return symbolTable.lookupSymbolIn<VariableOp>(cls, getVariableAttr());
}

mlir::LogicalResult
StartOp::getAccesses(llvm::SmallVectorImpl<VariableAccess> &result,
                     mlir::SymbolTableCollection &symbolTable) {
  auto yieldOp = mlir::cast<YieldOp>(getBody()->getTerminator());

  llvm::DenseMap<mlir::Value, unsigned int> inductionsPositionMap;

  if (mlir::failed(searchAccesses(result, symbolTable, inductionsPositionMap,
                                  yieldOp.getValues()[0],
                                  EquationPath(EquationPath::RIGHT, 0)))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult StartOp::searchAccesses(
    llvm::SmallVectorImpl<VariableAccess> &accesses,
    mlir::SymbolTableCollection &symbolTable,
    llvm::DenseMap<mlir::Value, unsigned int> &inductionsPositionMap,
    mlir::Value value, EquationPath path) {
  mlir::Operation *definingOp = value.getDefiningOp();

  if (!definingOp) {
    return mlir::success();
  }

  AdditionalInductions additionalInductions;
  llvm::SmallVector<std::unique_ptr<DimensionAccess>, 10> dimensionAccesses;

  if (auto expressionInt =
          mlir::dyn_cast<EquationExpressionOpInterface>(definingOp)) {
    return expressionInt.getEquationAccesses(
        accesses, symbolTable, inductionsPositionMap, additionalInductions,
        dimensionAccesses, std::move(path));
  }

  return mlir::failure();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// DefaultOp

namespace mlir::bmodelica {
VariableOp DefaultOp::getVariableOp(mlir::SymbolTableCollection &symbolTable) {
  auto cls = getOperation()->getParentOfType<ClassInterface>();
  return symbolTable.lookupSymbolIn<VariableOp>(cls, getVariableAttr());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// BindingEquationOp

namespace mlir::bmodelica {
VariableOp
BindingEquationOp::getVariableOp(mlir::SymbolTableCollection &symbolTable) {
  auto cls = getOperation()->getParentOfType<ClassInterface>();
  return symbolTable.lookupSymbolIn<VariableOp>(cls, getVariableAttr());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ForEquationOp

namespace {
struct EmptyForEquationOpErasePattern
    : public mlir::OpRewritePattern<ForEquationOp> {
  using mlir::OpRewritePattern<ForEquationOp>::OpRewritePattern;

  mlir::LogicalResult match(ForEquationOp op) const override {
    return mlir::LogicalResult::success(op.getOps().empty());
  }

  void rewrite(ForEquationOp op,
               mlir::PatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
  }
};
} // namespace

namespace mlir::bmodelica {
void ForEquationOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          long from, long to, long step) {
  mlir::OpBuilder::InsertionGuard guard(builder);

  state.addAttribute(getFromAttrName(state.name), builder.getIndexAttr(from));

  state.addAttribute(getToAttrName(state.name), builder.getIndexAttr(to));

  state.addAttribute(getStepAttrName(state.name), builder.getIndexAttr(step));

  mlir::Region *bodyRegion = state.addRegion();

  builder.createBlock(bodyRegion, {}, builder.getIndexType(),
                      builder.getUnknownLoc());
}

void ForEquationOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<EmptyForEquationOpErasePattern>(context);
}

mlir::Block *ForEquationOp::bodyBlock() {
  assert(getBodyRegion().getBlocks().size() == 1);
  return &getBodyRegion().front();
}

mlir::Value ForEquationOp::induction() {
  assert(getBodyRegion().getNumArguments() != 0);
  return getBodyRegion().getArgument(0);
}

mlir::ParseResult ForEquationOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  auto &builder = parser.getBuilder();

  mlir::OpAsmParser::Argument induction;

  int64_t from;
  int64_t to;
  int64_t step = 1;

  if (parser.parseArgument(induction) || parser.parseEqual() ||
      parser.parseInteger(from) || parser.parseKeyword("to") ||
      parser.parseInteger(to)) {
    return mlir::failure();
  }

  if (mlir::succeeded(parser.parseOptionalKeyword("step"))) {
    if (parser.parseInteger(step)) {
      return mlir::failure();
    }
  }

  induction.type = builder.getIndexType();

  result.attributes.append("from", builder.getIndexAttr(from));
  result.attributes.append("to", builder.getIndexAttr(to));
  result.attributes.append("step", builder.getIndexAttr(step));

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseRegion(*bodyRegion, induction)) {
    return mlir::failure();
  }

  return mlir::success();
}

void ForEquationOp::print(mlir::OpAsmPrinter &printer) {
  printer << " " << induction() << " = " << getFrom() << " to " << getTo();

  if (auto step = getStep(); step != 1) {
    printer << " step " << step;
  }

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs(),
                                           {"from", "to", "step"});

  printer << " ";
  printer.printRegion(getBodyRegion(), false);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// EquationTemplateOp

namespace {
struct UnusedEquationTemplatePattern
    : public mlir::OpRewritePattern<EquationTemplateOp> {
  using mlir::OpRewritePattern<EquationTemplateOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EquationTemplateOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->use_empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

namespace mlir::bmodelica {
mlir::ParseResult EquationTemplateOp::parse(mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::Argument, 3> inductions;
  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseKeyword("inductions") || parser.parseEqual() ||
      parser.parseArgumentList(inductions,
                               mlir::OpAsmParser::Delimiter::Square) ||
      parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  for (auto &induction : inductions) {
    induction.type = parser.getBuilder().getIndexType();
  }

  if (parser.parseRegion(*bodyRegion, inductions)) {
    return mlir::failure();
  }

  if (bodyRegion->empty()) {
    mlir::OpBuilder builder(bodyRegion);

    llvm::SmallVector<mlir::Type, 3> argTypes(inductions.size(),
                                              builder.getIndexType());

    llvm::SmallVector<mlir::Location, 3> argLocs(inductions.size(),
                                                 builder.getUnknownLoc());

    builder.createBlock(bodyRegion, {}, argTypes, argLocs);
  }

  result.addTypes(EquationType::get(parser.getContext()));
  return mlir::success();
}

void EquationTemplateOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer << "inductions = [";
  printer.printOperands(getInductionVariables());
  printer << "]";
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  printer << " ";
  printer.printRegion(getBodyRegion(), false);
}

void EquationTemplateOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<UnusedEquationTemplatePattern>(context);
}

mlir::Block *EquationTemplateOp::createBody(unsigned int numOfInductions) {
  mlir::OpBuilder builder(getContext());

  llvm::SmallVector<mlir::Type, 3> argTypes(numOfInductions,
                                            builder.getIndexType());

  llvm::SmallVector<mlir::Location, 3> argLocs(numOfInductions,
                                               builder.getUnknownLoc());

  return builder.createBlock(&getBodyRegion(), {}, argTypes, argLocs);
}

void EquationTemplateOp::printInline(llvm::raw_ostream &os) {
  llvm::DenseMap<mlir::Value, int64_t> inductions;
  auto inductionVars = getInductionVariables();

  for (size_t i = 0, e = inductionVars.size(); i < e; ++i) {
    inductions[inductionVars[i]] = static_cast<int64_t>(i);
  }

  if (auto expressionOp = mlir::cast<EquationExpressionOpInterface>(
          getBody()->getTerminator())) {
    expressionOp.printExpression(os, inductions);
  }
}

mlir::ValueRange EquationTemplateOp::getInductionVariables() {
  return getBodyRegion().getArguments();
}

llvm::SmallVector<mlir::Value>
EquationTemplateOp::getInductionVariablesAtPath(const EquationPath &path) {
  llvm::SmallVector<mlir::Value> result;
  auto equationInductions = getInductionVariables();
  result.append(equationInductions.begin(), equationInductions.end());

  mlir::Block *bodyBlock = getBody();
  EquationPath::EquationSide side = path.getEquationSide();

  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(bodyBlock->getTerminator());

  mlir::Value value = side == EquationPath::LEFT
                          ? equationSidesOp.getLhsValues()[path[0]]
                          : equationSidesOp.getRhsValues()[path[0]];

  for (size_t i = 1, e = path.size(); i < e; ++i) {
    mlir::Operation *op = value.getDefiningOp();
    assert(op != nullptr && "Invalid equation path");
    auto expressionInt = mlir::cast<EquationExpressionOpInterface>(op);
    auto additionalInductions = expressionInt.getAdditionalInductions();
    result.append(additionalInductions.begin(), additionalInductions.end());
    value = expressionInt.getExpressionElement(path[i]);
  }

  return result;
}

llvm::DenseMap<mlir::Value, unsigned int>
EquationTemplateOp::getInductionsPositionMap() {
  mlir::ValueRange inductionVariables = getInductionVariables();
  llvm::DenseMap<mlir::Value, unsigned int> inductionsPositionMap;

  for (auto inductionVariable : llvm::enumerate(inductionVariables)) {
    inductionsPositionMap[inductionVariable.value()] =
        inductionVariable.index();
  }

  return inductionsPositionMap;
}

mlir::LogicalResult
EquationTemplateOp::getAccesses(llvm::SmallVectorImpl<VariableAccess> &result,
                                mlir::SymbolTableCollection &symbolTable) {
  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(getBody()->getTerminator());

  // Get the induction variables and number them.
  auto inductionsPositionMap = getInductionsPositionMap();

  // Search the accesses starting from the left-hand side of the equation.
  if (mlir::failed(searchAccesses(result, symbolTable, inductionsPositionMap,
                                  equationSidesOp.getLhsValues()[0],
                                  EquationPath(EquationPath::LEFT, 0)))) {
    return mlir::failure();
  }

  // Search the accesses starting from the right-hand side of the equation.
  if (mlir::failed(searchAccesses(result, symbolTable, inductionsPositionMap,
                                  equationSidesOp.getRhsValues()[0],
                                  EquationPath(EquationPath::RIGHT, 0)))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EquationTemplateOp::getWriteAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    const IndexSet &equationIndices, llvm::ArrayRef<VariableAccess> accesses,
    const VariableAccess &matchedAccess) {
  const AccessFunction &matchedAccessFunction =
      matchedAccess.getAccessFunction();

  IndexSet matchedVariableIndices = matchedAccessFunction.map(equationIndices);

  for (const VariableAccess &access : accesses) {
    if (access.getVariable() != matchedAccess.getVariable()) {
      continue;
    }

    const AccessFunction &accessFunction = access.getAccessFunction();

    IndexSet accessedVariableIndices = accessFunction.map(equationIndices);

    if (matchedVariableIndices.empty() && accessedVariableIndices.empty()) {
      result.push_back(access);
    } else if (matchedVariableIndices.overlaps(accessedVariableIndices)) {
      result.push_back(access);
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationTemplateOp::getReadAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    const IndexSet &equationIndices, llvm::ArrayRef<VariableAccess> accesses,
    const VariableAccess &matchedAccess) {
  const AccessFunction &matchedAccessFunction =
      matchedAccess.getAccessFunction();

  IndexSet matchedVariableIndices = matchedAccessFunction.map(equationIndices);

  for (const VariableAccess &access : accesses) {
    if (access.getVariable() != matchedAccess.getVariable()) {
      result.push_back(access);
    } else {
      const AccessFunction &accessFunction = access.getAccessFunction();

      IndexSet accessedVariableIndices = accessFunction.map(equationIndices);

      if (!matchedVariableIndices.empty() && !accessedVariableIndices.empty()) {
        if (!matchedVariableIndices.contains(accessedVariableIndices)) {
          result.push_back(access);
        }
      }
    }
  }

  return mlir::success();
}

mlir::Value EquationTemplateOp::getValueAtPath(const EquationPath &path) {
  mlir::Block *bodyBlock = getBody();
  EquationPath::EquationSide side = path.getEquationSide();

  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(bodyBlock->getTerminator());

  mlir::Value value = side == EquationPath::LEFT
                          ? equationSidesOp.getLhsValues()[path[0]]
                          : equationSidesOp.getRhsValues()[path[0]];

  for (size_t i = 1, e = path.size(); i < e; ++i) {
    mlir::Operation *op = value.getDefiningOp();
    assert(op != nullptr && "Invalid equation path");
    auto expressionInt = mlir::cast<EquationExpressionOpInterface>(op);
    value = expressionInt.getExpressionElement(path[i]);
  }

  return value;
}

std::optional<VariableAccess>
EquationTemplateOp::getAccessAtPath(mlir::SymbolTableCollection &symbolTable,
                                    const EquationPath &path) {
  // Get the induction variables and number them.
  mlir::ValueRange inductionVariables = getInductionVariables();
  llvm::DenseMap<mlir::Value, unsigned int> inductionsPositionMap;

  for (auto inductionVariable : llvm::enumerate(inductionVariables)) {
    inductionsPositionMap[inductionVariable.value()] =
        inductionVariable.index();
  }

  // Get the access.
  llvm::SmallVector<VariableAccess, 1> accesses;
  mlir::Value access = getValueAtPath(path);

  if (mlir::failed(searchAccesses(accesses, symbolTable, inductionsPositionMap,
                                  access, path))) {
    return std::nullopt;
  }

  assert(accesses.size() == 1);
  return accesses[0];
}

mlir::LogicalResult EquationTemplateOp::searchAccesses(
    llvm::SmallVectorImpl<VariableAccess> &accesses,
    mlir::SymbolTableCollection &symbolTable,
    llvm::DenseMap<mlir::Value, unsigned int> &explicitInductionsPositionMap,
    mlir::Value value, EquationPath path) {
  mlir::Operation *definingOp = value.getDefiningOp();

  if (!definingOp) {
    return mlir::success();
  }

  AdditionalInductions additionalInductions;
  llvm::SmallVector<std::unique_ptr<DimensionAccess>, 10> dimensionAccesses;

  if (auto expressionInt =
          mlir::dyn_cast<EquationExpressionOpInterface>(definingOp)) {
    return expressionInt.getEquationAccesses(
        accesses, symbolTable, explicitInductionsPositionMap,
        additionalInductions, dimensionAccesses, std::move(path));
  }

  return mlir::failure();
}

mlir::LogicalResult EquationTemplateOp::cloneWithReplacedAccess(
    mlir::RewriterBase &rewriter,
    std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
    const VariableAccess &access, EquationTemplateOp replacementEquation,
    const VariableAccess &replacementAccess,
    llvm::SmallVectorImpl<std::pair<IndexSet, EquationTemplateOp>> &results) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Erase the operations in case of unrecoverable failure.
  auto cleanOnFailure = llvm::make_scope_exit([&]() {
    for (const auto &result : results) {
      rewriter.eraseOp(result.second);
    }
  });

  // The set of indices that are yet to be processed.
  IndexSet remainingEquationIndices;

  if (equationIndices) {
    remainingEquationIndices = equationIndices->get();
  }

  // Determine the access functions.
  mlir::Value destinationValue = getValueAtPath(access.getPath());
  int64_t destinationRank = 0;

  if (auto destinationShapedType =
          destinationValue.getType().dyn_cast<mlir::ShapedType>()) {
    destinationRank = destinationShapedType.getRank();
  }

  mlir::Value sourceValue =
      replacementEquation.getValueAtPath(replacementAccess.getPath());

  int64_t sourceRank = 0;

  if (auto sourceShapedType =
          sourceValue.getType().dyn_cast<mlir::ShapedType>()) {
    sourceRank = sourceShapedType.getRank();
  }

  if (destinationRank > sourceRank) {
    // The access to be replaced requires indices of the variables that are
    // potentially not handled by the source equation.
    return mlir::failure();
  }

  auto destinationAccessFunction = access.getAccessFunction().clone();

  // The extra subscription indices to be applied to the replacement value.
  llvm::SmallVector<mlir::Value> additionalSubscriptionIndices;

  if (destinationRank < sourceRank) {
    // The access to be replaced specifies more indices than the ones given
    // by the source equation. This means that the source equation writes to
    // more indices than the requested ones. Inlining the source equation
    // results in possibly wasted additional computations, but does lead to
    // a correct result.

    auto destinationDimensionAccesses =
        destinationAccessFunction->getGeneralizedAccesses();

    destinationAccessFunction =
        AccessFunction::build(destinationAccessFunction->getContext(),
                              destinationAccessFunction->getNumOfDims(),
                              llvm::ArrayRef(destinationDimensionAccesses)
                                  .drop_back(sourceRank - destinationRank));

    // If the destination access has more indices than the source one,
    // then collect the additional ones and apply them to the
    // replacement value.
    int64_t rankDifference = sourceRank - destinationRank;
    mlir::Operation *replacedValueOp = destinationValue.getDefiningOp();

    auto allAdditionalIndicesCollected = [&]() -> bool {
      return rankDifference ==
             static_cast<int64_t>(additionalSubscriptionIndices.size());
    };

    while (mlir::isa<TensorExtractOp, TensorViewOp>(replacedValueOp) &&
           !allAdditionalIndicesCollected()) {
      if (auto extractOp = mlir::dyn_cast<TensorExtractOp>(replacedValueOp)) {
        size_t numOfIndices = extractOp.getIndices().size();

        for (size_t i = 0; i < numOfIndices && !allAdditionalIndicesCollected();
             ++i) {
          additionalSubscriptionIndices.push_back(
              extractOp.getIndices()[numOfIndices - i - 1]);
        }

        replacedValueOp = extractOp.getTensor().getDefiningOp();
        continue;
      }

      if (auto viewOp = mlir::dyn_cast<TensorViewOp>(replacedValueOp)) {
        size_t numOfSubscripts = viewOp.getSubscriptions().size();

        for (size_t i = 0;
             i < numOfSubscripts && !allAdditionalIndicesCollected(); ++i) {
          additionalSubscriptionIndices.push_back(
              viewOp.getSubscriptions()[numOfSubscripts - i - 1]);
        }

        replacedValueOp = viewOp.getSource().getDefiningOp();
        continue;
      }

      return mlir::failure();
    }

    assert(allAdditionalIndicesCollected());

    // Indices have been collected in reverse order, due to the bottom-up
    // visit of the operations tree.
    std::reverse(additionalSubscriptionIndices.begin(),
                 additionalSubscriptionIndices.end());
  }

  VariableAccess destinationAccess(access.getPath(), access.getVariable(),
                                   std::move(destinationAccessFunction));

  // Try to perform a vectorized replacement first.
  if (mlir::failed(cloneWithReplacedVectorizedAccess(
          rewriter, equationIndices, access, replacementEquation,
          replacementAccess, additionalSubscriptionIndices, results,
          remainingEquationIndices))) {
    return mlir::failure();
  }

  // Perform scalar replacements on the remaining equation indices.
  // TODO
  // for (Point scalarEquationIndices : remainingEquationIndices) {
  //}

  if (remainingEquationIndices.empty()) {
    cleanOnFailure.release();
    return mlir::success();
  }

  return mlir::failure();
}

mlir::LogicalResult EquationTemplateOp::cloneWithReplacedVectorizedAccess(
    mlir::RewriterBase &rewriter,
    std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
    const VariableAccess &access, EquationTemplateOp replacementEquation,
    const VariableAccess &replacementAccess,
    llvm::ArrayRef<mlir::Value> additionalSubscriptions,
    llvm::SmallVectorImpl<std::pair<IndexSet, EquationTemplateOp>> &results,
    IndexSet &remainingEquationIndices) {
  const AccessFunction &destinationAccessFunction = access.getAccessFunction();

  const AccessFunction &sourceAccessFunction =
      replacementAccess.getAccessFunction();

  auto transformation = getReplacementTransformationAccess(
      destinationAccessFunction, sourceAccessFunction);

  if (transformation) {
    if (mlir::failed(cloneWithReplacedVectorizedAccess(
            rewriter, equationIndices, access, replacementEquation,
            replacementAccess, *transformation, additionalSubscriptions,
            results, remainingEquationIndices))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationTemplateOp::cloneWithReplacedVectorizedAccess(
    mlir::RewriterBase &rewriter,
    std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
    const VariableAccess &access, EquationTemplateOp replacementEquation,
    const VariableAccess &replacementAccess,
    const AccessFunction &transformation,
    llvm::ArrayRef<mlir::Value> additionalSubscriptions,
    llvm::SmallVectorImpl<std::pair<IndexSet, EquationTemplateOp>> &results,
    IndexSet &remainingEquationIndices) {
  if (equationIndices && !equationIndices->get().empty()) {
    for (const MultidimensionalRange &range :
         llvm::make_range(equationIndices->get().rangesBegin(),
                          equationIndices->get().rangesEnd())) {
      if (mlir::failed(cloneWithReplacedVectorizedAccess(
              rewriter, std::reference_wrapper(range), access,
              replacementEquation, replacementAccess, transformation,
              additionalSubscriptions, results, remainingEquationIndices))) {
        return mlir::failure();
      }
    }

    return mlir::success();
  }

  return cloneWithReplacedVectorizedAccess(
      rewriter,
      std::optional<std::reference_wrapper<const MultidimensionalRange>>(
          std::nullopt),
      access, replacementEquation, replacementAccess, transformation,
      additionalSubscriptions, results, remainingEquationIndices);
}

mlir::LogicalResult EquationTemplateOp::cloneWithReplacedVectorizedAccess(
    mlir::RewriterBase &rewriter,
    std::optional<std::reference_wrapper<const MultidimensionalRange>>
        equationIndices,
    const VariableAccess &access, EquationTemplateOp replacementEquation,
    const VariableAccess &replacementAccess,
    const AccessFunction &transformation,
    llvm::ArrayRef<mlir::Value> additionalSubscriptions,
    llvm::SmallVectorImpl<std::pair<IndexSet, EquationTemplateOp>> &results,
    IndexSet &remainingEquationIndices) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(getOperation());
  mlir::IRMapping mapping;

  // Create the equation template.
  auto newEquationTemplateOp = rewriter.create<EquationTemplateOp>(getLoc());
  newEquationTemplateOp->setAttrs(getOperation()->getAttrDictionary());

  if (equationIndices) {
    remainingEquationIndices -= equationIndices->get();
    results.emplace_back(IndexSet(equationIndices->get()),
                         newEquationTemplateOp);
  } else {
    results.emplace_back(IndexSet(), newEquationTemplateOp);
  }

  mlir::Block *newEquationBodyBlock =
      newEquationTemplateOp.createBody(getInductionVariables().size());

  rewriter.setInsertionPointToStart(newEquationBodyBlock);

  // The optional additional subscription indices.
  llvm::SmallVector<mlir::Value, 3> additionalMappedSubscriptions;

  // Clone the operations composing the destination equation.
  for (auto [oldInduction, newInduction] :
       llvm::zip(getInductionVariables(),
                 newEquationTemplateOp.getInductionVariables())) {
    mapping.map(oldInduction, newInduction);
  }

  for (auto &op : getOps()) {
    rewriter.clone(op, mapping);
  }

  mlir::Value originalReplacedValue = getValueAtPath(access.getPath());
  mlir::Value mappedReplacedValue = mapping.lookup(originalReplacedValue);
  rewriter.setInsertionPointAfterValue(mappedReplacedValue);

  // Clone the operations composing the replacement equation.
  if (mlir::failed(mapInductionVariables(
          rewriter, replacementEquation.getLoc(), mapping, replacementEquation,
          newEquationTemplateOp, access.getPath(), transformation))) {
    return mlir::failure();
  }

  for (auto &replacementOp : replacementEquation.getOps()) {
    if (!mlir::isa<EquationSideOp, EquationSidesOp>(replacementOp)) {
      rewriter.clone(replacementOp, mapping);
    }
  }

  // Get the replacement value.
  mlir::Value replacement = mapping.lookup(replacementEquation.getValueAtPath(
      EquationPath(EquationPath::RIGHT, replacementAccess.getPath()[0])));

  rewriter.replaceAllUsesWith(mappedReplacedValue, replacement);
  return mlir::success();
}

std::unique_ptr<AccessFunction>
EquationTemplateOp::getReplacementTransformationAccess(
    const AccessFunction &destinationAccess,
    const AccessFunction &sourceAccess) {
  if (auto sourceInverseAccess = sourceAccess.inverse()) {
    return destinationAccess.combine(*sourceInverseAccess);
  }

  // Check if the source access is invertible by removing the constant
  // accesses.

  if (!sourceAccess.isAffine() || !destinationAccess.isAffine()) {
    return nullptr;
  }

  // Determine the constant results to be removed.
  mlir::AffineMap sourceAffineMap = sourceAccess.getAffineMap();
  llvm::SmallVector<int64_t, 3> constantExprPositions;

  for (size_t i = 0, e = sourceAffineMap.getNumResults(); i < e; ++i) {
    if (mlir::isa<mlir::AffineConstantExpr>(sourceAffineMap.getResult(i))) {
      constantExprPositions.push_back(i);
    }
  }

  // Compute the reduced access functions.
  auto reducedSourceAccessFunction =
      AccessFunction::build(mlir::compressUnusedDims(
          sourceAccess.getAffineMap().dropResults(constantExprPositions)));

  auto reducedSourceInverseAccessFunction =
      reducedSourceAccessFunction->inverse();

  if (!reducedSourceInverseAccessFunction) {
    return nullptr;
  }

  auto reducedDestinationAccessFunction = AccessFunction::build(
      destinationAccess.getAffineMap().dropResults(constantExprPositions));

  auto combinedReducedAccess = reducedDestinationAccessFunction->combine(
      *reducedSourceInverseAccessFunction);

  mlir::AffineMap combinedAffineMap = combinedReducedAccess->getAffineMap();

  return AccessFunction::build(mlir::AffineMap::get(
      destinationAccess.getNumOfDims(), 0, combinedAffineMap.getResults(),
      combinedAffineMap.getContext()));
}

mlir::LogicalResult EquationTemplateOp::mapInductionVariables(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::IRMapping &mapping,
    EquationTemplateOp source, EquationTemplateOp destination,
    const EquationPath &destinationPath, const AccessFunction &transformation) {
  if (!transformation.isAffine()) {
    return mlir::failure();
  }

  mlir::AffineMap affineMap = transformation.getAffineMap();

  if (affineMap.getNumResults() < source.getInductionVariables().size()) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value> affineMapResults;

  auto inductionVariables =
      destination.getInductionVariablesAtPath(destinationPath);

  if (mlir::failed(materializeAffineMap(
          builder, loc, affineMap, inductionVariables, affineMapResults))) {
    return mlir::failure();
  }

  auto sourceInductionVariables = source.getInductionVariables();

  for (size_t i = 0, e = sourceInductionVariables.size(); i < e; ++i) {
    mapping.map(sourceInductionVariables[i], affineMapResults[i]);
  }

  return mlir::success();
}

IndexSet EquationTemplateOp::applyAccessFunction(
    const AccessFunction &accessFunction,
    std::optional<MultidimensionalRange> equationIndices,
    const EquationPath &path) {
  IndexSet result;

  if (equationIndices) {
    result = accessFunction.map(IndexSet(*equationIndices));
  }

  return result;
}

mlir::LogicalResult EquationTemplateOp::explicitate(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection,
    std::optional<MultidimensionalRange> equationIndices,
    const EquationPath &path) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  // Get all the paths that lead to accesses with the same accessed variable
  // and function.
  auto requestedAccess = getAccessAtPath(symbolTableCollection, path);

  if (!requestedAccess) {
    return mlir::failure();
  }

  const AccessFunction &requestedAccessFunction =
      requestedAccess->getAccessFunction();

  IndexSet requestedIndices =
      applyAccessFunction(requestedAccessFunction, equationIndices, path);

  llvm::SmallVector<VariableAccess, 10> accesses;

  if (mlir::failed(getAccesses(accesses, symbolTableCollection))) {
    return mlir::failure();
  }

  llvm::SmallVector<VariableAccess, 5> filteredAccesses;

  for (const VariableAccess &access : accesses) {
    if (requestedAccess->getVariable() != access.getVariable()) {
      continue;
    }

    const AccessFunction &currentAccessFunction = access.getAccessFunction();

    IndexSet currentIndices = applyAccessFunction(
        currentAccessFunction, equationIndices, access.getPath());

    if (requestedIndices == currentIndices) {
      filteredAccesses.push_back(access);
    }
  }

  assert(!filteredAccesses.empty());

  // If there is only one access, then it is sufficient to follow the path
  // and invert the operations.

  auto terminator = mlir::cast<EquationSidesOp>(getBody()->getTerminator());

  auto lhsOp = terminator.getLhs().getDefiningOp();
  auto rhsOp = terminator.getRhs().getDefiningOp();

  rewriter.setInsertionPoint(lhsOp);

  if (rhsOp->isBeforeInBlock(lhsOp)) {
    rewriter.setInsertionPoint(rhsOp);
  }

  if (filteredAccesses.size() == 1) {
    for (size_t i = 1, e = path.size(); i < e; ++i) {
      if (mlir::failed(
              explicitateLeaf(rewriter, path[i], path.getEquationSide()))) {
        return mlir::failure();
      }
    }

    if (path.getEquationSide() == EquationPath::RIGHT) {
      llvm::SmallVector<mlir::Value, 1> lhsValues;
      llvm::SmallVector<mlir::Value, 1> rhsValues;

      rewriter.setInsertionPointAfter(terminator);

      rewriter.create<EquationSidesOp>(
          terminator->getLoc(), terminator.getRhs(), terminator.getLhs());

      rewriter.eraseOp(terminator);
    }
  } else {
    // If there are multiple accesses, then we must group all of them and
    // extract the common multiplying factor.

    if (mlir::failed(groupLeftHandSide(rewriter, symbolTableCollection,
                                       equationIndices, *requestedAccess))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

EquationTemplateOp EquationTemplateOp::cloneAndExplicitate(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection,
    std::optional<MultidimensionalRange> equationIndices,
    const EquationPath &path) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(getOperation());

  auto clonedOp =
      mlir::cast<EquationTemplateOp>(rewriter.clone(*getOperation()));

  auto cleanOnFailure =
      llvm::make_scope_exit([&]() { rewriter.eraseOp(clonedOp); });

  if (mlir::failed(clonedOp.explicitate(rewriter, symbolTableCollection,
                                        equationIndices, path))) {
    return nullptr;
  }

  cleanOnFailure.release();
  return clonedOp;
}

mlir::LogicalResult
EquationTemplateOp::explicitateLeaf(mlir::RewriterBase &rewriter,
                                    size_t argumentIndex,
                                    EquationPath::EquationSide side) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  auto equationSidesOp =
      mlir::cast<EquationSidesOp>(getBody()->getTerminator());

  mlir::Value oldLhsValue = equationSidesOp.getLhsValues()[0];
  mlir::Value oldRhsValue = equationSidesOp.getRhsValues()[0];

  mlir::Value toExplicitate =
      side == EquationPath::LEFT ? oldLhsValue : oldRhsValue;

  mlir::Value otherExp =
      side == EquationPath::RIGHT ? oldLhsValue : oldRhsValue;

  mlir::Operation *op = toExplicitate.getDefiningOp();
  auto invertibleOp = mlir::dyn_cast<InvertibleOpInterface>(op);

  if (!invertibleOp) {
    return mlir::failure();
  }

  rewriter.setInsertionPoint(invertibleOp);

  if (auto otherExpOp = otherExp.getDefiningOp();
      otherExpOp && invertibleOp->isBeforeInBlock(otherExpOp)) {
    rewriter.setInsertionPointAfter(otherExpOp);
  }

  mlir::Value invertedOpResult =
      invertibleOp.inverse(rewriter, argumentIndex, otherExp);

  if (!invertedOpResult) {
    return mlir::failure();
  }

  llvm::SmallVector<mlir::Value, 1> newLhsValues;
  llvm::SmallVector<mlir::Value, 1> newRhsValues;

  if (side == EquationPath::LEFT) {
    newLhsValues.push_back(op->getOperand(argumentIndex));
  } else {
    newLhsValues.push_back(invertedOpResult);
  }

  if (side == EquationPath::LEFT) {
    newRhsValues.push_back(invertedOpResult);
  } else {
    newRhsValues.push_back(op->getOperand(argumentIndex));
  }

  // Create the new terminator.
  rewriter.setInsertionPoint(equationSidesOp);

  auto oldLhs =
      mlir::cast<EquationSideOp>(equationSidesOp.getLhs().getDefiningOp());

  auto oldRhs =
      mlir::cast<EquationSideOp>(equationSidesOp.getRhs().getDefiningOp());

  rewriter.replaceOpWithNewOp<EquationSideOp>(oldLhs, newLhsValues);
  rewriter.replaceOpWithNewOp<EquationSideOp>(oldRhs, newRhsValues);

  return mlir::success();
}

static mlir::LogicalResult removeSubtractions(mlir::RewriterBase &rewriter,
                                              mlir::Operation *root) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Operation *op = root;

  if (!op) {
    return mlir::success();
  }

  if (!mlir::isa<SubscriptionOp>(op) && !mlir::isa<LoadOp>(op)) {
    for (mlir::Value operand : op->getOperands()) {
      if (mlir::failed(removeSubtractions(rewriter, operand.getDefiningOp()))) {
        return mlir::failure();
      }
    }
  }

  if (auto subOp = mlir::dyn_cast<SubOp>(op)) {
    rewriter.setInsertionPoint(subOp);
    mlir::Value rhs = subOp.getRhs();

    mlir::Value negatedRhs =
        rewriter.create<NegateOp>(rhs.getLoc(), rhs.getType(), rhs);

    rewriter.replaceOpWithNewOp<AddOp>(subOp, subOp.getResult().getType(),
                                       subOp.getLhs(), negatedRhs);
  }

  return mlir::success();
}

static mlir::LogicalResult distributeMulAndDivOps(mlir::RewriterBase &rewriter,
                                                  mlir::Operation *root) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Operation *op = root;

  if (!op) {
    return mlir::success();
  }

  for (auto operand : op->getOperands()) {
    if (mlir::failed(
            distributeMulAndDivOps(rewriter, operand.getDefiningOp()))) {
      return mlir::failure();
    }
  }

  if (auto distributableOp = mlir::dyn_cast<DistributableOpInterface>(op)) {
    if (!mlir::isa<NegateOp>(op)) {
      rewriter.setInsertionPoint(distributableOp);
      llvm::SmallVector<mlir::Value, 1> results;

      if (mlir::succeeded(distributableOp.distribute(results, rewriter))) {
        for (size_t i = 0, e = distributableOp->getNumResults(); i < e; ++i) {
          mlir::Value oldValue = distributableOp->getResult(i);
          mlir::Value newValue = results[i];
          rewriter.replaceAllUsesWith(oldValue, newValue);
        }
      }
    }
  }

  return mlir::success();
}

static mlir::LogicalResult pushNegateOps(mlir::RewriterBase &rewriter,
                                         mlir::Operation *root) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  mlir::Operation *op = root;

  if (!op) {
    return mlir::success();
  }

  for (mlir::Value operand : op->getOperands()) {
    if (mlir::failed(pushNegateOps(rewriter, operand.getDefiningOp()))) {
      return mlir::failure();
    }
  }

  if (auto distributableOp = mlir::dyn_cast<NegateOp>(op)) {
    rewriter.setInsertionPoint(distributableOp);
    llvm::SmallVector<mlir::Value, 1> results;

    if (mlir::succeeded(distributableOp.distribute(results, rewriter))) {
      rewriter.replaceOp(distributableOp, results);
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationTemplateOp::collectSummedValues(
    llvm::SmallVectorImpl<std::pair<mlir::Value, EquationPath>> &result,
    mlir::Value root, EquationPath path) {
  if (auto definingOp = root.getDefiningOp()) {
    if (auto addOp = mlir::dyn_cast<AddOp>(definingOp)) {
      if (mlir::failed(collectSummedValues(result, addOp.getLhs(), path + 0))) {
        return mlir::failure();
      }

      if (mlir::failed(collectSummedValues(result, addOp.getRhs(), path + 1))) {
        return mlir::failure();
      }

      return mlir::success();
    }
  }

  result.push_back(std::make_pair(root, path));
  return mlir::success();
}

static void foldValue(mlir::RewriterBase &rewriter, mlir::Value value,
                      mlir::Block *block) {
  mlir::OperationFolder helper(value.getContext());
  llvm::SmallVector<mlir::Operation *> visitStack;
  llvm::SmallVector<mlir::Operation *, 3> ops;
  llvm::DenseSet<mlir::Operation *> processed;

  if (auto definingOp = value.getDefiningOp()) {
    visitStack.push_back(definingOp);
  }

  while (!visitStack.empty()) {
    auto op = visitStack.pop_back_val();
    ops.push_back(op);

    for (const auto &operand : op->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        visitStack.push_back(definingOp);
      }
    }
  }

  llvm::SmallVector<mlir::Operation *, 3> constants;

  for (mlir::Operation *op : llvm::reverse(ops)) {
    if (processed.contains(op)) {
      continue;
    }

    processed.insert(op);

    if (mlir::failed(helper.tryToFold(op))) {
      break;
    }
  }

  for (auto *op : llvm::reverse(constants)) {
    op->moveBefore(block, block->begin());
  }
}

static std::optional<bool> isZeroAttr(mlir::Attribute attribute) {
  if (auto booleanAttr = attribute.dyn_cast<BooleanAttr>()) {
    return !booleanAttr.getValue();
  }

  if (auto integerAttr = attribute.dyn_cast<IntegerAttr>()) {
    return integerAttr.getValue() == 0;
  }

  if (auto realAttr = attribute.dyn_cast<RealAttr>()) {
    return realAttr.getValue().isZero();
  }

  if (auto integerAttr = attribute.cast<mlir::IntegerAttr>()) {
    return integerAttr.getValue() == 0;
  }

  if (auto floatAttr = attribute.cast<mlir::FloatAttr>()) {
    return floatAttr.getValueAsDouble() == 0;
  }

  return std::nullopt;
}

std::optional<std::pair<unsigned int, mlir::Value>>
EquationTemplateOp::getMultiplyingFactor(
    mlir::OpBuilder &builder,
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::DenseMap<mlir::Value, unsigned int> &inductionsPositionMap,
    const IndexSet &equationIndices, mlir::Value value,
    llvm::StringRef variable, const IndexSet &variableIndices,
    EquationPath path) {
  mlir::OpBuilder::InsertionGuard guard(builder);

  auto isAccessToVarFn = [&](mlir::Value value, llvm::StringRef variable) {
    mlir::Operation *definingOp = value.getDefiningOp();

    if (!definingOp) {
      return false;
    }

    while (definingOp) {
      if (auto op = mlir::dyn_cast<VariableGetOp>(definingOp)) {
        return op.getVariable() == variable;
      }

      if (auto op = mlir::dyn_cast<TensorExtractOp>(definingOp)) {
        definingOp = op.getTensor().getDefiningOp();
        continue;
      }

      if (auto op = mlir::dyn_cast<TensorViewOp>(definingOp)) {
        definingOp = op.getSource().getDefiningOp();
        continue;
      }

      return false;
    }

    return false;
  };

  if (isAccessToVarFn(value, variable)) {
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(searchAccesses(accesses, symbolTableCollection,
                                    inductionsPositionMap, value, path)) ||
        accesses.size() != 1) {
      return std::nullopt;
    }

    if (accesses[0].getVariable().getRootReference() == variable) {
      const AccessFunction &accessFunction = accesses[0].getAccessFunction();
      auto accessedIndices = accessFunction.map(equationIndices);

      if (variableIndices == accessedIndices) {
        if (auto constantMaterializableType =
                value.getType()
                    .dyn_cast<ConstantMaterializableTypeInterface>()) {

          mlir::Value one = constantMaterializableType.materializeIntConstant(
              builder, value.getLoc(), 1);

          return std::make_pair(static_cast<unsigned int>(1), one);
        }

        return std::nullopt;
      }
    }
  }

  mlir::Operation *op = value.getDefiningOp();

  if (auto constantOp = mlir::dyn_cast<ConstantOp>(op)) {
    return std::make_pair(static_cast<unsigned int>(0), constantOp.getResult());
  }

  if (auto negateOp = mlir::dyn_cast<NegateOp>(op)) {
    auto operand = getMultiplyingFactor(
        builder, symbolTableCollection, inductionsPositionMap, equationIndices,
        negateOp.getOperand(), variable, variableIndices, path + 0);

    if (!operand) {
      return std::nullopt;
    }

    if (!operand->second) {
      return std::nullopt;
    }

    mlir::Value result = builder.create<NegateOp>(
        negateOp.getLoc(), negateOp.getResult().getType(), operand->second);

    return std::make_pair(operand->first, result);
  }

  if (auto mulOp = mlir::dyn_cast<MulOp>(op)) {
    auto lhs = getMultiplyingFactor(
        builder, symbolTableCollection, inductionsPositionMap, equationIndices,
        mulOp.getLhs(), variable, variableIndices, path + 0);

    auto rhs = getMultiplyingFactor(
        builder, symbolTableCollection, inductionsPositionMap, equationIndices,
        mulOp.getRhs(), variable, variableIndices, path + 1);

    if (!lhs || !rhs) {
      return std::nullopt;
    }

    if (!lhs->second || !rhs->second) {
      return std::make_pair(static_cast<unsigned int>(0), mlir::Value());
    }

    mlir::Value result = builder.create<MulOp>(
        mulOp.getLoc(), mulOp.getResult().getType(), lhs->second, rhs->second);

    return std::make_pair(lhs->first + rhs->first, result);
  }

  auto hasAccessToVar = [&](mlir::Value value,
                            EquationPath path) -> std::optional<bool> {
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(searchAccesses(accesses, symbolTableCollection,
                                    inductionsPositionMap, value, path))) {
      return std::nullopt;
    }

    bool hasAccess = llvm::any_of(accesses, [&](const VariableAccess &access) {
      if (access.getVariable().getRootReference().getValue() != variable) {
        return false;
      }

      const AccessFunction &accessFunction = access.getAccessFunction();
      IndexSet accessedIndices = accessFunction.map(equationIndices);

      if (accessedIndices.empty() && variableIndices.empty()) {
        return true;
      }

      return accessedIndices.overlaps(variableIndices);
    });

    if (hasAccess) {
      return true;
    }

    return false;
  };

  if (auto divOp = mlir::dyn_cast<DivOp>(op)) {
    auto dividend = getMultiplyingFactor(
        builder, symbolTableCollection, inductionsPositionMap, equationIndices,
        divOp.getLhs(), variable, variableIndices, path + 0);

    if (!dividend) {
      return std::nullopt;
    }

    if (!dividend->second) {
      return dividend;
    }

    // Check that the right-hand side value has no access to the variable
    // of interest.
    auto rhsHasAccess = hasAccessToVar(divOp.getRhs(), path + 1);

    if (!rhsHasAccess || *rhsHasAccess) {
      return std::nullopt;
    }

    mlir::Value result =
        builder.create<DivOp>(divOp.getLoc(), divOp.getResult().getType(),
                              dividend->second, divOp.getRhs());

    return std::make_pair(dividend->first, result);
  }

  // Check that the value is not the result of an operation using the
  // variable of interest. If it has such access, then we are not able to
  // extract the multiplying factor.
  if (hasAccessToVar(value, path)) {
    return std::make_pair(static_cast<unsigned int>(1), mlir::Value());
  }

  return std::make_pair(static_cast<unsigned int>(0), value);
}

bool EquationTemplateOp::checkAccessEquivalence(
    const IndexSet &equationIndices, const VariableAccess &firstAccess,
    const VariableAccess &secondAccess) {
  const AccessFunction &firstAccessFunction = firstAccess.getAccessFunction();

  const AccessFunction &secondAccessFunction = secondAccess.getAccessFunction();

  IndexSet firstIndices = firstAccessFunction.map(equationIndices);
  IndexSet secondIndices = secondAccessFunction.map(equationIndices);

  if (firstIndices.empty() && secondIndices.empty()) {
    return true;
  }

  if (firstAccessFunction == secondAccessFunction) {
    return true;
  }

  if (firstIndices.flatSize() == 1 && firstIndices == secondIndices) {
    return true;
  }

  return false;
}

mlir::LogicalResult EquationTemplateOp::groupLeftHandSide(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection,
    std::optional<MultidimensionalRange> equationRanges,
    const VariableAccess &requestedAccess) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  auto inductionsPositionMap = getInductionsPositionMap();
  uint64_t viewElementIndex = requestedAccess.getPath()[0];

  IndexSet equationIndices;

  if (equationRanges) {
    equationIndices += *equationRanges;
  }

  auto requestedValue = getValueAtPath(requestedAccess.getPath());

  // Determine whether the access to be grouped is inside both the equation's
  // sides or just one of them. When the requested access is found, also
  // check that the path goes through linear operations. If not,
  // explicitation is not possible.
  bool lhsHasAccess = false;
  bool rhsHasAccess = false;

  llvm::SmallVector<VariableAccess> accesses;

  if (mlir::failed(getAccesses(accesses, symbolTableCollection))) {
    return mlir::failure();
  }

  const AccessFunction &requestedAccessFunction =
      requestedAccess.getAccessFunction();

  auto requestedIndices = requestedAccessFunction.map(equationIndices);

  for (const VariableAccess &access : accesses) {
    if (access.getVariable() != requestedAccess.getVariable()) {
      continue;
    }

    const AccessFunction &currentAccessFunction = access.getAccessFunction();
    auto currentAccessIndices = currentAccessFunction.map(equationIndices);

    if ((requestedIndices.empty() && currentAccessIndices.empty()) ||
        requestedIndices.overlaps(currentAccessIndices)) {
      if (!checkAccessEquivalence(equationIndices, requestedAccess, access)) {
        return mlir::failure();
      }

      EquationPath::EquationSide side = access.getPath().getEquationSide();
      lhsHasAccess |= side == EquationPath::LEFT;
      rhsHasAccess |= side == EquationPath::RIGHT;
    }
  }

  // Convert the expression to a sum of values.
  auto convertToSumsFn =
      [&](std::function<std::pair<mlir::Value, EquationPath>()> rootFn)
      -> mlir::LogicalResult {
    if (auto root = rootFn(); mlir::failed(
            removeSubtractions(rewriter, root.first.getDefiningOp()))) {
      return mlir::failure();
    }

    if (auto root = rootFn(); mlir::failed(
            distributeMulAndDivOps(rewriter, root.first.getDefiningOp()))) {
      return mlir::failure();
    }

    if (auto root = rootFn();
        mlir::failed(pushNegateOps(rewriter, root.first.getDefiningOp()))) {
      return mlir::failure();
    }

    return mlir::success();
  };

  llvm::SmallVector<std::pair<mlir::Value, EquationPath>> lhsSummedValues;
  llvm::SmallVector<std::pair<mlir::Value, EquationPath>> rhsSummedValues;

  if (lhsHasAccess) {
    auto rootFn = [&]() -> std::pair<mlir::Value, EquationPath> {
      auto equationSidesOp =
          mlir::cast<EquationSidesOp>(getBody()->getTerminator());

      return std::make_pair(equationSidesOp.getLhsValues()[viewElementIndex],
                            EquationPath(EquationPath::LEFT, viewElementIndex));
    };

    if (mlir::failed(convertToSumsFn(rootFn))) {
      return mlir::failure();
    }

    if (auto root = rootFn(); mlir::failed(
            collectSummedValues(lhsSummedValues, root.first, root.second))) {
      return mlir::failure();
    }
  }

  if (rhsHasAccess) {
    auto rootFn = [&]() -> std::pair<mlir::Value, EquationPath> {
      auto equationSidesOp =
          mlir::cast<EquationSidesOp>(getBody()->getTerminator());

      return std::make_pair(
          equationSidesOp.getRhsValues()[viewElementIndex],
          EquationPath(EquationPath::RIGHT, viewElementIndex));
    };

    if (mlir::failed(convertToSumsFn(rootFn))) {
      return mlir::failure();
    }

    if (auto root = rootFn(); mlir::failed(
            collectSummedValues(rhsSummedValues, root.first, root.second))) {
      return mlir::failure();
    }
  }

  auto containsAccessFn =
      [&](bool &result, mlir::Value value, EquationPath path,
          const VariableAccess &access) -> mlir::LogicalResult {
    llvm::SmallVector<VariableAccess> accesses;

    if (mlir::failed(searchAccesses(accesses, symbolTableCollection,
                                    inductionsPositionMap, value, path))) {
      return mlir::failure();
    }

    const AccessFunction &accessFunction = access.getAccessFunction();

    result = llvm::any_of(accesses, [&](const VariableAccess &acc) {
      if (acc.getVariable() != access.getVariable()) {
        return false;
      }

      IndexSet requestedIndices = accessFunction.map(equationIndices);

      const AccessFunction &currentAccessFunction = acc.getAccessFunction();
      auto currentIndices = currentAccessFunction.map(equationIndices);

      assert(requestedIndices == currentIndices ||
             !requestedIndices.overlaps(currentIndices));
      return requestedIndices == currentIndices;
    });

    return mlir::success();
  };

  auto groupFactorsFn = [&](auto beginIt, auto endIt) -> mlir::Value {
    mlir::Value result = rewriter.create<ConstantOp>(
        getOperation()->getLoc(), RealAttr::get(rewriter.getContext(), 0));

    for (auto it = beginIt; it != endIt; ++it) {
      auto factor = getMultiplyingFactor(
          rewriter, symbolTableCollection, inductionsPositionMap,
          equationIndices, it->first,
          requestedAccess.getVariable().getRootReference().getValue(),
          requestedIndices, it->second);

      if (!factor) {
        return nullptr;
      }

      if (!factor->second || factor->first > 1) {
        return nullptr;
      }

      result = rewriter.create<AddOp>(
          it->first.getLoc(),
          getMostGenericScalarType(result.getType(), it->first.getType()),
          result, factor->second);
    }

    return result;
  };

  auto groupRemainingFn = [&](auto beginIt, auto endIt) -> mlir::Value {
    auto zeroConstantOp = rewriter.create<ConstantOp>(
        getOperation()->getLoc(), RealAttr::get(rewriter.getContext(), 0));

    mlir::Value result = zeroConstantOp.getResult();

    for (auto it = beginIt; it != endIt; ++it) {
      mlir::Value value = it->first;

      result = rewriter.create<AddOp>(
          value.getLoc(),
          getMostGenericScalarType(result.getType(), value.getType()), result,
          value);
    }

    return result;
  };

  if (lhsHasAccess && rhsHasAccess) {
    bool error = false;

    auto leftPos = llvm::partition(lhsSummedValues, [&](const auto &value) {
      bool result = false;

      if (mlir::failed(containsAccessFn(result, value.first, value.second,
                                        requestedAccess))) {
        error = true;
        return false;
      }

      return result;
    });

    auto rightPos = llvm::partition(rhsSummedValues, [&](const auto &value) {
      bool result = false;

      if (mlir::failed(containsAccessFn(result, value.first, value.second,
                                        requestedAccess))) {
        error = true;
        return false;
      }

      return result;
    });

    if (error) {
      return mlir::failure();
    }

    mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);
    mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

    if (!lhsFactor || !rhsFactor) {
      return mlir::failure();
    }

    mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());
    mlir::Value rhsRemaining =
        groupRemainingFn(rightPos, rhsSummedValues.end());

    auto rhs = rewriter.create<DivOp>(
        getLoc(), requestedValue.getType(),
        rewriter.create<SubOp>(getLoc(),
                               getMostGenericScalarType(rhsRemaining.getType(),
                                                        lhsRemaining.getType()),
                               rhsRemaining, lhsRemaining),
        rewriter.create<SubOp>(
            getLoc(),
            getMostGenericScalarType(lhsFactor.getType(), rhsFactor.getType()),
            lhsFactor, rhsFactor));

    // Check if we are dividing by zero.
    foldValue(rewriter, rhs.getRhs(), getBody());

    if (auto divisorOp =
            mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
      std::optional<bool> isZero = isZeroAttr(divisorOp.getValue());

      if (!isZero || *isZero) {
        return mlir::failure();
      }
    }

    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(getBody()->getTerminator());

    auto lhsOp = equationSidesOp.getLhs().getDefiningOp<EquationSideOp>();
    auto rhsOp = equationSidesOp.getRhs().getDefiningOp<EquationSideOp>();

    auto oldLhsValues = lhsOp.getValues();
    llvm::SmallVector<mlir::Value> newLhsValues(oldLhsValues.begin(),
                                                oldLhsValues.end());

    auto oldRhsValues = rhsOp.getValues();
    llvm::SmallVector<mlir::Value> newRhsValues(oldRhsValues.begin(),
                                                oldRhsValues.end());

    newLhsValues[viewElementIndex] = requestedValue;
    newRhsValues[viewElementIndex] = rhs.getResult();

    rewriter.setInsertionPoint(lhsOp);
    rewriter.replaceOpWithNewOp<EquationSideOp>(lhsOp, newLhsValues);

    rewriter.setInsertionPoint(rhsOp);
    rewriter.replaceOpWithNewOp<EquationSideOp>(rhsOp, newRhsValues);

    return mlir::success();
  }

  if (lhsHasAccess) {
    bool error = false;

    auto leftPos = llvm::partition(lhsSummedValues, [&](const auto &value) {
      bool result = false;

      if (mlir::failed(containsAccessFn(result, value.first, value.second,
                                        requestedAccess))) {
        error = true;
        return false;
      }

      return result;
    });

    if (error) {
      return mlir::failure();
    }

    mlir::Value lhsFactor = groupFactorsFn(lhsSummedValues.begin(), leftPos);

    if (!lhsFactor) {
      return mlir::failure();
    }

    mlir::Value lhsRemaining = groupRemainingFn(leftPos, lhsSummedValues.end());

    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(getBody()->getTerminator());

    auto rhs = rewriter.create<DivOp>(
        getLoc(), requestedValue.getType(),
        rewriter.create<SubOp>(
            getLoc(),
            getMostGenericScalarType(
                equationSidesOp.getRhsValues()[viewElementIndex].getType(),
                lhsRemaining.getType()),
            equationSidesOp.getRhsValues()[viewElementIndex], lhsRemaining),
        lhsFactor);

    // Check if we are dividing by zero.
    foldValue(rewriter, rhs.getRhs(), getBody());

    if (auto divisorOp =
            mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
      std::optional<bool> isZero = isZeroAttr(divisorOp.getValue());

      if (!isZero || *isZero) {
        return mlir::failure();
      }
    }

    auto lhsOp = equationSidesOp.getLhs().getDefiningOp<EquationSideOp>();
    auto rhsOp = equationSidesOp.getRhs().getDefiningOp<EquationSideOp>();

    auto oldLhsValues = lhsOp.getValues();
    llvm::SmallVector<mlir::Value> newLhsValues(oldLhsValues.begin(),
                                                oldLhsValues.end());

    auto oldRhsValues = rhsOp.getValues();
    llvm::SmallVector<mlir::Value> newRhsValues(oldRhsValues.begin(),
                                                oldRhsValues.end());

    newLhsValues[viewElementIndex] = requestedValue;
    newRhsValues[viewElementIndex] = rhs.getResult();

    rewriter.setInsertionPoint(lhsOp);
    rewriter.replaceOpWithNewOp<EquationSideOp>(lhsOp, newLhsValues);

    rewriter.setInsertionPoint(rhsOp);
    rewriter.replaceOpWithNewOp<EquationSideOp>(rhsOp, newRhsValues);

    return mlir::success();
  }

  if (rhsHasAccess) {
    bool error = false;

    auto rightPos = llvm::partition(rhsSummedValues, [&](const auto &value) {
      bool result = false;

      if (mlir::failed(containsAccessFn(result, value.first, value.second,
                                        requestedAccess))) {
        error = true;
        return false;
      }

      return result;
    });

    if (error) {
      return mlir::failure();
    }

    mlir::Value rhsFactor = groupFactorsFn(rhsSummedValues.begin(), rightPos);

    if (!rhsFactor) {
      return mlir::failure();
    }

    mlir::Value rhsRemaining =
        groupRemainingFn(rightPos, rhsSummedValues.end());

    auto equationSidesOp =
        mlir::cast<EquationSidesOp>(getBody()->getTerminator());

    auto rhs = rewriter.create<DivOp>(
        getLoc(), requestedValue.getType(),
        rewriter.create<SubOp>(
            getLoc(),
            getMostGenericScalarType(
                equationSidesOp.getLhsValues()[viewElementIndex].getType(),
                rhsRemaining.getType()),
            equationSidesOp.getLhsValues()[viewElementIndex], rhsRemaining),
        rhsFactor);

    // Check if we are dividing by zero.
    foldValue(rewriter, rhs.getRhs(), getBody());

    if (auto divisorOp =
            mlir::dyn_cast<ConstantOp>(rhs.getRhs().getDefiningOp())) {
      std::optional<bool> isZero = isZeroAttr(divisorOp.getValue());

      if (!isZero || *isZero) {
        return mlir::failure();
      }
    }

    auto lhsOp = equationSidesOp.getLhs().getDefiningOp<EquationSideOp>();
    auto rhsOp = equationSidesOp.getRhs().getDefiningOp<EquationSideOp>();

    auto oldLhsValues = lhsOp.getValues();
    llvm::SmallVector<mlir::Value> newLhsValues(oldLhsValues.begin(),
                                                oldLhsValues.end());

    auto oldRhsValues = rhsOp.getValues();
    llvm::SmallVector<mlir::Value> newRhsValues(oldRhsValues.begin(),
                                                oldRhsValues.end());

    newLhsValues[viewElementIndex] = requestedValue;
    newRhsValues[viewElementIndex] = rhs.getResult();

    rewriter.setInsertionPoint(lhsOp);
    rewriter.replaceOpWithNewOp<EquationSideOp>(lhsOp, newLhsValues);

    rewriter.setInsertionPoint(rhsOp);
    rewriter.replaceOpWithNewOp<EquationSideOp>(rhsOp, newRhsValues);

    return mlir::success();
  }

  llvm_unreachable("Access not found");
  return mlir::failure();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// EquationFunctionOp

namespace mlir::bmodelica {
mlir::ParseResult EquationFunctionOp::parse(mlir::OpAsmParser &parser,
                                            mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

void EquationFunctionOp::print(OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

void EquationFunctionOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &state,
                               llvm::StringRef name, uint64_t numOfInductions,
                               llvm::ArrayRef<mlir::NamedAttribute> attrs,
                               llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  llvm::SmallVector<mlir::Type> argTypes(numOfInductions * 2,
                                         builder.getIndexType());

  auto functionType = builder.getFunctionType(argTypes, std::nullopt);

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     mlir::TypeAttr::get(functionType));

  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty()) {
    return;
  }

  assert(functionType.getNumInputs() == argAttrs.size());

  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, std::nullopt, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));
}

mlir::Value EquationFunctionOp::getLowerBound(uint64_t induction) {
  return getArgument(induction * 2);
}

mlir::Value EquationFunctionOp::getUpperBound(uint64_t induction) {
  return getArgument(induction * 2 + 1);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// InitialOp

namespace {
struct EmptyInitialModelPattern : public mlir::OpRewritePattern<InitialOp> {
  using mlir::OpRewritePattern<InitialOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(InitialOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getBody()->empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

namespace mlir::bmodelica {
void InitialOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::MLIRContext *context) {
  patterns.add<EmptyInitialModelPattern>(context);
}

void InitialOp::collectSCCs(llvm::SmallVectorImpl<SCCOp> &SCCs) {
  for (SCCOp scc : getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }
}

void InitialOp::collectAlgorithms(
    llvm::SmallVectorImpl<AlgorithmOp> &algorithms) {
  for (AlgorithmOp algorithm : getOps<AlgorithmOp>()) {
    algorithms.push_back(algorithm);
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// DynamicOp

namespace {
struct EmptyMainModelPattern : public mlir::OpRewritePattern<DynamicOp> {
  using mlir::OpRewritePattern<DynamicOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(DynamicOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.getBody()->empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

namespace mlir::bmodelica {
void DynamicOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::MLIRContext *context) {
  patterns.add<EmptyMainModelPattern>(context);
}

void DynamicOp::collectSCCs(llvm::SmallVectorImpl<SCCOp> &SCCs) {
  for (SCCOp scc : getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }
}

void DynamicOp::collectAlgorithms(
    llvm::SmallVectorImpl<AlgorithmOp> &algorithms) {
  for (AlgorithmOp algorithm : getOps<AlgorithmOp>()) {
    algorithms.push_back(algorithm);
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// StartEquationInstanceOp

namespace mlir::bmodelica {
void StartEquationInstanceOp::build(mlir::OpBuilder &builder,
                                    mlir::OperationState &state,
                                    EquationTemplateOp equationTemplate) {
  build(builder, state, equationTemplate.getResult(), nullptr);
}

mlir::LogicalResult StartEquationInstanceOp::verify() {
  auto indicesRank =
      [&](std::optional<MultidimensionalRangeAttr> ranges) -> size_t {
    if (!ranges) {
      return 0;
    }

    return ranges->getValue().rank();
  };

  // Check the indices for the explicit inductions.
  size_t numOfExplicitInductions = getInductionVariables().size();

  if (size_t explicitIndicesRank = indicesRank(getIndices());
      numOfExplicitInductions != explicitIndicesRank) {
    return emitOpError() << "Unexpected rank of iteration indices (expected "
                         << numOfExplicitInductions << ", got "
                         << explicitIndicesRank << ")";
  }

  return mlir::success();
}

EquationTemplateOp StartEquationInstanceOp::getTemplate() {
  auto result = getBase().getDefiningOp<EquationTemplateOp>();
  assert(result != nullptr);
  return result;
}

void StartEquationInstanceOp::printInline(llvm::raw_ostream &os) {
  getTemplate().printInline(os);
}

mlir::ValueRange StartEquationInstanceOp::getInductionVariables() {
  return getTemplate().getInductionVariables();
}

IndexSet StartEquationInstanceOp::getIterationSpace() {
  if (auto indices = getIndices()) {
    return {indices->getValue()};
  }

  return {};
}

std::optional<VariableAccess> StartEquationInstanceOp::getWriteAccess(
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getAccessAtPath(symbolTableCollection,
                         EquationPath(EquationPath::LEFT, 0));
}

mlir::LogicalResult StartEquationInstanceOp::getAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTable) {
  return getTemplate().getAccesses(result, symbolTable);
}

mlir::LogicalResult StartEquationInstanceOp::getReadAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::ArrayRef<VariableAccess> accesses) {
  return getReadAccesses(result, symbolTableCollection, getIterationSpace(),
                         accesses);
}

mlir::LogicalResult StartEquationInstanceOp::getReadAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    const IndexSet &equationIndices, llvm::ArrayRef<VariableAccess> accesses) {
  std::optional<VariableAccess> writeAccess =
      getWriteAccess(symbolTableCollection);

  if (!writeAccess) {
    return mlir::failure();
  }

  return getTemplate().getReadAccesses(result, equationIndices, accesses,
                                       *writeAccess);
}

std::optional<VariableAccess> StartEquationInstanceOp::getAccessAtPath(
    mlir::SymbolTableCollection &symbolTable, const EquationPath &path) {
  return getTemplate().getAccessAtPath(symbolTable, path);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// EquationInstanceOp

namespace mlir::bmodelica {
void EquationInstanceOp::build(mlir::OpBuilder &builder,
                               mlir::OperationState &state,
                               EquationTemplateOp equationTemplate) {
  build(builder, state, equationTemplate.getResult(), nullptr);
}

mlir::LogicalResult EquationInstanceOp::verify() {
  auto indicesRank =
      [&](std::optional<MultidimensionalRangeAttr> ranges) -> size_t {
    if (!ranges) {
      return 0;
    }

    return ranges->getValue().rank();
  };

  // Check the indices for the explicit inductions.
  size_t numOfExplicitInductions = getInductionVariables().size();

  if (size_t explicitIndicesRank = indicesRank(getIndices());
      numOfExplicitInductions != explicitIndicesRank) {
    return emitOpError() << "Unexpected rank of iteration indices (expected "
                         << numOfExplicitInductions << ", got "
                         << explicitIndicesRank << ")";
  }

  return mlir::success();
}

EquationTemplateOp EquationInstanceOp::getTemplate() {
  auto result = getBase().getDefiningOp<EquationTemplateOp>();
  assert(result != nullptr);
  return result;
}

void EquationInstanceOp::printInline(llvm::raw_ostream &os) {
  getTemplate().printInline(os);
}

mlir::ValueRange EquationInstanceOp::getInductionVariables() {
  return getTemplate().getInductionVariables();
}

IndexSet EquationInstanceOp::getIterationSpace() {
  if (auto indices = getIndices()) {
    return IndexSet(indices->getValue());
  }

  return {};
}

mlir::LogicalResult
EquationInstanceOp::getAccesses(llvm::SmallVectorImpl<VariableAccess> &result,
                                mlir::SymbolTableCollection &symbolTable) {
  return getTemplate().getAccesses(result, symbolTable);
}

std::optional<VariableAccess>
EquationInstanceOp::getAccessAtPath(mlir::SymbolTableCollection &symbolTable,
                                    const EquationPath &path) {
  return getTemplate().getAccessAtPath(symbolTable, path);
}

mlir::LogicalResult EquationInstanceOp::cloneWithReplacedAccess(
    mlir::RewriterBase &rewriter,
    std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
    const VariableAccess &access, EquationTemplateOp replacementEquation,
    const VariableAccess &replacementAccess,
    llvm::SmallVectorImpl<EquationInstanceOp> &results) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  auto cleanTemplatesOnExit = llvm::make_scope_exit([&]() {
    llvm::SmallVector<EquationTemplateOp> templateOps;

    for (EquationInstanceOp equationOp : results) {
      templateOps.push_back(equationOp.getTemplate());
    }

    (void)cleanEquationTemplates(rewriter, templateOps);
  });

  llvm::SmallVector<std::pair<IndexSet, EquationTemplateOp>> templateResults;

  if (mlir::failed(getTemplate().cloneWithReplacedAccess(
          rewriter, equationIndices, access, replacementEquation,
          replacementAccess, templateResults))) {
    return mlir::failure();
  }

  rewriter.setInsertionPointAfter(getOperation());

  auto temporaryClonedOp =
      mlir::cast<EquationInstanceOp>(rewriter.clone(*getOperation()));

  for (auto &[assignedIndices, equationTemplateOp] : templateResults) {
    if (assignedIndices.empty()) {
      auto clonedOp = mlir::cast<EquationInstanceOp>(
          rewriter.clone(*temporaryClonedOp.getOperation()));

      clonedOp.setOperand(equationTemplateOp.getResult());
      clonedOp.removeIndicesAttr();
      results.push_back(clonedOp);
    } else {
      for (const MultidimensionalRange &assignedIndicesRange : llvm::make_range(
               assignedIndices.rangesBegin(), assignedIndices.rangesEnd())) {
        auto clonedOp = mlir::cast<EquationInstanceOp>(
            rewriter.clone(*temporaryClonedOp.getOperation()));

        clonedOp.setOperand(equationTemplateOp.getResult());

        if (auto explicitIndices = getIndices()) {
          MultidimensionalRange explicitRange =
              assignedIndicesRange.takeFirstDimensions(
                  explicitIndices->getValue().rank());

          clonedOp.setIndicesAttr(MultidimensionalRangeAttr::get(
              rewriter.getContext(), std::move(explicitRange)));
        }

        results.push_back(clonedOp);
      }
    }
  }

  rewriter.eraseOp(temporaryClonedOp);
  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// MatchedEquationInstanceOp

namespace mlir::bmodelica {
void MatchedEquationInstanceOp::build(mlir::OpBuilder &builder,
                                      mlir::OperationState &state,
                                      EquationTemplateOp equationTemplate,
                                      EquationPathAttr path) {
  build(builder, state, equationTemplate.getResult(), nullptr, path);
}

mlir::LogicalResult MatchedEquationInstanceOp::verify() {
  auto indicesRank =
      [&](std::optional<MultidimensionalRangeAttr> ranges) -> size_t {
    if (!ranges) {
      return 0;
    }

    return ranges->getValue().rank();
  };

  // Check the indices for the explicit inductions.
  size_t numOfExplicitInductions = getInductionVariables().size();

  if (size_t explicitIndicesRank = indicesRank(getIndices());
      numOfExplicitInductions != explicitIndicesRank) {
    return emitOpError() << "Unexpected rank of iteration indices (expected "
                         << numOfExplicitInductions << ", got "
                         << explicitIndicesRank << ")";
  }

  return mlir::success();
}

EquationTemplateOp MatchedEquationInstanceOp::getTemplate() {
  auto result = getBase().getDefiningOp<EquationTemplateOp>();
  assert(result != nullptr);
  return result;
}

void MatchedEquationInstanceOp::printInline(llvm::raw_ostream &os) {
  getTemplate().printInline(os);
}

mlir::ValueRange MatchedEquationInstanceOp::getInductionVariables() {
  return getTemplate().getInductionVariables();
}

IndexSet MatchedEquationInstanceOp::getIterationSpace() {
  if (auto indices = getIndices()) {
    return IndexSet(indices->getValue());
  }

  return {};
}

std::optional<VariableAccess> MatchedEquationInstanceOp::getMatchedAccess(
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getAccessAtPath(symbolTableCollection, getPath().getValue());
}

mlir::LogicalResult MatchedEquationInstanceOp::getAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTable) {
  return getTemplate().getAccesses(result, symbolTable);
}

mlir::LogicalResult MatchedEquationInstanceOp::getWriteAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::ArrayRef<VariableAccess> accesses) {
  return getWriteAccesses(result, symbolTableCollection, getIterationSpace(),
                          accesses);
}

mlir::LogicalResult MatchedEquationInstanceOp::getWriteAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    const IndexSet &equationIndices, llvm::ArrayRef<VariableAccess> accesses) {
  std::optional<VariableAccess> matchedAccess =
      getMatchedAccess(symbolTableCollection);

  if (!matchedAccess) {
    return mlir::failure();
  }

  return getTemplate().getWriteAccesses(result, equationIndices, accesses,
                                        *matchedAccess);
}

mlir::LogicalResult MatchedEquationInstanceOp::getReadAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::ArrayRef<VariableAccess> accesses) {
  return getReadAccesses(result, symbolTableCollection, getIterationSpace(),
                         accesses);
}

mlir::LogicalResult MatchedEquationInstanceOp::getReadAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    const IndexSet &equationIndices, llvm::ArrayRef<VariableAccess> accesses) {
  std::optional<VariableAccess> matchedAccess =
      getMatchedAccess(symbolTableCollection);

  if (!matchedAccess) {
    return mlir::failure();
  }

  return getTemplate().getReadAccesses(result, equationIndices, accesses,
                                       *matchedAccess);
}

std::optional<VariableAccess> MatchedEquationInstanceOp::getAccessAtPath(
    mlir::SymbolTableCollection &symbolTable, const EquationPath &path) {
  return getTemplate().getAccessAtPath(symbolTable, path);
}

mlir::LogicalResult MatchedEquationInstanceOp::explicitate(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  std::optional<MultidimensionalRange> indices = std::nullopt;

  if (auto indicesAttr = getIndices()) {
    indices = indicesAttr->getValue();
  }

  if (mlir::failed(getTemplate().explicitate(rewriter, symbolTableCollection,
                                             indices, getPath().getValue()))) {
    return mlir::failure();
  }

  setPathAttr(
      EquationPathAttr::get(getContext(), EquationPath(EquationPath::LEFT, 0)));

  return mlir::success();
}

MatchedEquationInstanceOp MatchedEquationInstanceOp::cloneAndExplicitate(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  std::optional<MultidimensionalRange> indices = std::nullopt;

  if (auto indicesAttr = getIndices()) {
    indices = indicesAttr->getValue();
  }

  EquationTemplateOp clonedTemplate = getTemplate().cloneAndExplicitate(
      rewriter, symbolTableCollection, indices, getPath().getValue());

  if (!clonedTemplate) {
    return nullptr;
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(getOperation());

  auto result = rewriter.create<MatchedEquationInstanceOp>(
      getLoc(), clonedTemplate,
      EquationPathAttr::get(getContext(), EquationPath(EquationPath::LEFT, 0)));

  if (indices) {
    result.setIndicesAttr(
        MultidimensionalRangeAttr::get(getContext(), *indices));
  }

  return result;
}

mlir::LogicalResult MatchedEquationInstanceOp::cloneWithReplacedAccess(
    mlir::RewriterBase &rewriter,
    std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
    const VariableAccess &access, EquationTemplateOp replacementEquation,
    const VariableAccess &replacementAccess,
    llvm::SmallVectorImpl<MatchedEquationInstanceOp> &results) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  auto cleanTemplatesOnExit = llvm::make_scope_exit([&]() {
    llvm::SmallVector<EquationTemplateOp> templateOps;

    for (MatchedEquationInstanceOp equationOp : results) {
      templateOps.push_back(equationOp.getTemplate());
    }

    (void)cleanEquationTemplates(rewriter, templateOps);
  });

  llvm::SmallVector<std::pair<IndexSet, EquationTemplateOp>> templateResults;

  if (mlir::failed(getTemplate().cloneWithReplacedAccess(
          rewriter, equationIndices, access, replacementEquation,
          replacementAccess, templateResults))) {
    return mlir::failure();
  }

  rewriter.setInsertionPointAfter(getOperation());

  auto temporaryClonedOp =
      mlir::cast<MatchedEquationInstanceOp>(rewriter.clone(*getOperation()));

  for (auto &[assignedIndices, equationTemplateOp] : templateResults) {
    if (assignedIndices.empty()) {
      auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
          rewriter.clone(*temporaryClonedOp.getOperation()));

      clonedOp.setOperand(equationTemplateOp.getResult());
      clonedOp.removeIndicesAttr();
      results.push_back(clonedOp);
    } else {
      for (const MultidimensionalRange &assignedIndicesRange : llvm::make_range(
               assignedIndices.rangesBegin(), assignedIndices.rangesEnd())) {
        auto clonedOp = mlir::cast<MatchedEquationInstanceOp>(
            rewriter.clone(*temporaryClonedOp.getOperation()));

        clonedOp.setOperand(equationTemplateOp.getResult());

        if (auto explicitIndices = getIndices()) {
          MultidimensionalRange explicitRange =
              assignedIndicesRange.takeFirstDimensions(
                  explicitIndices->getValue().rank());

          clonedOp.setIndicesAttr(MultidimensionalRangeAttr::get(
              rewriter.getContext(), std::move(explicitRange)));
        }

        results.push_back(clonedOp);
      }
    }
  }

  rewriter.eraseOp(temporaryClonedOp);
  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SCCGroupOp

namespace mlir::bmodelica {
mlir::RegionKind SCCGroupOp::getRegionKind(unsigned index) {
  return mlir::RegionKind::Graph;
}

void SCCGroupOp::collectSCCs(llvm::SmallVectorImpl<SCCOp> &SCCs) {
  for (SCCOp scc : getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// SCCOp

namespace {
struct EmptySCCPattern : public mlir::OpRewritePattern<SCCOp> {
  using mlir::OpRewritePattern<SCCOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(SCCOp op, mlir::PatternRewriter &rewriter) const override {
    if (op.getBody()->empty()) {
      rewriter.eraseOp(op);
      return mlir::success();
    }

    return mlir::failure();
  }
};
} // namespace

namespace mlir::bmodelica {
void SCCOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::MLIRContext *context) {
  patterns.add<EmptySCCPattern>(context);
}

mlir::RegionKind SCCOp::getRegionKind(unsigned index) {
  return mlir::RegionKind::Graph;
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ScheduledEquationInstanceOp

namespace mlir::bmodelica {
void ScheduledEquationInstanceOp::build(mlir::OpBuilder &builder,
                                        mlir::OperationState &state,
                                        EquationTemplateOp equationTemplate,
                                        EquationPathAttr path,
                                        mlir::ArrayAttr iterationDirections) {
  build(builder, state, equationTemplate.getResult(), nullptr, path,
        iterationDirections);
}

mlir::LogicalResult ScheduledEquationInstanceOp::verify() {
  auto indicesRankFn =
      [&](std::optional<MultidimensionalRangeAttr> ranges) -> size_t {
    if (!ranges) {
      return 0;
    }

    return ranges->getValue().rank();
  };

  // Check the indices for the explicit inductions.
  size_t numOfInductions = getInductionVariables().size();

  if (size_t indicesRank = indicesRankFn(getIndices());
      numOfInductions != indicesRank) {
    return emitOpError() << "Unexpected rank of iteration indices (expected "
                         << numOfInductions << ", got " << indicesRank << ")";
  }

  // Check the iteration directions.
  if (size_t numOfIterationDirections = getIterationDirections().size();
      numOfInductions != numOfIterationDirections) {
    return emitOpError()
           << "Unexpected number of iteration directions (expected "
           << numOfInductions << ", got " << numOfIterationDirections << ")";
  }

  return mlir::success();
}

EquationTemplateOp ScheduledEquationInstanceOp::getTemplate() {
  auto result = getBase().getDefiningOp<EquationTemplateOp>();
  assert(result != nullptr);
  return result;
}

void ScheduledEquationInstanceOp::printInline(llvm::raw_ostream &os) {
  getTemplate().printInline(os);
}

mlir::ValueRange ScheduledEquationInstanceOp::getInductionVariables() {
  return getTemplate().getInductionVariables();
}

mlir::LogicalResult ScheduledEquationInstanceOp::getAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTable) {
  return getTemplate().getAccesses(result, symbolTable);
}

std::optional<VariableAccess> ScheduledEquationInstanceOp::getAccessAtPath(
    mlir::SymbolTableCollection &symbolTable, const EquationPath &path) {
  return getTemplate().getAccessAtPath(symbolTable, path);
}

IndexSet ScheduledEquationInstanceOp::getIterationSpace() {
  if (auto indices = getIndices()) {
    return IndexSet(indices->getValue());
  }

  return {};
}

std::optional<VariableAccess> ScheduledEquationInstanceOp::getMatchedAccess(
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getAccessAtPath(symbolTableCollection, getPath().getValue());
}

mlir::LogicalResult ScheduledEquationInstanceOp::getWriteAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::ArrayRef<VariableAccess> accesses) {
  return getWriteAccesses(result, symbolTableCollection, getIterationSpace(),
                          accesses);
}

mlir::LogicalResult ScheduledEquationInstanceOp::getWriteAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    const IndexSet &equationIndices, llvm::ArrayRef<VariableAccess> accesses) {
  std::optional<VariableAccess> matchedAccess =
      getMatchedAccess(symbolTableCollection);

  if (!matchedAccess) {
    return mlir::failure();
  }

  return getTemplate().getWriteAccesses(result, equationIndices, accesses,
                                        *matchedAccess);
}

mlir::LogicalResult ScheduledEquationInstanceOp::getReadAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    llvm::ArrayRef<VariableAccess> accesses) {
  return getReadAccesses(result, symbolTableCollection, getIterationSpace(),
                         accesses);
}

mlir::LogicalResult ScheduledEquationInstanceOp::getReadAccesses(
    llvm::SmallVectorImpl<VariableAccess> &result,
    mlir::SymbolTableCollection &symbolTableCollection,
    const IndexSet &equationIndices, llvm::ArrayRef<VariableAccess> accesses) {
  std::optional<VariableAccess> matchedAccess =
      getMatchedAccess(symbolTableCollection);

  if (!matchedAccess) {
    return mlir::failure();
  }

  return getTemplate().getReadAccesses(result, equationIndices, accesses,
                                       *matchedAccess);
}

mlir::LogicalResult ScheduledEquationInstanceOp::explicitate(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  std::optional<MultidimensionalRange> indices = std::nullopt;

  if (auto indicesAttr = getIndices()) {
    indices = indicesAttr->getValue();
  }

  if (mlir::failed(getTemplate().explicitate(rewriter, symbolTableCollection,
                                             indices, getPath().getValue()))) {
    return mlir::failure();
  }

  setPathAttr(
      EquationPathAttr::get(getContext(), EquationPath(EquationPath::LEFT, 0)));

  return mlir::success();
}

ScheduledEquationInstanceOp ScheduledEquationInstanceOp::cloneAndExplicitate(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection) {
  std::optional<MultidimensionalRange> indices = std::nullopt;

  if (auto indicesAttr = getIndices()) {
    indices = indicesAttr->getValue();
  }

  EquationTemplateOp clonedTemplate = getTemplate().cloneAndExplicitate(
      rewriter, symbolTableCollection, indices, getPath().getValue());

  if (!clonedTemplate) {
    return nullptr;
  }

  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointAfter(getOperation());

  auto result = rewriter.create<ScheduledEquationInstanceOp>(
      getLoc(), clonedTemplate,
      EquationPathAttr::get(getContext(), EquationPath(EquationPath::LEFT, 0)),
      getIterationDirections());

  if (indices) {
    result.setIndicesAttr(
        MultidimensionalRangeAttr::get(getContext(), *indices));
  }

  return result;
}

mlir::LogicalResult ScheduledEquationInstanceOp::cloneWithReplacedAccess(
    mlir::RewriterBase &rewriter,
    std::optional<std::reference_wrapper<const IndexSet>> equationIndices,
    const VariableAccess &access, EquationTemplateOp replacementEquation,
    const VariableAccess &replacementAccess,
    llvm::SmallVectorImpl<ScheduledEquationInstanceOp> &results) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);

  auto cleanTemplatesOnExit = llvm::make_scope_exit([&]() {
    llvm::SmallVector<EquationTemplateOp> templateOps;

    for (ScheduledEquationInstanceOp equationOp : results) {
      templateOps.push_back(equationOp.getTemplate());
    }

    (void)cleanEquationTemplates(rewriter, templateOps);
  });

  llvm::SmallVector<std::pair<IndexSet, EquationTemplateOp>> templateResults;

  if (mlir::failed(getTemplate().cloneWithReplacedAccess(
          rewriter, equationIndices, access, replacementEquation,
          replacementAccess, templateResults))) {
    return mlir::failure();
  }

  rewriter.setInsertionPointAfter(getOperation());

  auto temporaryClonedOp =
      mlir::cast<ScheduledEquationInstanceOp>(rewriter.clone(*getOperation()));

  for (auto &[assignedIndices, equationTemplateOp] : templateResults) {
    if (assignedIndices.empty()) {
      auto clonedOp = mlir::cast<ScheduledEquationInstanceOp>(
          rewriter.clone(*temporaryClonedOp.getOperation()));

      clonedOp.setOperand(equationTemplateOp.getResult());
      clonedOp.removeIndicesAttr();
      results.push_back(clonedOp);
    } else {
      for (const MultidimensionalRange &assignedIndicesRange : llvm::make_range(
               assignedIndices.rangesBegin(), assignedIndices.rangesEnd())) {
        auto clonedOp = mlir::cast<ScheduledEquationInstanceOp>(
            rewriter.clone(*temporaryClonedOp.getOperation()));

        clonedOp.setOperand(equationTemplateOp.getResult());

        if (auto explicitIndices = getIndices()) {
          MultidimensionalRange explicitRange =
              assignedIndicesRange.takeFirstDimensions(
                  explicitIndices->getValue().rank());

          clonedOp.setIndicesAttr(MultidimensionalRangeAttr::get(
              rewriter.getContext(), std::move(explicitRange)));
        }

        results.push_back(clonedOp);
      }
    }
  }

  rewriter.eraseOp(temporaryClonedOp);
  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// EquationSideOp

namespace {
struct EquationSideTypePropagationPattern
    : public mlir::OpRewritePattern<EquationSideOp> {
  using mlir::OpRewritePattern<EquationSideOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(EquationSideOp op,
                  mlir::PatternRewriter &rewriter) const override {
    bool different = false;
    llvm::SmallVector<mlir::Type> newTypes;

    for (size_t i = 0, e = op.getValues().size(); i < e; ++i) {
      mlir::Type existingType = op.getResult().getType().getType(i);
      mlir::Type expectedType = op.getValues()[i].getType();

      if (existingType != expectedType) {
        different = true;
      }

      newTypes.push_back(expectedType);
    }

    if (!different) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<EquationSideOp>(op, op.getValues());
    return mlir::failure();
  }
};
} // namespace

namespace mlir::bmodelica {
mlir::ParseResult EquationSideOp::parse(mlir::OpAsmParser &parser,
                                        mlir::OperationState &result) {
  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand, 1> values;
  mlir::Type resultType;
  auto loc = parser.getCurrentLocation();

  if (parser.parseOperandList(values) || parser.parseColon() ||
      parser.parseType(resultType)) {
    return mlir::failure();
  }

  assert(resultType.isa<mlir::TupleType>());
  auto tupleType = resultType.cast<mlir::TupleType>();

  llvm::SmallVector<mlir::Type, 1> types(tupleType.begin(), tupleType.end());
  assert(types.size() == values.size());

  if (parser.resolveOperands(values, types, loc, result.operands)) {
    return mlir::failure();
  }

  result.addTypes(resultType);
  return mlir::success();
}

void EquationSideOp::print(mlir::OpAsmPrinter &printer) {
  printer.printOptionalAttrDict(getOperation()->getAttrs());
  printer << " " << getValues() << " : " << getResult().getType();
}

void EquationSideOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  patterns.add<EquationSideTypePropagationPattern>(context);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// EquationSidesOp

namespace mlir::bmodelica {
mlir::LogicalResult EquationSidesOp::verify() {
  auto lhsTypes = getLhs().getType().getTypes();
  auto rhsTypes = getRhs().getType().getTypes();

  if (lhsTypes.size() != rhsTypes.size()) {
    return emitOpError() << lhsTypes.size()
                         << " elements on the left-hand side and "
                         << rhsTypes.size() << " on the right-hand side";
  }

  for (auto [lhs, rhs] : llvm::zip(lhsTypes, rhsTypes)) {
    auto lhsShapedType = mlir::dyn_cast<mlir::ShapedType>(lhs);
    auto rhsShapedType = mlir::dyn_cast<mlir::ShapedType>(rhs);

    if (!lhsShapedType && !rhsShapedType) {
      continue;
    }

    if (!lhsShapedType || !rhsShapedType) {
      return emitOpError() << "incompatible types";
    }

    if (lhsShapedType.getRank() != rhsShapedType.getRank()) {
      return emitOpError() << "incompatible types";
    }

    for (int64_t dim = 0, rank = lhsShapedType.getRank(); dim < rank; ++dim) {
      if (mlir::failed(mlir::verifyCompatibleShape(lhsShapedType.getShape(),
                                                   rhsShapedType.getShape()))) {
        return emitOpError() << "incompatible types";
      }
    }
  }

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// FunctionOp

namespace mlir::bmodelica {
void FunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       llvm::StringRef name) {
  build(builder, state, name, nullptr);
}

llvm::SmallVector<mlir::Type> FunctionOp::getArgumentTypes() {
  llvm::SmallVector<mlir::Type> types;

  for (VariableOp variableOp : getVariables()) {
    VariableType variableType = variableOp.getVariableType();

    if (variableType.isInput()) {
      types.push_back(variableType.unwrap());
    }
  }

  return types;
}

llvm::SmallVector<mlir::Type> FunctionOp::getResultTypes() {
  llvm::SmallVector<mlir::Type> types;

  for (VariableOp variableOp : getVariables()) {
    VariableType variableType = variableOp.getVariableType();

    if (variableType.isOutput()) {
      types.push_back(variableType.unwrap());
    }
  }

  return types;
}

mlir::FunctionType FunctionOp::getFunctionType() {
  llvm::SmallVector<mlir::Type> argTypes;
  llvm::SmallVector<mlir::Type> resultTypes;

  for (VariableOp variableOp : getVariables()) {
    VariableType variableType = variableOp.getVariableType();

    if (variableType.isInput()) {
      argTypes.push_back(variableType.unwrap());
    } else if (variableType.isOutput()) {
      resultTypes.push_back(variableType.unwrap());
    }
  }

  return mlir::FunctionType::get(getContext(), argTypes, resultTypes);
}

bool FunctionOp::shouldBeInlined() {
  if (!getOperation()->hasAttrOfType<mlir::BoolAttr>("inline")) {
    return false;
  }

  auto inlineAttribute =
      getOperation()->getAttrOfType<mlir::BoolAttr>("inline");

  return inlineAttribute.getValue();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// DerFunctionOp

namespace mlir::bmodelica {
mlir::ParseResult DerFunctionOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  mlir::StringAttr nameAttr;

  if (parser.parseSymbolName(nameAttr, mlir::SymbolTable::getSymbolAttrName(),
                             result.attributes) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  return mlir::success();
}

void DerFunctionOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printSymbolName(getSymName());

  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(mlir::SymbolTable::getSymbolAttrName());

  printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RawFunctionOp

namespace mlir::bmodelica {
RawFunctionOp
RawFunctionOp::create(mlir::Location location, llvm::StringRef name,
                      mlir::FunctionType type,
                      llvm::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::OpBuilder builder(location->getContext());
  mlir::OperationState state(location, getOperationName());
  RawFunctionOp::build(builder, state, name, type, attrs);
  return mlir::cast<RawFunctionOp>(mlir::Operation::create(state));
}

RawFunctionOp RawFunctionOp::create(mlir::Location location,
                                    llvm::StringRef name,
                                    mlir::FunctionType type,
                                    mlir::Operation::dialect_attr_range attrs) {
  llvm::SmallVector<mlir::NamedAttribute, 8> attrRef(attrs);
  return create(location, name, type, llvm::ArrayRef(attrRef));
}

RawFunctionOp
RawFunctionOp::create(mlir::Location location, llvm::StringRef name,
                      mlir::FunctionType type,
                      llvm::ArrayRef<mlir::NamedAttribute> attrs,
                      llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
  RawFunctionOp func = create(location, name, type, attrs);
  func.setAllArgAttrs(argAttrs);
  return func;
}

void RawFunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          llvm::StringRef name, mlir::FunctionType type,
                          llvm::ArrayRef<mlir::NamedAttribute> attrs,
                          llvm::ArrayRef<mlir::DictionaryAttr> argAttrs) {
  state.addAttribute(mlir::SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));

  state.addAttribute(getFunctionTypeAttrName(state.name),
                     mlir::TypeAttr::get(type));

  state.attributes.append(attrs.begin(), attrs.end());
  state.addRegion();

  if (argAttrs.empty()) {
    return;
  }

  assert(type.getNumInputs() == argAttrs.size());

  function_interface_impl::addArgAndResultAttrs(
      builder, state, argAttrs, std::nullopt, getArgAttrsAttrName(state.name),
      getResAttrsAttrName(state.name));
}

mlir::ParseResult RawFunctionOp::parse(mlir::OpAsmParser &parser,
                                       mlir::OperationState &result) {
  auto buildFuncType =
      [](mlir::Builder &builder, llvm::ArrayRef<mlir::Type> argTypes,
         llvm::ArrayRef<mlir::Type> results,
         mlir::function_interface_impl::VariadicFlag,
         std::string &) { return builder.getFunctionType(argTypes, results); };

  return mlir::function_interface_impl::parseFunctionOp(
      parser, result, false, getFunctionTypeAttrName(result.name),
      buildFuncType, getArgAttrsAttrName(result.name),
      getResAttrsAttrName(result.name));
}

void RawFunctionOp::print(OpAsmPrinter &printer) {
  mlir::function_interface_impl::printFunctionOp(
      printer, *this, false, getFunctionTypeAttrName(), getArgAttrsAttrName(),
      getResAttrsAttrName());
}

bool RawFunctionOp::shouldBeInlined() {
  if (!getOperation()->hasAttrOfType<mlir::BoolAttr>("inline")) {
    return false;
  }

  auto inlineAttribute =
      getOperation()->getAttrOfType<mlir::BoolAttr>("inline");

  return inlineAttribute.getValue();
}

/// Clone the internal blocks from this function into dest and all attributes
/// from this function to dest.
void RawFunctionOp::cloneInto(RawFunctionOp dest, mlir::IRMapping &mapper) {
  // Add the attributes of this function to dest.
  llvm::MapVector<mlir::StringAttr, mlir::Attribute> newAttrMap;

  for (const auto &attr : dest->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }

  for (const auto &attr : (*this)->getAttrs()) {
    newAttrMap.insert({attr.getName(), attr.getValue()});
  }

  auto newAttrs = llvm::to_vector(llvm::map_range(
      newAttrMap, [](std::pair<StringAttr, Attribute> attrPair) {
        return NamedAttribute(attrPair.first, attrPair.second);
      }));

  dest->setAttrs(mlir::DictionaryAttr::get(getContext(), newAttrs));

  // Clone the body.
  getBody().cloneInto(&dest.getBody(), mapper);
}

/// Create a deep copy of this function and all of its blocks, remapping
/// any operands that use values outside of the function using the map that is
/// provided (leaving them alone if no entry is present). Replaces references
/// to cloned sub-values with the corresponding value that is copied, and adds
/// those mappings to the mapper.
RawFunctionOp RawFunctionOp::clone(mlir::IRMapping &mapper) {
  // Create the new function.
  RawFunctionOp newFunc =
      cast<RawFunctionOp>(getOperation()->cloneWithoutRegions());

  // If the function has a body, then the user might be deleting arguments to
  // the function by specifying them in the mapper. If so, we don't add the
  // argument to the input type vector.
  if (!isExternal()) {
    mlir::FunctionType oldType = getFunctionType();

    unsigned oldNumArgs = oldType.getNumInputs();
    llvm::SmallVector<Type, 4> newInputs;
    newInputs.reserve(oldNumArgs);

    for (unsigned i = 0; i != oldNumArgs; ++i) {
      if (!mapper.contains(getArgument(i))) {
        newInputs.push_back(oldType.getInput(i));
      }
    }

    // If any of the arguments were dropped, update the type and drop any
    // necessary argument attributes.
    if (newInputs.size() != oldNumArgs) {
      newFunc.setType(mlir::FunctionType::get(oldType.getContext(), newInputs,
                                              oldType.getResults()));

      if (mlir::ArrayAttr argAttrs = getAllArgAttrs()) {
        SmallVector<Attribute> newArgAttrs;
        newArgAttrs.reserve(newInputs.size());

        for (unsigned i = 0; i != oldNumArgs; ++i) {
          if (!mapper.contains(getArgument(i))) {
            newArgAttrs.push_back(argAttrs[i]);
          }
        }

        newFunc.setAllArgAttrs(newArgAttrs);
      }
    }
  }

  // Clone the current function into the new one and return it.
  cloneInto(newFunc, mapper);
  return newFunc;
}

RawFunctionOp RawFunctionOp::clone() {
  mlir::IRMapping mapper;
  return clone(mapper);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RawReturnOp

namespace mlir::bmodelica {
mlir::LogicalResult RawReturnOp::verify() {
  auto function = cast<RawFunctionOp>((*this)->getParentOp());

  // The operand number and types must match the function signature
  auto results = function.getFunctionType().getResults();

  if (getNumOperands() != results.size()) {
    return emitOpError("has ")
           << getNumOperands() << " operands, but enclosing function (@"
           << function.getName() << ") returns " << results.size();
  }

  for (unsigned i = 0, e = results.size(); i != e; ++i) {
    if (getOperand(i).getType() != results[i]) {
      return emitOpError() << "type of return operand " << i << " ("
                           << getOperand(i).getType()
                           << ") doesn't match function result type ("
                           << results[i] << ")"
                           << " in function @" << function.getName();
    }
  }

  return mlir::success();
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RawVariableOp

namespace mlir::bmodelica {
/*
mlir::ParseResult RawVariableOp::parse(
    mlir::OpAsmParser& parser, mlir::OperationState& result)
{
  auto& builder = parser.getBuilder();

  llvm::SmallVector<mlir::OpAsmParser::UnresolvedOperand> dynamicSizes;
  mlir::Type variableType;

  if (parser.parseOperandList(dynamicSizes) ||
      parser.resolveOperands(
          dynamicSizes, builder.getIndexType(), result.operands)) {
    return mlir::failure();
  }

  // Dimensions constraints.
  llvm::SmallVector<llvm::StringRef> dimensionsConstraints;

  if (mlir::succeeded(parser.parseOptionalLSquare())) {
    do {
      if (mlir::succeeded(
              parser.parseOptionalKeyword(kDimensionConstraintUnbounded))) {
        dimensionsConstraints.push_back(kDimensionConstraintUnbounded);
      } else {
        if (parser.parseKeyword(kDimensionConstraintFixed)) {
          return mlir::failure();
        }

        dimensionsConstraints.push_back(kDimensionConstraintFixed);
      }
    } while (mlir::succeeded(parser.parseOptionalComma()));

    if (parser.parseRSquare()) {
      return mlir::failure();
    }
  }

  result.attributes.append(
      getDimensionsConstraintsAttrName(result.name),
      builder.getStrArrayAttr(dimensionsConstraints));

  // Attributes.
  if (parser.parseOptionalAttrDict(result.attributes)) {
    return mlir::failure();
  }

  // Variable type.
  if (parser.parseColon() ||
      parser.parseType(variableType)) {
    return mlir::failure();
  }

  result.addTypes(variableType);

  return mlir::success();
}

void RawVariableOp::print(mlir::OpAsmPrinter& printer)
{
  if (auto dynamicSizes = getDynamicSizes(); !dynamicSizes.empty()) {
    printer << " " << dynamicSizes;
  }

  auto dimConstraints =
      getDimensionsConstraints().getAsRange<mlir::StringAttr>();

  if (llvm::any_of(dimConstraints, [](mlir::StringAttr constraint) {
        return constraint == kDimensionConstraintFixed;
      })) {
    printer << " [";

    for (const auto& constraint : llvm::enumerate(dimConstraints)) {
      if (constraint.index() != 0) {
        printer << ", ";
      }

      printer << constraint.value().getValue();
    }

    printer << "] ";
  }

  llvm::SmallVector<llvm::StringRef, 1> elidedAttrs;
  elidedAttrs.push_back(getDimensionsConstraintsAttrName());

  printer.printOptionalAttrDict(getOperation()->getAttrs(), elidedAttrs);

  printer << " : " << getVariable().getType();
}
*/

void RawVariableOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  auto variableMemRefType =
      getVariable().getType().dyn_cast<mlir::MemRefType>();

  if (variableMemRefType) {
    if (!getHeap() || isDynamicArrayVariable()) {
      effects.emplace_back(
          mlir::MemoryEffects::Allocate::get(), getResult(),
          mlir::SideEffects::AutomaticAllocationScopeResource::get());
    } else {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), getResult(),
                           mlir::SideEffects::DefaultResource::get());
    }
  }
}

/*
VariableType RawVariableOp::getVariableType()
{
  mlir::Type variableType = getVariable().getType();

  VariabilityProperty variabilityProperty = VariabilityProperty::none;
  IOProperty ioProperty = IOProperty::none;

  if (getOutput()) {
    ioProperty = IOProperty::output;
  }

  if (auto shapedType = variableType.dyn_cast<mlir::ShapedType>()) {
    return VariableType::get(
        shapedType.getShape(), shapedType.getElementType(),
        variabilityProperty, ioProperty);
  }

  return VariableType::get(
      std::nullopt, variableType, variabilityProperty, ioProperty);
}
 */

bool RawVariableOp::isScalarVariable(mlir::Type variableType) {
  auto variableShapedType = variableType.cast<mlir::ShapedType>();
  return variableShapedType.getShape().empty();
}

bool RawVariableOp::isStaticArrayVariable(mlir::Type variableType) {
  auto variableShapedType = variableType.cast<mlir::ShapedType>();

  return !variableShapedType.getShape().empty() &&
         variableShapedType.hasStaticShape();
}

bool RawVariableOp::isDynamicArrayVariable(mlir::Type variableType) {
  auto variableShapedType = variableType.cast<mlir::ShapedType>();

  return !variableShapedType.getShape().empty() &&
         !variableShapedType.hasStaticShape();
}

bool RawVariableOp::isScalarVariable() {
  return RawVariableOp::isScalarVariable(getVariable().getType());
}

bool RawVariableOp::isStaticArrayVariable() {
  return RawVariableOp::isStaticArrayVariable(getVariable().getType());
}

bool RawVariableOp::isDynamicArrayVariable() {
  return RawVariableOp::isDynamicArrayVariable(getVariable().getType());
}

bool RawVariableOp::isProtected() { return !getOutput(); }

bool RawVariableOp::isOutput() { return getOutput(); }
} // namespace mlir::bmodelica

namespace mlir::bmodelica {
void RawVariableDeallocOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  auto variableMemRefType =
      getVariable().getType().dyn_cast<mlir::MemRefType>();

  if (variableMemRefType) {
    effects.emplace_back(mlir::MemoryEffects::Free::get(), getVariable(),
                         mlir::SideEffects::DefaultResource::get());
  }
}

bool RawVariableDeallocOp::isScalarVariable() {
  return RawVariableOp::isScalarVariable(getVariable().getType());
}

bool RawVariableDeallocOp::isStaticArrayVariable() {
  return RawVariableOp::isStaticArrayVariable(getVariable().getType());
}

bool RawVariableDeallocOp::isDynamicArrayVariable() {
  return RawVariableOp::isDynamicArrayVariable(getVariable().getType());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RawVariableGetOp

namespace mlir::bmodelica {
void RawVariableGetOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Read::get(), getVariable(),
                       mlir::SideEffects::DefaultResource::get());
}

mlir::Type RawVariableGetOp::computeResultType(mlir::Type rawVariableType) {
  mlir::Type resultType = rawVariableType;
  bool isScalar = RawVariableOp::isScalarVariable(rawVariableType);

  if (isScalar) {
    auto shapedType = rawVariableType.cast<mlir::ShapedType>();
    resultType = shapedType.getElementType();
  }

  return resultType;
}

bool RawVariableGetOp::isScalarVariable() {
  return RawVariableOp::isScalarVariable(getVariable().getType());
}

bool RawVariableGetOp::isStaticArrayVariable() {
  return RawVariableOp::isStaticArrayVariable(getVariable().getType());
}

bool RawVariableGetOp::isDynamicArrayVariable() {
  return RawVariableOp::isDynamicArrayVariable(getVariable().getType());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RawVariableSetOp

namespace mlir::bmodelica {
void RawVariableSetOp::getEffects(
    mlir::SmallVectorImpl<
        mlir::SideEffects::EffectInstance<mlir::MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(mlir::MemoryEffects::Write::get(), getVariable(),
                       mlir::SideEffects::DefaultResource::get());
}

bool RawVariableSetOp::isScalarVariable() {
  return RawVariableOp::isScalarVariable(getVariable().getType());
}

bool RawVariableSetOp::isStaticArrayVariable() {
  return RawVariableOp::isStaticArrayVariable(getVariable().getType());
}

bool RawVariableSetOp::isDynamicArrayVariable() {
  return RawVariableOp::isDynamicArrayVariable(getVariable().getType());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// CallOp

namespace mlir::bmodelica {
void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   FunctionOp callee, mlir::ValueRange args,
                   std::optional<mlir::ArrayAttr> argNames) {
  mlir::SymbolRefAttr symbol = getSymbolRefFromRoot(callee);
  build(builder, state, symbol, callee.getResultTypes(), args, argNames);
}

void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   RawFunctionOp callee, mlir::ValueRange args) {
  mlir::SymbolRefAttr symbol = getSymbolRefFromRoot(callee);
  build(builder, state, symbol, callee.getResultTypes(), args);
}

void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   EquationFunctionOp callee, mlir::ValueRange args) {
  mlir::SymbolRefAttr symbol = getSymbolRefFromRoot(callee);
  build(builder, state, symbol, callee.getResultTypes(), args);
}

void CallOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   mlir::SymbolRefAttr callee, mlir::TypeRange resultTypes,
                   mlir::ValueRange args,
                   std::optional<mlir::ArrayAttr> argNames) {
  state.addOperands(args);
  state.addAttribute(getCalleeAttrName(state.name), callee);

  if (argNames) {
    state.addAttribute(getArgNamesAttrName(state.name), *argNames);
  }

  state.addTypes(resultTypes);
}

mlir::LogicalResult
CallOp::verifySymbolUses(mlir::SymbolTableCollection &symbolTable) {
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();
  mlir::Operation *callee = getFunction(moduleOp, symbolTable);

  if (!callee) {
    // TODO
    // At the moment verification would fail for derivatives of functions,
    // because they are declared through an attribute. We should look into
    // turning that attribute into an operation, so that the symbol becomes
    // declared within the module, and thus obtainable.
    return mlir::success();
  }

  if (mlir::isa<DerFunctionOp>(callee)) {
    // TODO implement proper verification of DerFunctionOp function type.
    return mlir::success();
  }

  // Verify that the operand and result types match the callee.
  if (auto functionOp = mlir::dyn_cast<FunctionOp>(callee)) {
    mlir::FunctionType functionType = functionOp.getFunctionType();

    llvm::SmallVector<mlir::StringAttr> inputVariables;
    llvm::DenseSet<mlir::StringAttr> inputVariablesSet;

    for (VariableOp variableOp : functionOp.getVariables()) {
      mlir::StringAttr variableName = variableOp.getSymNameAttr();

      if (variableOp.isInput()) {
        inputVariables.push_back(variableName);
        inputVariablesSet.insert(variableName);
      }
    }

    llvm::DenseSet<mlir::StringAttr> variablesWithDefaultValue;

    for (DefaultOp defaultOp : functionOp.getDefaultValues()) {
      variablesWithDefaultValue.insert(defaultOp.getVariableAttr());
    }

    auto args = getArgs();

    if (auto argNames = getArgNames()) {
      if (argNames->size() != args.size()) {
        return emitOpError()
               << "the number of arguments (" << args.size()
               << ") does not match the number of argument names ("
               << argNames->size() << ")";
      }

      llvm::DenseSet<mlir::StringAttr> specifiedInputs;

      for (mlir::FlatSymbolRefAttr argName :
           argNames->getAsRange<mlir::FlatSymbolRefAttr>()) {
        if (!inputVariablesSet.contains(argName.getAttr())) {
          return emitOpError()
                 << "unknown argument '" << argName.getValue() << "'";
        }

        if (specifiedInputs.contains(argName.getAttr())) {
          return emitOpError() << "multiple values for argument '"
                               << argName.getValue() << "'";
        }

        specifiedInputs.insert(argName.getAttr());
      }

      for (mlir::StringAttr variableName : inputVariables) {
        if (!variablesWithDefaultValue.contains(variableName) &&
            !specifiedInputs.contains(variableName)) {
          return emitOpError() << "missing value for argument '"
                               << variableName.getValue() << "'";
        }
      }
    } else {
      if (args.size() > inputVariables.size()) {
        return emitOpError()
               << "too many arguments specified (expected "
               << inputVariables.size() << ", got " << args.size() << ")";
      }

      for (mlir::StringAttr variableName :
           llvm::ArrayRef(inputVariables).drop_front(args.size())) {
        if (!variablesWithDefaultValue.contains(variableName)) {
          return emitOpError() << "missing value for argument '"
                               << variableName.getValue() << "'";
        }
      }
    }

    unsigned int expectedResults = functionType.getNumResults();
    unsigned int actualResults = getNumResults();

    if (expectedResults != actualResults) {
      return emitOpError()
             << "incorrect number of results for callee (expected "
             << expectedResults << ", got " << actualResults << ")";

      return mlir::failure();
    }

    return mlir::success();
  }

  if (auto equationFunctionOp = mlir::dyn_cast<EquationFunctionOp>(callee)) {
    mlir::FunctionType functionType = equationFunctionOp.getFunctionType();

    unsigned int expectedInputs = functionType.getNumInputs();
    unsigned int actualInputs = getNumOperands();

    if (expectedInputs != actualInputs) {
      return emitOpError()
             << "incorrect number of operands for callee (expected "
             << expectedInputs << ", got " << actualInputs << ")";
    }

    unsigned int expectedResults = functionType.getNumResults();
    unsigned int actualResults = getNumResults();

    if (expectedResults != actualResults) {
      return emitOpError()
             << "incorrect number of results for callee (expected "
             << expectedResults << ", got " << actualResults << ")";
    }

    return mlir::success();
  }

  if (auto rawFunctionOp = mlir::dyn_cast<RawFunctionOp>(callee)) {
    mlir::FunctionType functionType = rawFunctionOp.getFunctionType();

    unsigned int expectedInputs = functionType.getNumInputs();
    unsigned int actualInputs = getNumOperands();

    if (expectedInputs != actualInputs) {
      return emitOpError()
             << "incorrect number of operands for callee (expected "
             << expectedInputs << ", got " << actualInputs << ")";
    }

    unsigned int expectedResults = functionType.getNumResults();
    unsigned int actualResults = getNumResults();

    if (expectedResults != actualResults) {
      return emitOpError()
             << "incorrect number of results for callee (expected "
             << expectedResults << ", got " << actualResults << ")";
    }

    return mlir::success();
  }

  return emitOpError() << "'" << getCallee()
                       << "' does not reference a valid function";
}

void CallOp::getEffects(mlir::SmallVectorImpl<mlir::SideEffects::EffectInstance<
                            mlir::MemoryEffects::Effect>> &effects) {
  // The callee may have no arguments and no results, but still have side
  // effects (i.e. an external function writing elsewhere). Thus we need to
  // consider the call itself as if it is has side effects and prevent the
  // CSE pass to erase it.
  effects.emplace_back(mlir::MemoryEffects::Write::get(),
                       mlir::SideEffects::DefaultResource::get());

  for (mlir::Value result : getResults()) {
    if (auto arrayType = result.getType().dyn_cast<ArrayType>()) {
      effects.emplace_back(mlir::MemoryEffects::Allocate::get(), result,
                           mlir::SideEffects::DefaultResource::get());

      effects.emplace_back(mlir::MemoryEffects::Write::get(), result,
                           mlir::SideEffects::DefaultResource::get());
    }
  }
}

mlir::Operation *CallOp::getFunction(mlir::ModuleOp moduleOp,
                                     mlir::SymbolTableCollection &symbolTable) {
  mlir::SymbolRefAttr callee = getCallee();

  mlir::Operation *result =
      symbolTable.lookupSymbolIn(moduleOp, callee.getRootReference());

  for (mlir::FlatSymbolRefAttr flatSymbolRef : callee.getNestedReferences()) {
    if (result == nullptr) {
      return nullptr;
    }

    result = symbolTable.lookupSymbolIn(result, flatSymbolRef.getAttr());
  }

  return result;
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// ScheduleOp

namespace mlir::bmodelica {
void ScheduleOp::collectSCCGroups(
    llvm::SmallVectorImpl<SCCGroupOp> &SCCGroups) {
  for (SCCGroupOp sccGroup : getOps<SCCGroupOp>()) {
    SCCGroups.push_back(sccGroup);
  }
}

void ScheduleOp::collectSCCs(llvm::SmallVectorImpl<SCCOp> &SCCs) {
  for (SCCOp scc : getOps<SCCOp>()) {
    SCCs.push_back(scc);
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// RunScheduleOp

namespace mlir::bmodelica {
void RunScheduleOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                          ScheduleOp scheduleOp) {
  auto qualifiedRef = getSymbolRefFromRoot(scheduleOp);
  build(builder, state, qualifiedRef);
}

mlir::LogicalResult RunScheduleOp::verifySymbolUses(
    mlir::SymbolTableCollection &symbolTableCollection) {
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  mlir::Operation *symbolOp =
      resolveSymbol(moduleOp, symbolTableCollection, getSchedule());

  if (!symbolOp) {
    return emitError() << "symbol " << getSchedule() << " not found";
  }

  auto scheduleOp = mlir::dyn_cast<ScheduleOp>(symbolOp);

  if (!scheduleOp) {
    return emitError() << "symbol " << getSchedule() << " is not a schedule";
  }

  return mlir::success();
}

ScheduleOp RunScheduleOp::getScheduleOp() {
  mlir::SymbolTableCollection symbolTableCollection;
  return getScheduleOp(symbolTableCollection);
}

ScheduleOp RunScheduleOp::getScheduleOp(
    mlir::SymbolTableCollection &symbolTableCollection) {
  auto moduleOp = getOperation()->getParentOfType<mlir::ModuleOp>();

  mlir::Operation *variable =
      resolveSymbol(moduleOp, symbolTableCollection, getSchedule());

  return mlir::dyn_cast<ScheduleOp>(variable);
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Control flow operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// ForOp

namespace mlir::bmodelica {
llvm::SmallVector<mlir::Region *> ForOp::getLoopRegions() {
  llvm::SmallVector<mlir::Region *> result;
  result.push_back(&getBodyRegion());
  return result;
}

mlir::Block *ForOp::conditionBlock() {
  assert(!getConditionRegion().empty());
  return &getConditionRegion().front();
}

mlir::Block *ForOp::bodyBlock() {
  assert(!getBodyRegion().empty());
  return &getBodyRegion().front();
}

mlir::Block *ForOp::stepBlock() {
  assert(!getStepRegion().empty());
  return &getStepRegion().front();
}

mlir::ParseResult ForOp::parse(mlir::OpAsmParser &parser,
                               mlir::OperationState &result) {
  mlir::Region *conditionRegion = result.addRegion();

  if (mlir::succeeded(parser.parseOptionalLParen())) {
    if (mlir::failed(parser.parseOptionalRParen())) {
      do {
        mlir::OpAsmParser::UnresolvedOperand arg;
        mlir::Type argType;

        if (parser.parseOperand(arg) || parser.parseColonType(argType) ||
            parser.resolveOperand(arg, argType, result.operands))
          return mlir::failure();
      } while (mlir::succeeded(parser.parseOptionalComma()));
    }

    if (parser.parseRParen()) {
      return mlir::failure();
    }
  }

  if (parser.parseKeyword("condition")) {
    return mlir::failure();
  }

  if (parser.parseRegion(*conditionRegion)) {
    return mlir::failure();
  }

  if (parser.parseKeyword("body")) {
    return mlir::failure();
  }

  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  if (parser.parseKeyword("step")) {
    return mlir::failure();
  }

  mlir::Region *stepRegion = result.addRegion();

  if (parser.parseRegion(*stepRegion)) {
    return mlir::failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  return mlir::success();
}

void ForOp::print(mlir::OpAsmPrinter &printer) {
  if (auto values = getArgs(); !values.empty()) {
    printer << "(";

    for (auto arg : llvm::enumerate(values)) {
      if (arg.index() != 0) {
        printer << ", ";
      }

      printer << arg.value() << " : " << arg.value().getType();
    }

    printer << ")";
  }

  printer << " condition ";
  printer.printRegion(getConditionRegion(), true);
  printer << " body ";
  printer.printRegion(getBodyRegion(), true);
  printer << " step ";
  printer.printRegion(getStepRegion(), true);
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// IfOp

namespace mlir::bmodelica {
void IfOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                 mlir::Value condition, bool withElseRegion) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  state.addOperands(condition);

  // Create the "then" region.
  mlir::Region *thenRegion = state.addRegion();
  builder.createBlock(thenRegion);

  // Create the "else" region.
  mlir::Region *elseRegion = state.addRegion();

  if (withElseRegion) {
    builder.createBlock(elseRegion);
  }
}

mlir::Block *IfOp::thenBlock() { return &getThenRegion().front(); }

mlir::Block *IfOp::elseBlock() { return &getElseRegion().front(); }

mlir::ParseResult IfOp::parse(mlir::OpAsmParser &parser,
                              mlir::OperationState &result) {
  mlir::OpAsmParser::UnresolvedOperand condition;
  mlir::Type conditionType;

  if (parser.parseLParen() || parser.parseOperand(condition) ||
      parser.parseColonType(conditionType) || parser.parseRParen() ||
      parser.resolveOperand(condition, conditionType, result.operands)) {
    return mlir::failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  mlir::Region *thenRegion = result.addRegion();

  if (parser.parseRegion(*thenRegion)) {
    return mlir::failure();
  }

  mlir::Region *elseRegion = result.addRegion();

  if (mlir::succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion)) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

void IfOp::print(mlir::OpAsmPrinter &printer) {
  printer << " (" << getCondition() << " : " << getCondition().getType()
          << ") ";

  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
  printer.printRegion(getThenRegion());

  if (!getElseRegion().empty()) {
    printer << " else ";
    printer.printRegion(getElseRegion());
  }
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// WhileOp

namespace mlir::bmodelica {
mlir::ParseResult WhileOp::parse(mlir::OpAsmParser &parser,
                                 mlir::OperationState &result) {
  mlir::Region *conditionRegion = result.addRegion();
  mlir::Region *bodyRegion = result.addRegion();

  if (parser.parseRegion(*conditionRegion) || parser.parseKeyword("do") ||
      parser.parseRegion(*bodyRegion)) {
    return mlir::failure();
  }

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes)) {
    return mlir::failure();
  }

  if (conditionRegion->empty()) {
    conditionRegion->emplaceBlock();
  }

  if (bodyRegion->empty()) {
    bodyRegion->emplaceBlock();
  }

  return mlir::success();
}

void WhileOp::print(mlir::OpAsmPrinter &printer) {
  printer << " ";
  printer.printRegion(getConditionRegion(), false);
  printer << " do ";
  printer.printRegion(getBodyRegion(), false);
  printer.printOptionalAttrDictWithKeyword(getOperation()->getAttrs());
}

llvm::SmallVector<mlir::Region *> WhileOp::getLoopRegions() {
  llvm::SmallVector<mlir::Region *> result;
  result.push_back(&getBodyRegion());
  return result;
}
} // namespace mlir::bmodelica

//===---------------------------------------------------------------------===//
// Utility operations
//===---------------------------------------------------------------------===//

//===---------------------------------------------------------------------===//
// CastOp

namespace mlir::bmodelica {
mlir::OpFoldResult CastOp::fold(FoldAdaptor adaptor) {
  auto operand = adaptor.getValue();

  if (!operand) {
    return {};
  }

  auto resultType = getResult().getType();

  if (isScalar(operand)) {
    if (isScalarIntegerLike(operand)) {
      int64_t value = getScalarIntegerLikeValue(operand);
      return getAttr(resultType, value);
    }

    if (isScalarFloatLike(operand)) {
      double value = getScalarFloatLikeValue(operand);
      return getAttr(resultType, value);
    }
  }

  return {};
}
} // namespace mlir::bmodelica
