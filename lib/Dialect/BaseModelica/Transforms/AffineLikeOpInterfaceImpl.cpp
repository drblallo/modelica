#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationExpressionOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace ::mlir::bmodelica;

namespace {
std::optional<mlir::AffineExpr> getAffineExpression(
    mlir::Value value,
    const llvm::DenseMap<mlir::Value, int64_t> &inductionsPosMap) {
  if (inductionsPosMap.contains(value)) {
    return mlir::getAffineDimExpr(inductionsPosMap.lookup(value),
                                  value.getContext());
  }

  auto affineExpInt = value.getDefiningOp<AffineLikeOpInterface>();

  if (!affineExpInt) {
    return std::nullopt;
  }

  return affineExpInt.getAffineExpression(inductionsPosMap);
}
} // namespace

namespace {
struct ArithIndexCastOpInterface
    : public AffineLikeOpInterface::ExternalModel<::ArithIndexCastOpInterface,
                                                  mlir::arith::IndexCastOp> {
  std::optional<mlir::AffineExpr> getAffineExpression(
      mlir::Operation *op,
      const llvm::DenseMap<mlir::Value, int64_t> &inductionsPosMap) const {
    auto castedOp = mlir::cast<mlir::arith::IndexCastOp>(op);
    return ::getAffineExpression(castedOp.getIn(), inductionsPosMap);
  }
};

struct ArithConstantOpInterface
    : public AffineLikeOpInterface::ExternalModel<::ArithConstantOpInterface,
                                                  mlir::arith::ConstantOp> {
  std::optional<mlir::AffineExpr>
  getAffineExpression(mlir::Operation *op,
                      const llvm::DenseMap<mlir::Value, int64_t> &) const {
    auto castedOp = mlir::cast<mlir::arith::ConstantOp>(op);
    mlir::Attribute valueAttr = castedOp.getValue();

    if (auto integerAttr = mlir::dyn_cast<mlir::IntegerAttr>(valueAttr)) {
      return mlir::getAffineConstantExpr(integerAttr.getInt(),
                                         op->getContext());
    }

    return std::nullopt;
  }
};

struct ArithAddIOpInterface
    : public AffineLikeOpInterface::ExternalModel<::ArithAddIOpInterface,
                                                  mlir::arith::AddIOp> {
  std::optional<mlir::AffineExpr> getAffineExpression(
      mlir::Operation *op,
      const llvm::DenseMap<mlir::Value, int64_t> &inductionsPosMap) const {
    auto castedOp = mlir::cast<mlir::arith::AddIOp>(op);

    auto lhs = ::getAffineExpression(castedOp.getLhs(), inductionsPosMap);
    auto rhs = ::getAffineExpression(castedOp.getRhs(), inductionsPosMap);

    if (!lhs || !rhs) {
      return std::nullopt;
    }

    return *lhs + *rhs;
  }
};

struct ArithSubIOpInterface
    : public AffineLikeOpInterface::ExternalModel<::ArithSubIOpInterface,
                                                  mlir::arith::SubIOp> {
  std::optional<mlir::AffineExpr> getAffineExpression(
      mlir::Operation *op,
      const llvm::DenseMap<mlir::Value, int64_t> &inductionsPosMap) const {
    auto castedOp = mlir::cast<mlir::arith::SubIOp>(op);

    auto lhs = ::getAffineExpression(castedOp.getLhs(), inductionsPosMap);
    auto rhs = ::getAffineExpression(castedOp.getRhs(), inductionsPosMap);

    if (!lhs || !rhs) {
      return std::nullopt;
    }

    return *lhs - *rhs;
  }
};

struct ArithMulIOpInterface
    : public AffineLikeOpInterface::ExternalModel<::ArithMulIOpInterface,
                                                  mlir::arith::MulIOp> {
  std::optional<mlir::AffineExpr> getAffineExpression(
      mlir::Operation *op,
      const llvm::DenseMap<mlir::Value, int64_t> &inductionsPosMap) const {
    auto castedOp = mlir::cast<mlir::arith::MulIOp>(op);

    auto lhs = ::getAffineExpression(castedOp.getLhs(), inductionsPosMap);
    auto rhs = ::getAffineExpression(castedOp.getRhs(), inductionsPosMap);

    if (!lhs || !rhs) {
      return std::nullopt;
    }

    return *lhs * *rhs;
  }
};
} // namespace

namespace mlir::bmodelica {
void registerAffineLikeOpInterfaceExternalModels(
    mlir::DialectRegistry &registry) {
  registry.addExtension(
      +[](mlir::MLIRContext *context, BaseModelicaDialect *dialect) {
        context->loadDialect<mlir::arith::ArithDialect>();

        // clang-format off
        mlir::arith::IndexCastOp::attachInterface<::ArithIndexCastOpInterface>(*context);
        mlir::arith::ConstantOp::attachInterface<::ArithConstantOpInterface>(*context);
        mlir::arith::AddIOp::attachInterface<::ArithAddIOpInterface>(*context);
        mlir::arith::SubIOp::attachInterface<::ArithSubIOpInterface>(*context);
        mlir::arith::MulIOp::attachInterface<::ArithMulIOpInterface>(*context);
        // clang-format on
      });
}
} // namespace mlir::bmodelica
