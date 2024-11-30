#include "marco/Dialect/BaseModelica/Transforms/EquationFunctionAffineRaise.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONFUNCTIONAFFINERAISEPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationFunctionAffineRaisePass
    : public impl::EquationFunctionAffineRaisePassBase<
          EquationFunctionAffineRaisePass> {
public:
  using EquationFunctionAffineRaisePassBase<
      EquationFunctionAffineRaisePass>::EquationFunctionAffineRaisePassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processFuncOp(mlir::func::FuncOp);
};
} // namespace

void EquationFunctionAffineRaisePass::runOnOperation() {
  auto funcOp = getOperation();

  if (funcOp.getSymName().starts_with("equation")) {
    if (mlir::failed(processFuncOp(funcOp))) {
      return signalPassFailure();
    }
  }
}

namespace {
template <typename Op>
class RaiseSubscriptsPattern : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  virtual void getSubscripts(llvm::SmallVectorImpl<mlir::Value> &subscripts,
                             Op op) const = 0;

  virtual void
  replaceOp(mlir::PatternRewriter &rewriter, Op op,
            llvm::ArrayRef<mlir::Value> affineSubscripts) const = 0;

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const final {
    // Collect the induction variables.
    llvm::SmallVector<mlir::Value> inductions;
    mlir::Operation *parentOp = op->getParentOp();

    while (parentOp && isAffineLoop(parentOp)) {
      if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(parentOp)) {
        inductions.push_back(forOp.getInductionVar());
      } else if (auto parallelForOp =
                     mlir::dyn_cast<mlir::affine::AffineParallelOp>(parentOp)) {
        auto inductionVars = parallelForOp.getIVs();
        inductions.append(inductionVars.rbegin(), inductionVars.rend());
      }

      parentOp = parentOp->getParentOp();
    }

    std::reverse(inductions.begin(), inductions.end());

    // Map the positions of the induction variables.
    llvm::DenseMap<mlir::Value, int64_t> inductionsPosMap;

    for (auto induction : llvm::enumerate(inductions)) {
      inductionsPosMap[induction.value()] =
          static_cast<int64_t>(induction.index());
    }

    // Try to get the affine expressions for the subscripts.
    llvm::SmallVector<mlir::Value> subscripts;
    getSubscripts(subscripts, op);

    if (subscripts.empty()) {
      return mlir::failure();
    }

    llvm::SmallVector<mlir::AffineMap> affineMaps;

    for (mlir::Value subscript : subscripts) {
      if (subscript.getDefiningOp<mlir::affine::AffineApplyOp>()) {
        return mlir::failure();
      }

      if (auto affineSubscriptExp =
              getAffineExpr(subscript, inductionsPosMap)) {
        affineMaps.push_back(
            mlir::AffineMap::get(inductions.size(), 0, *affineSubscriptExp));
      } else {
        return mlir::failure();
      }
    }

    // Apply the affine expressions to the induction variables.
    llvm::SmallVector<mlir::Value> affineSubscripts;

    for (auto affineMap : llvm::enumerate(affineMaps)) {
      mlir::Value affineSubscript =
          rewriter.create<mlir::affine::AffineApplyOp>(
              subscripts[affineMap.index()].getLoc(), affineMap.value(),
              inductions);

      affineSubscripts.push_back(affineSubscript);
    }

    replaceOp(rewriter, op, affineSubscripts);
    return mlir::success();
  }

private:
  bool isAffineLoop(mlir::Operation *op) const {
    return mlir::isa<mlir::affine::AffineForOp, mlir::affine::AffineParallelOp>(
        op);
  }

  std::optional<mlir::AffineExpr> getAffineExpr(
      mlir::Value subscript,
      const llvm::DenseMap<mlir::Value, int64_t> &inductionsPosMap) const {
    if (inductionsPosMap.contains(subscript)) {
      return mlir::getAffineDimExpr(inductionsPosMap.lookup(subscript),
                                    subscript.getContext());
    }

    auto affineExpInt = subscript.getDefiningOp<AffineLikeOpInterface>();

    if (!affineExpInt) {
      return std::nullopt;
    }

    return affineExpInt.getAffineExpression(inductionsPosMap);
  }
};

class LoadOpPattern : public RaiseSubscriptsPattern<mlir::memref::LoadOp> {
public:
  using RaiseSubscriptsPattern<mlir::memref::LoadOp>::RaiseSubscriptsPattern;

  void getSubscripts(llvm::SmallVectorImpl<mlir::Value> &subscripts,
                     mlir::memref::LoadOp op) const override {
    llvm::append_range(subscripts, op.getIndices());
  }

  void replaceOp(mlir::PatternRewriter &rewriter, mlir::memref::LoadOp op,
                 llvm::ArrayRef<mlir::Value> affineSubscripts) const override {
    rewriter.replaceOpWithNewOp<mlir::affine::AffineLoadOp>(op, op.getMemRef(),
                                                            affineSubscripts);
  }
};

class StoreOpPattern : public RaiseSubscriptsPattern<mlir::memref::StoreOp> {
public:
  using RaiseSubscriptsPattern<mlir::memref::StoreOp>::RaiseSubscriptsPattern;

  void getSubscripts(llvm::SmallVectorImpl<mlir::Value> &subscripts,
                     mlir::memref::StoreOp op) const override {
    llvm::append_range(subscripts, op.getIndices());
  }

  void replaceOp(mlir::PatternRewriter &rewriter, mlir::memref::StoreOp op,
                 llvm::ArrayRef<mlir::Value> affineSubscripts) const override {
    rewriter.replaceOpWithNewOp<mlir::affine::AffineStoreOp>(
        op, op.getValue(), op.getMemRef(), affineSubscripts);
  }
};
} // namespace

mlir::LogicalResult
EquationFunctionAffineRaisePass::processFuncOp(mlir::func::FuncOp) {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<LoadOpPattern, StoreOpPattern>(&getContext());
  return applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationFunctionAffineRaisePass() {
  return std::make_unique<EquationFunctionAffineRaisePass>();
}
} // namespace mlir::bmodelica
