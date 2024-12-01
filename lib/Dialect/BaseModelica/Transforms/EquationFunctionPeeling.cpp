#include "marco/Dialect/BaseModelica/Transforms/EquationFunctionPeeling.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONFUNCTIONPEELINGPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

using NestedLoops = llvm::SmallVector<mlir::Operation *, 3>;

namespace {
class EquationFunctionPeelingPass
    : public impl::EquationFunctionPeelingPassBase<
          EquationFunctionPeelingPass> {
public:
  using EquationFunctionPeelingPassBase<
      EquationFunctionPeelingPass>::EquationFunctionPeelingPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult processFuncOp(mlir::func::FuncOp);

  void collectLoops(llvm::SmallVectorImpl<NestedLoops> &nestedLoops);

  mlir::LogicalResult peelLoops(llvm::ArrayRef<NestedLoops> nestedLoops);

  mlir::LogicalResult peelLoop(mlir::RewriterBase &rewriter,
                               mlir::affine::AffineForOp loop,
                               llvm::ArrayRef<mlir::Operation *> parentLoops,
                               int64_t widthFactor);

  mlir::LogicalResult peelLoop(mlir::RewriterBase &rewriter,
                               mlir::scf::ParallelOp loop,
                               llvm::ArrayRef<mlir::Operation *> parentLoops,
                               llvm::ArrayRef<int64_t> widthFactors);
};
} // namespace

void EquationFunctionPeelingPass::runOnOperation() {
  auto funcOp = getOperation();
  // funcOp.dump();

  if (funcOp.getSymName().starts_with("equation")) {
    if (mlir::failed(processFuncOp(funcOp))) {
      funcOp.dump();
      return signalPassFailure();
    }
  }

  // funcOp.dump();
}

mlir::LogicalResult
EquationFunctionPeelingPass::processFuncOp(mlir::func::FuncOp funcOp) {
  // Collect the loops.
  llvm::SmallVector<NestedLoops, 1> nestedLoops;

  for (auto forOp : funcOp.getOps<mlir::affine::AffineForOp>()) {
    auto &loops = nestedLoops.emplace_back();
    loops.push_back(forOp);
  }

  for (auto parallelOp : funcOp.getOps<mlir::scf::ParallelOp>()) {
    auto &loops = nestedLoops.emplace_back();
    loops.push_back(parallelOp);
  }

  collectLoops(nestedLoops);

  // Perform the peeling.
  return peelLoops(nestedLoops);
}

void EquationFunctionPeelingPass::collectLoops(
    llvm::SmallVectorImpl<NestedLoops> &nestedLoops) {
  llvm::SmallVector<NestedLoops> result;

  for (const NestedLoops &loops : nestedLoops) {
    assert(!loops.empty());
    llvm::SmallVector<NestedLoops> expanded;

    if (auto forOp = mlir::dyn_cast<mlir::affine::AffineForOp>(loops.back())) {
      for (auto nestedLoop : forOp.getOps<mlir::affine::AffineForOp>()) {
        auto &newLoops = expanded.emplace_back(loops);
        newLoops.push_back(nestedLoop);
      }

      for (auto nestedLoop : forOp.getOps<mlir::scf::ParallelOp>()) {
        auto &newLoops = expanded.emplace_back(loops);
        newLoops.push_back(nestedLoop);
      }
    }

    if (auto parallelOp = mlir::dyn_cast<mlir::scf::ParallelOp>(loops.back())) {
      for (auto nestedLoop : parallelOp.getOps<mlir::affine::AffineForOp>()) {
        auto &newLoops = expanded.emplace_back(loops);
        newLoops.push_back(nestedLoop);
      }

      for (auto nestedLoop : parallelOp.getOps<mlir::scf::ParallelOp>()) {
        auto &newLoops = expanded.emplace_back(loops);
        newLoops.push_back(nestedLoop);
      }
    }

    if (expanded.empty()) {
      // There are no more inner loops.
      result.push_back(loops);
    } else {
      for (const auto &toBeExpanded : expanded) {
        llvm::SmallVector<NestedLoops> recursiveExpansions;
        recursiveExpansions.push_back(toBeExpanded);
        collectLoops(recursiveExpansions);
        result.append(recursiveExpansions);
      }
    }
  }

  nestedLoops = result;
}

mlir::LogicalResult EquationFunctionPeelingPass::peelLoops(
    llvm::ArrayRef<NestedLoops> nestedLoops) {
  mlir::IRRewriter rewriter(&getContext());
  size_t peelRank = widthFactors.size();

  for (const NestedLoops &loops : nestedLoops) {
    size_t remainingFactors = peelRank;

    for (size_t i = 0, e = loops.size(); i < e && remainingFactors > 0; ++i) {
      mlir::Operation *loopOp = loops[e - i - 1];

      if (auto casted = mlir::dyn_cast<mlir::affine::AffineForOp>(loopOp)) {
        if (mlir::failed(peelLoop(rewriter, casted,
                                  llvm::ArrayRef(loops).drop_back(),
                                  widthFactors[remainingFactors - 1]))) {
          return mlir::failure();
        }

        --remainingFactors;
        continue;
      }

      if (auto casted = mlir::dyn_cast<mlir::scf::ParallelOp>(loopOp)) {
        size_t numOfInductions = casted.getInductionVars().size();
        size_t factorsForLoop = 0;

        if (remainingFactors >= numOfInductions) {
          factorsForLoop = numOfInductions;
        } else {
          factorsForLoop = remainingFactors;
        }

        if (mlir::failed(peelLoop(rewriter, casted,
                                  llvm::ArrayRef(loops).drop_back(),
                                  llvm::ArrayRef<int64_t>(widthFactors)
                                      .slice(remainingFactors - factorsForLoop,
                                             factorsForLoop)))) {
          return mlir::failure();
        }

        remainingFactors -= factorsForLoop;
        continue;
      }
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationFunctionPeelingPass::peelLoop(
    mlir::RewriterBase &rewriter, mlir::affine::AffineForOp loop,
    llvm::ArrayRef<mlir::Operation *> parentLoops, int64_t widthFactor) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(loop);

  llvm::SmallVector<mlir::Value> inductions;
  llvm::DenseMap<mlir::Value, int64_t> inductionsPosMap;
  int64_t numOfInductionVars = 0;

  for (mlir::Operation *parentLoop : parentLoops) {
    if (auto casted = mlir::dyn_cast<mlir::affine::AffineForOp>(parentLoop)) {
      inductions.push_back(casted.getInductionVar());
      inductionsPosMap[casted.getInductionVar()] = numOfInductionVars++;
      continue;
    }

    if (auto casted = mlir::dyn_cast<mlir::scf::ParallelOp>(parentLoop)) {
      for (mlir::Value inductionVar : casted.getInductionVars()) {
        llvm::append_range(inductions, casted.getInductionVars());
        inductionsPosMap[inductionVar] = numOfInductionVars++;
      }

      continue;
    }
  }

  // Compute the new boundaries.
  mlir::Value loopLowerBound = rewriter.create<mlir::affine::AffineApplyOp>(
      loop.getLoc(), loop.getLowerBound().getMap(),
      loop.getLowerBound().getOperands());

  mlir::Value loopUpperBound = rewriter.create<mlir::affine::AffineApplyOp>(
      loop.getLoc(), loop.getUpperBound().getMap(),
      loop.getUpperBound().getOperands());

  // Main loop.
  llvm::SmallVector<mlir::Value, 2> boundariesSymbols;
  boundariesSymbols.push_back(loopLowerBound);
  boundariesSymbols.push_back(loopUpperBound);

  auto loopLowerBoundExp = mlir::getAffineSymbolExpr(0, &getContext());
  auto loopUpperBoundExp = mlir::getAffineSymbolExpr(1, &getContext());

  auto mainRangeSizeExp =
      (loopUpperBoundExp - loopLowerBoundExp).floorDiv(widthFactor) *
      widthFactor;

  auto mainUpperBoundExp = loopLowerBoundExp + mainRangeSizeExp;

  mlir::Value mainUpperBound = rewriter.create<mlir::affine::AffineApplyOp>(
      loop.getLoc(), mlir::AffineMap::get(0, 2, mainUpperBoundExp),
      boundariesSymbols);

  // Update the main loop boundaries.
  loop.setUpperBound(mainUpperBound, rewriter.getMultiDimIdentityMap(1));

  // Set the peeled boundaries.
  rewriter.setInsertionPointAfter(loop);

  mlir::Value peelLowerBound = mainUpperBound;
  mlir::Value peelUpperBound = loopUpperBound;

  mlir::Value one =
      rewriter.create<mlir::arith::ConstantIndexOp>(loop.getLoc(), 1);

  auto peelForOp = rewriter.create<mlir::scf::ForOp>(
      loop.getLoc(), peelLowerBound, peelUpperBound, one);

  rewriter.setInsertionPointToStart(peelForOp.getBody());

  mlir::IRMapping mapping;
  mapping.map(loop.getInductionVar(), peelForOp.getInductionVar());

  for (auto &nestedOp : loop.getOps()) {
    if (!mlir::isa<mlir::affine::AffineYieldOp>(nestedOp)) {
      rewriter.clone(nestedOp, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationFunctionPeelingPass::peelLoop(
    mlir::RewriterBase &rewriter, mlir::scf::ParallelOp loop,
    llvm::ArrayRef<mlir::Operation *> parentLoops,
    llvm::ArrayRef<int64_t> widthFactors) {
  // TODO Implement peeling for parallel loops.
  return mlir::success();
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationFunctionPeelingPass() {
  return std::make_unique<EquationFunctionPeelingPass>();
}

std::unique_ptr<mlir::Pass> createEquationFunctionPeelingPass(
    const EquationFunctionPeelingPassOptions &options) {
  return std::make_unique<EquationFunctionPeelingPass>(options);
}
} // namespace mlir::bmodelica
