#include "marco/Dialect/BaseModelica/Transforms/EquationExplicitation.h"
#include "marco/Dialect/BaseModelica/Analysis/VariableAccessAnalysis.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_EQUATIONEXPLICITATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class EquationExplicitationPass
    : public impl::EquationExplicitationPassBase<EquationExplicitationPass> {
public:
  using EquationExplicitationPassBase<
      EquationExplicitationPass>::EquationExplicitationPassBase;

  void runOnOperation() override;

private:
  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(EquationTemplateOp equationTemplate,
                            mlir::SymbolTableCollection &symbolTableCollection);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(StartEquationInstanceOp equation,
                            mlir::SymbolTableCollection &symbolTableCollection);

  std::optional<std::reference_wrapper<VariableAccessAnalysis>>
  getVariableAccessAnalysis(ScheduledEquationInstanceOp equation,
                            mlir::SymbolTableCollection &symbolTableCollection);

  mlir::LogicalResult
  processScheduleOp(mlir::RewriterBase &rewriter,
                    mlir::SymbolTableCollection &symbolTableCollection,
                    mlir::ModuleOp moduleOp, ModelOp modelOp,
                    ScheduleOp scheduleOp);

  mlir::LogicalResult
  processStartEquations(mlir::RewriterBase &rewriter,
                        mlir::SymbolTableCollection &symbolTableCollection,
                        mlir::ModuleOp moduleOp, ModelOp modelOp,
                        llvm::ArrayRef<StartEquationInstanceOp> startEquations);

  mlir::LogicalResult
  processStartEquation(mlir::RewriterBase &rewriter,
                       mlir::SymbolTableCollection &symbolTableCollection,
                       mlir::ModuleOp moduleOp, ModelOp modelOp,
                       StartEquationInstanceOp equation);

  mlir::LogicalResult
  processSCCs(mlir::RewriterBase &rewriter,
              mlir::SymbolTableCollection &symbolTableCollection,
              mlir::ModuleOp moduleOp, ModelOp modelOp,
              llvm::ArrayRef<SCCOp> SCCs);

  mlir::LogicalResult
  processSCC(mlir::RewriterBase &rewriter,
             mlir::SymbolTableCollection &symbolTableCollection,
             mlir::ModuleOp moduleOp, ModelOp modelOp, SCCOp scc);

  EquationFunctionOp
  createEquationFunction(mlir::RewriterBase &rewriter,
                         mlir::SymbolTableCollection &symbolTableCollection,
                         mlir::ModuleOp moduleOp, ModelOp modelOp,
                         StartEquationInstanceOp equation,
                         llvm::SmallVectorImpl<uint64_t> &lowerBounds,
                         llvm::SmallVectorImpl<uint64_t> &upperBounds);

  EquationFunctionOp
  createEquationFunction(mlir::RewriterBase &rewriter,
                         mlir::SymbolTableCollection &symbolTableCollection,
                         mlir::ModuleOp moduleOp, ModelOp modelOp,
                         ScheduledEquationInstanceOp equation,
                         llvm::SmallVectorImpl<uint64_t> &lowerBounds,
                         llvm::SmallVectorImpl<uint64_t> &upperBounds);

  void shiftInductions(llvm::SmallVectorImpl<mlir::Value> &shiftedInductions,
                       mlir::OpBuilder &builder,
                       mlir::ArrayAttr iterationDirections,
                       const MultidimensionalRange &indices,
                       mlir::ValueRange loopInductions);

  mlir::LogicalResult cloneEquationTemplateIntoFunction(
      mlir::RewriterBase &rewriter,
      mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
      EquationTemplateOp templateOp, llvm::ArrayRef<mlir::Value> inductions);

  mlir::LogicalResult
  getAccessAttrs(llvm::SmallVectorImpl<Variable> &writtenVariables,
                 llvm::SmallVectorImpl<Variable> &readVariables,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 StartEquationInstanceOp equationOp);

  mlir::LogicalResult
  getAccessAttrs(llvm::SmallVectorImpl<Variable> &writtenVariables,
                 llvm::SmallVectorImpl<Variable> &readVariables,
                 mlir::SymbolTableCollection &symbolTableCollection,
                 ScheduledEquationInstanceOp equationOp);

  mlir::LogicalResult cleanModelOp(ModelOp modelOp);
};
} // namespace

void EquationExplicitationPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());
  mlir::SymbolTableCollection symbolTableCollection;
  llvm::SmallVector<ModelOp, 1> modelOps;

  walkClasses(getOperation(), [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    for (ScheduleOp scheduleOp :
         llvm::make_early_inc_range(modelOp.getOps<ScheduleOp>())) {
      if (mlir::failed(processScheduleOp(rewriter, symbolTableCollection,
                                         moduleOp, modelOp, scheduleOp))) {
        return signalPassFailure();
      }
    }

    if (mlir::failed(cleanModelOp(modelOp))) {
      return signalPassFailure();
    }
  }
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationExplicitationPass::getVariableAccessAnalysis(
    EquationTemplateOp equationTemplate,
    mlir::SymbolTableCollection &symbolTableCollection) {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::Operation *parentOp = equationTemplate->getParentOp();
  llvm::SmallVector<mlir::Operation *> parentOps;

  while (parentOp != moduleOp) {
    parentOps.push_back(parentOp);
    parentOp = parentOp->getParentOp();
  }

  mlir::AnalysisManager analysisManager = getAnalysisManager();

  for (mlir::Operation *op : llvm::reverse(parentOps)) {
    analysisManager = analysisManager.nest(op);
  }

  if (auto analysis =
          analysisManager.getCachedChildAnalysis<VariableAccessAnalysis>(
              equationTemplate)) {
    return *analysis;
  }

  auto &analysis = analysisManager.getChildAnalysis<VariableAccessAnalysis>(
      equationTemplate);

  if (mlir::failed(analysis.initialize(symbolTableCollection))) {
    return std::nullopt;
  }

  return std::reference_wrapper(analysis);
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationExplicitationPass::getVariableAccessAnalysis(
    StartEquationInstanceOp equation,
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getVariableAccessAnalysis(equation.getTemplate(),
                                   symbolTableCollection);
}

std::optional<std::reference_wrapper<VariableAccessAnalysis>>
EquationExplicitationPass::getVariableAccessAnalysis(
    ScheduledEquationInstanceOp equation,
    mlir::SymbolTableCollection &symbolTableCollection) {
  return getVariableAccessAnalysis(equation.getTemplate(),
                                   symbolTableCollection);
}

mlir::LogicalResult EquationExplicitationPass::processScheduleOp(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, ScheduleOp scheduleOp) {
  llvm::SmallVector<StartEquationInstanceOp> startEquations;
  llvm::SmallVector<SCCOp> SCCs;

  for (auto &op : scheduleOp.getOps()) {
    if (auto initialOp = mlir::dyn_cast<InitialOp>(op)) {
      initialOp.collectEquations(startEquations);
      initialOp.collectSCCs(SCCs);
      continue;
    }

    if (auto dynamicOp = mlir::dyn_cast<DynamicOp>(op)) {
      dynamicOp.collectSCCs(SCCs);
      continue;
    }
  }

  if (mlir::failed(processStartEquations(rewriter, symbolTableCollection,
                                         moduleOp, modelOp, startEquations))) {
    return mlir::failure();
  }

  if (mlir::failed(processSCCs(rewriter, symbolTableCollection, moduleOp,
                               modelOp, SCCs))) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processStartEquations(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<StartEquationInstanceOp> startEquations) {
  for (StartEquationInstanceOp equation : startEquations) {
    if (mlir::failed(processStartEquation(rewriter, symbolTableCollection,
                                          moduleOp, modelOp, equation))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processStartEquation(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, StartEquationInstanceOp equation) {
  llvm::SmallVector<uint64_t, 10> lowerBounds;
  llvm::SmallVector<uint64_t, 10> upperBounds;

  EquationFunctionOp eqFunc =
      createEquationFunction(rewriter, symbolTableCollection, moduleOp, modelOp,
                             equation, lowerBounds, upperBounds);

  if (!eqFunc) {
    return mlir::failure();
  }

  rewriter.setInsertionPoint(equation);

  auto scheduleBlockOp =
      rewriter.create<ScheduleBlockOp>(modelOp.getLoc(), false);

  if (mlir::failed(
          getAccessAttrs(scheduleBlockOp.getProperties().writtenVariables,
                         scheduleBlockOp.getProperties().readVariables,
                         symbolTableCollection, equation))) {
    return mlir::failure();
  }

  rewriter.createBlock(&scheduleBlockOp.getBodyRegion());
  rewriter.setInsertionPointToStart(scheduleBlockOp.getBody());

  llvm::SmallVector<Range> ranges;

  for (size_t i = 0, e = equation.getInductionVariables().size(); i < e; ++i) {
    ranges.push_back(Range(static_cast<Point::data_type>(lowerBounds[i]),
                           static_cast<Point::data_type>(upperBounds[i])));
  }

  auto callOp = rewriter.create<EquationCallOp>(
      eqFunc.getLoc(), eqFunc.getSymName(), nullptr, true);

  if (!ranges.empty()) {
    callOp.setIndicesAttr(MultidimensionalRangeAttr::get(
        rewriter.getContext(), MultidimensionalRange(ranges)));
  }

  rewriter.eraseOp(equation);
  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processSCCs(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, llvm::ArrayRef<SCCOp> SCCs) {
  for (SCCOp scc : SCCs) {
    if (mlir::failed(processSCC(rewriter, symbolTableCollection, moduleOp,
                                modelOp, scc))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::processSCC(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, SCCOp scc) {
  llvm::SmallVector<ScheduledEquationInstanceOp> equations;
  scc.collectEquations(equations);

  if (equations.size() > 1) {
    // Cycle among the equations.
    return mlir::success();
  }

  bool isSCCErasable = true;

  for (ScheduledEquationInstanceOp equation : equations) {
    ScheduledEquationInstanceOp explicitEquation =
        equation.cloneAndExplicitate(rewriter, symbolTableCollection);

    if (explicitEquation) {
      llvm::SmallVector<uint64_t, 10> lowerBounds;
      llvm::SmallVector<uint64_t, 10> upperBounds;

      EquationFunctionOp eqFunc = createEquationFunction(
          rewriter, symbolTableCollection, moduleOp, modelOp, explicitEquation,
          lowerBounds, upperBounds);

      if (!eqFunc) {
        return mlir::failure();
      }

      rewriter.setInsertionPoint(scc);

      auto scheduleBlockOp =
          rewriter.create<ScheduleBlockOp>(modelOp.getLoc(), true);

      if (mlir::failed(
              getAccessAttrs(scheduleBlockOp.getProperties().writtenVariables,
                             scheduleBlockOp.getProperties().readVariables,
                             symbolTableCollection, equation))) {
        return mlir::failure();
      }

      rewriter.createBlock(&scheduleBlockOp.getBodyRegion());
      rewriter.setInsertionPointToStart(scheduleBlockOp.getBody());

      llvm::SmallVector<Range> ranges;

      for (size_t i = 0, e = equation.getInductionVariables().size(); i < e;
           ++i) {
        ranges.push_back(Range(static_cast<Point::data_type>(lowerBounds[i]),
                               static_cast<Point::data_type>(upperBounds[i])));
      }

      auto iterationDirections = equation.getIterationDirections();
      bool independentIndices = !iterationDirections.empty();

      if (independentIndices) {
        independentIndices &= llvm::all_of(
            equation.getIterationDirections(), [](mlir::Attribute attr) {
              return attr.cast<EquationScheduleDirectionAttr>().getValue() ==
                     EquationScheduleDirection::Any;
            });
      }

      auto callOp = rewriter.create<EquationCallOp>(
          eqFunc.getLoc(), eqFunc.getSymName(), nullptr, independentIndices);

      if (!ranges.empty()) {
        callOp.setIndicesAttr(MultidimensionalRangeAttr::get(
            rewriter.getContext(), MultidimensionalRange(ranges)));
      }

      rewriter.eraseOp(explicitEquation);
    } else {
      isSCCErasable = false;
    }
  }

  if (isSCCErasable) {
    rewriter.eraseOp(scc);
  }

  return mlir::success();
}

EquationFunctionOp EquationExplicitationPass::createEquationFunction(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, StartEquationInstanceOp equation,
    llvm::SmallVectorImpl<uint64_t> &lowerBounds,
    llvm::SmallVectorImpl<uint64_t> &upperBounds) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto eqFunc = rewriter.create<EquationFunctionOp>(
      equation.getLoc(), "start", equation.getInductionVariables().size());

  symbolTableCollection.getSymbolTable(moduleOp).insert(eqFunc);

  mlir::Block *entryBlock = eqFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Value> shiftedInductions;

  if (auto indicesAttr = equation.getIndices()) {
    size_t rank = indicesAttr->getValue().rank();

    for (size_t i = 0; i < rank; ++i) {
      lowerBounds.push_back(0);

      auto upperBound = indicesAttr->getValue()[i].getEnd() -
                        indicesAttr->getValue()[i].getBegin();

      upperBounds.push_back(upperBound);
    }

    llvm::SmallVector<mlir::Value> inductions;

    for (size_t i = 0; i < rank; ++i) {
      auto forOp = rewriter.create<mlir::affine::AffineForOp>(
          equation.getLoc(), eqFunc.getLowerBound(i),
          rewriter.getMultiDimIdentityMap(1), eqFunc.getUpperBound(i),
          rewriter.getMultiDimIdentityMap(1));

      inductions.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    llvm::SmallVector<mlir::Attribute> iterationDirections(
        rank, EquationScheduleDirectionAttr::get(
                  rewriter.getContext(), EquationScheduleDirection::Any));

    shiftInductions(shiftedInductions, rewriter,
                    rewriter.getArrayAttr(iterationDirections),
                    indicesAttr->getValue(), inductions);
  }

  if (mlir::failed(cloneEquationTemplateIntoFunction(
          rewriter, symbolTableCollection, modelOp, equation.getTemplate(),
          shiftedInductions))) {
    return nullptr;
  }

  // Replace the VariableGetOps.
  eqFunc.walk([&](VariableGetOp getOp) {
    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, getOp.getVariableAttr());

    rewriter.setInsertionPoint(getOp);
    rewriter.replaceOpWithNewOp<QualifiedVariableGetOp>(getOp, variableOp);
  });

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<YieldOp>(eqFunc.getLoc());
  return eqFunc;
}

EquationFunctionOp EquationExplicitationPass::createEquationFunction(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, mlir::ModuleOp moduleOp,
    ModelOp modelOp, ScheduledEquationInstanceOp equation,
    llvm::SmallVectorImpl<uint64_t> &lowerBounds,
    llvm::SmallVectorImpl<uint64_t> &upperBounds) {
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToEnd(moduleOp.getBody());

  auto eqFunc = rewriter.create<EquationFunctionOp>(
      equation.getLoc(), "equation", equation.getInductionVariables().size());

  symbolTableCollection.getSymbolTable(moduleOp).insert(eqFunc);

  mlir::Block *entryBlock = eqFunc.addEntryBlock();
  rewriter.setInsertionPointToStart(entryBlock);

  llvm::SmallVector<mlir::Value> shiftedInductions;

  if (auto indicesAttr = equation.getIndices()) {
    size_t rank = indicesAttr->getValue().rank();
    auto iterationDirections = equation.getIterationDirections();

    for (size_t i = 0; i < rank; ++i) {
      auto iterationDirectionAttr =
          iterationDirections[i].cast<EquationScheduleDirectionAttr>();

      lowerBounds.push_back(0);
      auto iterationDirection = iterationDirectionAttr.getValue();

      if (iterationDirection == EquationScheduleDirection::Any ||
          iterationDirection == EquationScheduleDirection::Forward) {
        auto upperBound = indicesAttr->getValue()[i].getEnd() -
                          indicesAttr->getValue()[i].getBegin();

        upperBounds.push_back(upperBound);
      } else {
        assert(iterationDirection == EquationScheduleDirection::Backward);

        auto upperBound = indicesAttr->getValue()[i].getBegin() -
                          indicesAttr->getValue()[i].getEnd();

        upperBounds.push_back(upperBound);
      }
    }

    llvm::SmallVector<mlir::Value> inductions;

    for (size_t i = 0; i < rank; ++i) {
      auto forOp = rewriter.create<mlir::affine::AffineForOp>(
          equation.getLoc(), eqFunc.getLowerBound(i),
          rewriter.getMultiDimIdentityMap(1), eqFunc.getUpperBound(i),
          rewriter.getMultiDimIdentityMap(1));

      inductions.push_back(forOp.getInductionVar());
      rewriter.setInsertionPointToStart(forOp.getBody());
    }

    shiftInductions(shiftedInductions, rewriter, iterationDirections,
                    indicesAttr->getValue(), inductions);
  }

  if (mlir::failed(cloneEquationTemplateIntoFunction(
          rewriter, symbolTableCollection, modelOp, equation.getTemplate(),
          shiftedInductions))) {
    return nullptr;
  }

  // Replace the VariableGetOps.
  eqFunc.walk([&](VariableGetOp getOp) {
    auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
        modelOp, getOp.getVariableAttr());

    rewriter.setInsertionPoint(getOp);
    rewriter.replaceOpWithNewOp<QualifiedVariableGetOp>(getOp, variableOp);
  });

  rewriter.setInsertionPointToEnd(entryBlock);
  rewriter.create<YieldOp>(eqFunc.getLoc());
  return eqFunc;
}

void EquationExplicitationPass::shiftInductions(
    llvm::SmallVectorImpl<mlir::Value> &shiftedInductions,
    mlir::OpBuilder &builder, mlir::ArrayAttr iterationDirections,
    const MultidimensionalRange &indices, mlir::ValueRange loopInductions) {
  for (size_t i = 0, e = indices.rank(); i < e; ++i) {
    mlir::Value induction = loopInductions[i];

    mlir::Value fromValue = builder.create<ConstantOp>(
        induction.getLoc(), builder.getIndexAttr(indices[i].getBegin()));

    auto iterationDirectionAttr =
        iterationDirections[i].cast<EquationScheduleDirectionAttr>();

    auto iterationDirection = iterationDirectionAttr.getValue();

    if (iterationDirection == EquationScheduleDirection::Any ||
        iterationDirection == EquationScheduleDirection::Forward) {
      mlir::Value mappedInduction = builder.create<AddOp>(
          induction.getLoc(), builder.getIndexType(), fromValue, induction);

      shiftedInductions.push_back(mappedInduction);
    } else {
      assert(iterationDirection == EquationScheduleDirection::Backward);
      mlir::Value mappedInduction = builder.create<SubOp>(
          induction.getLoc(), builder.getIndexType(), fromValue, induction);

      shiftedInductions.push_back(mappedInduction);
    }
  }

  /*
  for (size_t i = 0, rank = indices.rank(); i < rank; ++i) {
    mlir::Value induction = loopInductions[i];

    auto iterationDirectionAttr =
        iterationDirections[i].cast<EquationScheduleDirectionAttr>();

    auto iterationDirection = iterationDirectionAttr.getValue();
    mlir::AffineExpr exp = mlir::getAffineDimExpr(i, builder.getContext());

    if (iterationDirection == EquationScheduleDirection::Any ||
        iterationDirection == EquationScheduleDirection::Forward) {
      exp = indices[i].getBegin() + exp;
    } else {
      exp = indices[i].getBegin() - exp;
    }

    auto affineMap = mlir::AffineMap::get(rank, 0, exp);
    shiftedInductions.push_back(builder.create<mlir::affine::AffineApplyOp>(induction.getLoc(),
  affineMap, loopInductions));
  }
   */
}

mlir::LogicalResult
EquationExplicitationPass::cloneEquationTemplateIntoFunction(
    mlir::RewriterBase &rewriter,
    mlir::SymbolTableCollection &symbolTableCollection, ModelOp modelOp,
    EquationTemplateOp templateOp, llvm::ArrayRef<mlir::Value> inductions) {
  assert(templateOp.getInductionVariables().size() == inductions.size());
  mlir::IRMapping mapping;

  for (const auto &[oldInduction, newInduction] :
       llvm::zip(templateOp.getInductionVariables(), inductions)) {
    mapping.map(oldInduction, newInduction);
  }

  for (auto &op : templateOp.getOps()) {
    if (mlir::isa<EquationSideOp>(op)) {
      continue;
    }

    if (auto equationSidesOp = mlir::dyn_cast<EquationSidesOp>(op)) {
      mlir::Value lhs = mapping.lookup(equationSidesOp.getLhsValues()[0]);
      mlir::Value rhs = mapping.lookup(equationSidesOp.getRhsValues()[0]);

      llvm::SmallVector<llvm::SmallVector<mlir::Value, 3>, 10> lhsSubscripts;
      mlir::Operation *lhsOldOp = lhs.getDefiningOp();

      while (lhsOldOp && !mlir::isa<QualifiedVariableGetOp>(lhsOldOp)) {
        if (auto tensorExtractOp = mlir::dyn_cast<TensorExtractOp>(lhsOldOp)) {
          auto subscripts = tensorExtractOp.getIndices();
          lhsSubscripts.emplace_back(subscripts.begin(), subscripts.end());
          lhsOldOp = tensorExtractOp.getTensor().getDefiningOp();
          continue;
        }

        if (auto tensorViewOp = mlir::dyn_cast<TensorViewOp>(lhsOldOp)) {
          auto subscripts = tensorViewOp.getSubscriptions();
          lhsSubscripts.emplace_back(subscripts.begin(), subscripts.end());
          lhsOldOp = tensorViewOp.getSource().getDefiningOp();
          continue;
        }

        llvm_unreachable("Unknown operation");
      }

      auto lhsGetOp = mlir::cast<QualifiedVariableGetOp>(lhsOldOp);

      if (lhsSubscripts.empty()) {
        rewriter.create<QualifiedVariableSetOp>(
            lhsGetOp.getLoc(), lhsGetOp.getVariable(), std::nullopt, rhs);
      } else {
        auto lhsTensorType =
            lhsGetOp.getResult().getType().cast<mlir::TensorType>();

        lhs = rewriter.create<QualifiedVariableGetOp>(
            lhsGetOp.getLoc(),
            ArrayType::get(lhsTensorType.getShape(),
                           lhsTensorType.getElementType()),
            lhsGetOp.getVariable());

        for (size_t i = 0, e = lhsSubscripts.size(); i < e; ++i) {
          lhs = rewriter.create<SubscriptionOp>(lhs.getLoc(), lhs,
                                                lhsSubscripts[e - i - 1]);
        }

        if (auto rhsArrayType = rhs.getType().dyn_cast<ArrayType>()) {
          rewriter.create<ArrayCopyOp>(equationSidesOp.getLoc(), rhs, lhs);
        } else {
          mlir::Type lhsElementType =
              lhs.getType().cast<ArrayType>().getElementType();

          if (rhs.getType() != lhsElementType) {
            rhs = rewriter.create<CastOp>(rhs.getLoc(), lhsElementType, rhs);
          }

          rewriter.create<StoreOp>(equationSidesOp.getLoc(), rhs, lhs,
                                   std::nullopt);
        }
      }
    } else if (auto variableGetOp = mlir::dyn_cast<VariableGetOp>(op)) {
      auto variableOp = symbolTableCollection.lookupSymbolIn<VariableOp>(
          modelOp, variableGetOp.getVariableAttr());

      auto qualifiedVariableGetOp =
          rewriter.create<QualifiedVariableGetOp>(op.getLoc(), variableOp);

      mapping.map(variableGetOp.getResult(),
                  qualifiedVariableGetOp.getResult());
    } else {
      rewriter.clone(op, mapping);
    }
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::getAccessAttrs(
    llvm::SmallVectorImpl<Variable> &writtenVariables,
    llvm::SmallVectorImpl<Variable> &readVariables,
    mlir::SymbolTableCollection &symbolTableCollection,
    StartEquationInstanceOp equationOp) {
  auto accessAnalysis =
      getVariableAccessAnalysis(equationOp, symbolTableCollection);

  if (!accessAnalysis) {
    return mlir::failure();
  }

  IndexSet equationIndices = equationOp.getIterationSpace();
  auto writeAccess = equationOp.getWriteAccess(symbolTableCollection);

  if (!writeAccess) {
    return mlir::failure();
  }

  IndexSet matchedVariableIndices =
      writeAccess->getAccessFunction().map(equationIndices);

  writtenVariables.emplace_back(writeAccess->getVariable(),
                                std::move(matchedVariableIndices));

  llvm::SmallVector<VariableAccess> readAccesses;

  auto accesses =
      accessAnalysis->get().getAccesses(equationOp, symbolTableCollection);

  if (!accesses) {
    return mlir::failure();
  }

  if (mlir::failed(equationOp.getReadAccesses(
          readAccesses, symbolTableCollection, *accesses))) {
    return mlir::failure();
  }

  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> readVariablesIndices;

  for (const VariableAccess &readAccess : readAccesses) {
    const AccessFunction &accessFunction = readAccess.getAccessFunction();
    IndexSet readIndices = accessFunction.map(equationIndices);
    readVariablesIndices[readAccess.getVariable()] += readIndices;
  }

  for (const auto &entry : readVariablesIndices) {
    readVariables.emplace_back(entry.getFirst(), entry.getSecond());
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::getAccessAttrs(
    llvm::SmallVectorImpl<Variable> &writtenVariables,
    llvm::SmallVectorImpl<Variable> &readVariables,
    mlir::SymbolTableCollection &symbolTableCollection,
    ScheduledEquationInstanceOp equationOp) {
  auto accessAnalysis =
      getVariableAccessAnalysis(equationOp, symbolTableCollection);

  if (!accessAnalysis) {
    return mlir::failure();
  }

  IndexSet equationIndices = equationOp.getIterationSpace();
  auto matchedAccess = equationOp.getMatchedAccess(symbolTableCollection);

  if (!matchedAccess) {
    return mlir::failure();
  }

  IndexSet matchedVariableIndices =
      matchedAccess->getAccessFunction().map(equationIndices);

  writtenVariables.emplace_back(matchedAccess->getVariable(),
                                std::move(matchedVariableIndices));

  llvm::SmallVector<VariableAccess> readAccesses;

  auto accesses =
      accessAnalysis->get().getAccesses(equationOp, symbolTableCollection);

  if (!accesses) {
    return mlir::failure();
  }

  if (mlir::failed(equationOp.getReadAccesses(
          readAccesses, symbolTableCollection, *accesses))) {
    return mlir::failure();
  }

  llvm::DenseMap<mlir::SymbolRefAttr, IndexSet> readVariablesIndices;

  for (const VariableAccess &readAccess : readAccesses) {
    const AccessFunction &accessFunction = readAccess.getAccessFunction();
    IndexSet readIndices = accessFunction.map(equationIndices);
    readVariablesIndices[readAccess.getVariable()] += readIndices;
  }

  for (const auto &entry : readVariablesIndices) {
    readVariables.emplace_back(entry.getFirst(), entry.getSecond());
  }

  return mlir::success();
}

mlir::LogicalResult EquationExplicitationPass::cleanModelOp(ModelOp modelOp) {
  mlir::RewritePatternSet patterns(&getContext());
  ModelOp::getCleaningPatterns(patterns, &getContext());
  return mlir::applyPatternsAndFoldGreedily(modelOp, std::move(patterns));
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createEquationExplicitationPass() {
  return std::make_unique<EquationExplicitationPass>();
}
} // namespace mlir::bmodelica
