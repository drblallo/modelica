#include "marco/Codegen/Transforms/EquationTemplatesCreation.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include <stack>

namespace mlir::modelica
{
#define GEN_PASS_DEF_EQUATIONTEMPLATESCREATIONPASS
#include "marco/Codegen/Transforms/Passes.h.inc"
}

using namespace ::mlir::modelica;

static std::vector<MultidimensionalRange> getRangesCombinations(
    size_t rank,
    std::function<llvm::ArrayRef<Range>(size_t)> rangesFn,
    size_t startingDimension)
{
  assert(startingDimension < rank);
  std::vector<MultidimensionalRange> result;

  if (startingDimension == rank - 1) {
    for (const Range& range : rangesFn(startingDimension)) {
      result.emplace_back(range);
    }

    return result;
  }

  auto subCombinations = getRangesCombinations(
      rank, rangesFn, startingDimension + 1);

  assert(!rangesFn(startingDimension).empty());

  for (const Range& range : rangesFn(startingDimension)) {
    for (const auto& subCombination : subCombinations) {
      llvm::SmallVector<Range, 1> current;
      current.push_back(range);

      for (size_t i = 0; i < subCombination.rank(); ++i) {
        current.push_back(subCombination[i]);
      }

      result.emplace_back(current);
    }
  }

  return result;
}

static std::vector<MultidimensionalRange> getRangesCombinations(
    size_t rank,
    std::function<llvm::ArrayRef<Range>(size_t)> rangesFn)
{
  return getRangesCombinations(rank, rangesFn, 0);
}

namespace
{
  template<typename Op>
  class EquationPattern : public mlir::OpRewritePattern<Op>
  {
    public:
      EquationPattern(
          mlir::MLIRContext* context)
          : mlir::OpRewritePattern<Op>(context)
      {
      }

      mlir::LogicalResult matchAndRewrite(
          Op op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto modelOp = op->template getParentOfType<ModelOp>();
        rewriter.setInsertionPointToEnd(modelOp.getBody());

        llvm::SmallVector<ForEquationOp, 3> loops;
        getForEquationOps(op, loops);

        llvm::SmallVector<mlir::Value, 3> oldInductions;

        for (ForEquationOp forEquationOp : loops) {
          oldInductions.push_back(forEquationOp.induction());
        }

        // Create the equation template.
        auto templateOp = rewriter.create<EquationTemplateOp>(loc);
        mlir::Block* templateBody = templateOp.createBody(loops.size());

        mlir::BlockAndValueMapping mapping;

        for (const auto& [oldInduction, newInduction] :
             llvm::zip(oldInductions, templateOp.getInductionVariables())) {
          mapping.map(oldInduction, newInduction);
        }

        rewriter.setInsertionPointToStart(templateBody);

        for (auto& nestedOp : op.getOps()) {
          rewriter.clone(nestedOp, mapping);
        }

        rewriter.setInsertionPointAfter(templateOp);

        // Get the iteration indices.
        IndexSet explicitIndices = getIndices(loops);

        if (mlir::failed(createEquationInstances(
                rewriter, op, templateOp, explicitIndices))) {
          return mlir::failure();
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      void getForEquationOps(
          Op op, llvm::SmallVectorImpl<ForEquationOp>& result) const
      {
        std::stack<ForEquationOp> loopsStack;
        auto parentLoop = op->template getParentOfType<ForEquationOp>();

        while (parentLoop) {
          loopsStack.push(parentLoop);
          parentLoop = parentLoop->template getParentOfType<ForEquationOp>();
        }

        while (!loopsStack.empty()) {
          result.push_back(loopsStack.top());
          loopsStack.pop();
        }
      }

      IndexSet getIndices(
          llvm::ArrayRef<ForEquationOp> loops) const
      {
        if (loops.empty()) {
          return {};
        }

        llvm::SmallVector<llvm::SmallVector<Range, 1>, 3> dimensionsRanges;
        dimensionsRanges.resize(loops.size());

        size_t forEquationIndex = 0;

        for (ForEquationOp forEquationOp : loops) {
          auto from = forEquationOp.getFrom().getSExtValue();
          auto to = forEquationOp.getTo().getSExtValue();
          auto step = forEquationOp.getStep().getSExtValue();

          if (step == 1) {
            dimensionsRanges[forEquationIndex].emplace_back(from, to + 1);
          } else {
            for (auto index = from; index < to + 1; index += step) {
              dimensionsRanges[forEquationIndex].emplace_back(index, index + 1);
            }
          }

          ++forEquationIndex;
        }

        return IndexSet(getRangesCombinations(
            dimensionsRanges.size(),
            [&](size_t dimension) -> llvm::ArrayRef<Range> {
              return dimensionsRanges[dimension];
            }));
      }

      mlir::LogicalResult createEquationInstances(
          mlir::OpBuilder& builder,
          Op equationOp,
          EquationTemplateOp templateOp,
          const IndexSet& explicitIndices) const
      {
        // Compute how many equalities the equation contains.
        auto equationSidesOp = mlir::cast<EquationSidesOp>(
            equationOp.bodyBlock()->getTerminator());

        size_t cardinality = equationSidesOp.getLhsValues().size();

        // Create an instance for each equality.
        for (size_t i = 0; i < cardinality; ++i) {
          llvm::Optional<IndexSet> implicitIndices =
              templateOp.computeImplicitIterationSpace(i);

          if (!explicitIndices.empty()) {
            if (implicitIndices) {
              if (mlir::failed(createInstanceWithExplicitAndImplicitIndices(
                      builder, templateOp, i,
                      explicitIndices, *implicitIndices))) {
                return mlir::failure();
              }
            } else {
              if (mlir::failed(createInstanceWithExplicitIndices(
                      builder, templateOp, i, explicitIndices))) {
                return mlir::failure();
              }
            }
          } else if (implicitIndices) {
            if (mlir::failed(createInstanceWithImplicitIndices(
                    builder, templateOp, i, *implicitIndices))) {
              return mlir::failure();
            }
          } else {
            if (mlir::failed(createScalarInstance(builder, templateOp, i))) {
              return mlir::failure();
            }
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createScalarInstance(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          uint64_t viewElementIndex) const
      {
        auto instanceOp = createInstance(builder, templateOp);

        if (!instanceOp) {
          return mlir::failure();
        }

        instanceOp.setViewElementIndex(viewElementIndex);
        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          uint64_t viewElementIndex,
          const IndexSet& explicitIndices) const
      {
        for (const MultidimensionalRange& range : llvm::make_range(
                 explicitIndices.rangesBegin(), explicitIndices.rangesEnd())) {
          if (mlir::failed(createInstanceWithExplicitIndices(
                  builder, templateOp, viewElementIndex, range))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          uint64_t viewElementIndex,
          const MultidimensionalRange& explicitIndices) const
      {
        auto instanceOp = createInstance(builder, templateOp);

        instanceOp.setIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), explicitIndices));

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          uint64_t viewElementIndex,
          const IndexSet& implicitIndices) const
      {
        for (const MultidimensionalRange& range : llvm::make_range(
                 implicitIndices.rangesBegin(), implicitIndices.rangesEnd())) {
          if (mlir::failed(createInstanceWithImplicitIndices(
                  builder, templateOp, viewElementIndex, range))) {
            return mlir::failure();
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          uint64_t viewElementIndex,
          const MultidimensionalRange& implicitIndices) const
      {
        auto instanceOp = createInstance(builder, templateOp);

        instanceOp.setImplicitIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), implicitIndices));

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitAndImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          uint64_t viewElementIndex,
          const IndexSet& explicitIndices,
          const IndexSet& implicitIndices) const
      {
        for (const MultidimensionalRange& explicitRange : llvm::make_range(
                 explicitIndices.rangesBegin(),
                 explicitIndices.rangesEnd())) {
          for (const MultidimensionalRange& implicitRange : llvm::make_range(
                   implicitIndices.rangesBegin(),
                   implicitIndices.rangesEnd())) {
            if (mlir::failed(createInstanceWithExplicitAndImplicitIndices(
                    builder, templateOp, viewElementIndex,
                    explicitRange, implicitRange))) {
              return mlir::failure();
            }
          }
        }

        return mlir::success();
      }

      mlir::LogicalResult createInstanceWithExplicitAndImplicitIndices(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp,
          uint64_t viewElementIndex,
          const MultidimensionalRange& explicitIndices,
          const MultidimensionalRange& implicitIndices) const
      {
        auto instanceOp = createInstance(builder, templateOp);

        instanceOp.setIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), explicitIndices));

        instanceOp.setImplicitIndicesAttr(MultidimensionalRangeAttr::get(
            builder.getContext(), implicitIndices));

        return mlir::success();
      }

    protected:
      virtual EquationInstanceOp createInstance(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp) const = 0;
  };

  class EquationOpPattern : public EquationPattern<EquationOp>
  {
    public:
      using EquationPattern<EquationOp>::EquationPattern;

    protected:
      EquationInstanceOp createInstance(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp) const override
      {
        return builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp, false);
      }
  };

  class InitialEquationOpPattern : public EquationPattern<InitialEquationOp>
  {
    public:
      using EquationPattern<InitialEquationOp>::EquationPattern;

    protected:
      EquationInstanceOp createInstance(
          mlir::OpBuilder& builder,
          EquationTemplateOp templateOp) const override
      {
        return builder.create<EquationInstanceOp>(
            templateOp.getLoc(), templateOp, true);
      }
  };
}

namespace
{
  class EquationTemplatesCreationPass
      : public impl::EquationTemplatesCreationPassBase<
            EquationTemplatesCreationPass>
  {
    public:
      using EquationTemplatesCreationPassBase
        ::EquationTemplatesCreationPassBase;

      void runOnOperation() override;
  };
}

void EquationTemplatesCreationPass::runOnOperation()
{
  ModelOp modelOp = getOperation();
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<EquationOp, InitialEquationOp>();

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<
      EquationOpPattern,
      InitialEquationOpPattern>(&getContext());

  if (mlir::failed(applyPartialConversion(
          modelOp, target, std::move(patterns)))) {
      return signalPassFailure();
  }
}

namespace mlir::modelica
{
  std::unique_ptr<mlir::Pass> createEquationTemplatesCreationPass()
  {
    return std::make_unique<EquationTemplatesCreationPass>();
  }
}
