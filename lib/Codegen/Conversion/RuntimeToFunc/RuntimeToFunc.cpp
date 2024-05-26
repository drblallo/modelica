#include "marco/Codegen/Conversion/RuntimeToFunc/RuntimeToFunc.h"
#include "marco/Dialect/Runtime/IR/RuntimeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include <map>
#include <set>

namespace mlir
{
#define GEN_PASS_DEF_RUNTIMETOFUNCCONVERSIONPASS
#include "marco/Codegen/Conversion/Passes.h.inc"
}

using namespace ::mlir::runtime;

namespace
{
  /// Generic rewrite pattern that provides some utility functions.
  template<typename Op>
  class RuntimeOpRewritePattern : public mlir::OpRewritePattern<Op>
  {
    public:
      using mlir::OpRewritePattern<Op>::OpRewritePattern;

    protected:
      mlir::Type getVoidPtrType() const
      {
        return mlir::LLVM::LLVMPointerType::get(this->getContext());
      }

      mlir::Value createGlobalString(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp moduleOp,
          mlir::StringRef name,
          mlir::StringRef value) const
      {
        mlir::LLVM::GlobalOp global;

        {
          // Create the global at the entry of the module.
          mlir::OpBuilder::InsertionGuard insertGuard(builder);
          builder.setInsertionPointToStart(moduleOp.getBody());

          auto type = mlir::LLVM::LLVMArrayType::get(
              mlir::IntegerType::get(
                  builder.getContext(), 8), value.size() + 1);

          global = builder.create<mlir::LLVM::GlobalOp>(
              loc, type, true, mlir::LLVM::Linkage::Internal, name,
              builder.getStringAttr(llvm::StringRef(
                  value.data(), value.size() + 1)));
        }

        // Get the pointer to the first character of the global string.
        mlir::Value globalPtr =
            builder.create<mlir::LLVM::AddressOfOp>(loc, global);

        mlir::Type type = mlir::LLVM::LLVMArrayType::get(
            mlir::IntegerType::get(builder.getContext(), 8),
            value.size() + 1);

        return builder.create<mlir::LLVM::GEPOp>(
            loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()), type,
            globalPtr, llvm::ArrayRef<mlir::LLVM::GEPArg>{0, 0});
      }
  };
}

namespace
{
  class ModelNameOpLowering : public RuntimeOpRewritePattern<ModelNameOp>
  {
    public:
      using RuntimeOpRewritePattern<ModelNameOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          ModelNameOp op, mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

        // char* type.
        mlir::Type opaquePtrType =
            mlir::LLVM::LLVMPointerType::get(getContext());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            loc, "getModelName",
            rewriter.getFunctionType(std::nullopt, opaquePtrType));

        mlir::Block* entryBlock = funcOp.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Create a global string containing the name of the model.
        mlir::Value result = createGlobalString(
            rewriter, loc, moduleOp, "modelName",
            op.getName());

        rewriter.create<mlir::func::ReturnOp>(loc, result);

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class NumberOfVariablesOpLowering
      : public RuntimeOpRewritePattern<NumberOfVariablesOp>
  {
    public:
      using RuntimeOpRewritePattern<NumberOfVariablesOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          NumberOfVariablesOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            loc, "getNumOfVariables",
            rewriter.getFunctionType(std::nullopt, rewriter.getI64Type()));

        mlir::Block* entryBlock = funcOp.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(op.getNumberOfVariables()));

        rewriter.create<mlir::func::ReturnOp>(loc, result);

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class VariableNamesOpLowering
      : public RuntimeOpRewritePattern<VariableNamesOp>
  {
    public:
      using RuntimeOpRewritePattern<VariableNamesOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          VariableNamesOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();

        // char* type.
        mlir::Type opaquePtrType =
            mlir::LLVM::LLVMPointerType::get(getContext());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            loc, "getVariableName",
            rewriter.getFunctionType(rewriter.getI64Type(), opaquePtrType));

        // Create the entry block.
        mlir::Block* entryBlock = funcOp.addEntryBlock();

        // Create the last block receiving the value to be returned.
        mlir::Block* returnBlock = rewriter.createBlock(
            &funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end(),
            opaquePtrType,
            loc);

        rewriter.setInsertionPointToEnd(returnBlock);

        rewriter.create<mlir::func::ReturnOp>(
            loc, returnBlock->getArgument(0));

        // Create the blocks and the switch.
        auto variableNameAttrs = op.getNames();

        size_t numCases = variableNameAttrs.size();
        llvm::SmallVector<int64_t> caseValues(numCases);
        llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
        llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

        for (size_t i = 0; i < numCases; ++i) {
          caseValues[i] = i;
          caseBlocks[i] = rewriter.createBlock(returnBlock);
          caseOperandsRefs[i] = std::nullopt;
        }

        rewriter.setInsertionPointToStart(entryBlock);

        mlir::Value unknownVariableName = createGlobalString(
            rewriter, loc, moduleOp, "var_name_unknown", "");

        rewriter.create<mlir::cf::SwitchOp>(
            loc,
            entryBlock->getArgument(0), returnBlock, unknownVariableName,
            rewriter.getI64TensorAttr(caseValues),
            caseBlocks, caseOperandsRefs);

        // Populate the case blocks.
        for (auto variableNameAttr : llvm::enumerate(
                 variableNameAttrs.getAsRange<mlir::StringAttr>())) {
          size_t i = variableNameAttr.index();
          rewriter.setInsertionPointToStart(caseBlocks[i]);

          std::string symbolName =
              "var_name_" + std::to_string(variableNameAttr.index());

          mlir::Value variableName = unknownVariableName;

          if (llvm::StringRef nameStr = variableNameAttr.value().getValue();
              !nameStr.empty()) {
            variableName = createGlobalString(
                rewriter, loc, moduleOp, symbolName, nameStr);
          }

          rewriter.create<mlir::cf::BranchOp>(loc, returnBlock, variableName);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class VariableRanksOpLowering
      : public RuntimeOpRewritePattern<VariableRanksOp>
  {
    public:
      using RuntimeOpRewritePattern<VariableRanksOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          VariableRanksOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            loc, "getVariableRank",
            rewriter.getFunctionType(
                rewriter.getI64Type(), rewriter.getI64Type()));

        // Create the entry block.
        mlir::Block* entryBlock = funcOp.addEntryBlock();

        // Create the last block receiving the value to be returned.
        mlir::Block* returnBlock = rewriter.createBlock(
            &funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end(),
            rewriter.getI64Type(), loc);

        rewriter.setInsertionPointToEnd(returnBlock);

        rewriter.create<mlir::func::ReturnOp>(
            loc, returnBlock->getArgument(0));

        // Create the blocks and the switch.
        mlir::ArrayAttr variableRankAttrs = op.getRanks();

        size_t numCases = variableRankAttrs.size();
        llvm::SmallVector<int64_t> caseValues(numCases);
        llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
        llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

        for (size_t i = 0; i < numCases; ++i) {
          caseValues[i] = i;
          caseBlocks[i] = rewriter.createBlock(returnBlock);
          caseOperandsRefs[i] = std::nullopt;
        }

        rewriter.setInsertionPointToStart(entryBlock);

        mlir::Value defaultOperand = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(0));

        rewriter.create<mlir::cf::SwitchOp>(
            loc,
            entryBlock->getArgument(0), returnBlock, defaultOperand,
            rewriter.getI64TensorAttr(caseValues),
            caseBlocks, caseOperandsRefs);

        rewriter.setInsertionPointToStart(entryBlock);

        // Populate the case blocks.
        for (auto rank : llvm::enumerate(
                 variableRankAttrs.getAsRange<mlir::IntegerAttr>())) {
          size_t i = rank.index();
          rewriter.setInsertionPointToStart(caseBlocks[i]);

          mlir::Value result =
              rewriter.create<mlir::arith::ConstantOp>(loc, rank.value());

          rewriter.create<mlir::cf::BranchOp>(loc, returnBlock, result);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class PrintableIndicesOpLowering
      : public RuntimeOpRewritePattern<PrintableIndicesOp>
  {
    public:
      using RuntimeOpRewritePattern<PrintableIndicesOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          PrintableIndicesOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        if (mlir::failed(createIsPrintableFunction(rewriter, op))) {
          return mlir::failure();
        }

        // Get the canonical version of index sets in order to guarantee
        // consistency among functions and minimal code size.
        llvm::DenseMap<int64_t, IndexSet> printableIndicesMap;

        for (auto attr : llvm::enumerate(op.getValue())) {
          if (auto indexSetAttr = attr.value().dyn_cast<IndexSetAttr>()) {
            const IndexSet& indexSet = indexSetAttr.getValue();

            if (!indexSet.empty()) {
              printableIndicesMap[attr.index()] =
                  indexSet.getCanonicalRepresentation();
            }
          }
        }

        if (mlir::failed(createGetVariableNumOfPrintableRangesFunction(
                rewriter, op.getLoc(), printableIndicesMap))) {
          return mlir::failure();
        }

        if (mlir::failed(createGetVariablePrintableRangeBeginFunction(
                rewriter, op.getLoc(), printableIndicesMap))) {
          return mlir::failure();
        }

        if (mlir::failed(createGetVariablePrintableRangeEndFunction(
                rewriter, op.getLoc(), printableIndicesMap))) {
          return mlir::failure();
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }

    private:
      mlir::LogicalResult createIsPrintableFunction(
        mlir::OpBuilder& builder,
        PrintableIndicesOp op) const
      {
        mlir::Location loc = op.getLoc();

        auto funcOp = builder.create<mlir::func::FuncOp>(
            loc, "isPrintable",
            builder.getFunctionType(builder.getI64Type(), builder.getI1Type()));

        // Create the entry block.
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        mlir::Value falseValue = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getBoolAttr(false));

        // Create the last block receiving the value to be returned.
        mlir::Block* returnBlock = builder.createBlock(
            &funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end(),
            builder.getI1Type(), loc);

        builder.setInsertionPointToEnd(returnBlock);
        builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

        // Create the blocks and the switch.
        mlir::ArrayAttr printabilityAttrs = op.getValue();
        llvm::DenseSet<int64_t> printableVariables;

        for (auto attr : llvm::enumerate(printabilityAttrs)) {
          if (auto boolAttr = attr.value().dyn_cast<mlir::BoolAttr>()) {
            if (boolAttr.getValue()) {
              printableVariables.insert(attr.index());
            }

            continue;
          }

          if (auto indexSetAttr = attr.value().dyn_cast<IndexSetAttr>()) {
            if (!indexSetAttr.getValue().empty()) {
              printableVariables.insert(attr.index());
            }
          }
        }

        llvm::SmallVector<int64_t> caseValues;
        llvm::SmallVector<mlir::Block*> caseBlocks;
        llvm::SmallVector<mlir::ValueRange> caseOperandsRefs;

        if (!printableVariables.empty()) {
          mlir::Block* printableVariableBlock = builder.createBlock(returnBlock);
          builder.setInsertionPointToStart(printableVariableBlock);

          mlir::Value trueValue = builder.create<mlir::arith::ConstantOp>(
              loc, builder.getBoolAttr(true));

          builder.create<mlir::cf::BranchOp>(loc, returnBlock, trueValue);

          for (int64_t variablePos : printableVariables) {
            caseValues.push_back(variablePos);
            caseBlocks.push_back(printableVariableBlock);
            caseOperandsRefs.push_back(std::nullopt);
          }
        }

        builder.setInsertionPointToEnd(entryBlock);

        builder.create<mlir::cf::SwitchOp>(
            loc,
            entryBlock->getArgument(0), returnBlock, falseValue,
            builder.getI64TensorAttr(caseValues),
            caseBlocks, caseOperandsRefs);

        return mlir::success();
      }

      mlir::LogicalResult createGetVariableNumOfPrintableRangesFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          const llvm::DenseMap<int64_t, IndexSet> printableIndicesMap) const
      {
        auto funcOp = builder.create<mlir::func::FuncOp>(
            loc, "getVariableNumOfPrintableRanges",
            builder.getFunctionType(
                builder.getI64Type(), builder.getI64Type()));

        // Create the entry block.
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block* entryBlock = funcOp.addEntryBlock();

        // Create the last block receiving the value to be returned.
        mlir::Block* returnBlock = builder.createBlock(
            &funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end(),
            builder.getI64Type(),
            loc);

        builder.setInsertionPointToEnd(returnBlock);
        builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

        // Collect the results.
        llvm::DenseMap<int64_t, llvm::DenseSet<int64_t>> resultsVariablesMap;

        for (const auto& entry : printableIndicesMap) {
          const IndexSet& indexSet = entry.getSecond();

          if (!indexSet.empty()) {
            size_t rangesAmount =
                std::distance(indexSet.rangesBegin(), indexSet.rangesEnd());

            resultsVariablesMap[rangesAmount].insert(entry.getFirst());
          }
        }

        // Create the blocks and the switch.
        llvm::SmallVector<int64_t> caseValues;
        llvm::SmallVector<mlir::Block*> caseBlocks;
        llvm::SmallVector<mlir::ValueRange> caseOperandsRefs;

        for (const auto& entry : resultsVariablesMap) {
          int64_t numOfRanges = entry.getFirst();

          mlir::Block* caseBlock = builder.createBlock(returnBlock);
          builder.setInsertionPointToStart(caseBlock);

          mlir::Value result = builder.create<mlir::arith::ConstantOp>(
              loc, builder.getI64IntegerAttr(numOfRanges));

          builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);

          for (int64_t variable : entry.getSecond()) {
            caseValues.push_back(variable);
            caseBlocks.push_back(caseBlock);
            caseOperandsRefs.push_back(std::nullopt);
          }
        }

        builder.setInsertionPointToStart(entryBlock);

        mlir::Value defaultOperand = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getI64IntegerAttr(0));

        builder.create<mlir::cf::SwitchOp>(
            loc,
            entryBlock->getArgument(0), returnBlock, defaultOperand,
            builder.getI64TensorAttr(caseValues),
            caseBlocks, caseOperandsRefs);

        return mlir::success();
      }

      mlir::LogicalResult createGetVariablePrintableRangeBeginFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          const llvm::DenseMap<int64_t, IndexSet> printableIndicesMap) const
      {
        auto callback = [](const Range& range) -> int64_t {
          return range.getBegin();
        };

        return createGetVariablePrintableRangeBoundariesFunction(
            builder, loc,
            "getVariablePrintableRangeBegin",
            printableIndicesMap,
            callback);
      }

      mlir::LogicalResult createGetVariablePrintableRangeEndFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          const llvm::DenseMap<int64_t, IndexSet> printableIndicesMap) const
      {
        auto callback = [](const Range& range) -> int64_t {
          return range.getEnd();
        };

        return createGetVariablePrintableRangeBoundariesFunction(
            builder, loc,
            "getVariablePrintableRangeEnd",
            printableIndicesMap,
            callback);
      }

      mlir::LogicalResult createGetVariablePrintableRangeBoundariesFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          llvm::StringRef functionName,
          const llvm::DenseMap<int64_t, IndexSet> printableIndicesMap,
          llvm::function_ref<int64_t(const Range&)> boundaryGetterCallback) const
      {
        llvm::SmallVector<mlir::Type, 3> argTypes;
        argTypes.push_back(builder.getI64Type());
        argTypes.push_back(builder.getI64Type());
        argTypes.push_back(builder.getI64Type());

        auto funcOp = builder.create<mlir::func::FuncOp>(
            loc, functionName,
            builder.getFunctionType(argTypes, builder.getI64Type()));

        // Create the entry block.
        mlir::OpBuilder::InsertionGuard guard(builder);
        mlir::Block* entryBlock = funcOp.addEntryBlock();
        builder.setInsertionPointToStart(entryBlock);

        mlir::Value unknownResult = builder.create<mlir::arith::ConstantOp>(
            loc, builder.getI64IntegerAttr(-1));

        // Create the last block receiving the value to be returned.
        mlir::Block* returnBlock = builder.createBlock(
            &funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end(),
            builder.getI64Type(),
            loc);

        builder.setInsertionPointToEnd(returnBlock);
        builder.create<mlir::func::ReturnOp>(loc, returnBlock->getArgument(0));

        // Collect the unique ranges and multidimensional ranges.
        std::set<Range> uniqueRanges;
        std::set<MultidimensionalRange> uniqueMultidimensionalRanges;

        for (const auto& entry : printableIndicesMap) {
          const IndexSet& indexSet = entry.getSecond();
          assert(!indexSet.empty());

          for (const MultidimensionalRange& multidimensionalRange :
               llvm::make_range(indexSet.rangesBegin(), indexSet.rangesEnd())) {
            uniqueMultidimensionalRanges.insert(multidimensionalRange);

            for (size_t i = 0, e = multidimensionalRange.rank(); i < e; ++i) {
              uniqueRanges.insert(multidimensionalRange[i]);
            }
          }
        }

        // Create a block for each unique range.
        std::map<Range, mlir::Block*> rangeBlocks;
        mlir::Block* firstRangeBlock = nullptr;

        for (const Range& range : uniqueRanges) {
          mlir::Block* block = builder.createBlock(returnBlock);
          rangeBlocks[range] = block;

          if (firstRangeBlock == nullptr) {
            firstRangeBlock = block;
          }

          builder.setInsertionPointToStart(block);

          mlir::Value result = builder.create<mlir::arith::ConstantOp>(
              loc, builder.getI64IntegerAttr(boundaryGetterCallback(range)));

          builder.create<mlir::cf::BranchOp>(loc, returnBlock, result);
        }

        // Create a block for each unique multidimensional range.
        std::map<
            MultidimensionalRange,
            mlir::Block*> multidimensionalRangeBlocks;

        mlir::Block* firstMultidimensionalRangeBlock = nullptr;

        for (const MultidimensionalRange& multidimensionalRange :
             uniqueMultidimensionalRanges) {
          llvm::SmallVector<int64_t, 3> caseValues;
          llvm::SmallVector<mlir::Block*, 3> caseBlocks;
          llvm::SmallVector<mlir::ValueRange, 3> caseOperandsRefs;

          for (size_t i = 0, e = multidimensionalRange.rank(); i < e; ++i) {
            const Range& range = multidimensionalRange[i];
            caseValues.push_back(i);
            caseBlocks.push_back(rangeBlocks[range]);
            caseOperandsRefs.push_back(std::nullopt);
          }

          assert(firstRangeBlock != nullptr);

          // The block takes as argument the index of the dimension of
          // interest.
          llvm::SmallVector<mlir::Type, 1> blockArgTypes(
              1, builder.getI64Type());

          llvm::SmallVector<mlir::Location, 1> blockArgLocations(1, loc);

          mlir::Block* block = builder.createBlock(
              firstRangeBlock, blockArgTypes, blockArgLocations);

          multidimensionalRangeBlocks[multidimensionalRange] = block;

          if (firstMultidimensionalRangeBlock == nullptr) {
            firstMultidimensionalRangeBlock = block;
          }

          builder.setInsertionPointToStart(block);

          // The switch operates on the index of the dimension of interest.
          builder.create<mlir::cf::SwitchOp>(
              loc,
              block->getArgument(0), returnBlock, unknownResult,
              builder.getI64TensorAttr(caseValues),
              caseBlocks, caseOperandsRefs);
        }

        // Create a block for each variable and the switch inside the entry
        // block.
        llvm::SmallVector<int64_t> variablesCaseValues;
        llvm::SmallVector<mlir::Block*> variablesCaseBlocks;
        llvm::SmallVector<mlir::ValueRange> variablesCaseOperandsRefs;

        for (const auto& entry : printableIndicesMap) {
          const IndexSet& indexSet = entry.getSecond();

          // Create the block for the variable.
          // The arguments are the index of the multidimensional range and its
          // dimension of interest.
          variablesCaseValues.push_back(entry.getFirst());

          assert(firstMultidimensionalRangeBlock != nullptr);

          llvm::SmallVector<mlir::Type, 2> blockArgTypes(
              2, builder.getI64Type());

          llvm::SmallVector<mlir::Location, 2> blockArgLocations(2, loc);

          mlir::Block* block = builder.createBlock(
              firstMultidimensionalRangeBlock,
              blockArgTypes,
              blockArgLocations);

          variablesCaseBlocks.push_back(block);

          variablesCaseOperandsRefs.push_back(
              entryBlock->getArguments().drop_front());

          // Create the switch for the multidimensional ranges.
          // The switch operates on the index of the multidimensional range.
          llvm::SmallVector<int64_t, 1> caseValues;
          llvm::SmallVector<mlir::Block*, 1> caseBlocks;
          llvm::SmallVector<llvm::SmallVector<mlir::Value, 1>> caseOperands;
          llvm::SmallVector<mlir::ValueRange, 1> caseOperandsRefs;

          for (const MultidimensionalRange& multidimensionalRange :
               llvm::make_range(
                   indexSet.rangesBegin(), indexSet.rangesEnd())) {
            caseValues.push_back(caseValues.size());

            caseBlocks.push_back(
                multidimensionalRangeBlocks[multidimensionalRange]);

            auto& caseOperandsVector = caseOperands.emplace_back();
            caseOperandsVector.push_back(block->getArgument(1));
            caseOperandsRefs.push_back(caseOperandsVector);
          }

          builder.setInsertionPointToStart(block);

          builder.create<mlir::cf::SwitchOp>(
              loc,
              block->getArgument(0), returnBlock, unknownResult,
              builder.getI64TensorAttr(caseValues),
              caseBlocks, caseOperandsRefs);
        }

        // Create the switch inside the entry block.
        // The switch operates on the index of the variable.
        builder.setInsertionPointToEnd(entryBlock);

        builder.create<mlir::cf::SwitchOp>(
            loc,
            entryBlock->getArgument(0), returnBlock, unknownResult,
            builder.getI64TensorAttr(variablesCaseValues),
            variablesCaseBlocks, variablesCaseOperandsRefs);

        return mlir::success();
      }
  };

  class DerivativesMapOpLowering
      : public RuntimeOpRewritePattern<DerivativesMapOp>
  {
    public:
      using RuntimeOpRewritePattern<DerivativesMapOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          DerivativesMapOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            loc, "getDerivative",
            rewriter.getFunctionType(
                rewriter.getI64Type(), rewriter.getI64Type()));

        // Create the entry block.
        mlir::Block* entryBlock = funcOp.addEntryBlock();

        // Create the last block receiving the value to be returned.
        mlir::Block* returnBlock = rewriter.createBlock(
            &funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end(),
            rewriter.getI64Type(), loc);

        rewriter.setInsertionPointToEnd(returnBlock);

        rewriter.create<mlir::func::ReturnOp>(
            loc, returnBlock->getArgument(0));

        // Create the blocks and the switch.
        mlir::ArrayAttr derivativeAttrs = op.getDerivatives();

        size_t numCases = derivativeAttrs.size();
        llvm::SmallVector<int64_t> caseValues(numCases);
        llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
        llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

        for (size_t i = 0; i < numCases; ++i) {
          caseValues[i] = i;
          caseBlocks[i] = rewriter.createBlock(returnBlock);
          caseOperandsRefs[i] = std::nullopt;
        }

        rewriter.setInsertionPointToStart(entryBlock);

        mlir::Value defaultOperand = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getI64IntegerAttr(-1));

        rewriter.create<mlir::cf::SwitchOp>(
            loc,
            entryBlock->getArgument(0), returnBlock, defaultOperand,
            rewriter.getI64TensorAttr(caseValues),
            caseBlocks, caseOperandsRefs);

        rewriter.setInsertionPointToStart(entryBlock);

        // Populate the case blocks.
        for (auto derivative : llvm::enumerate(
                 derivativeAttrs.getAsRange<mlir::IntegerAttr>())) {
          size_t i = derivative.index();
          rewriter.setInsertionPointToStart(caseBlocks[i]);

          mlir::Value result = rewriter.create<mlir::arith::ConstantOp>(
              loc, derivative.value());

          rewriter.create<mlir::cf::BranchOp>(loc, returnBlock, result);
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class VariableGetterOpLowering
      : public RuntimeOpRewritePattern<VariableGetterOp>
  {
    public:
      using RuntimeOpRewritePattern<VariableGetterOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          VariableGetterOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        mlir::Type ptrType =
            mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            loc, op.getSymName(),
            rewriter.getFunctionType(ptrType, op.getResultTypes()));

        mlir::IRMapping mapping;

        // Create the entry block.
        mlir::Block* entryBlock = funcOp.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        // Move the function block.
        size_t numOfIndices = op.getIndices().size();
        mlir::Block* firstSourceBlock = &op.getFunctionBody().front();

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        // Extract and map the indices.
        rewriter.setInsertionPointToStart(entryBlock);
        llvm::SmallVector<mlir::Value, 3> mappedIndices;

        for (size_t i = 0; i < numOfIndices; ++i) {
          mlir::Value address = rewriter.create<mlir::LLVM::GEPOp>(
              loc, ptrType, rewriter.getI64Type(), funcOp.getArgument(0),
              llvm::ArrayRef<mlir::LLVM::GEPArg>(static_cast<int32_t>(i)));

          mlir::Value index = rewriter.create<mlir::LLVM::LoadOp>(
              loc, rewriter.getI64Type(), address);

          index = rewriter.create<mlir::arith::IndexCastOp>(
              loc, rewriter.getIndexType(), index);

          mappedIndices.push_back(index);
        }

        // Branch to the moved body region.
        rewriter.create<mlir::cf::BranchOp>(
            op.getLoc(), firstSourceBlock, mappedIndices);

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class VariableGettersOpLowering
      : public RuntimeOpRewritePattern<VariableGettersOp>
  {
    public:
      using RuntimeOpRewritePattern<VariableGettersOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          VariableGettersOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        mlir::Location loc = op.getLoc();

        mlir::Type ptrType =
            mlir::LLVM::LLVMPointerType::get(rewriter.getContext());

        llvm::SmallVector<mlir::Type, 2> argTypes;
        argTypes.push_back(rewriter.getI64Type());
        argTypes.push_back(ptrType);

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            loc, "getVariableValue",
            rewriter.getFunctionType(argTypes, rewriter.getF64Type()));

        // Create the entry block.
        mlir::Block* entryBlock = funcOp.addEntryBlock();

        // Create the last block receiving the value to be returned.
        mlir::Block* returnBlock = rewriter.createBlock(
            &funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end(),
            rewriter.getF64Type(), loc);

        rewriter.setInsertionPointToEnd(returnBlock);

        rewriter.create<mlir::func::ReturnOp>(
            loc, returnBlock->getArgument(0));

        // Create the blocks and the switch.
        mlir::ArrayAttr getterNameAttrs = op.getNames();

        size_t numCases = getterNameAttrs.size();
        llvm::SmallVector<int64_t> caseValues(numCases);
        llvm::SmallVector<mlir::Block*> caseBlocks(numCases);
        llvm::SmallVector<mlir::ValueRange> caseOperandsRefs(numCases);

        for (size_t i = 0; i < numCases; ++i) {
          caseValues[i] = i;
          caseBlocks[i] = rewriter.createBlock(returnBlock);
          caseOperandsRefs[i] = std::nullopt;
        }

        rewriter.setInsertionPointToStart(entryBlock);

        mlir::Value defaultOperand = rewriter.create<mlir::arith::ConstantOp>(
            loc, rewriter.getF64FloatAttr(0));

        rewriter.create<mlir::cf::SwitchOp>(
            loc,
            entryBlock->getArgument(0), returnBlock, defaultOperand,
            rewriter.getI64TensorAttr(caseValues),
            caseBlocks, caseOperandsRefs);

        rewriter.setInsertionPointToStart(entryBlock);

        // Populate the case blocks.
        for (auto getterName : llvm::enumerate(
                 getterNameAttrs.getAsRange<mlir::FlatSymbolRefAttr>())) {
          size_t i = getterName.index();
          rewriter.setInsertionPointToStart(caseBlocks[i]);

          auto callOp = rewriter.create<mlir::func::CallOp>(
              loc, getterName.value().getValue(), rewriter.getF64Type(),
              funcOp.getArgument(1));

          rewriter.create<mlir::cf::BranchOp>(
              loc, returnBlock, callOp.getResult(0));
        }

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class InitFunctionOpLowering
      : public RuntimeOpRewritePattern<InitFunctionOp>
  {
    public:
      using RuntimeOpRewritePattern<InitFunctionOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          InitFunctionOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), "init",
            rewriter.getFunctionType(std::nullopt, std::nullopt));

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        auto terminator = mlir::cast<YieldOp>(
            funcOp.getFunctionBody().back().getTerminator());

        rewriter.setInsertionPoint(terminator);
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class DeinitFunctionOpLowering
      : public RuntimeOpRewritePattern<DeinitFunctionOp>
  {
      public:
      using RuntimeOpRewritePattern<DeinitFunctionOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          DeinitFunctionOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), "deinit",
            rewriter.getFunctionType(std::nullopt, std::nullopt));

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        auto terminator = mlir::cast<YieldOp>(
            funcOp.getFunctionBody().back().getTerminator());

        rewriter.setInsertionPoint(terminator);
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(terminator);

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class ICModelBeginOpLowering
      : public RuntimeOpRewritePattern<ICModelBeginOp>
  {
    public:
      using RuntimeOpRewritePattern<ICModelBeginOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          ICModelBeginOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), "icModelBegin",
            rewriter.getFunctionType(std::nullopt, std::nullopt));

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
        rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class ICModelEndOpLowering
      : public RuntimeOpRewritePattern<ICModelEndOp>
  {
    public:
      using RuntimeOpRewritePattern<ICModelEndOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          ICModelEndOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), "icModelEnd",
            rewriter.getFunctionType(std::nullopt, std::nullopt));

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
        rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class DynamicModelBeginOpLowering
      : public RuntimeOpRewritePattern<DynamicModelBeginOp>
  {
    public:
      using RuntimeOpRewritePattern<DynamicModelBeginOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          DynamicModelBeginOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), "dynamicModelBegin",
            rewriter.getFunctionType(std::nullopt, std::nullopt));

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
        rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class DynamicModelEndOpLowering
      : public RuntimeOpRewritePattern<DynamicModelEndOp>
  {
    public:
      using RuntimeOpRewritePattern<DynamicModelEndOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          DynamicModelEndOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), "dynamicModelEnd",
            rewriter.getFunctionType(std::nullopt, std::nullopt));

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        rewriter.setInsertionPointToEnd(&funcOp.getBody().back());
        rewriter.create<mlir::func::ReturnOp>(funcOp.getLoc());

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class EquationFunctionOpLowering
      : public RuntimeOpRewritePattern<EquationFunctionOp>
  {
    public:
      using RuntimeOpRewritePattern<EquationFunctionOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          EquationFunctionOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        llvm::SmallVector<mlir::Type, 1> argsTypes;

        argsTypes.push_back(mlir::LLVM::LLVMPointerType::get(
            rewriter.getContext()));

        auto functionType = rewriter.getFunctionType(argsTypes, std::nullopt);

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), op.getSymName(), functionType);

        mlir::Block* entryBlock = funcOp.addEntryBlock();
        rewriter.setInsertionPointToStart(entryBlock);

        llvm::SmallVector<mlir::Value> mappedBoundaries;
        mlir::Value equationBoundariesPtr = funcOp.getArgument(0);

        for (auto arg : llvm::enumerate(op.getArguments())) {
          mlir::Value index = rewriter.create<mlir::arith::ConstantOp>(
              arg.value().getLoc(),
              rewriter.getI64IntegerAttr(arg.index()));

          mlir::Value boundaryPtr = rewriter.create<mlir::LLVM::GEPOp>(
              arg.value().getLoc(),
              equationBoundariesPtr.getType(),
              rewriter.getI64Type(),
              equationBoundariesPtr,
              index);

          mlir::Value mappedBoundary = rewriter.create<mlir::LLVM::LoadOp>(
              boundaryPtr.getLoc(), rewriter.getI64Type(), boundaryPtr);

          mappedBoundary = rewriter.create<mlir::arith::IndexCastOp>(
              mappedBoundary.getLoc(), rewriter.getIndexType(), mappedBoundary);

          mappedBoundaries.push_back(mappedBoundary);
        }

        rewriter.create<mlir::cf::BranchOp>(
            funcOp.getLoc(), &op.getBody().front(), mappedBoundaries);

        rewriter.inlineRegionBefore(
            op.getBody(), funcOp.getFunctionBody(), funcOp.end());

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  class FunctionOpLowering
      : public RuntimeOpRewritePattern<FunctionOp>
  {
    public:
      using RuntimeOpRewritePattern<FunctionOp>
          ::RuntimeOpRewritePattern;

      mlir::LogicalResult matchAndRewrite(
          FunctionOp op,
          mlir::PatternRewriter& rewriter) const override
      {
        if (op.isDeclaration()) {
          return rewriter.notifyMatchFailure(op, "Declaration");
        }

        auto moduleOp = op->getParentOfType<mlir::ModuleOp>();
        rewriter.setInsertionPointToEnd(moduleOp.getBody());

        auto funcOp = rewriter.create<mlir::func::FuncOp>(
            op.getLoc(), op.getSymName(), op.getFunctionType());

        rewriter.inlineRegionBefore(
            op.getBodyRegion(),
            funcOp.getFunctionBody(),
            funcOp.getFunctionBody().end());

        rewriter.eraseOp(op);
        return mlir::success();
      }
  };

  struct ReturnOpLowering : public RuntimeOpRewritePattern<ReturnOp>
  {
    using RuntimeOpRewritePattern<ReturnOp>::RuntimeOpRewritePattern;

    mlir::LogicalResult matchAndRewrite(
        ReturnOp op, mlir::PatternRewriter& rewriter) const override
    {
      rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(op, op.getOperands());
      return mlir::success();
    }
  };
}

namespace
{
  class RuntimeToFuncConversionPass
      : public mlir::impl::RuntimeToFuncConversionPassBase<
          RuntimeToFuncConversionPass>
  {
    public:
      using RuntimeToFuncConversionPassBase<RuntimeToFuncConversionPass>
        ::RuntimeToFuncConversionPassBase;

      void runOnOperation() override;

    private:
      mlir::LogicalResult groupModelOps();

      mlir::LogicalResult convertOps();
  };
}

void RuntimeToFuncConversionPass::runOnOperation()
{
  if (mlir::failed(groupModelOps())) {
    return signalPassFailure();
  }

  if (mlir::failed(convertOps())) {
    return signalPassFailure();
  }
}

mlir::LogicalResult RuntimeToFuncConversionPass::groupModelOps()
{
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter(&getContext());

  llvm::SmallVector<ICModelBeginOp> icModelBeginOps;
  llvm::SmallVector<ICModelEndOp> icModelEndOps;
  llvm::SmallVector<DynamicModelBeginOp> dynamicModelBeginOps;
  llvm::SmallVector<DynamicModelEndOp> dynamicModelEndOps;

  for (auto& op : moduleOp.getOps()) {
    if (auto icModelBeginOp = mlir::dyn_cast<ICModelBeginOp>(op)) {
      icModelBeginOps.push_back(icModelBeginOp);
      continue;
    }

    if (auto icModelEndOp = mlir::dyn_cast<ICModelEndOp>(op)) {
      icModelEndOps.push_back(icModelEndOp);
      continue;
    }

    if (auto dynamicModelBeginOp = mlir::dyn_cast<DynamicModelBeginOp>(op)) {
      dynamicModelBeginOps.push_back(dynamicModelBeginOp);
      continue;
    }

    if (auto dynamicModelEndOp = mlir::dyn_cast<DynamicModelEndOp>(op)) {
      dynamicModelEndOps.push_back(dynamicModelEndOp);
      continue;
    }
  }

  if (icModelBeginOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<ICModelBeginOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (ICModelBeginOp op : icModelBeginOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  if (icModelEndOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<ICModelEndOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (ICModelEndOp op : icModelEndOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  if (dynamicModelBeginOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<DynamicModelBeginOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (DynamicModelBeginOp op : dynamicModelBeginOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  if (dynamicModelEndOps.size() > 1) {
    rewriter.setInsertionPointToEnd(moduleOp.getBody());

    auto mergedOp = rewriter.create<DynamicModelEndOp>(moduleOp.getLoc());
    rewriter.createBlock(&mergedOp.getBodyRegion());
    rewriter.setInsertionPointToStart(mergedOp.getBody());

    for (DynamicModelEndOp op : dynamicModelEndOps) {
      rewriter.mergeBlocks(op.getBody(), mergedOp.getBody());
      rewriter.eraseOp(op);
    }
  }

  return mlir::success();
}

mlir::LogicalResult RuntimeToFuncConversionPass::convertOps()
{
  mlir::ConversionTarget target(getContext());

  target.addIllegalOp<
      VariableGetterOp,
      InitFunctionOp,
      DeinitFunctionOp,
      ICModelBeginOp,
      ICModelEndOp,
      DynamicModelBeginOp,
      DynamicModelEndOp,
      EquationFunctionOp,
      ReturnOp>();

  target.addDynamicallyLegalOp<FunctionOp>([](FunctionOp op) {
    return op.isDeclaration();
  });

  target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
    return true;
  });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<
      VariableGetterOpLowering,
      InitFunctionOpLowering,
      DeinitFunctionOpLowering,
      ICModelBeginOpLowering,
      ICModelEndOpLowering,
      DynamicModelBeginOpLowering,
      DynamicModelEndOpLowering,
      EquationFunctionOpLowering,
      FunctionOpLowering,
      ReturnOpLowering>(&getContext());

  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

namespace mlir
{
  std::unique_ptr<mlir::Pass> createRuntimeToFuncConversionPass()
  {
    return std::make_unique<RuntimeToFuncConversionPass>();
  }
}
