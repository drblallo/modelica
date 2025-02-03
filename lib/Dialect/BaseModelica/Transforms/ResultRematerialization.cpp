#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/IR/Properties.h"
#include "marco/Dialect/BaseModelica/Transforms/ResultRematerialization.h"

namespace mlir::bmodelica {
#define GEN_PASS_DEF_RESULTREMATERIALIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
class ResultRematerializationPass
    : public impl::ResultRematerializationPassBase<
          ResultRematerializationPass> {

public:
  using ResultRematerializationPassBase<
      ResultRematerializationPass>::ResultRematerializationPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult
  handleModel(ModelOp modelOp,
              mlir::SymbolTableCollection &symbolTableCollection);

  llvm::SmallVector<VariableOp>
  collectVariables(ModelOp modelOp, mlir::SymbolTableCollection &symTables) {
    llvm::SmallVector<VariableOp> result{};

    for (VariableOp var : modelOp.getVariables()) {
      result.push_back(var);
      llvm::dbgs() << "Found variable " << var.getName() << "\n";
    }

    return result;
  }

  //============================================================
  // Utility functions
  //============================================================
  llvm::SmallVector<ScheduleOp> getSchedules(ModelOp modelOp);
  llvm::SmallVector<ScheduleBlockOp> getScheduleBlocks(ScheduleOp scheduleOp);

  llvm::SmallVector<std::pair<VariableOp, VariableOp>>
  getVariablePairs(ModelOp modelOp, llvm::SmallVector<VariableOp> &variableOps,
                   mlir::SymbolTableCollection &symbolTableCollection,
                   const DerivativesMap &derivativesMap);
};
} // namespace

void ResultRematerializationPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();
  mlir::IRRewriter rewriter{&getContext()};

  mlir::SymbolTableCollection symTables{};

  llvm::SmallVector<ModelOp, 1> modelOps;

  // Capture the models
  walkClasses(moduleOp, [&](mlir::Operation *op) {
    if (auto modelOp = mlir::dyn_cast<ModelOp>(op)) {
      modelOps.push_back(modelOp);
    }
  });

  for (ModelOp modelOp : modelOps) {
    auto res = handleModel(modelOp, symTables);

    if (res.failed()) {
      return signalPassFailure();
    }
  }
}

struct RMScheduleBlockNode {
  ScheduleOp parentScheduleOp;
  ScheduleBlockOp scheduleBlockOp;

  llvm::SmallVector<Variable> reads;
  llvm::SmallVector<Variable> writes;
};

mlir::LogicalResult ResultRematerializationPass::handleModel(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection) {
  llvm::dbgs() << "Handling model: " << modelOp.getName() << "\n";

  // Get all model variables
  auto variableOps = collectVariables(modelOp, symbolTableCollection);

  // Get state variables and their derivatives
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;
  auto variablePairs = getVariablePairs(modelOp, variableOps,
                                        symbolTableCollection, derivativesMap);

  llvm::DenseMap<llvm::StringRef, std::deque<RMScheduleBlockNode>>
      scheduleGraphs;

  auto scheduleOps = getSchedules(modelOp);

  for (ScheduleOp scheduleOp : scheduleOps) {

    std::deque<RMScheduleBlockNode> list;
    llvm::dbgs() << "Handling schedule " << scheduleOp.getName() << "\n";

    auto scheduleBlockOps = getScheduleBlocks(scheduleOp);

    for (ScheduleBlockOp scheduleBlockOp : scheduleBlockOps) {
      VariablesList writes =
          scheduleBlockOp.getProperties().getWrittenVariables();

      RMScheduleBlockNode node{};

      for (const auto &write : writes) {
        node.writes.push_back(write);
      }

      for (const auto &read :
           scheduleBlockOp.getProperties().getReadVariables()) {
        node.reads.push_back(read);
      }

      node.parentScheduleOp = scheduleOp;
      node.scheduleBlockOp = scheduleBlockOp;

      list.emplace_back(std::move(node));
    }

    scheduleGraphs[scheduleOp.getName()] = std::move(list);
  }

  for (auto &[name, list] : scheduleGraphs) {
    llvm::dbgs() << "For schedule " << name << "\n";

    for (const RMScheduleBlockNode &node : list) {
      node.scheduleBlockOp->dump();
    }
  }

  return mlir::success();
}

llvm::SmallVector<std::pair<VariableOp, VariableOp>>
ResultRematerializationPass::getVariablePairs(
    ModelOp modelOp, llvm::SmallVector<VariableOp> &variableOps,
    mlir::SymbolTableCollection &symbolTableCollection,
    const DerivativesMap &derivativesMap) {
  llvm::SmallVector<std::pair<VariableOp, VariableOp>> result;

  for (VariableOp variableOp : variableOps) {
    llvm::dbgs() << "Handling variable " << variableOp.getName() << "\n";
    if (auto derivativeName = derivativesMap.getDerivative(
            mlir::FlatSymbolRefAttr::get(variableOp.getSymNameAttr()))) {
      auto derivativeVariableOp =
          symbolTableCollection.lookupSymbolIn<VariableOp>(modelOp,
                                                           *derivativeName);

      result.push_back(std::make_pair(variableOp, derivativeVariableOp));
    }
  }

  return result;
}

llvm::SmallVector<ScheduleOp>
ResultRematerializationPass::getSchedules(ModelOp modelOp) {
  // Get the schedules
  llvm::SmallVector<ScheduleOp> result{};

  modelOp.walk([&](mlir::Operation *op) {
    if (ScheduleOp scheduleOp = mlir::dyn_cast<ScheduleOp>(op)) {
      // TODO: Remove this condition or refine it
      if (scheduleOp.getName() == "dynamic") {
        result.push_back(scheduleOp);
      }
    }
  });

  return result;
}

llvm::SmallVector<ScheduleBlockOp>
ResultRematerializationPass::getScheduleBlocks(ScheduleOp scheduleOp) {
  // Get the schedules
  llvm::SmallVector<ScheduleBlockOp> result{};

  scheduleOp.walk([&](mlir::Operation *op) {
    if (ScheduleBlockOp scheduleBlockOp = mlir::dyn_cast<ScheduleBlockOp>(op)) {
      result.push_back(scheduleBlockOp);
    }
  });

  return result;
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createResultRematerializationPass() {
  return std::make_unique<ResultRematerializationPass>();
  {}
}
} // namespace mlir::bmodelica
