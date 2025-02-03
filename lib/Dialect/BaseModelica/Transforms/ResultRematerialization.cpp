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

  mlir::LogicalResult handleModel(ModelOp modelOp,
                                  mlir::SymbolTableCollection &symbolTableCollection);

  llvm::SmallVector<VariableOp> collectVariables(ModelOp modelOp,
                                  mlir::SymbolTableCollection &symTables)
  {
    llvm::SmallVector<VariableOp> result{};

    for ( VariableOp var : modelOp.getVariables() ) {
        result.push_back(var);
        llvm::dbgs() << "Found variable " << var.getName() << "\n";
    }

    return result;
  }

llvm::SmallVector<std::pair<VariableOp, VariableOp>>
getVariablePairs(ModelOp modelOp, llvm::SmallVector<VariableOp> &variableOps, mlir::SymbolTableCollection &symbolTableCollection, const DerivativesMap &derivativesMap) ;


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

    if ( res.failed() ) {
      return signalPassFailure();
    }
  }
}


mlir::LogicalResult ResultRematerializationPass::handleModel(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection)
{
  llvm::dbgs() << "Handling model: " << modelOp.getName() << "\n";

  // Get all model variables
  auto variableOps = collectVariables(modelOp, symbolTableCollection);

  // Get state variables and their derivatives
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;
  auto variablePairs = getVariablePairs(modelOp, variableOps, symbolTableCollection, derivativesMap);

  // Get the schedules
  llvm::SmallVector<ScheduleOp> scheduleOps{};


  modelOp.walk([&] (mlir::Operation *op) {
    if ( ScheduleOp scheduleOp = mlir::dyn_cast<ScheduleOp>(op) ) {
      // TODO: Remove this condition or refine it
      if ( scheduleOp.getName() == "dynamic" ) {
        scheduleOps.push_back(scheduleOp);
      }
    }
  });

  for ( ScheduleOp scheduleOp : scheduleOps ) {
    llvm::dbgs() << "Handling schedule " << scheduleOp.getName() << "\n";

    llvm::SmallVector<ScheduleBlockOp> scheduleBlockOps{};

    scheduleOp.walk([&] (mlir::Operation *op) {
      if ( ScheduleBlockOp scheduleBlockOp = mlir::dyn_cast<ScheduleBlockOp>(op) ) {
        scheduleBlockOps.push_back(scheduleBlockOp);
      }
    });

    for ( ScheduleBlockOp scheduleBlockOp : scheduleBlockOps ) {
      VariablesList list = scheduleBlockOp.getProperties().getWrittenVariables();


    llvm::dbgs() << "For schedule : " << scheduleOp.getName() << "\n";
      for ( auto x : list ) {
        llvm::dbgs() << "Written variable: " << x.name << "\n";
      }

      for ( auto x : scheduleBlockOp.getProperties().getReadVariables() ) {
        llvm::dbgs() << "Read variable: " << x.name << "\n";
      }


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


namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createResultRematerializationPass() {
  return std::make_unique<ResultRematerializationPass>();
  {}
}
} // namespace mlir::bmodelica
