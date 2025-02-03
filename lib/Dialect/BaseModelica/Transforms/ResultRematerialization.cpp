#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
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
                                  mlir::SymbolTableCollection &symTables);

  llvm::SmallVector<VariableOp, 8> collectVariables(ModelOp modelOp,
                                  mlir::SymbolTableCollection &symTables)
  {
    llvm::SmallVector<VariableOp, 8> result{};

    modelOp.walk([&](mlir::Operation *op) {
      if ( auto var = mlir::dyn_cast<VariableOp>(op) ) {
        result.push_back(var);
      }
    });

    return result;
  }

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
    ModelOp modelOp, mlir::SymbolTableCollection &symTables)
{
  auto variables = collectVariables(modelOp, symTables);

  for ( auto v : variables ) {
    llvm::outs() << v.getName() << "\n";
  }

  return mlir::success();

}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createResultRematerializationPass() {
  return std::make_unique<ResultRematerializationPass>();
  {}
}
} // namespace mlir::bmodelica
