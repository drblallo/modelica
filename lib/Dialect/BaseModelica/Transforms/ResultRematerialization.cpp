#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/IR/Properties.h"
#include "marco/Modeling/Graph.h"
#include "marco/Dialect/BaseModelica/Transforms/ResultRematerialization.h"
#include <deque>

namespace mlir::bmodelica {
#define GEN_PASS_DEF_RESULTREMATERIALIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
struct RMScheduleBlockNode {
  ScheduleOp parentScheduleOp;
  ScheduleBlockOp scheduleBlockOp;

  llvm::SmallVector<Variable> reads;
  llvm::SmallVector<Variable> writes;
};
} // namespace

namespace {

// Get the definitions from the graphing library
using namespace marco::modeling::internal;

class ResultRematerializationPass
    : public ::mlir::bmodelica::impl::ResultRematerializationPassBase<
          ResultRematerializationPass> {

  using GraphType = UndirectedGraph<RMScheduleBlockNode>;

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

  //===---------------------------------------------------------===//
  // Utility functions
  //===---------------------------------------------------------===//
  llvm::SmallVector<ScheduleOp> getSchedules(ModelOp modelOp);
  llvm::SmallVector<ScheduleBlockOp> getScheduleBlocks(ScheduleOp scheduleOp);

  llvm::SmallVector<std::pair<VariableOp, VariableOp>>
  getVariablePairs(ModelOp modelOp, llvm::SmallVector<VariableOp> &variableOps,
                   mlir::SymbolTableCollection &symbolTableCollection,
                   const DerivativesMap &derivativesMap);

  GraphType buildScheduleGraph(ScheduleOp scheduleOp);


  void walkGraph(GraphType &graph, const std::function<void (GraphType::VertexProperty &)> &);

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


mlir::LogicalResult ResultRematerializationPass::handleModel(
    ModelOp modelOp, mlir::SymbolTableCollection &symbolTableCollection) {
  llvm::dbgs() << "Handling model: " << modelOp.getName() << "\n";

  // Get all model variables
  auto variableOps = collectVariables(modelOp, symbolTableCollection);

  // Get state variables and their derivatives
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;
  auto variablePairs = getVariablePairs(modelOp, variableOps,
                                        symbolTableCollection, derivativesMap);

  llvm::DenseMap<llvm::StringRef, marco::modeling::internal::UndirectedGraph<RMScheduleBlockNode>>
    scheduleGraphs;

  auto scheduleOps = getSchedules(modelOp);

  for (ScheduleOp scheduleOp : scheduleOps) {
    scheduleGraphs[scheduleOp.getName()] = buildScheduleGraph(scheduleOp);
  }

  for (auto &[name, list] : scheduleGraphs) {
    llvm::dbgs() << "For schedule " << name << "\n";


    auto &graph = scheduleGraphs[name];


    walkGraph(graph, [] ( RMScheduleBlockNode &node) {
      llvm::dbgs() << node.parentScheduleOp.getName() << "\n";
    });

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

ResultRematerializationPass::GraphType
ResultRematerializationPass::buildScheduleGraph(ScheduleOp scheduleOp) {

  using namespace marco::modeling::internal;

  using GraphType = UndirectedGraph<RMScheduleBlockNode>;

  GraphType graph;
  llvm::dbgs() << "Handling schedule " << scheduleOp.getName() << "\n";

  auto scheduleBlockOps = getScheduleBlocks(scheduleOp);

  bool init = false;
  GraphType::VertexDescriptor currentVertex{};

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

    GraphType::VertexDescriptor newVertex = graph.addVertex(std::move(node));

    if ( ! init ) {
      init = true;
    } else {
      graph.addEdge(currentVertex, newVertex);
    }

    currentVertex = newVertex;
  }

  return graph;
}

void ResultRematerializationPass::walkGraph(
    GraphType &graph,
    const std::function<void (typename GraphType::VertexProperty &)> &callBack)
{
  auto vertex = *graph.verticesBegin();

  // BFS, preorder visit
  std::vector<decltype(vertex)> stack;
  stack.emplace_back(vertex);

  // Ensure single visitation.
  llvm::DenseSet<GraphType::VertexDescriptor> visited;

  while ( ! stack.empty() ) {
    vertex = stack.back();
    stack.pop_back();

    if ( visited.contains(vertex) ) {
      continue;
    }

    for ( auto eIt = graph.outgoingEdgesBegin(vertex); eIt != graph.outgoingEdgesEnd(vertex); eIt++ )
    {
      auto target = (*eIt).to;
      if ( ! visited.contains(target) ) {
        stack.emplace_back(target);
      }
    }

    visited.insert(vertex);

    callBack(*(*vertex.value));
  }
}

namespace mlir::bmodelica {
std::unique_ptr<mlir::Pass> createResultRematerializationPass() {
  return std::make_unique<ResultRematerializationPass>();
  {}
}
} // namespace mlir::bmodelica
