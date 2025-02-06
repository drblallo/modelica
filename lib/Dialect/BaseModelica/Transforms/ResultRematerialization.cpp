#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/IR/DerivativesMap.h"
#include "marco/Dialect/BaseModelica/IR/Ops.h"
#include "marco/Dialect/BaseModelica/IR/Properties.h"
#include "marco/Dialect/BaseModelica/Transforms/ResultRematerialization.h"
#include "marco/Modeling/Graph.h"
#include <deque>

namespace mlir::bmodelica {
#define GEN_PASS_DEF_RESULTREMATERIALIZATIONPASS
#include "marco/Dialect/BaseModelica/Transforms/Passes.h.inc"
} // namespace mlir::bmodelica

using namespace ::mlir::bmodelica;

namespace {
struct EquationWrapper {
  ScheduleOp parentScheduleOp;
  ScheduleBlockOp scheduleBlockOp;
  EquationFunctionOp equationFunctionOp;

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

public:
  using GraphType = UndirectedGraph<EquationWrapper>;
  using ResultRematerializationPassBase<
      ResultRematerializationPass>::ResultRematerializationPassBase;

  void runOnOperation() override;

private:
  mlir::LogicalResult
  handleModel(mlir::ModuleOp module, ModelOp modelOp,
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

  GraphType
  buildScheduleGraph(ScheduleOp scheduleOp, mlir::ModuleOp moduleOp,
                     mlir::SymbolTableCollection &symbolTableCollection);

  void walkGraph(GraphType &graph,
                 const std::function<void(GraphType::VertexProperty &)> &);
};

template <class GraphType, class VPrinter, class EPrinter>
struct GraphDumperImpl {
  // using Traits = typename GraphType::Traits;
  using VertexDescriptor = typename GraphType::VertexDescriptor;
  using EdgeDescriptor = typename GraphType::EdgeDescriptor;

  using VertexProperty = typename GraphType::VertexProperty;
  using EdgeProperty = typename GraphType::EdgeProperty;

  using VertexPrinter =
      std::function<void(VertexProperty &, llvm::raw_ostream &)>;
  using EdgePrinter = std::function<void(EdgeProperty &, llvm::raw_ostream &)>;

private:
  GraphDumperImpl(GraphType *graph) : graph{graph} {}

  void dump(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep) {
    GraphDumperImpl dumper(graph);
    dumper.dumpImpl(os, std::forward<VPrinter>(vp), std::forward<EPrinter>(ep));
  }

  std::string indexToStringIdentifier(int idx) {
    std::string result;
    constexpr int base = 26;

    while (idx >= 0) {
      result = char('A' + (idx % base)) + result;
      idx = (idx / base) - 1;
    }

    return result;
  }

  void dumpImpl(llvm::raw_ostream &os, VPrinter &&vp, EPrinter &&ep) {

    auto mappings = computeMappings();

    outputNodes(os, mappings, std::forward<VPrinter>(vp));

    for (auto eIt = graph->edgesBegin(); eIt != graph->edgesEnd(); eIt++) {

      auto fromVertex = (*eIt).from;
      auto toVertex = (*eIt).to;

      const std::string &identifier = mappings.at(fromVertex);
      const std::string &toIdentifier = mappings.at(toVertex);

      os << identifier << " --> " << toIdentifier << "\n";
    }
  }

  template <class VPrinterT>
  void callVertexPrinter(VPrinterT &&vp, VertexProperty &prop,
                         llvm::raw_ostream &os) {

    if constexpr (!std::is_same_v<VPrinterT, std::nullptr_t>) {
      os << "(\"";
      std::forward<VPrinter>(vp)(prop, os);
      os << "\")";
    }
  }

  void outputNodes(llvm::raw_ostream &os,
                   llvm::DenseMap<VertexDescriptor, std::string> &mappings,
                   VPrinter &&vp) {
    for (auto vIt = graph->verticesBegin(); vIt != graph->verticesEnd();
         vIt++) {
      std::string identifier = mappings.at(*vIt);

      VertexProperty &prop = (**(*vIt).value);

      os << identifier;

      callVertexPrinter<VPrinter>(std::forward<VPrinter>(vp), prop, os);

      os << "\n";
    }
  }

  llvm::DenseMap<VertexDescriptor, std::string> computeMappings() {
    llvm::DenseMap<VertexDescriptor, std::string> result;

    int idx = 0;
    for (auto vIt = graph->verticesBegin(); vIt != graph->verticesEnd();
         vIt++) {
      std::string identifier = indexToStringIdentifier(idx++);
      result[*vIt] = identifier;
    }

    return result;
  }

  friend struct GraphDumper;

private:
  GraphType *graph;
};

struct GraphDumper {
  template <class GraphType>
  static void dump(GraphType *graph, llvm::raw_ostream &os) {

    dump(graph, os, nullptr, nullptr);
  }

  template <class GraphType, class VPrinter>
  static void dump(GraphType *graph, llvm::raw_ostream &os,
                   VPrinter &&vprinter) {
    dump(graph, os, vprinter, nullptr);
  }

  template <class GraphType, class VPrinter, class EPrinter>
  static void dump(GraphType *graph, llvm::raw_ostream &os, VPrinter &&vprinter,
                   EPrinter &&eprinter) {
    GraphDumperImpl<GraphType, VPrinter, EPrinter> instance{graph};
    instance.dump(os, std::forward<VPrinter>(vprinter),
                  std::forward<EPrinter>(eprinter));
  }
};

} // namespace

void ResultRematerializationPass::runOnOperation() {
  mlir::ModuleOp moduleOp = getOperation();

  moduleOp.dump();
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
    auto res = handleModel(moduleOp, modelOp, symTables);

    if (res.failed()) {
      return signalPassFailure();
    }
  }
}

mlir::LogicalResult ResultRematerializationPass::handleModel(
    mlir::ModuleOp moduleOp, ModelOp modelOp,
    mlir::SymbolTableCollection &symbolTableCollection) {
  llvm::dbgs() << "Handling model: " << modelOp.getName() << "\n";

  // Get all model variables
  auto variableOps = collectVariables(modelOp, symbolTableCollection);

  // Get state variables and their derivatives
  const DerivativesMap &derivativesMap = modelOp.getProperties().derivativesMap;
  auto variablePairs = getVariablePairs(modelOp, variableOps,
                                        symbolTableCollection, derivativesMap);

  llvm::DenseMap<llvm::StringRef,
                 marco::modeling::internal::UndirectedGraph<EquationWrapper>>
      scheduleGraphs;

  auto scheduleOps = getSchedules(modelOp);

  for (ScheduleOp scheduleOp : scheduleOps) {
    scheduleGraphs[scheduleOp.getName()] =
        buildScheduleGraph(scheduleOp, moduleOp, symbolTableCollection);
  }

  for (auto &[name, graph] : scheduleGraphs) {
    llvm::dbgs() << "--- Graph for " << name << " ---\n";

    auto vertexPrinter = [](EquationWrapper &eq, llvm::raw_ostream &os) {
      bool first = true;
      for (Variable &write : eq.writes) {
        std::string res{};
        llvm::raw_string_ostream ss{res};
        write.indices.dump(ss);

        if (!first)
          os << ", ";
        first = false;
        os << write.name << " " << ss.str();
      }
    };

    GraphDumper::dump(&graph, llvm::dbgs(), vertexPrinter, nullptr);
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
ResultRematerializationPass::buildScheduleGraph(
    ScheduleOp scheduleOp, mlir::ModuleOp moduleOp,
    mlir::SymbolTableCollection &symbolTableCollection) {

  using namespace marco::modeling::internal;

  scheduleOp.dump();

  using GraphType = ResultRematerializationPass::GraphType;
  using VertexDescriptor = typename GraphType::VertexDescriptor;

  GraphType graph;
  llvm::dbgs() << "Handling schedule " << scheduleOp.getName() << "\n";

  auto scheduleBlockOps = getScheduleBlocks(scheduleOp);

  VertexDescriptor currentVertex{};

  llvm::DenseSet<VertexDescriptor> currentScheduleBlockEquations{};
  llvm::DenseSet<VertexDescriptor> nextScheduleBlockEquations{};

  for (ScheduleBlockOp scheduleBlockOp : scheduleBlockOps) {
    VariablesList writes =
        scheduleBlockOp.getProperties().getWrittenVariables();

    EquationWrapper node{};

    // Get the read and written variables
    for (const auto &write : writes) {
      node.writes.push_back(write);
    }

    for (const auto &read :
         scheduleBlockOp.getProperties().getReadVariables()) {
      node.reads.push_back(read);
    }

    node.parentScheduleOp = scheduleOp;
    node.scheduleBlockOp = scheduleBlockOp;

    // Get the callee
    llvm::SmallVector<EquationCallOp> equationCallOps{};

    scheduleBlockOp.walk([&](mlir::Operation *op) {
      if (EquationCallOp equationCallOp = mlir::dyn_cast<EquationCallOp>(op)) {
        equationCallOps.push_back(equationCallOp);
      }
    });

    for (EquationCallOp equationCallOp : equationCallOps) {
      llvm::dbgs() << "Callee: " << equationCallOp.getCallee().str() << "\n";

      // Try to get the equation

      mlir::Operation *calleeSym = symbolTableCollection.lookupSymbolIn(
          moduleOp,
          mlir::FlatSymbolRefAttr::get(equationCallOp.getCalleeAttr()));

      if (EquationFunctionOp equationFunctionOp =
              mlir::dyn_cast<EquationFunctionOp>(calleeSym)) {

        node.equationFunctionOp = equationFunctionOp;

        VertexDescriptor newVertex = graph.addVertex(std::move(node));
        nextScheduleBlockEquations.insert(newVertex);

        for (VertexDescriptor prevBlockVertex : currentScheduleBlockEquations) {
          graph.addEdge(prevBlockVertex, newVertex);
        }
      }
    }

    currentScheduleBlockEquations = std::move(nextScheduleBlockEquations);
    nextScheduleBlockEquations.clear();
  }

  return graph;
}

[[maybe_unused]]
void ResultRematerializationPass::walkGraph(
    GraphType &graph,
    const std::function<void(typename GraphType::VertexProperty &)> &callBack) {
  auto vertex = *graph.verticesBegin();

  // BFS, preorder visit
  std::vector<decltype(vertex)> stack;
  stack.emplace_back(vertex);

  // Ensure single visitation.
  llvm::DenseSet<GraphType::VertexDescriptor> visited;

  while (!stack.empty()) {
    vertex = stack.back();
    stack.pop_back();

    if (visited.contains(vertex)) {
      continue;
    }

    for (auto eIt = graph.outgoingEdgesBegin(vertex);
         eIt != graph.outgoingEdgesEnd(vertex); eIt++) {
      auto target = (*eIt).to;
      if (!visited.contains(target)) {
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
  {
  }
}
} // namespace mlir::bmodelica
