#ifndef MARCO_MODELING_MATCHING_H
#define MARCO_MODELING_MATCHING_H

#ifndef DEBUG_TYPE
#define DEBUG_TYPE "matching"
#endif

#include "marco/Modeling/AccessFunction.h"
#include "marco/Modeling/Dumpable.h"
#include "marco/Modeling/Graph.h"
#include "marco/Modeling/LocalMatchingSolutions.h"
#include "marco/Modeling/MCIM.h"
#include "marco/Modeling/Range.h"
#include "marco/Modeling/TreeOStream.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <atomic>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <variant>

namespace marco::modeling {
namespace matching {
// This class must be specialized for the variable type that is used during
// the matching process.
template <typename VariableType>
struct VariableTraits {
  // Elements to provide:
  //
  // typedef Id : the ID type of the variable.
  //
  // static Id getId(const VariableType*)
  //    return the ID of the variable.
  //
  // static size_t getRank(const VariableType*)
  //    return the number of dimensions.
  //
  // static IndexSet getIndices(const VariableType*)
  //    return the indices of a variable.
  //
  // static llvm::raw_ostream& dump(const VariableType*, llvm::raw_ostream&)
  //    print debug information.

  using Id = typename VariableType::UnknownVariableTypeError;
};

// This class must be specialized for the equation type that is used during
// the matching process.
template <typename EquationType>
struct EquationTraits {
  // Elements to provide:
  //
  // typedef Id : the ID type of the equation.
  //
  // static Id getId(const EquationType*)
  //    return the ID of the equation.
  //
  // static size_t getNumOfIterationVars(const EquationType*)
  //    return the number of induction variables.
  //
  // static MultidimensionalRange getIterationRanges(const EquationType*)
  //    return the iteration ranges.
  //
  // typedef VariableType : the type of the accessed variable
  //
  // typedef AccessProperty : the access property (this is optional, and if
  //                          not specified an empty one is used)
  //
  // static std::vector<Access<VariableType, AccessProperty>>
  // getAccesses(const EquationType*)
  //    return the accesses done by the equation.
  //
  // static llvm::raw_ostream& dump(const EquationType*, llvm::raw_ostream&)
  //    print debug information.

  using Id = typename EquationType::UnknownEquationTypeError;
};
} // namespace matching

namespace internal {
namespace matching {
/// Represent a generic vectorized entity whose scalar elements
/// can be matched with the scalar elements of other arrays.
/// The relationship is tracked by means of an incidence matrix.
class Matchable {
public:
  explicit Matchable(IndexSet matchableIndices);

  const IndexSet &getMatched() const;

  IndexSet getUnmatched() const;

  /// Check whether all the scalar elements of this array have been matched.
  bool allComponentsMatched() const;

  void addMatch(const IndexSet &newMatch);

  void removeMatch(const IndexSet &removedMatch);

private:
  IndexSet matchableIndices;
  IndexSet match;
};

/// Graph node representing a variable.
template <typename VariableProperty>
class VariableVertex : public Matchable, public Dumpable {
public:
  using Property = VariableProperty;
  using Traits =
      typename ::marco::modeling::matching::VariableTraits<VariableProperty>;
  using Id = typename Traits::Id;

  explicit VariableVertex(VariableProperty property)
      : Matchable(getIndices(property)), property(property), visible(true) {
    // Scalar variables can be represented by means of an array with just one
    // element
    assert(getRank() > 0 && "Scalar variables are not supported");
  }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Variable\n";
    os << "  - ID: " << getId() << "\n";
    os << "  - Rank: " << getRank() << "\n";
    os << "  - Indices: " << getIndices() << "\n";
    os << "  - Matched: " << getMatched() << "\n";
    os << "  - Details: ";
    Traits::dump(&property, os);
    os << "\n";
  }

  VariableProperty &getProperty() { return property; }

  const VariableProperty &getProperty() const { return property; }

  Id getId() const { return Traits::getId(&property); }

  size_t getRank() const {
    auto result = getRank(property);
    assert(result > 0);
    return result;
  }

  IndexSet getIndices() const {
    auto result = getIndices(property);
    assert(!result.empty());
    return result;
  }

  unsigned int flatSize() const { return getIndices().flatSize(); }

  bool isVisible() const { return visible; }

  void setVisibility(bool visibility) { visible = visibility; }

private:
  static size_t getRank(const VariableProperty &p) {
    return Traits::getRank(&p);
  }

  static IndexSet getIndices(const VariableProperty &p) {
    return Traits::getIndices(&p);
  }

  // Custom equation property
  VariableProperty property;

  // Whether the node is visible or has been erased
  bool visible;
};

class EmptyAccessProperty {};

template <class EquationProperty>
struct get_access_property {
  template <typename U>
  using Traits = ::marco::modeling::matching::EquationTraits<U>;

  template <class U, typename = typename Traits<U>::AccessProperty>
  static typename Traits<U>::AccessProperty property(int);

  template <class U>
  static EmptyAccessProperty property(...);

  using type = decltype(property<EquationProperty>(0));
};
} // namespace matching
} // namespace internal

namespace matching {
template <typename VariableProperty,
          typename AccessProperty = internal::matching::EmptyAccessProperty>
class Access {
public:
  using Property = AccessProperty;

  Access(const VariableProperty &variable,
         std::unique_ptr<AccessFunction> accessFunction,
         AccessProperty property = {})
      : variable(VariableTraits<VariableProperty>::getId(&variable)),
        accessFunction(std::move(accessFunction)),
        property(std::move(property)) {}

  Access(const Access &other)
      : variable(other.variable), accessFunction(other.accessFunction->clone()),
        property(other.property) {}

  ~Access() = default;

  typename VariableTraits<VariableProperty>::Id getVariable() const {
    return variable;
  }

  const AccessFunction &getAccessFunction() const {
    assert(accessFunction != nullptr);
    return *accessFunction;
  }

  const AccessProperty &getProperty() const { return property; }

private:
  typename VariableTraits<VariableProperty>::Id variable;
  std::unique_ptr<AccessFunction> accessFunction;
  AccessProperty property;
};
} // namespace matching

namespace internal::matching {
template <typename T>
void insertOrAdd(std::map<T, IndexSet> &map, T key, IndexSet value) {
  if (auto it = map.find(key); it != map.end()) {
    it->second += std::move(value);
  } else {
    map.emplace(key, std::move(value));
  }
}

/// Graph node representing an equation.
template <typename EquationProperty>
class EquationVertex : public Matchable, public Dumpable {
public:
  using Property = EquationProperty;
  using Traits =
      typename ::marco::modeling::matching::EquationTraits<EquationProperty>;
  using Id = typename Traits::Id;

  using Access = ::marco::modeling::matching::Access<
      typename Traits::VariableType,
      typename get_access_property<EquationProperty>::type>;

  EquationVertex(EquationProperty property)
      : Matchable(IndexSet(getIterationRanges(property))), property(property),
        visible(true) {}

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Equation\n";
    os << "  - ID: " << getId() << "\n";
    os << "  - Iteration ranges: " << getIterationRanges() << "\n";
    os << "  - Matched: " << getMatched() << "\n";
    os << "  - Details: ";
    Traits::dump(&property, os);
    os << "\n";
  }

  EquationProperty &getProperty() { return property; }

  const EquationProperty &getProperty() const { return property; }

  Id getId() const { return Traits::getId(&property); }

  size_t getNumOfIterationVars() const {
    auto result = getNumOfIterationVars(property);
    assert(result > 0);
    return result;
  }

  IndexSet getIterationRanges() const {
    auto result = getIterationRanges(property);
    assert(!result.empty());
    return result;
  }

  unsigned int flatSize() const { return getIterationRanges().flatSize(); }

  std::vector<Access> getVariableAccesses() const {
    return Traits::getAccesses(&property);
  }

  bool isVisible() const { return visible; }

  void setVisibility(bool visibility) { visible = visibility; }

private:
  static size_t getNumOfIterationVars(const EquationProperty &p) {
    return Traits::getNumOfIterationVars(&p);
  }

  static IndexSet getIterationRanges(const EquationProperty &p) {
    return Traits::getIterationRanges(&p);
  }

  // Custom equation property
  EquationProperty property;

  // Whether the node is visible or has been erased
  bool visible;
};

template <typename Variable, typename Equation>
class Edge : public Dumpable {
public:
  using AccessProperty = typename Equation::Access::Property;

  Edge(typename Equation::Id equation, typename Variable::Id variable,
       IndexSet equationRanges, IndexSet variableRanges,
       typename Equation::Access access)
      : equation(std::move(equation)), variable(std::move(variable)),
        accessFunction(access.getAccessFunction().clone()),
        accessProperty(access.getProperty()),
        incidenceMatrix(equationRanges, variableRanges),
        matchMatrix(equationRanges, variableRanges), visible(true) {
    incidenceMatrix.apply(getAccessFunction());
  }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Edge\n";
    os << "  - Equation: " << equation << "\n";
    os << "  - Variable: " << variable << "\n";
    os << "  - Incidence matrix:\n" << incidenceMatrix << "\n";
    os << "  - Matched equations: " << getMatched().flattenColumns() << "\n";
    os << "  - Matched variables: " << getMatched().flattenRows() << "\n";
    os << "  - Match matrix:\n" << getMatched() << "\n";
  }

  const AccessFunction &getAccessFunction() const { return *accessFunction; }

  const AccessProperty &getAccessProperty() const { return accessProperty; }

  const MCIM &getIncidenceMatrix() const { return incidenceMatrix; }

  void addMatch(const MCIM &match) { matchMatrix += match; }

  void removeMatch(const MCIM &match) { matchMatrix -= match; }

  const MCIM &getMatched() const { return matchMatrix; }

  MCIM getUnmatched() const { return incidenceMatrix - matchMatrix; }

  bool isVisible() const { return visible; }

  void setVisibility(bool visibility) { visible = visibility; }

private:
  // Equation's ID. Just for debugging purpose
  typename Equation::Id equation;

  // Variable's ID. Just for debugging purpose
  typename Variable::Id variable;

  std::unique_ptr<AccessFunction> accessFunction;
  AccessProperty accessProperty;
  MCIM incidenceMatrix;
  MCIM matchMatrix;

  bool visible;
};

template <typename Graph, typename Variable, typename Equation>
class BFSStep : public Dumpable {
public:
  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

  using VertexProperty = typename Graph::VertexProperty;

  BFSStep(const Graph &graph, VertexDescriptor node, IndexSet candidates)
      : graph(&graph), previous(nullptr), node(std::move(node)),
        candidates(std::move(candidates)), edge(std::nullopt),
        mappedFlow(std::nullopt) {}

  BFSStep(const Graph &graph, std::shared_ptr<BFSStep> previous,
          EdgeDescriptor edge, VertexDescriptor node, IndexSet candidates,
          MCIM mappedFlow)
      : graph(&graph), previous(previous), node(std::move(node)),
        candidates(std::move(candidates)), edge(std::move(edge)),
        mappedFlow(std::move(mappedFlow)) {}

  BFSStep(const BFSStep &other) = default;

  BFSStep(BFSStep &&other) = default;

  ~BFSStep() = default;

  BFSStep &operator=(const BFSStep &other) = default;

  BFSStep &operator=(BFSStep &&other) = default;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "BFS step\n";

    os << "Node: ";
    dumpId(os, getNode());
    os << "\n";

    os << "Candidates:\n" << getCandidates();

    if (hasPrevious()) {
      os << "\n";
      os << "Edge: ";
      dumpId(os, getEdge().from);
      os << " - ";
      dumpId(os, getEdge().to);
      os << "\n";

      os << "Mapped flow:\n" << getMappedFlow() << "\n";
      os << "Previous:\n";
      getPrevious()->dump(os);
    }
  }

  bool hasPrevious() const { return previous != nullptr; }

  const BFSStep *getPrevious() const { return previous.get(); }

  const VertexDescriptor &getNode() const { return node; }

  const IndexSet &getCandidates() const { return candidates; }

  const EdgeDescriptor &getEdge() const {
    assert(edge.has_value());
    return *edge;
  }

  const MCIM &getMappedFlow() const {
    assert(mappedFlow.has_value());
    return *mappedFlow;
  }

private:
  void dumpId(llvm::raw_ostream &os, VertexDescriptor descriptor) const {
    const VertexProperty &nodeProperty = (*graph)[descriptor];

    if (std::holds_alternative<Variable>(nodeProperty)) {
      os << std::get<Variable>(nodeProperty).getId();
    } else {
      os << std::get<Equation>(nodeProperty).getId();
    }
  }

private:
  // Stored for debugging purpose
  const Graph *graph;

  std::shared_ptr<BFSStep> previous;
  VertexDescriptor node;
  IndexSet candidates;
  std::optional<EdgeDescriptor> edge;
  std::optional<MCIM> mappedFlow;
};

template <typename BFSStep>
class Frontier : public Dumpable {
private:
  template <typename T>
  using Container = std::vector<std::shared_ptr<T>>;

public:
  using iterator = typename Container<BFSStep>::iterator;
  using const_iterator = typename Container<BFSStep>::const_iterator;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Frontier\n";

    for (const auto &step : steps) {
      step->dump(os);
      os << "\n";
    }
  }

  friend void swap(Frontier &first, Frontier &second) {
    using std::swap;

    swap(first.steps, second.steps);
  }

  BFSStep &operator[](size_t index) {
    assert(index < steps.size());
    return steps[index];
  }

  const BFSStep &operator[](size_t index) const {
    assert(index < steps.size());
    return steps[index];
  }

  bool empty() const { return steps.empty(); }

  template <typename... Args>
  void emplace_back(Args &&...args) {
    steps.emplace_back(std::make_shared<BFSStep>(std::forward<Args>(args)...));
  }

  void push_back(std::shared_ptr<BFSStep> step) {
    steps.push_back(std::move(step));
  }

  void clear() { steps.clear(); }

  void swap(Frontier &other) { steps.swap(other.steps); }

  iterator begin() { return steps.begin(); }

  const_iterator begin() const { return steps.begin(); }

  iterator end() { return steps.end(); }

  const_iterator end() const { return steps.end(); }

private:
  Container<BFSStep> steps;
};

template <typename Graph, typename Variable, typename Equation>
class Flow : public Dumpable {
private:
  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

  using VertexProperty = typename Graph::VertexProperty;

public:
  Flow(const Graph &graph, VertexDescriptor source, EdgeDescriptor edge,
       const MCIM &delta)
      : graph(&graph), source(std::move(source)), edge(std::move(edge)),
        delta(std::move(delta)) {
    assert(this->source == this->edge.from || this->source == this->edge.to);
  }

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Flow\n";

    os << "  - Source: ";
    dumpId(os, source);
    os << "\n";

    os << "  - Edge: ";
    dumpId(os, edge.from);
    os << " - ";
    dumpId(os, edge.to);
    os << "\n";

    os << "  - Delta:\n" << delta;
  }

private:
  void dumpId(llvm::raw_ostream &os, VertexDescriptor descriptor) const {
    const VertexProperty &nodeProperty = (*graph)[descriptor];

    if (std::holds_alternative<Variable>(nodeProperty)) {
      os << std::get<Variable>(nodeProperty).getId();
    } else {
      os << std::get<Equation>(nodeProperty).getId();
    }
  }

private:
  // Stored for debugging purpose
  const Graph *graph;

public:
  const VertexDescriptor source;
  const EdgeDescriptor edge;
  const MCIM delta;
};

template <typename Flow>
class AugmentingPath : public Dumpable {
private:
  template <typename T>
  using Container = std::vector<T>;

public:
  using const_iterator = typename Container<Flow>::const_iterator;

  template <typename Flows>
  explicit AugmentingPath(const Flows &flows)
      : flows(flows.begin(), flows.end()) {}

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "Augmenting path\n";

    for (const auto &flow : flows) {
      os << "  ";
      flow.dump(os);
      os << "\n";
    }
  }

  const Flow &operator[](size_t index) const {
    assert(index < flows.size());
    return flows[index];
  }

  const_iterator begin() const { return flows.begin(); }

  const_iterator end() const { return flows.end(); }

private:
  Container<Flow> flows;
};

/// Represents how an equation has been matched (i.e. the selected indexes and
/// access).
template <typename EquationProperty, typename VariableProperty,
          typename AccessProperty>
class MatchingSolution {
public:
  MatchingSolution(EquationProperty equation, VariableProperty variable,
                   IndexSet indexes, AccessProperty access)
      : equation(std::move(equation)), variable(std::move(variable)),
        indexes(std::move(indexes)), access(std::move(access)) {}

  EquationProperty &getEquation() { return equation; }

  const EquationProperty &getEquation() const { return equation; }

  const VariableProperty &getVariable() const { return variable; }

  const AccessProperty &getAccess() const { return access; }

  const IndexSet &getIndexes() const { return indexes; }

private:
  EquationProperty equation;
  VariableProperty variable;
  IndexSet indexes;
  AccessProperty access;
};
} // namespace internal::matching

template <typename VariableProperty, typename EquationProperty>
class MatchingGraph : public internal::Dumpable {
public:
  using Variable = internal::matching::VariableVertex<VariableProperty>;
  using Equation = internal::matching::EquationVertex<EquationProperty>;
  using Vertex = std::variant<Variable, Equation>;
  using Edge = internal::matching::Edge<Variable, Equation>;

private:
  using Graph = internal::UndirectedGraph<Vertex, Edge>;

  using VertexDescriptor = typename Graph::VertexDescriptor;
  using EdgeDescriptor = typename Graph::EdgeDescriptor;

  using VertexIterator = typename Graph::VertexIterator;
  using EdgeIterator = typename Graph::EdgeIterator;
  using VisibleIncidentEdgeIterator =
      typename Graph::FilteredIncidentEdgeIterator;

  using MCIM = internal::MCIM;
  using BFSStep = internal::matching::BFSStep<Graph, Variable, Equation>;
  using Frontier = internal::matching::Frontier<BFSStep>;
  using Flow = internal::matching::Flow<Graph, Variable, Equation>;
  using AugmentingPath = internal::matching::AugmentingPath<Flow>;

public:
  using VariableIterator = typename Graph::FilteredVertexIterator;
  using EquationIterator = typename Graph::FilteredVertexIterator;

  using AccessProperty = typename Equation::Access::Property;
  using Access = matching::Access<VariableProperty, AccessProperty>;
  using MatchingSolution =
      internal::matching::MatchingSolution<EquationProperty, VariableProperty,
                                           AccessProperty>;

  explicit MatchingGraph(mlir::MLIRContext *context) : context(context) {}

  MatchingGraph(const MatchingGraph &other) = delete;

  MatchingGraph(MatchingGraph &&other) {
    std::lock_guard<std::mutex> lockGuard(other.mutex);

    context = std::move(other.context);
    graph = std::move(other.graph);
    variablesMap = std::move(other.variablesMap);
    equationsMap = std::move(other.equationsMap);
  }

  ~MatchingGraph() = default;

  MatchingGraph &operator=(const MatchingGraph &other) = delete;

  MatchingGraph &operator=(MatchingGraph &&other) = default;

  using Dumpable::dump;

  void dump(llvm::raw_ostream &os) const override {
    os << "--------------------------------------------------\n";
    os << "Matching graph\n";
    os << "- Vertices:\n";

    for (auto vertexDescriptor :
         llvm::make_range(graph.verticesBegin(), graph.verticesEnd())) {

      std::visit([&](const auto &vertex) { vertex.dump(os); },
                 graph[vertexDescriptor]);

      os << "\n";
    }

    os << "- Equations:\n";

    for (auto equationDescriptor :
         llvm::make_range(graph.edgesBegin(), graph.edgesEnd())) {
      graph[equationDescriptor].dump(os);
      os << "\n";
    }

    os << "--------------------------------------------------\n";
  }

  mlir::MLIRContext *getContext() const {
    assert(context != nullptr);
    return context;
  }

  bool hasVariable(typename Variable::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return hasVariableWithId(id);
  }

  VariableProperty &getVariable(typename Variable::Id id) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablePropertyFromId(id);
  }

  const VariableProperty &getVariable(typename Variable::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablePropertyFromId(id);
  }

  Variable &getVariable(VertexDescriptor descriptor) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariableFromDescriptor(descriptor);
  }

  const Variable &getVariable(VertexDescriptor descriptor) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariableFromDescriptor(descriptor);
  }

  VariableIterator variablesBegin() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablesBeginIt();
  }

  VariableIterator variablesEnd() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getVariablesEndIt();
  }

  void addVariable(VariableProperty property) {
    std::lock_guard<std::mutex> lockGuard(mutex);

    Variable variable(std::move(property));
    auto id = variable.getId();
    assert(!hasVariableWithId(id) && "Already existing variable");
    VertexDescriptor variableDescriptor = graph.addVertex(std::move(variable));
    variablesMap[id] = variableDescriptor;
  }

  bool hasEquation(typename Equation::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return hasEquationWithId(id);
  }

  EquationProperty &getEquation(typename Equation::Id id) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationPropertyFromId(id);
  }

  const EquationProperty &getEquation(typename Equation::Id id) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationPropertyFromId(id);
  }

  Equation &getEquation(VertexDescriptor descriptor) {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationFromDescriptor(descriptor);
  }

  const Equation &getEquation(VertexDescriptor descriptor) const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationFromDescriptor(descriptor);
  }

  EquationIterator equationsBegin() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationsBeginIt();
  }

  EquationIterator equationsEnd() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    return getEquationsEndIt();
  }

  void addEquation(EquationProperty property) {
    Equation eq(std::move(property));
    [[maybe_unused]] auto id = eq.getId();

    std::unique_lock<std::mutex> lockGuard(mutex);
    assert(!hasEquationWithId(id) && "Already existing equation");

    // Insert the equation into the graph and get a reference to the new vertex
    VertexDescriptor equationDescriptor = graph.addVertex(std::move(eq));
    equationsMap[id] = equationDescriptor;
    Equation &equation = getEquationFromDescriptor(equationDescriptor);
    lockGuard.unlock();

    // The equation may access multiple variables or even multiple indexes
    // of the same variable. Add an edge to the graph for each of those
    // accesses.

    IndexSet equationRanges = equation.getIterationRanges();

    for (const auto &access : equation.getVariableAccesses()) {
      lockGuard.lock();
      VertexDescriptor variableDescriptor =
          getVariableDescriptorFromId(access.getVariable());
      Variable &variable = getVariableFromDescriptor(variableDescriptor);
      lockGuard.unlock();

      IndexSet indices = variable.getIndices().getCanonicalRepresentation();

      for (const MultidimensionalRange &range :
           llvm::make_range(indices.rangesBegin(), indices.rangesEnd())) {
        Edge edge(equation.getId(), variable.getId(), equationRanges,
                  IndexSet(range), access);

        lockGuard.lock();
        graph.addEdge(equationDescriptor, variableDescriptor, std::move(edge));
        lockGuard.unlock();
      }
    }
  }

  /// Get the total amount of scalar variables inside the graph.
  size_t getNumberOfScalarVariables() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    size_t result = 0;

    auto variables =
        llvm::make_range(getVariablesBeginIt(), getVariablesEndIt());

    for (VertexDescriptor variableDescriptor : variables) {
      result += getVariableFromDescriptor(variableDescriptor).flatSize();
    }

    return result;
  }

  /// Get the total amount of scalar equations inside the graph.
  /// With "scalar equations" we mean the ones generated by unrolling
  /// the loops defining them.
  size_t getNumberOfScalarEquations() const {
    std::lock_guard<std::mutex> lockGuard(mutex);
    size_t result = 0;

    auto equations =
        llvm::make_range(getEquationsBeginIt(), getEquationsEndIt());

    for (VertexDescriptor equationDescriptor : equations) {
      result += getEquationFromDescriptor(equationDescriptor).flatSize();
    }

    return result;
  }

  // Warning: highly inefficient, use for testing purposes only.
  bool hasEdge(typename Equation::Id equationId,
               typename Variable::Id variableId) const {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (findEdge<Equation, Variable>(equationId, variableId).first) {
      return true;
    }

    return findEdge<Variable, Equation>(variableId, equationId).first;
  }

  /// Apply the simplification algorithm in order to perform all
  /// the obligatory matches, that is the variables and equations
  /// having only one incident edge.
  ///
  /// @return true if the simplification algorithm didn't find any inconsistency
  bool simplify() {
    std::lock_guard<std::mutex> lockGuard(mutex);

    // Vertices that are candidate for the first simplification phase.
    // They are the ones having only one incident edge.
    std::list<VertexDescriptor> candidates;

    if (!collectSimplifiableNodes(candidates)) {
      return false;
    }

    // Check that the list of simplifiable nodes does not contain
    // duplicates.
    assert(llvm::all_of(candidates,
                        [&](const VertexDescriptor &vertex) {
                          return llvm::count(candidates, vertex) == 1;
                        }) &&
           "Duplicates found in the list of simplifiable nodes");

    // Iterate on the candidate vertices and apply the simplification algorithm
    auto isVisibleFn = [](const auto &obj) -> bool { return obj.isVisible(); };

    auto allComponentsMatchedFn = [](const auto &vertex) -> bool {
      return vertex.allComponentsMatched();
    };

    while (!candidates.empty()) {
      VertexDescriptor v1 = candidates.front();
      candidates.pop_front();

      if (const Vertex &v = graph[v1]; !std::visit(isVisibleFn, v)) {
        // The current node, which initially had one and only one incident
        // edge, has been removed more by simplifications performed in the
        // previous iterations. We could just remove the vertex while the
        // edge was removed, but that would have required iterating over
        // the whole candidates list, thus worsening the overall complexity
        // of the algorithm.

        assert(std::visit(allComponentsMatchedFn, v));
        continue;
      }

      EdgeDescriptor edgeDescriptor = getFirstOutVisibleEdge(v1);
      Edge &edge = graph[edgeDescriptor];

      VertexDescriptor v2 =
          edgeDescriptor.from == v1 ? edgeDescriptor.to : edgeDescriptor.from;

      const auto &u = edge.getIncidenceMatrix();

      auto matchOptions = internal::solveLocalMatchingProblem(
          u.getEquationRanges(), u.getVariableRanges(),
          edge.getAccessFunction().clone());

      // The simplification steps is executed only in case of a single
      // matching option. In case of multiple ones, in fact, the choice
      // would be arbitrary and may affect the feasibility of the
      // array-aware matching problem.

      if (matchOptions.size() == 1) {
        const MCIM &match = matchOptions[0];

        Variable &variable = isVariable(v1) ? getVariableFromDescriptor(v1)
                                            : getVariableFromDescriptor(v2);
        Equation &equation = isEquation(v1) ? getEquationFromDescriptor(v1)
                                            : getEquationFromDescriptor(v2);

        IndexSet proposedVariableMatch = match.flattenRows();
        IndexSet proposedEquationMatch = match.flattenColumns();

        MCIM reducedMatch =
            match.filterColumns(proposedVariableMatch - variable.getMatched());
        reducedMatch = reducedMatch.filterRows(proposedEquationMatch -
                                               equation.getMatched());

        edge.addMatch(reducedMatch);

        IndexSet reducedVariableMatch = reducedMatch.flattenRows();
        IndexSet reducedEquationMatch = reducedMatch.flattenColumns();

        variable.addMatch(reducedVariableMatch);
        equation.addMatch(reducedEquationMatch);

        if (!std::visit(allComponentsMatchedFn, graph[v1])) {
          return false;
        }

        bool shouldRemoveOppositeNode =
            std::visit(allComponentsMatchedFn, graph[v2]);

        // Remove the edge and the current candidate vertex.
        remove(edgeDescriptor);
        remove(v1);

        if (shouldRemoveOppositeNode) {
          // When a node is removed, then also its incident edges are
          // removed. This can lead to new obliged matches, like in the
          // following example:
          //        |-- v3 ----
          // v1 -- v2         |
          //        |-- v4 -- v5
          // v1 is the current candidate and thus is removed.
          // v2 is removed because fully matched.
          // v3 and v4 become new candidates for the simplification pass.

          for (EdgeDescriptor e : llvm::make_range(
                   graph.outgoingEdgesBegin(v2), graph.outgoingEdgesEnd(v2))) {
            remove(e);

            VertexDescriptor v = e.from == v2 ? e.to : e.from;

            if (!std::visit(isVisibleFn, graph[v])) {
              continue;
            }

            size_t visibilityDegree = getVertexVisibilityDegree(v);

            if (visibilityDegree == 0) {
              // Chained simplifications may have led the 'v' vertex
              // without any edge. In that case, it must have been fully
              // matched during the process.

              if (!std::visit(allComponentsMatchedFn, graph[v])) {
                return false;
              }

              // 'v' will also be present for sure in the candidates list.
              // However, being fully matched and having no outgoing edge,
              // we now must remove it.
              remove(v);
            } else if (visibilityDegree == 1) {
              candidates.push_back(v);
            }
          }

          // Remove the v2 vertex.
          remove(v2);
        } else {
          // When an edge is removed but one of its vertices survives, we must
          // check if the remaining vertex has an obliged match.

          size_t visibilityDegree = getVertexVisibilityDegree(v2);

          if (visibilityDegree == 1) {
            candidates.push_back(v2);
          }
        }
      }
    }

    return true;
  }

  /// Apply the matching algorithm.
  ///
  /// @return true if the matching algorithm managed to fully match all the
  /// nodes
  bool match() {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (allNodesMatched()) {
      return true;
    }

    bool success;
    bool complete;

    do {
      success = matchIteration();
      complete = allNodesMatched();

      LLVM_DEBUG({
        llvm::dbgs() << "Match iteration completed\n";
        dump(llvm::dbgs());
      });
    } while (success && !complete);

    LLVM_DEBUG({
      if (success) {
        llvm::dbgs() << "Matching completed successfully\n";
      } else {
        llvm::dbgs() << "Matching failed\n";
      }
    });

    return success;
  }

  /// Get the solution of the matching problem.
  bool getMatch(llvm::SmallVectorImpl<MatchingSolution> &result) const {
    std::lock_guard<std::mutex> lockGuard(mutex);

    if (!allNodesMatched()) {
      return false;
    }

    auto equations =
        llvm::make_range(getEquationsBeginIt(), getEquationsEndIt());

    for (VertexDescriptor equationDescriptor : equations) {
      auto edges = llvm::make_range(edgesBegin(equationDescriptor),
                                    edgesEnd(equationDescriptor));

      for (EdgeDescriptor edgeDescriptor : edges) {
        const Edge &edge = graph[edgeDescriptor];

        if (const auto &matched = edge.getMatched(); !matched.empty()) {
          auto variableDescriptor = edgeDescriptor.from == equationDescriptor
                                        ? edgeDescriptor.to
                                        : edgeDescriptor.from;

          IndexSet matchedEquationIndices = matched.flattenColumns();
          assert(!matchedEquationIndices.empty());

          result.emplace_back(
              getEquationFromDescriptor(equationDescriptor).getProperty(),
              getVariableFromDescriptor(variableDescriptor).getProperty(),
              std::move(matchedEquationIndices), edge.getAccessProperty());
        }
      }
    }

    return true;
  }

private:
  /// Check if a variable with a given ID exists.
  bool hasVariableWithId(typename Variable::Id id) const {
    return variablesMap.find(id) != variablesMap.end();
  }

  bool isVariable(VertexDescriptor vertex) const {
    return std::holds_alternative<Variable>(graph[vertex]);
  }

  VertexDescriptor getVariableDescriptorFromId(typename Variable::Id id) const {
    auto it = variablesMap.find(id);
    assert(it != variablesMap.end() && "Variable not found");
    return it->second;
  }

  VariableProperty &getVariablePropertyFromId(typename Variable::Id id) {
    VertexDescriptor vertex = getVariableDescriptorFromId(id);
    return std::get<Variable>(graph[vertex]).getProperty();
  }

  const VariableProperty &
  getVariablePropertyFromId(typename Variable::Id id) const {
    VertexDescriptor vertex = getVariableDescriptorFromId(id);
    return std::get<Variable>(graph[vertex]).getProperty();
  }

  Variable &getVariableFromDescriptor(VertexDescriptor descriptor) {
    Vertex &vertex = graph[descriptor];
    assert(std::holds_alternative<Variable>(vertex));
    return std::get<Variable>(vertex);
  }

  const Variable &getVariableFromDescriptor(VertexDescriptor descriptor) const {
    const Vertex &vertex = graph[descriptor];
    assert(std::holds_alternative<Variable>(vertex));
    return std::get<Variable>(vertex);
  }

  /// Check if an equation with a given ID exists.
  bool hasEquationWithId(typename Equation::Id id) const {
    return equationsMap.find(id) != equationsMap.end();
  }

  bool isEquation(VertexDescriptor vertex) const {
    return std::holds_alternative<Equation>(graph[vertex]);
  }

  VertexDescriptor getEquationDescriptorFromId(typename Equation::Id id) const {
    auto it = equationsMap.find(id);
    assert(it != equationsMap.end() && "Equation not found");
    return it->second;
  }

  EquationProperty &getEquationPropertyFromId(typename Equation::Id id) {
    VertexDescriptor vertex = getEquationDescriptorFromId(id);
    return std::get<Equation>(graph[vertex]).getProperty();
  }

  const EquationProperty &
  getEquationPropertyFromId(typename Equation::Id id) const {
    VertexDescriptor vertex = getEquationDescriptorFromId(id);
    return std::get<Equation>(graph[vertex]).getProperty();
  }

  Equation &getEquationFromDescriptor(VertexDescriptor descriptor) {
    Vertex &vertex = graph[descriptor];
    assert(std::holds_alternative<Equation>(vertex));
    return std::get<Equation>(vertex);
  }

  const Equation &getEquationFromDescriptor(VertexDescriptor descriptor) const {
    const Vertex &vertex = graph[descriptor];
    assert(std::holds_alternative<Equation>(vertex));
    return std::get<Equation>(vertex);
  }

  /// Get the begin iterator for the variables of the graph.
  VariableIterator getVariablesBeginIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Variable>(vertex);
    };

    return graph.verticesBegin(filter);
  }

  /// Get the end iterator for the variables of the graph.
  VariableIterator getVariablesEndIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Variable>(vertex);
    };

    return graph.verticesEnd(filter);
  }

  /// Get the begin iterator for the equations of the graph.
  EquationIterator getEquationsBeginIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Equation>(vertex);
    };

    return graph.verticesBegin(filter);
  }

  /// Get the end iterator for the equations of the graph.
  EquationIterator getEquationsEndIt() const {
    auto filter = [](const Vertex &vertex) -> bool {
      return std::holds_alternative<Equation>(vertex);
    };

    return graph.verticesEnd(filter);
  }

  /// Check if all the scalar variables and equations have been matched.
  bool allNodesMatched() const {
    auto allComponentsMatchedFn = [](const auto &obj) {
      return obj.allComponentsMatched();
    };

    return mlir::succeeded(mlir::failableParallelForEach(
        getContext(), graph.verticesBegin(), graph.verticesEnd(),
        [&](VertexDescriptor vertex) -> mlir::LogicalResult {
          return mlir::LogicalResult::success(
              std::visit(allComponentsMatchedFn, graph[vertex]));
        }));
  }

  size_t getVertexVisibilityDegree(VertexDescriptor vertex) const {
    auto edges =
        llvm::make_range(visibleEdgesBegin(vertex), visibleEdgesEnd(vertex));
    return std::distance(edges.begin(), edges.end());
  }

  void remove(VertexDescriptor vertex) {
    std::visit([](auto &obj) -> void { obj.setVisibility(false); },
               graph[vertex]);
  }

  // Warning: highly inefficient, use for testing purposes only.
  template <typename From, typename To>
  std::pair<bool, EdgeIterator> findEdge(typename From::Id from,
                                         typename To::Id to) const {
    auto edges = llvm::make_range(graph.edgesBegin(), graph.edgesEnd());

    EdgeIterator it =
        std::find_if(edges.begin(), edges.end(), [&](const EdgeDescriptor &e) {
          const Vertex &source = graph[e.from];
          const Vertex &target = graph[e.to];

          if (!std::holds_alternative<From>(source) ||
              !std::holds_alternative<To>(target)) {
            return false;
          }

          return std::get<From>(source).getId() == from &&
                 std::get<To>(target).getId() == to;
        });

    return std::make_pair(it != edges.end(), it);
  }

  auto edgesBegin(VertexDescriptor vertex) const {
    return graph.outgoingEdgesBegin(vertex);
  }

  auto edgesEnd(VertexDescriptor vertex) const {
    return graph.outgoingEdgesEnd(vertex);
  }

  VisibleIncidentEdgeIterator visibleEdgesBegin(VertexDescriptor vertex) const {
    auto filter = [&](const Edge &edge) -> bool { return edge.isVisible(); };

    return graph.outgoingEdgesBegin(vertex, filter);
  }

  VisibleIncidentEdgeIterator visibleEdgesEnd(VertexDescriptor vertex) const {
    auto filter = [&](const Edge &edge) -> bool { return edge.isVisible(); };

    return graph.outgoingEdgesEnd(vertex, filter);
  }

  EdgeDescriptor getFirstOutVisibleEdge(VertexDescriptor vertex) const {
    auto edges =
        llvm::make_range(visibleEdgesBegin(vertex), visibleEdgesEnd(vertex));
    assert(edges.begin() != edges.end() && "Vertex doesn't belong to any edge");
    return *edges.begin();
  }

  void remove(EdgeDescriptor edge) { graph[edge].setVisibility(false); }

  /// Collect the list of vertices with exactly one incident edge.
  /// The function returns 'false' if there exist a node with no incident
  /// edges (which would make the matching process to fail in aby case).
  bool collectSimplifiableNodes(std::list<VertexDescriptor> &nodes) const {
    std::mutex resultMutex;

    auto collectFn = [&](VertexDescriptor vertex) -> mlir::LogicalResult {
      size_t incidentEdges = getVertexVisibilityDegree(vertex);

      if (incidentEdges == 0) {
        return mlir::failure();
      }

      if (incidentEdges == 1) {
        std::lock_guard<std::mutex> resultLockGuard(resultMutex);
        nodes.push_back(vertex);
      }

      return mlir::success();
    };

    return mlir::succeeded(mlir::failableParallelForEach(
        getContext(), graph.verticesBegin(), graph.verticesEnd(), collectFn));
  }

  bool matchIteration() {
    std::vector<AugmentingPath> augmentingPaths;
    getAugmentingPaths(augmentingPaths);

    if (augmentingPaths.empty()) {
      return false;
    }

    for (auto &path : augmentingPaths) {
      applyPath(path);
    }

    return true;
  }

  void getAugmentingPaths(std::vector<AugmentingPath> &augmentingPaths) const {
    // Get the possible augmenting paths.
    std::vector<std::shared_ptr<BFSStep>> paths = getCandidateAugmentingPaths();

    // For each traversed node, keep track of the indices that have already
    // been traversed by some augmenting path. A new candidate path can be
    // accepted only if it does not traverse any of them.
    std::map<VertexDescriptor, IndexSet> visited;

    for (const std::shared_ptr<BFSStep> &pathEnd : paths) {
      // All the candidate paths consist in at least two nodes by construction
      assert(pathEnd->hasPrevious());

      std::list<Flow> flows;

      // The path's validity is unknown, so we must avoid polluting the
      // list of visited scalar nodes. If the path will be marked as valid,
      // then the new visits will be merged with the already found ones.
      std::map<VertexDescriptor, IndexSet> newVisits;

      const BFSStep *curStep = pathEnd.get();
      MCIM map = curStep->getMappedFlow();
      bool validPath = true;

      while (curStep && validPath) {
        if (curStep->hasPrevious()) {
          if (!flows.empty()) {
            // Restrict the flow
            const auto &prevMap = flows.front().delta;

            if (isVariable(curStep->getNode())) {
              map =
                  curStep->getMappedFlow().filterColumns(prevMap.flattenRows());
            } else {
              map =
                  curStep->getMappedFlow().filterRows(prevMap.flattenColumns());
            }
          }

          flows.emplace(flows.begin(), graph, curStep->getPrevious()->getNode(),
                        curStep->getEdge(), map);
        }

        auto touchedIndexes = isVariable(curStep->getNode())
                                  ? map.flattenRows()
                                  : map.flattenColumns();

        if (auto it = visited.find(curStep->getNode()); it != visited.end()) {
          auto &alreadyTouchedIndices = it->second;

          if (touchedIndexes.overlaps(alreadyTouchedIndices)) {
            // The current path intersects another one, so we need to discard it
            validPath = false;
          } else {
            insertOrAdd(newVisits, curStep->getNode(),
                        alreadyTouchedIndices + touchedIndexes);
          }
        } else {
          insertOrAdd(newVisits, curStep->getNode(), touchedIndexes);
        }

        // Move backwards inside the candidate augmenting path
        curStep = curStep->getPrevious();
      }

      if (validPath) {
        augmentingPaths.emplace_back(std::move(flows));

        for (auto &p : newVisits) {
          visited.insert_or_assign(p.first, p.second);
        }
      }
    }
  }

  std::vector<std::shared_ptr<BFSStep>> getCandidateAugmentingPaths() const {
    std::vector<std::shared_ptr<BFSStep>> paths;

    auto sortHeuristic = [](const std::shared_ptr<BFSStep> &first,
                            const std::shared_ptr<BFSStep> &second) {
      return first->getCandidates().flatSize() >
             second->getCandidates().flatSize();
    };

    Frontier frontier;

    // Computation of the initial frontier.
    auto equations =
        llvm::make_range(getEquationsBeginIt(), getEquationsEndIt());

    for (VertexDescriptor equationDescriptor : equations) {
      const Equation &equation = getEquationFromDescriptor(equationDescriptor);

      if (IndexSet unmatchedEquations = equation.getUnmatched();
          !unmatchedEquations.empty()) {
        frontier.emplace_back(graph, equationDescriptor,
                              std::move(unmatchedEquations));
      }
    }

    // Breadth-first search.
    Frontier newFrontier;

    std::mutex newFrontierMutex;
    std::mutex pathsMutex;

    while (!frontier.empty() && paths.empty()) {
      llvm::sort(frontier, sortHeuristic);

      auto stepFn = [&](const std::shared_ptr<BFSStep> &step) {
        std::vector<std::shared_ptr<BFSStep>> localFrontier;
        std::vector<std::shared_ptr<BFSStep>> localPaths;

        expandFrontier(step, localFrontier, localPaths);

        if (!localFrontier.empty()) {
          std::lock_guard<std::mutex> lock(newFrontierMutex);

          for (auto &localStep : localFrontier) {
            newFrontier.push_back(std::move(localStep));
          }
        }

        if (!localPaths.empty()) {
          std::lock_guard<std::mutex> lock(pathsMutex);

          for (auto &localStep : localPaths) {
            paths.push_back(std::move(localStep));
          }
        }
      };

      mlir::parallelForEach(getContext(), frontier, stepFn);

      // Set the new frontier for the next iteration.
      frontier = std::move(newFrontier);
    }

    llvm::sort(paths, sortHeuristic);
    return paths;
  }

  void expandFrontier(const std::shared_ptr<BFSStep> &step,
                      std::vector<std::shared_ptr<BFSStep>> &newFrontier,
                      std::vector<std::shared_ptr<BFSStep>> &paths) const {
    const VertexDescriptor &vertexDescriptor = step->getNode();

    for (EdgeDescriptor edgeDescriptor : llvm::make_range(
             edgesBegin(vertexDescriptor), edgesEnd(vertexDescriptor))) {
      assert(edgeDescriptor.from == vertexDescriptor);
      VertexDescriptor nextNode = edgeDescriptor.to;
      const Edge &edge = graph[edgeDescriptor];

      if (isEquation(vertexDescriptor)) {
        assert(isVariable(nextNode));
        auto unmatchedMatrix = edge.getUnmatched();
        auto filteredMatrix = unmatchedMatrix.filterRows(step->getCandidates());

        internal::LocalMatchingSolutions solutions =
            internal::solveLocalMatchingProblem(filteredMatrix);

        for (auto solution : solutions) {
          Variable var = getVariableFromDescriptor(edgeDescriptor.to);
          auto unmatchedScalarVariables = var.getUnmatched();
          auto matched = solution.filterColumns(unmatchedScalarVariables);

          if (!matched.empty()) {
            paths.push_back(
                std::make_shared<BFSStep>(graph, step, edgeDescriptor, nextNode,
                                          matched.flattenRows(), matched));
          } else {
            newFrontier.push_back(
                std::make_shared<BFSStep>(graph, step, edgeDescriptor, nextNode,
                                          solution.flattenRows(), solution));
          }
        }
      } else {
        assert(isEquation(nextNode));
        auto filteredMatrix =
            edge.getMatched().filterColumns(step->getCandidates());

        internal::LocalMatchingSolutions solutions =
            internal::solveLocalMatchingProblem(filteredMatrix);

        for (auto solution : solutions) {
          newFrontier.push_back(
              std::make_shared<BFSStep>(graph, step, edgeDescriptor, nextNode,
                                        solution.flattenColumns(), solution));
        }
      }
    }
  }

  /// Apply an augmenting path to the graph.
  void applyPath(const AugmentingPath &path) {
    // In order to preserve consistency of the match information among
    // edges and nodes, we need to separately track the modifications
    // created by the augmenting path on the vertices and apply all the
    // removals before the additions.
    // Consider in fact the example path [eq1 -> x -> eq2]: the first
    // move would add some match information to eq1 and x, while the
    // subsequent x -> eq2 would remove some from x. However, being the
    // match matrices made of booleans, the components of x that are
    // matched by eq1 would result as unmatched. If we instead first
    // apply the removals, the new matches are not wrongly erased anymore.

    std::map<VertexDescriptor, IndexSet> removedMatches;
    std::map<VertexDescriptor, IndexSet> newMatches;

    // Update the match matrices on the edges and store the information
    // about the vertices to be updated later.

    for (auto &flow : path) {
      Edge &edge = graph[flow.edge];

      VertexDescriptor from = flow.source;
      VertexDescriptor to =
          flow.edge.from == from ? flow.edge.to : flow.edge.from;

      auto deltaEquations = flow.delta.flattenColumns();
      auto deltaVariables = flow.delta.flattenRows();

      if (isVariable(from)) {
        // Backward node
        insertOrAdd(removedMatches, from, deltaVariables);
        insertOrAdd(removedMatches, to, deltaEquations);
        edge.removeMatch(flow.delta);
      } else {
        // Forward node
        insertOrAdd(newMatches, from, deltaEquations);
        insertOrAdd(newMatches, to, deltaVariables);
        edge.addMatch(flow.delta);
      }
    }

    // Update the match information stored on the vertices

    for (const auto &match : removedMatches) {
      std::visit([&match](auto &node) { node.removeMatch(match.second); },
                 graph[match.first]);
    }

    for (const auto &match : newMatches) {
      std::visit([&match](auto &node) { node.addMatch(match.second); },
                 graph[match.first]);
    }
  }

private:
  mlir::MLIRContext *context;
  Graph graph;

  // Maps user for faster lookups.
  std::map<typename Variable::Id, VertexDescriptor> variablesMap;
  std::map<typename Equation::Id, VertexDescriptor> equationsMap;

  // Multithreading.
  mutable std::mutex mutex;
};
} // namespace marco::modeling

#endif // MARCO_MODELING_MATCHING_H
