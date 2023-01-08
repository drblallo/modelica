#ifndef MARCO_CYCLESSYMBOLICSOLVER_H
#define MARCO_CYCLESSYMBOLICSOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Matching.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/STLExtras.h"

#include <algorithm>

namespace marco::codegen {
  class CyclesSymbolicSolver
  {
  private:
    mlir::OpBuilder& builder;

  public:
    CyclesSymbolicSolver(mlir::OpBuilder& builder);

    bool solve(Model<MatchedEquation>& model);

  };

  class OperandIterator {
    private:

  };

  class OperationNode {
  private:
    mlir::Operation* operation;
    OperationNode* next;
    OperationNode* prev;
    OperationNode* father;
    OperationNode* child;
    size_t childNumber;
    size_t numberOfChildren;
  public:
    OperationNode(mlir::Operation* operation,
                OperationNode* next,
                OperationNode* prev,
                OperationNode* father,
                OperationNode* child,
                size_t childNumber,
                size_t numberOfChildren);
    mlir::Operation* getOperation();
    void setNext(OperationNode* next);
    void setChild(OperationNode* child);
    OperationNode* getChild();
    OperationNode* getNext();
  };

  class EquationGraph {
  private:
    marco::codegen::MatchedEquation* equation;
    OperationNode* entryNode;
  public:
    explicit EquationGraph(MatchedEquation* equation);
    OperationNode* getEntryNode();
    void print();
  };
}


namespace llvm
{
  template<>
  struct GraphTraits<const marco::codegen::EquationGraph>
  {
    using Graph = const marco::codegen::EquationGraph;
    using GraphPtr = Graph*;

    using NodeRef = typename mlir::Operation*;
    // Need an iterator that dereferences to a NodeRef
    using ChildIteratorType = typename marco::codegen::OperationNode*;

    static NodeRef getEntryNode(const GraphPtr& graph) {
      //return graph->getEntryNode().getOperation();
    }

    static ChildIteratorType child_begin(NodeRef node) {

    }

    static ChildIteratorType child_end(NodeRef node) {

    }
  };
}

#endif//MARCO_CYCLESSYMBOLICSOLVER_H
