#ifndef MARCO_AST_NODE_STATEMENT_H
#define MARCO_AST_NODE_STATEMENT_H

#include "marco/AST/Node/ASTNode.h"
#include "llvm/ADT/STLExtras.h"

namespace marco::ast
{
  class Statement : public ASTNode
  {
    public:
      using ASTNode::ASTNode;

      Statement(const Statement& other);

      ~Statement();

      static bool classof(const ASTNode* node)
      {
        return node->getKind() >= ASTNode::Kind::Statement &&
            node->getKind() <= ASTNode::Kind::Statement_LastStatement;
      }
  };
}

#endif // MARCO_AST_NODE_STATEMENT_H
