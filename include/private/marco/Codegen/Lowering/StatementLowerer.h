#ifndef MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class StatementLowerer : public Lowerer
  {
    public:
      StatementLowerer(BridgeInterface* bridge);

      void lower(const ast::Statement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_STATEMENTLOWERER_H
