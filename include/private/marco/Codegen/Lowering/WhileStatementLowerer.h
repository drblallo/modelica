#ifndef MARCO_CODEGEN_LOWERING_WHILESTATEMENTLOWERER_H
#define MARCO_CODEGEN_LOWERING_WHILESTATEMENTLOWERER_H

#include "marco/AST/AST.h"
#include "marco/Codegen/Lowering/Lowerer.h"
#include "marco/Codegen/BridgeInterface.h"

namespace marco::codegen::lowering
{
  class WhileStatementLowerer : public Lowerer
  {
    public:
      WhileStatementLowerer(BridgeInterface* bridge);

      void lower(const ast::WhileStatement& statement) override;

    protected:
      using Lowerer::lower;
  };
}

#endif // MARCO_CODEGEN_LOWERING_WHILESTATEMENTLOWERER_H
