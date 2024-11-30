#include "marco/Dialect/BaseModelica/Transforms/AllInterfaces.h"
#include "marco/Dialect/BaseModelica/Transforms/AffineLikeOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/AllocationOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/BufferizableOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/ConstantMaterializableTypeInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/DerivableOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/DerivableTypeInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/EquationExpressionOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/InvertibleOpInterfaceImpl.h"
#include "marco/Dialect/BaseModelica/Transforms/VectorizableOpInterfaceImpl.h"

namespace mlir::bmodelica {
void registerAllDialectInterfaceImplementations(
    mlir::DialectRegistry &registry) {
  // Operation interfaces.
  registerAffineLikeOpInterfaceExternalModels(registry);
  registerAllocationOpInterfaceExternalModels(registry);
  registerBufferizableOpInterfaceExternalModels(registry);
  registerDerivableOpInterfaceExternalModels(registry);
  registerEquationExpressionOpInterfaceExternalModels(registry);
  registerInvertibleOpInterfaceExternalModels(registry);
  registerVectorizableOpInterfaceExternalModels(registry);

  // Type interfaces.
  registerConstantMaterializableTypeInterfaceExternalModels(registry);
  registerDerivableTypeInterfaceExternalModels(registry);
}
} // namespace mlir::bmodelica
