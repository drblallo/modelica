#ifndef MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AFFINELIKEOPINTERFACEIMPL_H
#define MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AFFINELIKEOPINTERFACEIMPL_H

namespace mlir {
class DialectRegistry;

namespace bmodelica {
void registerAffineLikeOpInterfaceExternalModels(
    mlir::DialectRegistry &registry);
}
} // namespace mlir

#endif // MARCO_DIALECT_BASEMODELICA_TRANSFORMS_AFFINELIKEOPINTERFACEIMPL_H
