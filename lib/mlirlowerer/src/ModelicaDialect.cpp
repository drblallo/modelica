#include <modelica/mlirlowerer/ModelicaDialect.h>

using namespace modelica;

ModelicaDialect::ModelicaDialect(mlir::MLIRContext* context)
		: Dialect("modelica", context, mlir::TypeID::get<ModelicaDialect>())
{
	addTypes<BooleanType, IntegerType, RealType, PointerType>();

	// Basic operations
	addOperations<AssignmentOp>();
	addOperations<CastOp>();
	addOperations<CastCommonOp>();

	// MMemory operations
	addOperations<AllocaOp, AllocOp>();
	addOperations<FreeOp, DimOp, SubscriptionOp>();
	addOperations<LoadOp, StoreOp>();
	addOperations<ArrayCopyOp>();

	// Math operations
	addOperations<AddOp, SubOp, MulOp, DivOp, PowOp>();

	// Logic operations
	addOperations<NegateOp>();
	addOperations<EqOp>();
	addOperations<NotEqOp>();
	addOperations<GtOp>();
	addOperations<GteOp>();
	addOperations<LtOp>();
	addOperations<LteOp>();

	// Control flow operations
	addOperations<IfOp>();
	addOperations<ForOp>();
	addOperations<WhileOp>();
	addOperations<ConditionOp>();
	addOperations<YieldOp>();
}

mlir::StringRef ModelicaDialect::getDialectNamespace()
{
	return "modelica";
}

void ModelicaDialect::printType(mlir::Type type, mlir::DialectAsmPrinter& printer) const {
	return printModelicaType(const_cast<ModelicaDialect*>(this), type, printer);
}
