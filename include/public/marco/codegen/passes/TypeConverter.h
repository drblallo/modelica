#pragma once

#include "marco/codegen/dialects/modelica/Type.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Transforms/DialectConversion.h"

namespace marco::codegen {
	class TypeConverter : public mlir::LLVMTypeConverter {
		public:
		TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth);

		private:
		mlir::Type convertBooleanType(modelica::BooleanType type);
		mlir::Type convertIntegerType(modelica::IntegerType type);
		mlir::Type convertRealType(modelica::RealType type);

		mlir::Type convertArrayType(modelica::ArrayType type);
		mlir::Type convertUnsizedArrayType(modelica::UnsizedArrayType type);
		mlir::Type convertStructType(modelica::StructType type);

		llvm::SmallVector<mlir::Type, 3> getArrayDescriptorFields(modelica::ArrayType type);
		llvm::SmallVector<mlir::Type, 3> getUnsizedArrayDescriptorFields(modelica::UnsizedArrayType type);

		unsigned int bitWidth;
	};
}
