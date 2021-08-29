#pragma once

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/BuiltinDialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Transforms/DialectConversion.h>
#include <marco/mlirlowerer/dialects/ida/Type.h>
#include <marco/mlirlowerer/dialects/modelica/Type.h>

namespace marco::codegen {
	class TypeConverter : public mlir::LLVMTypeConverter {
		public:
		TypeConverter(mlir::MLIRContext* context, mlir::LowerToLLVMOptions options, unsigned int bitWidth);

		private:
		mlir::Type convertBooleanType(mlir::Type type);
		mlir::Type convertIntegerType(mlir::Type type);
		mlir::Type convertRealType(mlir::Type type);
		mlir::Type convertArrayType(modelica::ArrayType type);
		mlir::Type convertUnsizedArrayType(modelica::UnsizedArrayType type);
		mlir::Type convertOpaquePointerType(mlir::Type type);
		mlir::Type convertStructType(modelica::StructType type);

		llvm::SmallVector<mlir::Type, 3> getArrayDescriptorFields(modelica::ArrayType type);
		llvm::SmallVector<mlir::Type, 3> getUnsizedArrayDescriptorFields(modelica::UnsizedArrayType type);

		unsigned int bitWidth;
	};
}
