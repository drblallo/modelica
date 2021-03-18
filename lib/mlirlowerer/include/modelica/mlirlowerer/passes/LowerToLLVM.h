#pragma once

#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <modelica/mlirlowerer/Ops.h>
#include <modelica/mlirlowerer/passes/TypeConverter.h>

namespace modelica
{
	struct ModelicaToLLVMConversionOptions
	{
		bool emitCWrappers = false;

		/**
		 * Get a statically allocated copy of the default options.
		 *
		 * @return default options
		 */
		static const ModelicaToLLVMConversionOptions& getDefaultOptions() {
			static ModelicaToLLVMConversionOptions options;
			return options;
		}
	};

	std::unique_ptr<mlir::Pass> createLLVMLoweringPass(ModelicaToLLVMConversionOptions options = ModelicaToLLVMConversionOptions::getDefaultOptions());
}
