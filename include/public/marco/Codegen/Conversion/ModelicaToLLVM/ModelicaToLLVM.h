#ifndef MARCO_CODEN_CONVERSION_MODELICA_MODELICACONVERSION_H
#define MARCO_CODEN_CONVERSION_MODELICA_MODELICACONVERSION_H

#include "mlir/Pass/Pass.h"

namespace marco::codegen
{
	struct ModelicaToLLVMOptions
	{
		bool assertions = true;

		static const ModelicaToLLVMOptions& getDefaultOptions();
	};

	std::unique_ptr<mlir::Pass> createModelicaToLLVMPass(
      ModelicaToLLVMOptions options = ModelicaToLLVMOptions::getDefaultOptions(),
			unsigned int bitWidth = 64);
}

#endif // MARCO_CODEN_CONVERSION_MODELICA_MODELICACONVERSION_H
