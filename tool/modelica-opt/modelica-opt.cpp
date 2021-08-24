#include <mlir/IR/Dialect.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/InitAllPasses.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/FileUtilities.h>
#include <mlir/Support/MlirOptMain.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/ToolOutputFile.h>
#include <marco/mlirlowerer/CodeGen.h>
#include <marco/mlirlowerer/ModelicaDialect.h>

using namespace marco::codegen;

int main(int argc, char* argv[])
{
	mlir::registerAllPasses();
	marco::codegen::registerModelicaPasses();

	mlir::DialectRegistry registry;
	registry.insert<marco::codegen::ModelicaDialect>();
	registry.insert<mlir::BuiltinDialect>();
	registry.insert<mlir::StandardOpsDialect>();

	auto result = mlir::MlirOptMain(argc, argv, "Modelica optimizer driver\n", registry);
	return mlir::failed(result) ? 1 : 0;
}
