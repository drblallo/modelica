#ifndef MARCO_FRONTEND_FRONTENDACTIONS_H
#define MARCO_FRONTEND_FRONTENDACTIONS_H

#include "marco/AST/AST.h"
#include "marco/Frontend/DiagnosticHandler.h"
#include "marco/Frontend/FrontendAction.h"
#include "marco/IO/InputFile.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "clang/CodeGen/BackendUtil.h"
#include "llvm/Target/TargetMachine.h"

namespace marco::frontend {
//===---------------------------------------------------------------------===//
// Custom actions
//===---------------------------------------------------------------------===//

class InitOnlyAction : public FrontendAction {
public:
  void executeAction() override;

private:
  void printCategory(llvm::raw_ostream &os, llvm::StringRef category) const;

  void printOption(llvm::raw_ostream &os, llvm::StringRef name,
                   llvm::StringRef value);

  void printOption(llvm::raw_ostream &os, llvm::StringRef name, bool value);

  void printOption(llvm::raw_ostream &os, llvm::StringRef name, long value);

  void printOption(llvm::raw_ostream &os, llvm::StringRef name, double value);
};

//===---------------------------------------------------------------------===//
// Preprocessing actions
//===---------------------------------------------------------------------===//

class PreprocessingAction : public FrontendAction {
public:
  bool beginSourceFilesAction() override;

protected:
  std::string flattened;
};

class EmitBaseModelicaAction : public PreprocessingAction {
public:
  void executeAction() override;
};

//===---------------------------------------------------------------------===//
// AST actions
//===---------------------------------------------------------------------===//

enum class ASTActionKind { Parse };

class ASTAction : public PreprocessingAction {
protected:
  explicit ASTAction(ASTActionKind action);

public:
  bool beginSourceFilesAction() override;

private:
  ASTActionKind action;

protected:
  std::unique_ptr<ast::ASTNode> ast;
};

class EmitASTAction : public ASTAction {
public:
  EmitASTAction();

  void executeAction() override;
};

//===---------------------------------------------------------------------===//
// CodeGen actions
//===---------------------------------------------------------------------===//

enum class CodeGenActionKind {
  GenerateMLIR,
  GenerateMLIRModelica,
  GenerateMLIRLLVM,
  GenerateLLVMIR
};

class CodeGenAction : public ASTAction {
protected:
  explicit CodeGenAction(CodeGenActionKind action);

public:
  ~CodeGenAction() override;

  bool beginSourceFilesAction() override;

protected:
  /// Set up the LLVM's TargetMachine.
  bool setUpTargetMachine();

  /// Get the LLVM's TargetMachine.
  llvm::TargetMachine &getTargetMachine();

  /// Get the LLVM's TargetMachine.
  const llvm::TargetMachine &getTargetMachine() const;

  /// Get the data layout of the machine for which the code is being
  /// compiled.
  llvm::DataLayout getDataLayout() const;

  /// @name MLIR
  /// {

  /// Register the MLIR dialects that can be encountered while parsing.
  void registerMLIRDialects();

  /// Register the MLIR extensions.
  void registerMLIRExtensions();

  /// Create the MLIR context.
  void createMLIRContext();

  /// Get the MLIR context.
  mlir::MLIRContext &getMLIRContext();

  /// Create the LLVM context.
  void createLLVMContext();

  /// Get the LLVM context.
  llvm::LLVMContext &getLLVMContext();

  /// Parse MLIR from the source file if in MLIR format, or generate it
  /// according to the source file type.
  bool generateMLIR();

  /// Parse MLIR from the source file if in MLIR format (or generate it)
  /// and obtain the Modelica dialect.
  bool generateMLIRModelica();

  /// Parse MLIR from the source file if in MLIR format (or generate it)
  /// and obtain the LLVM dialect.
  bool generateMLIRLLVM();

  /// Set the target triple inside the MLIR module.
  void setMLIRModuleTargetTriple();

  /// Set the data layout inside the MLIR module.
  void setMLIRModuleDataLayout();

  /// Given a pass manager, append the passes to lower the MLIR module to
  /// the LLVM dialect.
  void buildMLIRLoweringPipeline(mlir::PassManager &pm);

  /// }
  /// @name Modelica passes.
  /// {

  std::unique_ptr<mlir::Pass> createMLIRFunctionScalarizationPass();
  std::unique_ptr<mlir::Pass> createMLIRReadOnlyVariablesPropagationPass();

  std::unique_ptr<mlir::Pass> createMLIRRungeKuttaPass(llvm::StringRef variant);
  std::unique_ptr<mlir::Pass> createMLIREulerForwardPass();
  std::unique_ptr<mlir::Pass> createMLIRIDAPass();

  std::unique_ptr<mlir::Pass> createMLIRSCCSolvingBySubstitutionPass();
  std::unique_ptr<mlir::Pass> createMLIRSCCSolvingWithKINSOLPass();

  std::unique_ptr<mlir::Pass> createMLIREquationFunctionPeelingPass();

  /// }
  /// @name Conversion passes.
  /// {

  std::unique_ptr<mlir::Pass> createMLIRBaseModelicaToRuntimeConversionPass();

  /// }
  /// @name MLIR built-in passes.
  /// {

  std::unique_ptr<mlir::Pass> createMLIROneShotBufferizePass();
  void buildMLIRBufferDeallocationPipeline(mlir::OpPassManager &pm);
  void addMLIRLoopTilingPass(mlir::OpPassManager &pm);
  std::unique_ptr<mlir::Pass> createMLIRPromoteBuffersToStackPass();

  std::unique_ptr<mlir::Pass> createMLIRAffineVectorizePass();
  std::unique_ptr<mlir::Pass> createMLIRVectorToLLVMConversionPass();

  /// }
  /// @name LLVM-IR
  /// {

  /// Register the translations needed to emit LLVM-IR.
  void registerMLIRToLLVMIRTranslations();

  /// Parse LLVM-IR from the source file if in LLVM-IR format, or generate
  /// it according to the source file type.
  bool generateLLVMIR();

  /// Set the target triple inside the LLVM module.
  void setLLVMModuleTargetTriple();

  /// Set the data layout inside the LLVM module.
  void setLLVMModuleDataLayout();

  /// Run the optimization (aka middle-end) pipeline on the LLVM module
  /// associated with this action.
  void runOptimizationPipeline();

  /// }
  /// @name Backend
  /// {

  void emitBackendOutput(clang::BackendAction backendAction,
                         std::unique_ptr<llvm::raw_pwrite_stream> os);

  /// }

private:
  CodeGenActionKind action;
  std::unique_ptr<llvm::TargetMachine> targetMachine;
  mlir::DialectRegistry mlirDialectRegistry;
  std::unique_ptr<mlir::MLIRContext> mlirContext;
  std::unique_ptr<DiagnosticHandler> diagnosticHandler;
  std::unique_ptr<llvm::LLVMContext> llvmContext;

protected:
  std::unique_ptr<mlir::ModuleOp> mlirModule;
  std::unique_ptr<llvm::Module> llvmModule;
};

class EmitMLIRAction : public CodeGenAction {
public:
  EmitMLIRAction();

  void executeAction() override;
};

class EmitMLIRModelicaAction : public CodeGenAction {
public:
  EmitMLIRModelicaAction();

  void executeAction() override;
};

class EmitMLIRLLVMAction : public CodeGenAction {
public:
  EmitMLIRLLVMAction();

  void executeAction() override;
};

class EmitLLVMIRAction : public CodeGenAction {
public:
  EmitLLVMIRAction();

  void executeAction() override;
};

class EmitBitcodeAction : public CodeGenAction {
public:
  EmitBitcodeAction();

  void executeAction() override;
};

class EmitAssemblyAction : public CodeGenAction {
public:
  EmitAssemblyAction();

  void executeAction() override;
};

class EmitObjAction : public CodeGenAction {
public:
  EmitObjAction();

  void executeAction() override;
};
} // namespace marco::frontend

#endif // MARCO_FRONTEND_FRONTENDACTIONS_H
