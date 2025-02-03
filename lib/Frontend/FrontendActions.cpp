#include "marco/Frontend/FrontendActions.h"
#include "marco/Codegen/Conversion/Passes.h"
#include "marco/Codegen/Lowering/Bridge.h"
#include "marco/Codegen/Verifier.h"
#include "marco/Dialect/BaseModelica/IR/BaseModelica.h"
#include "marco/Dialect/BaseModelica/Transforms/AllInterfaces.h"
#include "marco/Dialect/BaseModelica/Transforms/Passes.h"
#include "marco/Dialect/IDA/IR/IDA.h"
#include "marco/Dialect/KINSOL/IR/KINSOL.h"
#include "marco/Dialect/Modeling/IR/Modeling.h"
#include "marco/Dialect/Runtime/IR/Runtime.h"
#include "marco/Dialect/Runtime/Transforms/AllInterfaces.h"
#include "marco/Dialect/Runtime/Transforms/Passes.h"
#include "marco/Frontend/CompilerInstance.h"
#include "marco/IO/Command.h"
#include "marco/Parser/Parser.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/OpenMP/OpenMPToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "clang/Basic/DiagnosticFrontend.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Bitcode/BitcodeWriterPass.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/StandardInstrumentations.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/TargetParser/Host.h"

using namespace ::marco;
using namespace ::marco::frontend;
using namespace ::marco::io;

//===---------------------------------------------------------------------===//
// Utility functions
//===---------------------------------------------------------------------===//

static llvm::CodeGenOptLevel
mapOptimizationLevelToCodeGenLevel(llvm::OptimizationLevel level, bool debug) {
  if (level == llvm::OptimizationLevel::O0) {
    return llvm::CodeGenOptLevel::None;
  }

  if (debug || level == llvm::OptimizationLevel::O1) {
    return llvm::CodeGenOptLevel::Less;
  }

  if (level == llvm::OptimizationLevel::O2 ||
      level == llvm::OptimizationLevel::Os ||
      level == llvm::OptimizationLevel::Oz) {
    return llvm::CodeGenOptLevel::Default;
  }

  assert(level == llvm::OptimizationLevel::O3);
  return llvm::CodeGenOptLevel::Aggressive;
}

namespace marco::frontend {
bool PreprocessingAction::beginSourceFilesAction() {
  CompilerInstance &ci = getInstance();

  if (ci.getFrontendOptions().omcBypass) {
    return true;
  }

  // Use OMC to generate the Base Modelica code.
  auto omcPath = llvm::sys::findProgramByName("omc");

  if (!omcPath) {
    auto &diag = ci.getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                     "Can't obtain path to omc executable"));

    return false;
  }

  Command cmd(*omcPath);

  // Add the input files.
  for (const InputFile &inputFile : ci.getFrontendOptions().inputs) {
    if (inputFile.getKind().getLanguage() != Language::Modelica) {
      auto &diag = ci.getDiagnostics();

      diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                       "Invalid input type: flattening can be "
                                       "performed only on Modelica files"));

      return false;
    }

    cmd.appendArg(inputFile.getFile().str());
  }

  // Set the model to be flattened.
  if (const auto &modelName = ci.getSimulationOptions().modelName;
      modelName.empty()) {
    auto &diag = ci.getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                     "Model name not specified"));
  } else {
    cmd.appendArg("+i=" + ci.getSimulationOptions().modelName);
  }

  // Enable the output of base Modelica.
  cmd.appendArg("-f");

  // Add the extra arguments to OMC.
  for (const auto &arg : ci.getFrontendOptions().omcCustomArgs) {
    cmd.appendArg(arg);
  }

  // Create the file in which OMC will write.
  llvm::SmallVector<char, 128> tmpPath;
  llvm::sys::path::system_temp_directory(true, tmpPath);
  llvm::sys::path::append(tmpPath, "marco_omc_%%%%%%.bmo");

  llvm::Expected<llvm::sys::fs::TempFile> tempFile =
      llvm::sys::fs::TempFile::create(tmpPath);

  if (!tempFile) {
    auto &diag = ci.getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                     "Can't create OMC output file"));

    return false;
  }

  // Redirect stdout of omc.
  cmd.setStdoutRedirect(tempFile->TmpName);

  // Run the command.
  int resultCode = cmd.exec();

  if (resultCode == EXIT_SUCCESS) {
    auto buffer = llvm::errorOrToExpected(
        ci.getFileManager().getBufferForFile(tempFile->TmpName));

    if (!buffer) {
      auto &diag = ci.getDiagnostics();
      auto errorCode = llvm::errorToErrorCode(buffer.takeError());

      diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                       "Unable to open '%0'. %1"))
          << tempFile->TmpName << errorCode.message();

      return false;
    }

    flattened = buffer->get()->getBuffer().str();

    // Overlay the current file system with an in-memory one containing the
    // Base Modelica source file.
    auto &fileManager = ci.getFileManager();

    auto inMemoryFS =
        llvm::makeIntrusiveRefCnt<llvm::vfs::InMemoryFileSystem>();

    inMemoryFS->addFile(
        tempFile->TmpName, 0,
        llvm::MemoryBuffer::getMemBuffer(flattened, tempFile->TmpName, false));

    auto overlayFS = llvm::makeIntrusiveRefCnt<llvm::vfs::OverlayFileSystem>(
        fileManager.getVirtualFileSystemPtr());

    overlayFS->pushOverlay(inMemoryFS);
    fileManager.setVirtualFileSystem(overlayFS);
  }

  setCurrentInputs(
      io::InputFile(tempFile->TmpName,
                    InputKind(io::Language::BaseModelica, io::Format::Source)));

  if (auto ec = tempFile->discard()) {
    auto &diag = ci.getDiagnostics();

    diag.Report(
        diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                             "Can't erase the temporary OMC output file"));

    return false;
  }

  return resultCode == EXIT_SUCCESS;
}

void EmitBaseModelicaAction::executeAction() {
  CompilerInstance &ci = getInstance();
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  // Get the default output stream, if none was specified.
  if (ci.isOutputStreamNull()) {
    if (!(os = ci.createDefaultOutputFile(
              false, ci.getSimulationOptions().modelName, "bmo"))) {
      return;
    }
  }

  auto inputFile = getCurrentInputs()[0].getFile();
  auto &fileManager = ci.getFileManager();

  auto fileBuffer =
      llvm::errorOrToExpected(fileManager.getBufferForFile(inputFile));

  if (!fileBuffer) {
    auto &diags = ci.getDiagnostics();
    auto errorCode = llvm::errorToErrorCode(fileBuffer.takeError());

    if (inputFile != "-") {
      diags.Report(clang::diag::err_fe_error_reading)
          << inputFile << errorCode.message();
    } else {
      diags.Report(clang::diag::err_fe_error_reading_stdin)
          << errorCode.message();
    }

    return;
  }

  *os << fileBuffer->get()->getBuffer().str();
}

ASTAction::ASTAction(ASTActionKind action) : action(action) {}

bool ASTAction::beginSourceFilesAction() {
  if (!PreprocessingAction::beginSourceFilesAction()) {
    return false;
  }

  CompilerInstance &ci = getInstance();
  auto &diags = ci.getDiagnostics();

  // Parse the source code.
  auto &fileManager = ci.getFileManager();
  auto inputFile = getCurrentInputs()[0].getFile();

  auto fileRef = inputFile == "-" ? fileManager.getSTDIN()
                                  : fileManager.getFileRef(inputFile, true);

  if (!fileRef) {
    auto errorCode = llvm::errorToErrorCode(fileRef.takeError());

    if (inputFile != "-") {
      diags.Report(clang::diag::err_fe_error_reading)
          << inputFile << errorCode.message();
    } else {
      diags.Report(clang::diag::err_fe_error_reading_stdin)
          << errorCode.message();
    }

    return false;
  }

  auto sourceFile = std::make_shared<SourceFile>(inputFile);

  auto fileBuffer = fileManager.getBufferForFile(*fileRef);

  if (!fileBuffer) {
    auto errorCode = llvm::errorToErrorCode(fileRef.takeError());

    if (inputFile != "-") {
      diags.Report(clang::diag::err_fe_error_reading)
          << inputFile << errorCode.message();
    } else {
      diags.Report(clang::diag::err_fe_error_reading_stdin)
          << errorCode.message();
    }

    return false;
  }

  sourceFile->setMemoryBuffer(fileBuffer->get());
  parser::Parser parser(diags, ci.getSourceManager(), sourceFile);
  auto cls = parser.parseRoot();

  if (!cls.has_value()) {
    diags.Report(diags.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                       "AST generation failed"));

    return false;
  }

  ast = std::move(*cls);

  if (!ci.getFrontendOptions().omcBypass) {
    // Remove the outer package.
    auto *root = ast->cast<ast::Root>();
    assert(root->getInnerClasses().size() == 1);
    assert(root->getInnerClasses()[0]->isa<ast::Package>());
    auto *package = root->getInnerClasses()[0]->cast<ast::Package>();
    auto newRoot = std::make_unique<ast::Root>(root->getLocation());
    newRoot->setInnerClasses(package->getInnerClasses());
    ast = std::move(newRoot);
  }

  return true;
}

EmitASTAction::EmitASTAction() : ASTAction(ASTActionKind::Parse) {}

void EmitASTAction::executeAction() {
  CompilerInstance &ci = getInstance();
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  // Get the default output stream, if none was specified.
  if (ci.isOutputStreamNull()) {
    if (!(os = ci.createDefaultOutputFile(
              false, ci.getSimulationOptions().modelName, "ast"))) {
      return;
    }
  }

  // Get the JSON object for the AST.
  llvm::json::Value json = ast->toJSON();

  // Print the JSON object to file.
  if (ci.isOutputStreamNull()) {
    *os << llvm::formatv("{0:2}", json);
  } else {
    ci.getOutputStream() << llvm::formatv("{0:2}", json);
  }
}

void InitOnlyAction::executeAction() {
  CompilerInstance &ci = getInstance();
  auto &diag = ci.getDiagnostics();

  diag.Report(
      diag.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                           "Use '--init-only' for testing purposes only"));

  auto &os = llvm::outs();

  const CodegenOptions &codegenOptions = ci.getCodeGenOptions();
  printCategory(os, "Code generation");
  printOption(os, "Time optimization level",
              static_cast<long>(codegenOptions.optLevel.getSpeedupLevel()));
  printOption(os, "Size optimization level",
              static_cast<long>(codegenOptions.optLevel.getSizeLevel()));
  printOption(os, "Debug information", codegenOptions.debug);
  printOption(os, "Assertions", codegenOptions.assertions);
  printOption(os, "Inlining", codegenOptions.inlining);
  printOption(os, "Output arrays promotion",
              codegenOptions.outputArraysPromotion);
  printOption(os, "Read-only variables propagation",
              codegenOptions.readOnlyVariablesPropagation);
  printOption(os, "CSE", codegenOptions.cse);
  printOption(os, "OpenMP", codegenOptions.omp);
  os << "\n";

  const SimulationOptions &simulationOptions = ci.getSimulationOptions();
  printCategory(os, "Simulation");
  printOption(os, "Model", simulationOptions.modelName);
  printOption(os, "Solver", simulationOptions.solver);
  os << "\n";
}

void InitOnlyAction::printCategory(llvm::raw_ostream &os,
                                   llvm::StringRef category) const {
  os << "[" << category << "]\n";
}

void InitOnlyAction::printOption(llvm::raw_ostream &os, llvm::StringRef name,
                                 llvm::StringRef value) {
  os << " - " << name << ": " << value << "\n";
}

void InitOnlyAction::printOption(llvm::raw_ostream &os, llvm::StringRef name,
                                 bool value) {
  os << " - " << name << ": " << (value ? "true" : "false") << "\n";
}

void InitOnlyAction::printOption(llvm::raw_ostream &os, llvm::StringRef name,
                                 long value) {
  os << " - " << name << ": " << value << "\n";
}

void InitOnlyAction::printOption(llvm::raw_ostream &os, llvm::StringRef name,
                                 double value) {
  os << " - " << name << ": " << value << "\n";
}

CodeGenAction::CodeGenAction(CodeGenActionKind action)
    : ASTAction(ASTActionKind::Parse), action(action) {
  registerMLIRDialects();
  registerMLIRExtensions();
}

CodeGenAction::~CodeGenAction() {
  if (mlirModule != nullptr) {
    mlirModule->erase();
  }
}

bool CodeGenAction::beginSourceFilesAction() {
  if (!mlirContext) {
    createMLIRContext();
  }

  if (!setUpTargetMachine()) {
    return false;
  }

  if (action == CodeGenActionKind::GenerateMLIR) {
    return generateMLIR();
  }

  if (action == CodeGenActionKind::GenerateMLIRModelica) {
    return generateMLIRModelica();
  }

  if (action == CodeGenActionKind::GenerateMLIRLLVM) {
    return generateMLIRLLVM();
  }

  if (!llvmContext) {
    createLLVMContext();
  }

  assert(action == CodeGenActionKind::GenerateLLVMIR);
  return generateLLVMIR();
}

bool CodeGenAction::setUpTargetMachine() {
  CompilerInstance &ci = getInstance();
  const std::string &triple = ci.getTarget().getTriple().str();

  // Get the LLVM target.
  std::string targetError;

  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, targetError);

  if (!target) {
    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialize the
    // TargetRegistry or if we have a bogus target triple.
    auto &diag = ci.getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                     "Invalid target triple '%0'"))
        << triple;

    return false;
  }

  // Create the LLVM TargetMachine.
  const CodegenOptions &codegenOptions = ci.getCodeGenOptions();

  llvm::CodeGenOptLevel optLevel = mapOptimizationLevelToCodeGenLevel(
      codegenOptions.optLevel, codegenOptions.debug);

  std::string cpu = ci.getCodeGenOptions().cpu;
  std::string features = llvm::join(ci.getCodeGenOptions().features, ",");

  auto relocationModel = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);

  targetMachine.reset(
      target->createTargetMachine(triple, cpu, features, llvm::TargetOptions(),
                                  relocationModel, std::nullopt, optLevel));

  if (!targetMachine) {
    auto &diag = ci.getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                     "Can't create TargetMachine %0"))
        << triple;

    return false;
  }

  return true;
}

llvm::TargetMachine &CodeGenAction::getTargetMachine() {
  assert(targetMachine && "TargetMachine has not been initialized yet");
  return *targetMachine;
}

const llvm::TargetMachine &CodeGenAction::getTargetMachine() const {
  assert(targetMachine && "TargetMachine has not been initialized yet");
  return *targetMachine;
}

llvm::DataLayout CodeGenAction::getDataLayout() const {
  return getTargetMachine().createDataLayout();
}

void CodeGenAction::registerMLIRDialects() {
  // Register all the MLIR native dialects. This does not impact performance,
  // because of lazy loading.
  mlir::registerAllDialects(mlirDialectRegistry);

  // Register the custom dialects.
  mlirDialectRegistry
      .insert<mlir::modeling::ModelingDialect,
              mlir::bmodelica::BaseModelicaDialect, mlir::ida::IDADialect,
              mlir::kinsol::KINSOLDialect, mlir::runtime::RuntimeDialect>();
}

void CodeGenAction::registerMLIRExtensions() {
  mlir::registerAllExtensions(mlirDialectRegistry);

  // Register the extensions of custom dialects.
  mlir::bmodelica::registerAllDialectInterfaceImplementations(
      mlirDialectRegistry);

  mlir::runtime::registerAllDialectInterfaceImplementations(
      mlirDialectRegistry);
}

void CodeGenAction::createMLIRContext() {
  CompilerInstance &ci = getInstance();

  mlirContext = std::make_unique<mlir::MLIRContext>(mlirDialectRegistry);
  mlirContext->enableMultithreading(ci.getFrontendOptions().multithreading);

  // Register the handler for the diagnostics.
  diagnosticHandler = std::make_unique<DiagnosticHandler>(ci);

  mlirContext->getDiagEngine().registerHandler(
      [&](mlir::Diagnostic &diag) -> mlir::LogicalResult {
        llvm::SmallVector<std::string> notes;

        for (const auto &note : diag.getNotes()) {
          notes.push_back(note.str());
        }

        return diagnosticHandler->emit(diag.getSeverity(), diag.getLocation(),
                                       diag.str(), notes);
      });

  // Load dialects.
  mlirContext->loadDialect<mlir::bmodelica::BaseModelicaDialect>();
  mlirContext->loadDialect<mlir::DLTIDialect>();
  mlirContext->loadDialect<mlir::LLVM::LLVMDialect>();
}

mlir::MLIRContext &CodeGenAction::getMLIRContext() {
  assert(mlirContext && "MLIR context has not been initialized");
  return *mlirContext;
}

void CodeGenAction::createLLVMContext() {
  llvmContext = std::make_unique<llvm::LLVMContext>();
}

llvm::LLVMContext &CodeGenAction::getLLVMContext() {
  assert(llvmContext && "LLVM context has not been initialized");
  return *llvmContext;
}

bool CodeGenAction::generateMLIR() {
  CompilerInstance &ci = getInstance();

  if (getCurrentInputs()[0].getKind().getLanguage() == Language::MLIR) {
    // If the input is an MLIR file, parse it.
    llvm::SourceMgr sourceMgr;

    auto &fileManager = ci.getFileManager();
    auto inputFile = getCurrentInputs()[0].getFile();

    auto fileRef = inputFile == "-" ? fileManager.getSTDIN()
                                    : fileManager.getFileRef(inputFile, true);

    if (!fileRef) {
      auto &diags = ci.getDiagnostics();
      auto errorCode = llvm::errorToErrorCode(fileRef.takeError());

      if (inputFile != "-") {
        diags.Report(clang::diag::err_fe_error_reading)
            << inputFile << errorCode.message();
      } else {
        diags.Report(clang::diag::err_fe_error_reading_stdin)
            << errorCode.message();
      }

      return false;
    }

    ci.getSourceManager().createFileID(*fileRef, clang::SourceLocation(),
                                       clang::SrcMgr::C_User);

    auto fileBuffer = fileManager.getBufferForFile(*fileRef);

    if (!fileBuffer) {
      auto &diags = ci.getDiagnostics();
      auto errorCode = llvm::errorToErrorCode(fileRef.takeError());

      if (inputFile != "-") {
        diags.Report(clang::diag::err_fe_error_reading)
            << inputFile << errorCode.message();
      } else {
        diags.Report(clang::diag::err_fe_error_reading_stdin)
            << errorCode.message();
      }

      return false;
    }

    sourceMgr.AddNewSourceBuffer(std::move(*fileBuffer), llvm::SMLoc());

    // Parse the module.
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &getMLIRContext());

    if (!module || mlir::failed(module->verifyInvariants())) {
      auto &diag = ci.getDiagnostics();

      diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                       "Could not parse MLIR"));

      return false;
    }

    mlirModule = std::make_unique<mlir::ModuleOp>(module.release());
  } else {
    // If the input is not an MLIR file, then the MLIR module will be
    // obtained starting from the AST.

    if (!ASTAction::beginSourceFilesAction()) {
      return false;
    }

    // Convert the AST to MLIR.
    marco::codegen::lowering::Bridge bridge(getMLIRContext());
    if (!bridge.lower(*ast->cast<ast::Root>())) {
      return false;
    }

    mlirModule = std::move(bridge.getMLIRModule());
  }

  // Set target triple and data layout inside the MLIR module.
  setMLIRModuleTargetTriple();
  setMLIRModuleDataLayout();

  // Verify the IR.
  mlir::PassManager pm(&getMLIRContext());
  pm.addPass(std::make_unique<codegen::lowering::VerifierPass>());

  if (!mlir::succeeded(pm.run(*mlirModule))) {
    auto &diag = ci.getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                     "Verification of MLIR code failed"));

    return false;
  }

  return true;
}

bool CodeGenAction::generateMLIRModelica() { return generateMLIR(); }

bool CodeGenAction::generateMLIRLLVM() {
  if (!generateMLIR()) {
    return false;
  }

  mlir::PassManager pm(&getMLIRContext());
  CompilerInstance &ci = getInstance();

  // Enable verification.
  pm.enableVerifier(true);

  // If requested, print the statistics.
  if (ci.getFrontendOptions().printStatistics) {
    pm.enableTiming();
    pm.enableStatistics();
  }

  if (ci.getFrontendOptions().shouldPrintIR()) {
    // IR printing requires multithreading to be disabled.
    pm.getContext()->disableMultithreading();

    auto shouldPrintBeforePass = [&](mlir::Pass *pass,
                                     mlir::Operation *) -> bool {
      return pass->getArgument() == ci.getFrontendOptions().printIRBeforePass;
    };

    auto shouldPrintAfterPass = [&](mlir::Pass *pass,
                                    mlir::Operation *) -> bool {
      return pass->getArgument() == ci.getFrontendOptions().printIRAfterPass;
    };

    pm.enableIRPrinting(shouldPrintBeforePass, shouldPrintAfterPass, true,
                        false, false, llvm::errs());
  }

  buildMLIRLoweringPipeline(pm);

  if (mlir::failed(pm.run(*mlirModule))) {
    if (ci.getFrontendOptions().printIROnFailure) {
      llvm::errs() << *mlirModule << "\n";
    }

    return false;
  }

  return true;
}

void CodeGenAction::setMLIRModuleTargetTriple() {
  assert(mlirModule && "MLIR module has not been created yet");
  const std::string &triple = getTargetMachine().getTargetTriple().str();

  llvm::StringRef tripleAttrName =
      mlir::LLVM::LLVMDialect::getTargetTripleAttrName();

  auto tripleAttr = mlir::StringAttr::get(mlirModule->getContext(), triple);

  auto existingTripleAttr =
      (*mlirModule)->getAttrOfType<mlir::StringAttr>(tripleAttrName);

  if (existingTripleAttr && existingTripleAttr != tripleAttr) {
    // The LLVM module already has a target triple which is different from
    // the one specified through the command-line.
    auto &diag = getInstance().getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                     "The target triple is being overridden"));
  }

  (*mlirModule)->setAttr(tripleAttrName, tripleAttr);
}

void CodeGenAction::setMLIRModuleDataLayout() {
  assert(mlirModule && "MLIR module has not been created yet");
  const llvm::DataLayout &dl = getTargetMachine().createDataLayout();

  llvm::StringRef dlAttrName = mlir::LLVM::LLVMDialect::getDataLayoutAttrName();

  auto dlAttr =
      mlir::StringAttr::get(&getMLIRContext(), dl.getStringRepresentation());

  auto existingDlAttr =
      (*mlirModule)->getAttrOfType<mlir::StringAttr>(dlAttrName);

  llvm::StringRef dlSpecAttrName = mlir::DLTIDialect::kDataLayoutAttrName;

  mlir::DataLayoutSpecInterface dlSpecAttr =
      mlir::translateDataLayout(dl, &getMLIRContext());

  auto existingDlSpecAttr =
      (*mlirModule)
          ->getAttrOfType<mlir::DataLayoutSpecInterface>(dlSpecAttrName);

  if ((existingDlAttr && existingDlAttr != dlAttr) ||
      (existingDlSpecAttr &&
       existingDlSpecAttr != mlir::cast<mlir::Attribute>(dlSpecAttr))) {
    // The MLIR module already has a data layout which is different from the
    // one given by the target machine.
    auto &diag = getInstance().getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                     "The data layout is being overridden"));
  }

  (*mlirModule)->setAttr(dlAttrName, dlAttr);

  (*mlirModule)
      ->setAttr(dlSpecAttrName, mlir::cast<mlir::Attribute>(dlSpecAttr));
}

void CodeGenAction::buildMLIRLoweringPipeline(mlir::PassManager &pm) {
  CompilerInstance &ci = getInstance();

  if (!ci.getCodeGenOptions().debug) {
    // Remove the debug information if a non-debuggable executable has been
    // requested. By doing this at the beginning of the compilation pipeline
    // we reduce the time needed for the pass itself, as the code inevitably
    // grows while traversing the pipeline.
    pm.addPass(mlir::createStripDebugInfoPass());
  }

  // Try to simplify the starting IR.
  pm.addPass(mlir::createCanonicalizerPass());

  if (ci.getCodeGenOptions().readOnlyVariablesPropagation) {
    // Propagate the read-only variables and try to fold the constants right
    // after.
    pm.addPass(createMLIRReadOnlyVariablesPropagationPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }

  pm.addPass(mlir::bmodelica::createAutomaticDifferentiationPass());
  pm.addPass(mlir::bmodelica::createDerivativeChainRulePass());

  // Inline the functions marked as "inlinable", in order to enable
  // simplifications for the model solving process.
  pm.addPass(mlir::bmodelica::createFunctionInliningPass());

  pm.addPass(mlir::bmodelica::createRecordInliningPass());
  pm.addPass(mlir::bmodelica::createFunctionUnwrapPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Infer the range boundaries for subscriptions.
  pm.addPass(mlir::bmodelica::createRangeBoundariesInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Make the equations having only one element on its left and right-hand
  // sides.
  pm.addPass(mlir::bmodelica::createEquationSidesSplitPass());

  // Add additional inductions in case of equalities between arrays.
  pm.addPass(mlir::bmodelica::createEquationInductionsExplicitationPass());


  // Fold accesses operating on views.
  pm.addPass(mlir::bmodelica::createViewAccessFoldingPass());

  // Lift the equations.
  pm.addPass(mlir::bmodelica::createEquationTemplatesCreationPass());

  // Eliminate repeated function calls
  if (ci.getCodeGenOptions().functionCallsCSE) {
    pm.addPass(mlir::bmodelica::createCallCSEPass());
  }

  // Materialize the derivatives.
  pm.addPass(mlir::bmodelica::createDerivativesMaterializationPass());

  // Legalize the model.
  pm.addPass(mlir::bmodelica::createBindingEquationConversionPass());
  pm.addPass(mlir::bmodelica::createExplicitStartValueInsertionPass());
  pm.addPass(mlir::bmodelica::createModelAlgorithmConversionPass());
  pm.addPass(mlir::bmodelica::createExplicitInitialEquationsInsertionPass());

  if (ci.getFrontendOptions().printModelInfo) {
    pm.addPass(mlir::bmodelica::createPrintModelInfoPass());
  }

  // Solve the model.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::bmodelica::createMatchingPass());
  pm.addPass(mlir::bmodelica::createEquationAccessSplitPass());

  if (ci.getCodeGenOptions().singleValuedInductionElimination) {
    pm.addPass(mlir::bmodelica::createSingleValuedInductionEliminationPass());
  }

  if (ci.getCodeGenOptions().variablesPruning) {
    pm.addPass(mlir::bmodelica::createVariablesPruningPass());
  }

  pm.addPass(mlir::bmodelica::createSCCDetectionPass());

  if (ci.getCodeGenOptions().variablesToParametersPromotion) {
    pm.addPass(mlir::bmodelica::createVariablesPromotionPass());
  }

  // Try to solve the cycles by substitution.
  pm.addPass(createMLIRSCCSolvingBySubstitutionPass());

  // Simplify the possibly complex accesses introduced by equations
  // substitutions.
  pm.addPass(mlir::createCanonicalizerPass());

  if (ci.getCodeGenOptions().singleValuedInductionElimination) {
    pm.addPass(mlir::bmodelica::createSingleValuedInductionEliminationPass());
  }

  // Apply the selected solver.
  pm.addPass(
      llvm::StringSwitch<std::unique_ptr<mlir::Pass>>(
          ci.getSimulationOptions().solver)
          .Case("euler-forward", createMLIREulerForwardPass())
          .Case("ida", createMLIRIDAPass())
          .Case("rk4", createMLIRRungeKuttaPass("rk4"))
          .Case("rk-euler-forward", createMLIRRungeKuttaPass("euler-forward"))
          .Case("rk-midpoint", createMLIRRungeKuttaPass("midpoint"))
          .Case("rk-heun", createMLIRRungeKuttaPass("heun"))
          .Case("rk-ralston", createMLIRRungeKuttaPass("ralston"))
          .Case("rk-heun-euler", createMLIRRungeKuttaPass("heun-euler"))
          .Case("rk-bogacki-shampine",
                createMLIRRungeKuttaPass("bogacki-shampine"))
          .Case("rk-fehlberg", createMLIRRungeKuttaPass("fehlberg"))
          .Case("rk-cash-karp", createMLIRRungeKuttaPass("cash-karp"))
          .Case("rk-dormand-prince", createMLIRRungeKuttaPass("dormand-prince"))
          .Case("rk-euler-backward", createMLIRRungeKuttaPass("euler-backward"))
          .Default(mlir::bmodelica::createEulerForwardPass()));

  // Solve the initial conditions model.
  pm.addPass(mlir::bmodelica::createInitialConditionsSolvingPass());

  // Schedule the equations.
  pm.addPass(mlir::bmodelica::createSchedulingPass());


  // Explicitate the equations.
  pm.addPass(mlir::bmodelica::createEquationExplicitationPass());

  if ( ci.getCodeGenOptions().resultRematerialization ) {
    // Perform experimental memory balancing optimization
    //   recomputes results when operands are estimated likely to
    //   reside in caches.
    pm.addPass(mlir::bmodelica::createResultRematerializationPass());
  }

  // Lift loop-independent code from loops of equations.
  pm.addNestedPass<mlir::bmodelica::EquationFunctionOp>(
      mlir::bmodelica::createEquationFunctionLoopHoistingPass());

  // Export the unsolved SCCs to KINSOL.
  pm.addPass(createMLIRSCCSolvingWithKINSOLPass());


  // Parallelize the scheduled blocks.
  pm.addPass(mlir::bmodelica::createScheduleParallelizationPass());

  if (ci.getCodeGenOptions().equationsRuntimeScheduling) {
    // Delegate the calls to the equation functions to the runtime library.
    pm.addPass(mlir::bmodelica::createSchedulersInstantiationPass());
  }

  // Check that no SCC is left unsolved.
  pm.addPass(mlir::bmodelica::createSCCAbsenceVerificationPass());

  pm.addPass(createMLIRBaseModelicaToRuntimeConversionPass());

  pm.addPass(createMLIRFunctionScalarizationPass());
  pm.addPass(mlir::bmodelica::createExplicitCastInsertionPass());
  pm.addPass(mlir::createCanonicalizerPass());

  if (ci.getCodeGenOptions().cse) {
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<mlir::bmodelica::FunctionOp>(mlir::createCSEPass());
    pm.addNestedPass<mlir::func::FuncOp>(mlir::createCSEPass());
  }

  pm.addPass(mlir::bmodelica::createFunctionDefaultValuesConversionPass());
  pm.addPass(mlir::createBaseModelicaToCFConversionPass());

  if (ci.getCodeGenOptions().inlining) {
    // Inline the functions with the 'inline' annotation.
    pm.addPass(mlir::createInlinerPass());
  }

  // Lower to MLIR core dialects.
  pm.addPass(mlir::createBaseModelicaToMLIRCoreConversionPass());
  pm.addPass(mlir::createSUNDIALSToFuncConversionPass());
  pm.addPass(mlir::createIDAToFuncConversionPass());
  pm.addPass(mlir::createKINSOLToFuncConversionPass());
  pm.addPass(mlir::createRuntimeToFuncConversionPass());
  pm.addPass(mlir::createCanonicalizerPass());

  // Recreate structured control flow whenever possible.
  pm.addPass(mlir::createLiftControlFlowToSCFPass());

  // Generalize linalg operations.
  pm.addPass(mlir::createLinalgGeneralizeNamedOpsPass());

  // Perform bufferization.
  pm.addPass(createMLIROneShotBufferizePass());

  if (ci.getCodeGenOptions().outputArraysPromotion) {
    pm.addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
  }

  // Lower the linalg dialect.
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());

  // Optimize loops.
  if (ci.getCodeGenOptions().loopFusion) {
    pm.addNestedPass<mlir::func::FuncOp>(mlir::affine::createLoopFusionPass());

    // Try to get rid of privatized memrefs that may have been introduced by
    // the loop fusion pass.
    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createAffineScalarReplacementPass());
  }

  if (ci.getCodeGenOptions().loopTiling) {
    addMLIRLoopTilingPass(pm);
  }

  if (ci.getCodeGenOptions().heapToStackPromotion) {
    pm.addNestedPass<mlir::func::FuncOp>(createMLIRPromoteBuffersToStackPass());
  }

  // Buffer deallocations placements must be performed after loop
  // optimizations because they may introduce additional heap allocations.
  buildMLIRBufferDeallocationPipeline(pm);
  pm.addPass(mlir::createBufferizationToMemRefPass());

  // Convert the raw variables.
  // This must be performed only after the insertion of buffer deallocations,
  // so that the indirection level introduced by dynamically-sized
  // variables does not interfere.
  pm.addPass(mlir::createBaseModelicaRawVariablesConversionPass());

  // Lower to LLVM dialect.
  pm.addPass(mlir::createRuntimeModelMetadataConversionPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());

  // Perform again the conversion of bmodelica to core dialects,
  // as additional operations may have been introduced by the canonicalization
  // patterns.
  pm.addPass(mlir::createBaseModelicaToMLIRCoreConversionPass());

  // Perform the default conversions towards LLVM dialect.
  pm.addPass(mlir::createConvertToLLVMPass());

  // The conversion of the internal dialects to LLVM must be performed
  // separately because it requires a symbol table for efficiency reasons.
  // If MLIR will ever provide a SymbolTableCollection within the ToLLVM
  // conversion infrastructure, or if such information will be embedded in
  // the symbol table operations, then this separation will be not needed
  // anymore.
  pm.addPass(mlir::createBaseModelicaToLLVMConversionPass());
  pm.addPass(mlir::createIDAToLLVMConversionPass());
  pm.addPass(mlir::createKINSOLToLLVMConversionPass());
  pm.addPass(mlir::createRuntimeToLLVMConversionPass());

  // Finalization passes.
  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(
      mlir::createReconcileUnrealizedCastsPass());

  pm.addNestedPass<mlir::LLVM::LLVMFuncOp>(mlir::createCanonicalizerPass());

  pm.addPass(mlir::runtime::createHeapFunctionsReplacementPass());
  pm.addPass(mlir::LLVM::createLegalizeForExportPass());
}

std::unique_ptr<mlir::Pass>
CodeGenAction::createMLIRFunctionScalarizationPass() {
  CompilerInstance &ci = getInstance();

  mlir::bmodelica::FunctionScalarizationPassOptions options;
  options.assertions = ci.getCodeGenOptions().assertions;

  return mlir::bmodelica::createFunctionScalarizationPass(options);
}

std::unique_ptr<mlir::Pass>
CodeGenAction::createMLIRReadOnlyVariablesPropagationPass() {
  CompilerInstance &ci = getInstance();

  mlir::bmodelica::ReadOnlyVariablesPropagationPassOptions options;
  std::string modelName = ci.getSimulationOptions().modelName;

  if (!ci.getFrontendOptions().omcBypass) {
    auto pos = modelName.find_last_of('.');

    if (pos != std::string::npos) {
      modelName = llvm::StringRef(modelName).drop_front(pos).str();
    }
  }

  options.modelName = modelName;
  return mlir::bmodelica::createReadOnlyVariablesPropagationPass(options);
}

std::unique_ptr<mlir::Pass>
CodeGenAction::createMLIRRungeKuttaPass(llvm::StringRef variant) {
  mlir::bmodelica::RungeKuttaPassOptions options;
  options.variant = variant.str();

  return mlir::bmodelica::createRungeKuttaPass(options);
}

std::unique_ptr<mlir::Pass> CodeGenAction::createMLIREulerForwardPass() {
  auto &ci = getInstance();

  mlir::bmodelica::EulerForwardPassOptions options;

  options.rangedStateUpdateFunctions =
      ci.getCodeGenOptions().equationsRuntimeScheduling;

  return mlir::bmodelica::createEulerForwardPass(options);
}

std::unique_ptr<mlir::Pass> CodeGenAction::createMLIRIDAPass() {
  CompilerInstance &ci = getInstance();

  mlir::bmodelica::IDAPassOptions options;
  options.reducedSystem = ci.getSimulationOptions().IDAReducedSystem;
  options.reducedDerivatives = ci.getSimulationOptions().IDAReducedDerivatives;
  options.jacobianOneSweep = ci.getSimulationOptions().IDAJacobianOneSweep;
  options.debugInformation = ci.getCodeGenOptions().debug;

  return mlir::bmodelica::createIDAPass(options);
}

std::unique_ptr<mlir::Pass>
CodeGenAction::createMLIRSCCSolvingBySubstitutionPass() {
  CompilerInstance &ci = getInstance();
  mlir::bmodelica::SCCSolvingBySubstitutionPassOptions options;

  options.maxIterations =
      ci.getCodeGenOptions().sccSolvingBySubstitutionMaxIterations;

  options.maxEquationsInSCC =
      ci.getCodeGenOptions().sccSolvingBySubstitutionMaxEquationsInSCC;

  return mlir::bmodelica::createSCCSolvingBySubstitutionPass(options);
}

std::unique_ptr<mlir::Pass>
CodeGenAction::createMLIRSCCSolvingWithKINSOLPass() {
  CompilerInstance &ci = getInstance();

  mlir::bmodelica::SCCSolvingWithKINSOLPassOptions options;
  // TODO create options for KINSOL
  options.reducedDerivatives = ci.getSimulationOptions().IDAReducedDerivatives;
  options.jacobianOneSweep = ci.getSimulationOptions().IDAJacobianOneSweep;
  options.debugInformation = ci.getCodeGenOptions().debug;

  return mlir::bmodelica::createSCCSolvingWithKINSOLPass(options);
}

std::unique_ptr<mlir::Pass>
CodeGenAction::createMLIRBaseModelicaToRuntimeConversionPass() {
  CompilerInstance &ci = getInstance();

  mlir::BaseModelicaToRuntimeConversionPassOptions options;
  options.variableFilter = ci.getFrontendOptions().variableFilter;

  return mlir::createBaseModelicaToRuntimeConversionPass(options);
}

std::unique_ptr<mlir::Pass> CodeGenAction::createMLIROneShotBufferizePass() {
  mlir::bufferization::OneShotBufferizationOptions options;
  options.bufferizeFunctionBoundaries = true;

  return mlir::bufferization::createOneShotBufferizePass(options);
}

void CodeGenAction::buildMLIRBufferDeallocationPipeline(
    mlir::OpPassManager &pm) {
  mlir::bufferization::BufferDeallocationPipelineOptions options;
  mlir::bufferization::buildBufferDeallocationPipeline(pm, options);
}

void CodeGenAction::addMLIRLoopTilingPass(mlir::OpPassManager &pm) {
  const auto *subtargetInfo = getTargetMachine().getMCSubtargetInfo();

  if (auto cacheSize = subtargetInfo->getCacheSize(0);
      cacheSize && *cacheSize >= 1024) {

    pm.addNestedPass<mlir::func::FuncOp>(
        mlir::affine::createLoopTilingPass(*cacheSize));
  }
}

std::unique_ptr<mlir::Pass>
CodeGenAction::createMLIRPromoteBuffersToStackPass() {
  // TODO: control with CLI
  unsigned int maxAllocSizeInBytes = 1024;

  auto isSmallAllocFn = [=](mlir::Value alloc) -> bool {
    auto type = mlir::dyn_cast<mlir::ShapedType>(alloc.getType());

    if (!type ||
        !alloc.getDefiningOp<mlir::bufferization::AllocationOpInterface>()) {
      return false;
    }

    if (!type.hasStaticShape()) {
      return false;
    }

    unsigned int bitwidth = mlir::DataLayout::closest(alloc.getDefiningOp())
                                .getTypeSizeInBits(type.getElementType());

    return type.getNumElements() * bitwidth <= maxAllocSizeInBytes * 8;
  };

  return mlir::bufferization::createPromoteBuffersToStackPass(isSmallAllocFn);
}

void CodeGenAction::registerMLIRToLLVMIRTranslations() {
  mlir::registerBuiltinDialectTranslation(getMLIRContext());
  mlir::registerLLVMDialectTranslation(getMLIRContext());
  mlir::registerOpenMPDialectTranslation(getMLIRContext());
}

bool CodeGenAction::generateLLVMIR() {
  CompilerInstance &ci = getInstance();

  if (getCurrentInputs()[0].getKind().getLanguage() == Language::LLVM_IR) {
    // If the input is an LLVM file, parse it return.
    llvm::SMDiagnostic err;

    llvmModule = llvm::parseIRFile(getCurrentInputs()[0].getFile(), err,
                                   getLLVMContext());

    if (!llvmModule || llvm::verifyModule(*llvmModule, &llvm::errs())) {
      err.print("marco", llvm::errs());
      auto &diag = ci.getDiagnostics();

      diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                       "Could not parse LLVM-IR"));

      return false;
    }
  } else {
    // Obtain the LLVM dialect.
    if (!generateMLIRLLVM()) {
      return false;
    }

    // Register the MLIR translations to obtain LLVM-IR.
    registerMLIRToLLVMIRTranslations();

    // Translate to LLVM-IR.
    std::optional<llvm::StringRef> moduleName = mlirModule->getName();

    if (!moduleName) {
      // Fallback to the name of the compiled model.
      if (llvm::StringRef modelName = ci.getSimulationOptions().modelName;
          !modelName.empty()) {
        moduleName = modelName;
      }
    }

    llvmModule = mlir::translateModuleToLLVMIR(*mlirModule, getLLVMContext(),
                                               moduleName ? *moduleName
                                                          : "ModelicaModule");

    if (!llvmModule) {
      auto &diag = ci.getDiagnostics();

      diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Fatal,
                                       "Failed to create the LLVM module"));

      return false;
    }
  }

  // Set target triple and data layout inside the LLVM module.
  setLLVMModuleTargetTriple();
  setLLVMModuleDataLayout();

  // Apply the optimizations to the LLVM module.
  runOptimizationPipeline();

  return true;
}

void CodeGenAction::setLLVMModuleTargetTriple() {
  assert(llvmModule && "LLVM module has not been created yet");
  const std::string &triple = getTargetMachine().getTargetTriple().str();

  if (llvmModule->getTargetTriple() != triple) {
    // The LLVM module already has a target triple which is different from
    // the one specified through the command-line.
    auto &diag = getInstance().getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                     "The target triple is being overridden"));
  }

  llvmModule->setTargetTriple(triple);
}

void CodeGenAction::setLLVMModuleDataLayout() {
  assert(llvmModule && "LLVM module has not been created yet");
  const llvm::DataLayout &dataLayout = getTargetMachine().createDataLayout();

  if (llvmModule->getDataLayout() != dataLayout) {
    // The LLVM module already has a data layout which is different from the
    // one given by the target machine.
    auto &diag = getInstance().getDiagnostics();

    diag.Report(diag.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                     "The data layout is being overridden"));
  }

  llvmModule->setDataLayout(dataLayout);
}

void CodeGenAction::runOptimizationPipeline() {
  CompilerInstance &ci = getInstance();
  const CodegenOptions &codegenOptions = ci.getCodeGenOptions();

  // Create the analysis managers.
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  // Create the pass manager builder.
  llvm::PassInstrumentationCallbacks pic;
  llvm::PipelineTuningOptions pto;
  std::optional<llvm::PGOOptions> pgoOpt;

  llvm::StandardInstrumentations si(getLLVMContext(), false);
  si.registerCallbacks(pic, &mam);

  llvm::PassBuilder pb(&getTargetMachine(), pto, pgoOpt, &pic);

  // Register all the basic analyses with the managers.
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  // Create the pass manager.
  llvm::ModulePassManager mpm;

  if (codegenOptions.optLevel == llvm::OptimizationLevel::O0) {
    mpm = pb.buildO0DefaultPipeline(codegenOptions.optLevel, false);
  } else {
    mpm = pb.buildPerModuleDefaultPipeline(codegenOptions.optLevel);
  }

  // Run the passes.
  mpm.run(*llvmModule, mam);
}

void CodeGenAction::emitBackendOutput(
    clang::BackendAction backendAction,
    std::unique_ptr<llvm::raw_pwrite_stream> os) {
  auto &ci = getInstance();
  clang::HeaderSearchOptions headerSearchOptions;

  EmitBackendOutput(
      ci.getDiagnostics(), headerSearchOptions, ci.getCodeGenOptions(),
      ci.getTarget().getTargetOpts(), ci.getLanguageOptions(),
      ci.getTarget().getDataLayoutString(), llvmModule.get(), backendAction,
      ci.getFileManager().getVirtualFileSystemPtr(), std::move(os));
}

EmitMLIRAction::EmitMLIRAction()
    : CodeGenAction(CodeGenActionKind::GenerateMLIR) {}

void EmitMLIRAction::executeAction() {
  CompilerInstance &ci = getInstance();
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  // Get the default output stream, if none was specified.
  if (ci.isOutputStreamNull()) {
    auto fileOrBufferNames = getCurrentFilesOrBufferNames();

    if (!(os = ci.createDefaultOutputFile(
              false, ci.getSimulationOptions().modelName, "mlir"))) {
      return;
    }
  }

  // Emit MLIR.
  mlirModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream());
}

EmitMLIRModelicaAction::EmitMLIRModelicaAction()
    : CodeGenAction(CodeGenActionKind::GenerateMLIRModelica) {}

void EmitMLIRModelicaAction::executeAction() {
  CompilerInstance &ci = getInstance();
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  // Get the default output stream, if none was specified.
  if (ci.isOutputStreamNull()) {
    auto fileOrBufferNames = getCurrentFilesOrBufferNames();

    if (!(os = ci.createDefaultOutputFile(
              false, ci.getSimulationOptions().modelName, "mo.mlir"))) {
      return;
    }
  }

  // Emit MLIR.
  mlirModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream());
}

EmitMLIRLLVMAction::EmitMLIRLLVMAction()
    : CodeGenAction(CodeGenActionKind::GenerateMLIRLLVM) {}

void EmitMLIRLLVMAction::executeAction() {
  CompilerInstance &ci = getInstance();
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  // Get the default output stream, if none was specified.
  if (ci.isOutputStreamNull()) {
    auto fileOrBufferNames = getCurrentFilesOrBufferNames();

    if (!(os = ci.createDefaultOutputFile(
              false, ci.getSimulationOptions().modelName, "llvm.mlir"))) {
      return;
    }
  }

  // Emit MLIR.
  mlirModule->print(ci.isOutputStreamNull() ? *os : ci.getOutputStream());
}

EmitLLVMIRAction::EmitLLVMIRAction()
    : CodeGenAction(CodeGenActionKind::GenerateLLVMIR) {}

void EmitLLVMIRAction::executeAction() {
  CompilerInstance &ci = getInstance();

  // Get the default output stream, if none was specified.
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  if (ci.isOutputStreamNull()) {
    if (!(os = ci.createDefaultOutputFile(
              false, ci.getSimulationOptions().modelName, "ll"))) {
      return;
    }
  } else {
    os = ci.takeOutputStream();
  }

  // Emit LLVM-IR.
  emitBackendOutput(clang::BackendAction::Backend_EmitLL, std::move(os));
}

EmitBitcodeAction::EmitBitcodeAction()
    : CodeGenAction(CodeGenActionKind::GenerateLLVMIR) {}

void EmitBitcodeAction::executeAction() {
  CompilerInstance &ci = getInstance();

  // Get the default output stream, if none was specified.
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  if (ci.isOutputStreamNull()) {
    if (!(os = ci.createDefaultOutputFile(
              true, ci.getSimulationOptions().modelName, "bc"))) {
      return;
    }
  } else {
    os = ci.takeOutputStream();
  }

  // Emit the bitcode.
  emitBackendOutput(clang::BackendAction::Backend_EmitBC, std::move(os));
}

EmitAssemblyAction::EmitAssemblyAction()
    : CodeGenAction(CodeGenActionKind::GenerateLLVMIR) {}

void EmitAssemblyAction::executeAction() {
  CompilerInstance &ci = getInstance();

  // Get the default output stream, if none was specified.
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  if (ci.isOutputStreamNull()) {
    if (!(os = ci.createDefaultOutputFile(
              false, ci.getSimulationOptions().modelName, "s"))) {
      return;
    }
  } else {
    os = ci.takeOutputStream();
  }

  // Emit the assembly code.
  emitBackendOutput(clang::BackendAction::Backend_EmitAssembly, std::move(os));
}

EmitObjAction::EmitObjAction()
    : CodeGenAction(CodeGenActionKind::GenerateLLVMIR) {}

void EmitObjAction::executeAction() {
  CompilerInstance &ci = getInstance();

  // Get the default output stream, if none was specified.
  std::unique_ptr<llvm::raw_pwrite_stream> os;

  if (ci.isOutputStreamNull()) {
    if (!(os = ci.createDefaultOutputFile(
              true, ci.getSimulationOptions().modelName, "o"))) {
      return;
    }
  } else {
    os = ci.takeOutputStream();
  }

  // Emit the object file.
  emitBackendOutput(clang::BackendAction::Backend_EmitObj, std::move(os));
}
} // namespace marco::frontend
