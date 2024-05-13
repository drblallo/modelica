#include "marco/Frontend/CompilerInvocation.h"
#include "mlir/IR/Diagnostics.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/Support/Process.h"

using namespace ::marco;
using namespace ::marco::frontend;
using namespace ::marco::io;
using namespace clang::driver;

//===---------------------------------------------------------------------===//
// CompilerInvocation
//===---------------------------------------------------------------------===//

/// Tweak the frontend configuration based on the frontend action
static void setUpFrontendBasedOnAction(FrontendOptions& options)
{
  assert(options.programAction != InvalidAction && "Frontend action not set!");
}

static bool parseDiagnosticArgs(
    clang::DiagnosticOptions& diagnosticOptions,
    llvm::opt::ArgList& args,
    clang::DiagnosticsEngine* diags = nullptr,
    bool defaultDiagColor = true)
{
  return clang::ParseDiagnosticArgs(
      diagnosticOptions, args, diags, defaultDiagColor);
}

static void parseFrontendArgs(
    FrontendOptions& options,
    llvm::opt::ArgList& args,
    clang::DiagnosticsEngine& diagnostics)
{
  // Default action
  options.programAction = EmitObject;

  if (args.hasArg(options::OPT_emit_ast)) {
    // -emit-ast is not part of the Action group in clang's options.
    options.programAction = EmitAST;
  } else if (const llvm::opt::Arg* a =
                 args.getLastArg(options::OPT_Action_Group)) {
    switch (a->getOption().getID()) {
      case options::OPT_init_only:
        options.programAction = InitOnly;
        break;

      case options::OPT_emit_base_modelica:
        options.programAction = EmitBaseModelica;
        break;

      case options::OPT_emit_mlir:
        options.programAction = EmitMLIR;
        break;

      case options::OPT_emit_mlir_modelica:
        options.programAction = EmitMLIRModelica;
        break;

      case options::OPT_emit_mlir_llvm:
        options.programAction = EmitMLIRLLVM;
        break;

      case options::OPT_emit_llvm:
        options.programAction = EmitLLVMIR;
        break;

      case options::OPT_emit_llvm_bc:
        options.programAction = EmitLLVMBitcode;
        break;

      // Old OPT_compile_only.
      case options::OPT_S:
        options.programAction = EmitAssembly;
        break;

      case options::OPT_emit_obj:
        options.programAction = EmitObject;
        break;

      default: {
        llvm_unreachable("Invalid option in group!");
      }
    }
  }

  options.outputFile = args.getLastArgValue(options::OPT_o);
  options.showHelp = args.hasArg(options::OPT_help);
  options.showVersion = args.hasArg(options::OPT_version);
  options.printModelInfo = args.hasArg(options::OPT_print_model_info);
  options.printStatistics = args.hasArg(options::OPT_print_statistics);

  options.multithreading = args.hasFlag(
      options::OPT_multithreading,
      options::OPT_no_multithreading,
      options.multithreading);

  // Collect the input files and save them in our instance of FrontendOptions
  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  options.inputs.clear();

  if (inputs.empty()) {
    // '-' is the default input if none is given
    inputs.push_back("-");
  }

  for (size_t i = 0, e = inputs.size(); i != e; ++i) {
    options.inputs.emplace_back(
        inputs[i], InputKind::getFromFullFileName(inputs[i]));
  }

  setUpFrontendBasedOnAction(options);

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_omc_path)) {
    llvm::StringRef value = arg->getValue();
    options.omcPath = value.str();
  }

  for (const auto& omcArg : args.getAllArgValues(options::OPT_omc_arg)) {
    options.omcCustomArgs.push_back(omcArg);
  }

  options.omcBypass = args.hasArg(options::OPT_omc_bypass);

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_filter)) {
    options.variablesFilter = arg->getValue();
  }
}

static void parseCodegenArgs(
    CodegenOptions& options,
    llvm::opt::ArgList& args,
    clang::DiagnosticsEngine& diagnostics)
{
  // Determine the optimization level
  for (const auto& arg : args.getAllArgValues(options::OPT_O)) {
    if (arg == "0") {
      options.optLevel = llvm::OptimizationLevel::O0;
    } else if (arg == "1") {
      options.optLevel = llvm::OptimizationLevel::O1;
    } else if (arg == "2") {
      options.optLevel = llvm::OptimizationLevel::O2;
    } else if (arg == "3") {
      options.optLevel = llvm::OptimizationLevel::O3;
    } else if (arg == "fast") {
      options.optLevel = llvm::OptimizationLevel::O3;
    } else if (arg == "s") {
      options.optLevel = llvm::OptimizationLevel::Os;
    } else if (arg == "z") {
      options.optLevel = llvm::OptimizationLevel::Oz;
    } else {
      // "Unknown optimization option: %s"
      diagnostics.Report(diagnostics.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "Unknown optimization option '%0'")) << arg;
    }
  }

  // Set the default options based on the optimization level
  if (options.optLevel.getSpeedupLevel() > 0) {
    options.debug = false;
    options.assertions = false;
  }

  if (options.optLevel.getSpeedupLevel() > 1) {
    options.outputArraysPromotion = true;
    options.readOnlyVariablesPropagation = true;
    options.variablesToParametersPromotion = true;
    options.inlining = true;
    options.cse = true;
  }

  if (options.optLevel.getSizeLevel() > 0) {
    options.debug = false;
    options.cse = true;
  }

  if (options.optLevel.getSizeLevel() > 1) {
    options.assertions = false;
  }

  // Continue in processing the user-provided options, which may override the
  // default options given by the optimization level.

  if (!options.debug) {
    options.debug = args.hasArg(options::OPT_g_Flag);
  }

  options.assertions = args.hasFlag(
      options::OPT_assertions,
      options::OPT_no_assertions,
      options.assertions);

  options.inlining = args.hasFlag(
      options::OPT_function_inlining,
      options::OPT_no_function_inlining,
      options.inlining);

  options.outputArraysPromotion = args.hasFlag(
      options::OPT_output_arrays_promotion,
      options::OPT_no_output_arrays_promotion,
      options.outputArraysPromotion);

  options.readOnlyVariablesPropagation = args.hasFlag(
      options::OPT_read_only_variables_propagation,
      options::OPT_no_read_only_variables_propagation,
      options.readOnlyVariablesPropagation);

  options.variablesToParametersPromotion = args.hasFlag(
      options::OPT_variables_to_parameters_promotion,
      options::OPT_no_variables_to_parameters_promotion,
      options.variablesToParametersPromotion);

  options.cse = args.hasFlag(
      options::OPT_cse,
      options::OPT_no_cse,
      options.cse);

  options.equationsRuntimeScheduling = args.hasFlag(
      options::OPT_equations_runtime_scheduling,
      options::OPT_no_equations_runtime_scheduling,
      options.equationsRuntimeScheduling);

  options.omp = args.hasFlag(
      options::OPT_omp,
      options::OPT_no_omp,
      options.omp);

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_bit_width)) {
    llvm::StringRef value = arg->getValue();
    llvm::APSInt numericValue(value);
    options.bitWidth = numericValue.getSExtValue();
  }

  // Cross-compilation options

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_target)) {
    llvm::StringRef value = arg->getValue();
    options.target = value.str();
  } else {
    // Default: native compilation
    options.target = llvm::sys::getDefaultTargetTriple();
  }

  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_mcpu_EQ)) {
    llvm::StringRef value = arg->getValue();
    options.cpu = value.str();

    if (options.cpu == "native") {
      // Get the host CPU name
      options.cpu = llvm::sys::getHostCPUName().str();
    }
  } else {
    // Default: native compilation
    options.cpu = llvm::sys::getHostCPUName().str();
  }
}

static void parseSimulationArgs(
    SimulationOptions& options,
    llvm::opt::ArgList& args,
    clang::DiagnosticsEngine& diagnostics)
{
  if (const llvm::opt::Arg* arg = args.getLastArg(options::OPT_model)) {
    llvm::StringRef value = arg->getValue();
    options.modelName = value.str();
  }

  // Determine the solver to be used.
  for (const auto& arg : args.getAllArgValues(options::OPT_solver)) {
    if (arg == "euler-forward") {
      options.solver = "euler-forward";
    } else if (arg == "ida") {
      options.solver = "ida";
    } else {
      diagnostics.Report(diagnostics.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "Unknown solver '%0'")) << arg;
    }
  }

  // IDA: reduced system computation.
  options.IDAReducedSystem = args.hasFlag(
      options::OPT_ida_reduced_system,
      options::OPT_no_ida_reduced_system,
      options.IDAReducedSystem);

  // IDA: reduced derivatives.
  options.IDAReducedDerivatives = args.hasFlag(
      options::OPT_ida_reduced_derivatives,
      options::OPT_no_ida_reduced_derivatives,
      options.IDAReducedDerivatives);

  // IDA: AD seeds optimization.
  options.IDAJacobianOneSweep = args.hasFlag(
      options::OPT_ida_jacobian_one_sweep,
      options::OPT_no_ida_jacobian_one_sweep,
      options.IDAJacobianOneSweep);
}

namespace {
  template <class T> std::shared_ptr<T> makeSharedCopy(const T &X) {
    return std::make_shared<T>(X);
  }

  template <class T>
  llvm::IntrusiveRefCntPtr<T> makeIntrusiveRefCntCopy(const T &X) {
    return llvm::makeIntrusiveRefCnt<T>(X);
  }
}

namespace marco::frontend
{
  CompilerInvocationBase::CompilerInvocationBase()
      : languageOptions(llvm::makeIntrusiveRefCnt<LanguageOptions>()),
        diagnosticOptions(
            llvm::makeIntrusiveRefCnt<clang::DiagnosticOptions>()),
        fileSystemOptions(std::make_shared<clang::FileSystemOptions>()),
        frontendOptions(std::make_shared<FrontendOptions>()),
        codegenOptions(std::make_shared<CodegenOptions>()),
        simulationOptions(std::make_shared<SimulationOptions>())
  {
  }

  CompilerInvocationBase::CompilerInvocationBase(EmptyConstructor)
  {
  }

  CompilerInvocationBase::CompilerInvocationBase(
      CompilerInvocationBase&& other) = default;

  CompilerInvocationBase& CompilerInvocationBase::deepCopyAssign(
      const CompilerInvocationBase& other)
  {
    if (this != &other) {
      languageOptions = makeIntrusiveRefCntCopy(other.getLanguageOptions());

      diagnosticOptions =
          makeIntrusiveRefCntCopy(other.getDiagnosticOptions());

      fileSystemOptions = makeSharedCopy(other.getFileSystemOptions());
      frontendOptions = makeSharedCopy(other.getFrontendOptions());
      codegenOptions = makeSharedCopy(other.getCodeGenOptions());
      simulationOptions = makeSharedCopy(other.getSimulationOptions());
    }

    return *this;
  }

  CompilerInvocationBase& CompilerInvocationBase::shallowCopyAssign(
      const CompilerInvocationBase& other)
  {
    if (this != &other) {
      languageOptions = other.languageOptions;
      diagnosticOptions = other.diagnosticOptions;
      fileSystemOptions = other.fileSystemOptions;
      frontendOptions = other.frontendOptions;
      codegenOptions = other.codegenOptions;
      simulationOptions = other.simulationOptions;
    }

    return *this;
  }

  CompilerInvocationBase& CompilerInvocationBase::operator=(
      CompilerInvocationBase&& other) = default;

  CompilerInvocationBase::~CompilerInvocationBase() = default;

  LanguageOptions& CompilerInvocationBase::getLanguageOptions()
  {
    assert(languageOptions && "Compiler invocation has no language options");
    return *languageOptions;
  }

  const LanguageOptions& CompilerInvocationBase::getLanguageOptions() const
  {
    assert(languageOptions && "Compiler invocation has no language options");
    return *languageOptions;
  }

  clang::DiagnosticOptions& CompilerInvocationBase::getDiagnosticOptions()
  {
    assert(diagnosticOptions &&
           "Compiler invocation has no diagnostic options");

    return *diagnosticOptions;
  }

  const clang::DiagnosticOptions&
  CompilerInvocationBase::getDiagnosticOptions() const
  {
    assert(diagnosticOptions &&
           "Compiler invocation has no diagnostic options");

    return *diagnosticOptions;
  }

  clang::FileSystemOptions& CompilerInvocationBase::getFileSystemOptions()
  {
    assert(fileSystemOptions &&
           "Compiler invocation has no file system options");

    return *fileSystemOptions;
  }

  const clang::FileSystemOptions&
  CompilerInvocationBase::getFileSystemOptions() const
  {
    assert(fileSystemOptions &&
           "Compiler invocation has no file system options");

    return *fileSystemOptions;
  }

  FrontendOptions& CompilerInvocationBase::getFrontendOptions()
  {
    assert(frontendOptions && "Compiler invocation has no frontend options");
    return *frontendOptions;
  }

  const FrontendOptions& CompilerInvocationBase::getFrontendOptions() const
  {
    assert(frontendOptions && "Compiler invocation has no frontend options");
    return *frontendOptions;
  }

  CodegenOptions& CompilerInvocationBase::getCodeGenOptions()
  {
    assert(codegenOptions && "Compiler invocation has no codegen options");
    return *codegenOptions;
  }

  const CodegenOptions& CompilerInvocationBase::getCodeGenOptions() const
  {
    assert(codegenOptions && "Compiler invocation has no codegen options");
    return *codegenOptions;
  }

  SimulationOptions& CompilerInvocationBase::getSimulationOptions()
  {
    assert(simulationOptions &&
           "Compiler invocation has no simulation options");

    return *simulationOptions;
  }

  const SimulationOptions& CompilerInvocationBase::getSimulationOptions() const
  {
    assert(simulationOptions &&
           "Compiler invocation has no simulation options");

    return *simulationOptions;
  }

  bool CompilerInvocation::createFromArgs(
      CompilerInvocation& res,
      llvm::ArrayRef<const char*> commandLineArgs,
      clang::DiagnosticsEngine& diagnostics)
  {
    auto numOfErrors = diagnostics.getNumErrors();

    // Parse the arguments
    const llvm::opt::OptTable& opts = getDriverOptTable();
    unsigned missingArgIndex, missingArgCount;

    llvm::opt::InputArgList args = opts.ParseArgs(
        commandLineArgs, missingArgIndex, missingArgCount, llvm::opt::Visibility(clang::driver::options::MC1Option));

    // Check for missing argument error
    if (missingArgCount != 0) {
      diagnostics.Report(diagnostics.getCustomDiagID(
          clang::DiagnosticsEngine::Error,
          "Missing value for argument '%0'"))
          << args.getArgString(missingArgIndex);
    }

    // Issue errors on unknown arguments
    for (const auto* a: args.filtered(options::OPT_UNKNOWN)) {
      auto argString = a->getAsString(args);
      diagnostics.Report(diagnostics.getCustomDiagID(
          clang::DiagnosticsEngine::Warning,
          "Unknown argument '%0'")) << argString;
    }

    if (!parseDiagnosticArgs(res.getDiagnosticOptions(), args, &diagnostics)) {
      return false;
    }

    parseFrontendArgs(res.getFrontendOptions(), args, diagnostics);
    parseCodegenArgs(res.getCodeGenOptions(), args, diagnostics);
    parseSimulationArgs(res.getSimulationOptions(), args, diagnostics);

    return numOfErrors == diagnostics.getNumErrors();
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  createVFSFromCompilerInvocation(
      const CompilerInvocation& ci, clang::DiagnosticsEngine& diags)
  {
    return createVFSFromCompilerInvocation(
        ci, diags, llvm::vfs::getRealFileSystem());
  }

  llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
  createVFSFromCompilerInvocation(
      const CompilerInvocation& ci,
      clang::DiagnosticsEngine& diags,
      llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFS)
  {
    // TODO Check if it makes sense to use VFS overlay files with Modelica.
    return baseFS;
  }
}
