#include "marco/Frontend/CompilerInvocation.h"
#include "mlir/IR/Diagnostics.h"
#include "clang/Basic/DiagnosticDriver.h"
#include "clang/Driver/Options.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/Arg.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/TargetParser/Host.h"

using namespace ::marco;
using namespace ::clang;
using namespace driver;
using namespace options;
using namespace llvm::opt;

// Code taken from clang.
// Unfortunately the code is private and the only wait to reuse the
// infrastructure is by copy-pasting.

//===----------------------------------------------------------------------===//
// Normalizers
//===----------------------------------------------------------------------===//

using ArgumentConsumer = CompilerInvocation::ArgumentConsumer;

#define SIMPLE_ENUM_VALUE_TABLE
#include "clang/Driver/Options.inc"
#undef SIMPLE_ENUM_VALUE_TABLE

[[maybe_unused]] static std::optional<bool>
normalizeSimpleFlag(OptSpecifier Opt, unsigned TableIndex, const ArgList &Args,
                    DiagnosticsEngine &Diags) {
  if (Args.hasArg(Opt))
    return true;
  return std::nullopt;
}

[[maybe_unused]] static std::optional<bool>
normalizeSimpleNegativeFlag(OptSpecifier Opt, unsigned, const ArgList &Args,
                            DiagnosticsEngine &) {
  if (Args.hasArg(Opt))
    return false;
  return std::nullopt;
}

/// The tblgen-erated code passes in a fifth parameter of an arbitrary type, but
/// denormalizeSimpleFlags never looks at it. Avoid bloating compile-time with
/// unnecessary template instantiations and just ignore it with a variadic
/// argument.
[[maybe_unused]] static void denormalizeSimpleFlag(ArgumentConsumer Consumer,
                                                   const Twine &Spelling,
                                                   Option::OptionClass,
                                                   unsigned, /*T*/...) {
  Consumer(Spelling);
}

template <typename T>
static constexpr bool is_uint64_t_convertible() {
  return !std::is_same_v<T, uint64_t> && llvm::is_integral_or_enum<T>::value;
}

template <typename T,
          std::enable_if_t<!is_uint64_t_convertible<T>(), bool> = false>
static auto makeFlagToValueNormalizer(T Value) {
  return [Value](OptSpecifier Opt, unsigned, const ArgList &Args,
                 DiagnosticsEngine &) -> std::optional<T> {
    if (Args.hasArg(Opt))
      return Value;
    return std::nullopt;
  };
}

template <typename T,
          std::enable_if_t<is_uint64_t_convertible<T>(), bool> = false>
static auto makeFlagToValueNormalizer(T Value) {
  return makeFlagToValueNormalizer(uint64_t(Value));
}

static auto makeBooleanOptionNormalizer(bool Value, bool OtherValue,
                                        OptSpecifier OtherOpt) {
  return [Value, OtherValue,
          OtherOpt](OptSpecifier Opt, unsigned, const ArgList &Args,
                    DiagnosticsEngine &) -> std::optional<bool> {
    if (const Arg *A = Args.getLastArg(Opt, OtherOpt)) {
      return A->getOption().matches(Opt) ? Value : OtherValue;
    }
    return std::nullopt;
  };
}

[[maybe_unused]] static auto makeBooleanOptionDenormalizer(bool Value) {
  return [Value](ArgumentConsumer Consumer, const Twine &Spelling,
                 Option::OptionClass, unsigned, bool KeyPath) {
    if (KeyPath == Value)
      Consumer(Spelling);
  };
}

static void denormalizeStringImpl(ArgumentConsumer Consumer,
                                  const Twine &Spelling,
                                  Option::OptionClass OptClass, unsigned,
                                  const Twine &Value) {
  switch (OptClass) {
  case Option::SeparateClass:
  case Option::JoinedOrSeparateClass:
  case Option::JoinedAndSeparateClass:
    Consumer(Spelling);
    Consumer(Value);
    break;
  case Option::JoinedClass:
  case Option::CommaJoinedClass:
    Consumer(Spelling + Value);
    break;
  default:
    llvm_unreachable("Cannot denormalize an option with option class "
                     "incompatible with string denormalization.");
  }
}

template <typename T>
static void denormalizeString(ArgumentConsumer Consumer, const Twine &Spelling,
                              Option::OptionClass OptClass, unsigned TableIndex,
                              T Value) {
  denormalizeStringImpl(Consumer, Spelling, OptClass, TableIndex, Twine(Value));
}

static std::optional<SimpleEnumValue>
findValueTableByName(const SimpleEnumValueTable &Table, StringRef Name) {
  for (int I = 0, E = Table.Size; I != E; ++I)
    if (Name == Table.Table[I].Name)
      return Table.Table[I];

  return std::nullopt;
}

static std::optional<SimpleEnumValue>
findValueTableByValue(const SimpleEnumValueTable &Table, unsigned Value) {
  for (int I = 0, E = Table.Size; I != E; ++I)
    if (Value == Table.Table[I].Value)
      return Table.Table[I];

  return std::nullopt;
}

static std::optional<unsigned> normalizeSimpleEnum(OptSpecifier Opt,
                                                   unsigned TableIndex,
                                                   const ArgList &Args,
                                                   DiagnosticsEngine &Diags) {
  assert(TableIndex < SimpleEnumValueTablesSize);
  const SimpleEnumValueTable &Table = SimpleEnumValueTables[TableIndex];

  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return std::nullopt;

  StringRef ArgValue = Arg->getValue();
  if (auto MaybeEnumVal = findValueTableByName(Table, ArgValue))
    return MaybeEnumVal->Value;

  Diags.Report(diag::err_drv_invalid_value)
      << Arg->getAsString(Args) << ArgValue;
  return std::nullopt;
}

static void denormalizeSimpleEnumImpl(ArgumentConsumer Consumer,
                                      const Twine &Spelling,
                                      Option::OptionClass OptClass,
                                      unsigned TableIndex, unsigned Value) {
  assert(TableIndex < SimpleEnumValueTablesSize);
  const SimpleEnumValueTable &Table = SimpleEnumValueTables[TableIndex];
  if (auto MaybeEnumVal = findValueTableByValue(Table, Value)) {
    denormalizeString(Consumer, Spelling, OptClass, TableIndex,
                      MaybeEnumVal->Name);
  } else {
    llvm_unreachable("The simple enum value was not correctly defined in "
                     "the tablegen option description");
  }
}

template <typename T>
static void denormalizeSimpleEnum(ArgumentConsumer Consumer,
                                  const Twine &Spelling,
                                  Option::OptionClass OptClass,
                                  unsigned TableIndex, T Value) {
  return denormalizeSimpleEnumImpl(Consumer, Spelling, OptClass, TableIndex,
                                   static_cast<unsigned>(Value));
}

static std::optional<std::string> normalizeString(OptSpecifier Opt,
                                                  int TableIndex,
                                                  const ArgList &Args,
                                                  DiagnosticsEngine &Diags) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return std::nullopt;
  return std::string(Arg->getValue());
}

template <typename IntTy>
static std::optional<IntTy> normalizeStringIntegral(OptSpecifier Opt, int,
                                                    const ArgList &Args,
                                                    DiagnosticsEngine &Diags) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return std::nullopt;
  IntTy Res;
  if (StringRef(Arg->getValue()).getAsInteger(0, Res)) {
    Diags.Report(diag::err_drv_invalid_int_value)
        << Arg->getAsString(Args) << Arg->getValue();
    return std::nullopt;
  }
  return Res;
}

static std::optional<std::vector<std::string>>
normalizeStringVector(OptSpecifier Opt, int, const ArgList &Args,
                      DiagnosticsEngine &) {
  return Args.getAllArgValues(Opt);
}

[[maybe_unused]] static void
denormalizeStringVector(ArgumentConsumer Consumer, const Twine &Spelling,
                        Option::OptionClass OptClass, unsigned TableIndex,
                        const std::vector<std::string> &Values) {
  switch (OptClass) {
  case Option::CommaJoinedClass: {
    std::string CommaJoinedValue;
    if (!Values.empty()) {
      CommaJoinedValue.append(Values.front());
      for (const std::string &Value : llvm::drop_begin(Values, 1)) {
        CommaJoinedValue.append(",");
        CommaJoinedValue.append(Value);
      }
    }
    denormalizeString(Consumer, Spelling, Option::OptionClass::JoinedClass,
                      TableIndex, CommaJoinedValue);
    break;
  }
  case Option::JoinedClass:
  case Option::SeparateClass:
  case Option::JoinedOrSeparateClass:
    for (const std::string &Value : Values)
      denormalizeString(Consumer, Spelling, OptClass, TableIndex, Value);
    break;
  default:
    llvm_unreachable("Cannot denormalize an option with option class "
                     "incompatible with string vector denormalization.");
  }
}

static std::optional<std::string> normalizeTriple(OptSpecifier Opt,
                                                  int TableIndex,
                                                  const ArgList &Args,
                                                  DiagnosticsEngine &Diags) {
  auto *Arg = Args.getLastArg(Opt);
  if (!Arg)
    return std::nullopt;
  return llvm::Triple::normalize(Arg->getValue());
}

template <typename T, typename U>
static T mergeForwardValue(T KeyPath, U Value) {
  return static_cast<T>(Value);
}

template <typename T, typename U>
static T mergeMaskValue(T KeyPath, U Value) {
  return KeyPath | Value;
}

template <typename T>
static T extractForwardValue(T KeyPath) {
  return KeyPath;
}

template <typename T, typename U, U Value>
static T extractMaskValue(T KeyPath) {
  return ((KeyPath & Value) == Value) ? static_cast<T>(Value) : T();
}

#define PARSE_OPTION_WITH_MARSHALLING(                                         \
    ARGS, DIAGS, PREFIX_TYPE, SPELLING, ID, KIND, GROUP, ALIAS, ALIASARGS,     \
    FLAGS, VISIBILITY, PARAM, HELPTEXT, METAVAR, VALUES, SHOULD_PARSE,         \
    ALWAYS_EMIT, KEYPATH, DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE,         \
    NORMALIZER, DENORMALIZER, MERGER, EXTRACTOR, TABLE_INDEX)                  \
  if ((VISIBILITY) & options::MC1Option) {                                     \
    KEYPATH = MERGER(KEYPATH, DEFAULT_VALUE);                                  \
    if (IMPLIED_CHECK)                                                         \
      KEYPATH = MERGER(KEYPATH, IMPLIED_VALUE);                                \
    if (SHOULD_PARSE)                                                          \
      if (auto MaybeValue = NORMALIZER(OPT_##ID, TABLE_INDEX, ARGS, DIAGS))    \
        KEYPATH =                                                              \
            MERGER(KEYPATH, static_cast<decltype(KEYPATH)>(*MaybeValue));      \
  }

// Capture the extracted value as a lambda argument to avoid potential issues
// with lifetime extension of the reference.
#define GENERATE_OPTION_WITH_MARSHALLING(                                      \
    CONSUMER, PREFIX_TYPE, SPELLING, ID, KIND, GROUP, ALIAS, ALIASARGS, FLAGS, \
    VISIBILITY, PARAM, HELPTEXT, METAVAR, VALUES, SHOULD_PARSE, ALWAYS_EMIT,   \
    KEYPATH, DEFAULT_VALUE, IMPLIED_CHECK, IMPLIED_VALUE, NORMALIZER,          \
    DENORMALIZER, MERGER, EXTRACTOR, TABLE_INDEX)                              \
  if ((VISIBILITY) & options::MC1Option) {                                     \
    [&](const auto &Extracted) {                                               \
      if (ALWAYS_EMIT ||                                                       \
          (Extracted !=                                                        \
           static_cast<decltype(KEYPATH)>((IMPLIED_CHECK) ? (IMPLIED_VALUE)    \
                                                          : (DEFAULT_VALUE)))) \
        DENORMALIZER(CONSUMER, SPELLING, Option::KIND##Class, TABLE_INDEX,     \
                     Extracted);                                               \
    }(EXTRACTOR(KEYPATH));                                                     \
  }

//===---------------------------------------------------------------------===//
// CompilerInvocation
//===---------------------------------------------------------------------===//

/// Tweak the frontend configuration based on the frontend action
static void
setUpFrontendBasedOnAction(marco::frontend::FrontendOptions &options) {
  assert(options.programAction != marco::frontend::InvalidAction &&
         "Frontend action not set!");
}

static bool parseDiagnosticArgs(clang::DiagnosticOptions &diagnosticOptions,
                                llvm::opt::ArgList &args,
                                clang::DiagnosticsEngine *diags = nullptr,
                                bool defaultDiagColor = true) {
  return clang::ParseDiagnosticArgs(diagnosticOptions, args, diags,
                                    defaultDiagColor);
}

static bool parseTargetArgs(clang::TargetOptions &targetOptions,
                            llvm::opt::ArgList &args,
                            clang::DiagnosticsEngine &diags) {
  using namespace options;
  using namespace llvm::opt;

  unsigned numErrorsBefore = diags.getNumErrors();
  clang::TargetOptions *TargetOpts = &targetOptions;

#define TARGET_OPTION_WITH_MARSHALLING(...)                                    \
  PARSE_OPTION_WITH_MARSHALLING(args, diags, __VA_ARGS__)
#include "clang/Driver/Options.inc"
#undef TARGET_OPTION_WITH_MARSHALLING

  if (llvm::opt::Arg *A = args.getLastArg(options::OPT_target_sdk_version_EQ)) {
    llvm::VersionTuple Version;
    if (Version.tryParse(A->getValue()))
      diags.Report(clang::diag::err_drv_invalid_value)
          << A->getAsString(args) << A->getValue();
    else
      targetOptions.SDKVersion = Version;
  }

  if (Arg *A =
          args.getLastArg(options::OPT_darwin_target_variant_sdk_version_EQ)) {
    llvm::VersionTuple Version;
    if (Version.tryParse(A->getValue()))
      diags.Report(clang::diag::err_drv_invalid_value)
          << A->getAsString(args) << A->getValue();
    else
      targetOptions.DarwinTargetVariantSDKVersion = Version;
  }

  return diags.getNumErrors() == numErrorsBefore;
}

static void parseFrontendArgs(marco::frontend::FrontendOptions &options,
                              llvm::opt::ArgList &args,
                              clang::DiagnosticsEngine &diagnostics) {
  using namespace ::marco::frontend;

  // Default action
  options.programAction = EmitObject;

  if (args.hasArg(options::OPT_emit_ast)) {
    // -emit-ast is not part of the Action group in clang's options.
    options.programAction = EmitAST;
  } else if (const llvm::opt::Arg *a =
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

  options.multithreading =
      args.hasFlag(options::OPT_multithreading, options::OPT_no_multithreading,
                   options.multithreading);

  if (const llvm::opt::Arg *arg =
          args.getLastArg(options::OPT_print_ir_before_pass)) {
    llvm::StringRef value = arg->getValue();
    options.printIRBeforePass = value.str();
  }

  if (const llvm::opt::Arg *arg =
          args.getLastArg(options::OPT_print_ir_after_pass)) {
    llvm::StringRef value = arg->getValue();
    options.printIRAfterPass = value.str();
  }

  options.printIROnFailure = args.hasArg(options::OPT_print_ir_on_failure);

  // Collect the input files and save them in our instance of FrontendOptions
  std::vector<std::string> inputs = args.getAllArgValues(options::OPT_INPUT);
  options.inputs.clear();

  if (inputs.empty()) {
    // '-' is the default input if none is given
    inputs.push_back("-");
  }

  for (size_t i = 0, e = inputs.size(); i != e; ++i) {
    options.inputs.emplace_back(
        inputs[i], marco::io::InputKind::getFromFullFileName(inputs[i]));
  }

  setUpFrontendBasedOnAction(options);

  if (const llvm::opt::Arg *arg = args.getLastArg(options::OPT_omc_path)) {
    llvm::StringRef value = arg->getValue();
    options.omcPath = value.str();
  }

  for (const auto &omcArg : args.getAllArgValues(options::OPT_Xomc)) {
    options.omcCustomArgs.push_back(omcArg);
  }

  options.omcBypass = args.hasArg(options::OPT_omc_bypass);

  if (const llvm::opt::Arg *arg =
          args.getLastArg(options::OPT_variable_filter)) {
    options.variableFilter = arg->getValue();
  }
}

static void parseCodegenArgs(marco::frontend::CodegenOptions &options,
                             llvm::opt::ArgList &args,
                             clang::DiagnosticsEngine &diagnostics) {
  // Determine the optimization level
  for (const auto &arg : args.getAllArgValues(options::OPT_O)) {
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
      diagnostics.Report(
          diagnostics.getCustomDiagID(clang::DiagnosticsEngine::Warning,
                                      "Unknown optimization option '%0'"))
          << arg;
    }
  }

  // Set the default options based on the optimization level
  if (options.optLevel.getSpeedupLevel() > 0) {
    options.debug = false;
    options.assertions = false;
    options.singleValuedInductionElimination = true;
  }

  if (options.optLevel.getSpeedupLevel() > 1) {
    options.outputArraysPromotion = true;
    options.heapToStackPromotion = true;
    options.readOnlyVariablesPropagation = true;
    options.variablesPruning = true;
    options.variablesToParametersPromotion = true;
    options.inlining = true;
    options.cse = true;
    options.functionCallsCSE = true;
    options.loopFusion = true;
  }

  if (options.optLevel.getSpeedupLevel() > 2) {
    options.loopTiling = true;
    options.vectorization = true;
  }

  if (options.optLevel.getSizeLevel() > 0) {
    options.debug = false;
    options.cse = true;
    options.functionCallsCSE = true;
    options.singleValuedInductionElimination = true;
    options.variablesPruning = true;
    options.loopFusion = true;
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
      options::OPT_assertions, options::OPT_no_assertions, options.assertions);

  options.inlining =
      args.hasFlag(options::OPT_function_inlining,
                   options::OPT_no_function_inlining, options.inlining);

  options.outputArraysPromotion = args.hasFlag(
      options::OPT_output_arrays_promotion,
      options::OPT_no_output_arrays_promotion, options.outputArraysPromotion);

  options.readOnlyVariablesPropagation =
      args.hasFlag(options::OPT_read_only_variables_propagation,
                   options::OPT_no_read_only_variables_propagation,
                   options.readOnlyVariablesPropagation);

  options.variablesToParametersPromotion =
      args.hasFlag(options::OPT_variables_to_parameters_promotion,
                   options::OPT_no_variables_to_parameters_promotion,
                   options.variablesToParametersPromotion);

  if (const llvm::opt::Arg *arg =
          args.getLastArg(options::OPT_scc_substitution_max_iterations)) {
    llvm::StringRef value = arg->getValue();
    llvm::APSInt numericValue(value);
    options.sccSolvingBySubstitutionMaxIterations = numericValue.getSExtValue();
  }

  if (const llvm::opt::Arg *arg =
          args.getLastArg(options::OPT_scc_substitution_max_equations)) {
    llvm::StringRef value = arg->getValue();
    llvm::APSInt numericValue(value);

    options.sccSolvingBySubstitutionMaxEquationsInSCC =
        numericValue.getSExtValue();
  }

  options.cse =
      args.hasFlag(options::OPT_cse, options::OPT_no_cse, options.cse);

  options.functionCallsCSE = args.hasFlag(options::OPT_function_calls_cse,
                                          options::OPT_no_function_calls_cse,
                                          options.functionCallsCSE);

  options.equationsRuntimeScheduling =
      args.hasFlag(options::OPT_equations_runtime_scheduling,
                   options::OPT_no_equations_runtime_scheduling,
                   options.equationsRuntimeScheduling);

  options.omp =
      args.hasFlag(options::OPT_omp, options::OPT_no_omp, options.omp);

  if (const llvm::opt::Arg *arg = args.getLastArg(options::OPT_bit_width)) {
    llvm::StringRef value = arg->getValue();
    llvm::APSInt numericValue(value);
    options.bitWidth = numericValue.getSExtValue();
  }

  // Target-specific options.
  if (const llvm::opt::Arg *arg = args.getLastArg(options::OPT_target_cpu)) {
    llvm::StringRef value = arg->getValue();
    options.cpu = value.str();
  }

  options.features = args.getAllArgValues(options::OPT_target_feature);

  // Static schedule only optimizations.
  options.loopTiling &= !options.equationsRuntimeScheduling;
  options.vectorization &= !options.equationsRuntimeScheduling;
}

static void parseSimulationArgs(marco::frontend::SimulationOptions &options,
                                llvm::opt::ArgList &args,
                                clang::DiagnosticsEngine &diagnostics) {
  if (const llvm::opt::Arg *arg = args.getLastArg(options::OPT_model)) {
    llvm::StringRef value = arg->getValue();
    options.modelName = value.str();
  }

  // Determine the solver to be used.
  for (const auto &arg : args.getAllArgValues(options::OPT_solver)) {
    if (arg == "euler-forward") {
      options.solver = "euler-forward";
    } else if (arg == "ida") {
      options.solver = "ida";
    } else if (arg == "rk4") {
      options.solver = "rk4";
    } else if (llvm::StringRef(arg).starts_with("rk-")) {
      options.solver = arg;
    } else {
      diagnostics.Report(diagnostics.getCustomDiagID(
          clang::DiagnosticsEngine::Warning, "Unknown solver '%0'"))
          << arg;
    }
  }

  // IDA: reduced system computation.
  options.IDAReducedSystem = args.hasFlag(options::OPT_ida_reduced_system,
                                          options::OPT_no_ida_reduced_system,
                                          options.IDAReducedSystem);

  // IDA: reduced derivatives.
  options.IDAReducedDerivatives = args.hasFlag(
      options::OPT_ida_reduced_derivatives,
      options::OPT_no_ida_reduced_derivatives, options.IDAReducedDerivatives);

  // IDA: AD seeds optimization.
  options.IDAJacobianOneSweep = args.hasFlag(
      options::OPT_ida_jacobian_one_sweep,
      options::OPT_no_ida_jacobian_one_sweep, options.IDAJacobianOneSweep);
}

static bool fixupInvocation(marco::frontend::CompilerInvocation &invocation,
                            clang::DiagnosticsEngine &diags,
                            const llvm::opt::ArgList &args) {
  auto numErrorsBefore = diags.getNumErrors();

  auto &LangOpts = invocation.getLanguageOptions();
  auto &CodeGenOpts = invocation.getCodeGenOptions();
  auto &TargetOpts = invocation.getTargetOptions();

  CodeGenOpts.XRayInstrumentFunctions = LangOpts.XRayInstrument;
  CodeGenOpts.XRayAlwaysEmitCustomEvents = LangOpts.XRayAlwaysEmitCustomEvents;
  CodeGenOpts.XRayAlwaysEmitTypedEvents = LangOpts.XRayAlwaysEmitTypedEvents;

  LangOpts.SanitizeCoverage = CodeGenOpts.hasSanitizeCoverage();
  LangOpts.ForceEmitVTables = CodeGenOpts.ForceEmitVTables;
  LangOpts.SpeculativeLoadHardening = CodeGenOpts.SpeculativeLoadHardening;
  LangOpts.CurrentModule = LangOpts.ModuleName;

  CodeGenOpts.CodeModel = TargetOpts.CodeModel;
  CodeGenOpts.LargeDataThreshold = TargetOpts.LargeDataThreshold;

  return diags.getNumErrors() == numErrorsBefore;
}

namespace {
template <class T>
std::shared_ptr<T> makeSharedCopy(const T &X) {
  return std::make_shared<T>(X);
}

template <class T>
llvm::IntrusiveRefCntPtr<T> makeIntrusiveRefCntCopy(const T &X) {
  return llvm::makeIntrusiveRefCnt<T>(X);
}
} // namespace

namespace marco::frontend {
CompilerInvocationBase::CompilerInvocationBase()
    : languageOptions(llvm::makeIntrusiveRefCnt<LanguageOptions>()),
      targetOptions(std::make_shared<clang::TargetOptions>()),
      diagnosticOptions(llvm::makeIntrusiveRefCnt<clang::DiagnosticOptions>()),
      fileSystemOptions(std::make_shared<clang::FileSystemOptions>()),
      frontendOptions(std::make_shared<FrontendOptions>()),
      codegenOptions(std::make_shared<CodegenOptions>()),
      simulationOptions(std::make_shared<SimulationOptions>()) {}

CompilerInvocationBase::CompilerInvocationBase(EmptyConstructor) {}

CompilerInvocationBase::CompilerInvocationBase(CompilerInvocationBase &&other) =
    default;

CompilerInvocationBase &
CompilerInvocationBase::deepCopyAssign(const CompilerInvocationBase &other) {
  if (this != &other) {
    languageOptions = makeIntrusiveRefCntCopy(other.getLanguageOptions());
    targetOptions = makeSharedCopy(other.getTargetOptions());

    diagnosticOptions = makeIntrusiveRefCntCopy(other.getDiagnosticOptions());

    fileSystemOptions = makeSharedCopy(other.getFileSystemOptions());
    frontendOptions = makeSharedCopy(other.getFrontendOptions());
    codegenOptions = makeSharedCopy(other.getCodeGenOptions());
    simulationOptions = makeSharedCopy(other.getSimulationOptions());
  }

  return *this;
}

CompilerInvocationBase &
CompilerInvocationBase::shallowCopyAssign(const CompilerInvocationBase &other) {
  if (this != &other) {
    languageOptions = other.languageOptions;
    targetOptions = other.targetOptions;
    diagnosticOptions = other.diagnosticOptions;
    fileSystemOptions = other.fileSystemOptions;
    frontendOptions = other.frontendOptions;
    codegenOptions = other.codegenOptions;
    simulationOptions = other.simulationOptions;
  }

  return *this;
}

CompilerInvocationBase &
CompilerInvocationBase::operator=(CompilerInvocationBase &&other) = default;

CompilerInvocationBase::~CompilerInvocationBase() = default;

LanguageOptions &CompilerInvocationBase::getLanguageOptions() {
  assert(languageOptions && "Compiler invocation has no language options");
  return *languageOptions;
}

const LanguageOptions &CompilerInvocationBase::getLanguageOptions() const {
  assert(languageOptions && "Compiler invocation has no language options");
  return *languageOptions;
}

clang::TargetOptions &CompilerInvocationBase::getTargetOptions() {
  assert(targetOptions && "Compiler invocation has no target options");
  return *targetOptions;
}

const clang::TargetOptions &CompilerInvocationBase::getTargetOptions() const {
  assert(targetOptions && "Compiler invocation has no target options");
  return *targetOptions;
}

std::shared_ptr<clang::TargetOptions>
CompilerInvocationBase::getTargetOptionsPtr() {
  assert(targetOptions && "Compiler invocation has no target options");
  return targetOptions;
}

clang::DiagnosticOptions &CompilerInvocationBase::getDiagnosticOptions() {
  assert(diagnosticOptions && "Compiler invocation has no diagnostic options");

  return *diagnosticOptions;
}

const clang::DiagnosticOptions &
CompilerInvocationBase::getDiagnosticOptions() const {
  assert(diagnosticOptions && "Compiler invocation has no diagnostic options");

  return *diagnosticOptions;
}

clang::FileSystemOptions &CompilerInvocationBase::getFileSystemOptions() {
  assert(fileSystemOptions && "Compiler invocation has no file system options");

  return *fileSystemOptions;
}

const clang::FileSystemOptions &
CompilerInvocationBase::getFileSystemOptions() const {
  assert(fileSystemOptions && "Compiler invocation has no file system options");

  return *fileSystemOptions;
}

FrontendOptions &CompilerInvocationBase::getFrontendOptions() {
  assert(frontendOptions && "Compiler invocation has no frontend options");
  return *frontendOptions;
}

const FrontendOptions &CompilerInvocationBase::getFrontendOptions() const {
  assert(frontendOptions && "Compiler invocation has no frontend options");
  return *frontendOptions;
}

CodegenOptions &CompilerInvocationBase::getCodeGenOptions() {
  assert(codegenOptions && "Compiler invocation has no codegen options");
  return *codegenOptions;
}

const CodegenOptions &CompilerInvocationBase::getCodeGenOptions() const {
  assert(codegenOptions && "Compiler invocation has no codegen options");
  return *codegenOptions;
}

SimulationOptions &CompilerInvocationBase::getSimulationOptions() {
  assert(simulationOptions && "Compiler invocation has no simulation options");

  return *simulationOptions;
}

const SimulationOptions &CompilerInvocationBase::getSimulationOptions() const {
  assert(simulationOptions && "Compiler invocation has no simulation options");

  return *simulationOptions;
}

bool CompilerInvocation::createFromArgs(
    CompilerInvocation &res, llvm::ArrayRef<const char *> commandLineArgs,
    clang::DiagnosticsEngine &diagnostics) {
  auto numOfErrors = diagnostics.getNumErrors();

  // Parse the arguments
  const llvm::opt::OptTable &opts = getDriverOptTable();
  unsigned missingArgIndex, missingArgCount;

  llvm::opt::InputArgList args =
      opts.ParseArgs(commandLineArgs, missingArgIndex, missingArgCount,
                     llvm::opt::Visibility(clang::driver::options::MC1Option));

  // Check for missing argument error
  if (missingArgCount != 0) {
    diagnostics.Report(diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Error, "Missing value for argument '%0'"))
        << args.getArgString(missingArgIndex);
  }

  // Issue errors on unknown arguments
  for (const auto *a : args.filtered(options::OPT_UNKNOWN)) {
    auto argString = a->getAsString(args);
    diagnostics.Report(diagnostics.getCustomDiagID(
        clang::DiagnosticsEngine::Warning, "Unknown argument '%0'"))
        << argString;
  }

  if (!parseDiagnosticArgs(res.getDiagnosticOptions(), args, &diagnostics)) {
    return false;
  }

  if (!parseTargetArgs(res.getTargetOptions(), args, diagnostics)) {
    return false;
  }

  parseFrontendArgs(res.getFrontendOptions(), args, diagnostics);
  parseCodegenArgs(res.getCodeGenOptions(), args, diagnostics);
  parseSimulationArgs(res.getSimulationOptions(), args, diagnostics);

  fixupInvocation(res, diagnostics, args);

  return numOfErrors == diagnostics.getNumErrors();
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem>
createVFSFromCompilerInvocation(const CompilerInvocation &ci,
                                clang::DiagnosticsEngine &diags) {
  return createVFSFromCompilerInvocation(ci, diags,
                                         llvm::vfs::getRealFileSystem());
}

llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> createVFSFromCompilerInvocation(
    const CompilerInvocation &ci, clang::DiagnosticsEngine &diags,
    llvm::IntrusiveRefCntPtr<llvm::vfs::FileSystem> baseFS) {
  // TODO Check if it makes sense to use VFS overlay files with Modelica.
  return baseFS;
}
} // namespace marco::frontend
