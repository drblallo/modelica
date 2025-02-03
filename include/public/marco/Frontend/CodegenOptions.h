#ifndef MARCO_FRONTEND_CODEGENOPTIONS_H
#define MARCO_FRONTEND_CODEGENOPTIONS_H

#include "clang/Basic/CodeGenOptions.h"
#include "llvm/Passes/OptimizationLevel.h"
#include <string>
#include <vector>

namespace marco::frontend {
/// Code generation options.
/// The default values are for compiling without optimizations.
/// The class extends the language options for C / C++ to enable the
/// integration with clang's diagnostics infrastructure.
struct CodegenOptions : public clang::CodeGenOptions {
  llvm::OptimizationLevel optLevel = llvm::OptimizationLevel::O0;

  bool debug = true;
  bool assertions = true;
  bool inlining = false;
  bool outputArraysPromotion = false;
  bool heapToStackPromotion = false;
  bool readOnlyVariablesPropagation = false;
  bool variablesPruning = false;
  bool variablesToParametersPromotion = false;
  int64_t sccSolvingBySubstitutionMaxIterations = 100;
  int64_t sccSolvingBySubstitutionMaxEquationsInSCC = 5;
  bool cse = false;
  bool functionCallsCSE = false;
  bool equationsRuntimeScheduling = false;
  bool omp = false;
  bool singleValuedInductionElimination = false;
  bool loopFusion = false;
  bool loopTiling = false;

  // TODO(Tor): Default this to false
  // TODO(Tor): Add this option to the clang driver in MARCO's customized clang.
  bool resultRematerialization = true;

  unsigned int bitWidth = 64;

  std::string cpu = "generic";
  std::vector<std::string> features;
};
} // namespace marco::frontend

#endif // MARCO_FRONTEND_CODEGENOPTIONS_H
