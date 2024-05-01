#ifndef MARCO_CODEGEN_CONVERSION_PASSES_H
#define MARCO_CODEGEN_CONVERSION_PASSES_H

#include "marco/Codegen/Conversion/IDAToFunc/IDAToFunc.h"
#include "marco/Codegen/Conversion/IDAToLLVM/IDAToLLVM.h"
#include "marco/Codegen/Conversion/KINSOLToFunc/KINSOLToFunc.h"
#include "marco/Codegen/Conversion/KINSOLToLLVM/KINSOLToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaToArith/ModelicaToArith.h"
#include "marco/Codegen/Conversion/ModelicaToCF/ModelicaToCF.h"
#include "marco/Codegen/Conversion/ModelicaToFunc/ModelicaToFunc.h"
#include "marco/Codegen/Conversion/ModelicaToLLVM/ModelicaToLLVM.h"
#include "marco/Codegen/Conversion/ModelicaToMemRef/ModelicaToMemRef.h"
#include "marco/Codegen/Conversion/ModelicaToRuntime/ModelicaToRuntime.h"
#include "marco/Codegen/Conversion/ModelicaToVector/ModelicaToVector.h"
#include "marco/Codegen/Conversion/RuntimeToFunc/RuntimeToFunc.h"
#include "marco/Codegen/Conversion/RuntimeToLLVM/RuntimeToLLVM.h"
#include "marco/Codegen/Conversion/SUNDIALSToFunc/SUNDIALSToFunc.h"

namespace marco::codegen
{
  /// Generate the code for registering passes
  #define GEN_PASS_REGISTRATION
  #include "marco/Codegen/Conversion/Passes.h.inc"
}

#endif // MARCO_CODEGEN_CONVERSION_PASSES_H
