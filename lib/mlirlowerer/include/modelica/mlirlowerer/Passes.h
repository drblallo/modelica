#pragma once

// Just a convenience header file to include the Modelica passes

#include "passes/BufferDeallocation.h"
#include "passes/BufferLoopHoisting.h"
#include "passes/ExplicitCastInsertion.h"
#include "passes/LowerToLLVM.h"
#include "passes/ModelicaConversion.h"
#include "passes/ResultBuffersToArgs.h"
#include "passes/SolveModel.h"
