#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_SOLVER_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_SOLVER_H

#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/VariableFilter/VariableFilter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace marco::codegen
{
  /// The purpose of this class is to generate the algorithmic functions that
  /// compose the simulation. The usage of such functions is delegated to the
  /// runtime library, which is statically linked with the code generated by
  /// the compiler. This decoupling allows to relieve the code generation phase
  /// from the generation of functions that are independent from the model
  /// being processed.
  class ModelSolver
  {
    public:
      // Name for the functions of the simulation.
      static constexpr llvm::StringLiteral getModelNameFunctionName = "getModelName";
      static constexpr llvm::StringLiteral getNumOfVariablesFunctionName = "getNumOfVariables";
      static constexpr llvm::StringLiteral getVariableNameFunctionName = "getVariableName";
      static constexpr llvm::StringLiteral getVariableRankFunctionName = "getVariableRank";
      static constexpr llvm::StringLiteral getVariableNumOfPrintableRangesFunctionName = "getVariableNumOfPrintableRanges";
      static constexpr llvm::StringLiteral getVariablePrintableRangeBeginFunctionName = "getVariablePrintableRangeBegin";
      static constexpr llvm::StringLiteral getVariablePrintableRangeEndFunctionName = "getVariablePrintableRangeEnd";
      static constexpr llvm::StringLiteral getVariableValueFunctionName = "getVariableValue";
      static constexpr llvm::StringLiteral getDerivativeFunctionName = "getDerivative";
      static constexpr llvm::StringLiteral getTimeFunctionName = "getTime";
      static constexpr llvm::StringLiteral setTimeFunctionName = "setTime";
      static constexpr llvm::StringLiteral initFunctionName = "init";
      static constexpr llvm::StringLiteral deinitFunctionName = "deinit";
      static constexpr llvm::StringLiteral mainFunctionName = "main";
      static constexpr llvm::StringLiteral runFunctionName = "runSimulation";

    protected:
      static constexpr size_t solversDataPosition = 0;
      static constexpr size_t timeVariablePosition = 1;
      static constexpr size_t variablesOffset = 2;

    public:
      ModelSolver(
        mlir::LLVMTypeConverter& typeConverter,
        VariableFilter& variablesFilter);

      virtual ~ModelSolver();

      /// Create the function to be called to retrieve the name of the compiled
      /// model.
      mlir::LogicalResult createGetModelNameFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the function to be called to retrieve the number of variables
      /// of the compiled model.
      mlir::LogicalResult createGetNumOfVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the function to be called to retrieve the name of variables of
      /// the compiled model.
      mlir::LogicalResult createGetVariableNameFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the function to be called to retrieve the name of variables of
      /// the compiled model.
      mlir::LogicalResult createGetVariableRankFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the function to be called to retrieve the number of printable
      /// indices ranges for a given variable.
      mlir::LogicalResult createGetVariableNumOfPrintableRangesFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp,
          const DerivativesMap& derivativesMap) const;

      /// Create the function to be called to retrieve the begin index of a
      /// printable range for a given variable and dimension.
      mlir::LogicalResult createGetVariablePrintableRangeBeginFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp,
          const DerivativesMap& derivativesMap) const;

      /// Create the function to be called to retrieve the end index of a
      /// printable range for a given variable and dimension.
      mlir::LogicalResult createGetVariablePrintableRangeEndFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp,
          const DerivativesMap& derivativesMap) const;

      /// Create the function to be called to retrieve the value of a scalar
      /// variable.
      mlir::LogicalResult createGetVariableValueFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the function to be called to retrieve the index of the
      /// derivative of a variable.
      mlir::LogicalResult createGetDerivativeFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp,
          const DerivativesMap& derivativesMap) const;

      /// Create the function to be called to retrieve the current time of the
      /// simulation.
      mlir::LogicalResult createGetTimeFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the function to be called to set the current time of the
      /// simulation.
      mlir::LogicalResult createSetTimeFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the initialization function that allocates the variables and
      /// stores them into an appropriate data structure to be passed to the
      /// other simulation functions.
      mlir::LogicalResult createInitFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create a function to be called when the simulation has finished and
      /// the variables together with its data structure are not required
      /// anymore and thus can be deallocated.
      mlir::LogicalResult createDeinitFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Create the main function, which is called when the executable of the
      /// simulation is run. In order to keep the code generation simpler, the
      /// real implementation of the function managing the simulation lives
      /// within the runtime library and the main just consists in a call to
      /// such function.
      mlir::LogicalResult createMainFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp) const;

      /// Convert the initial scheduled model into the algorithmic functions
      /// used to determine the initial values of the simulation.
      virtual mlir::LogicalResult solveICModel(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) = 0;

      /// Convert the main scheduled model into the algorithmic functions that
      /// compose the simulation.
      virtual mlir::LogicalResult solveMainModel(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model) = 0;

    protected:
      /// Get the MLIR type corresponding to void.
      mlir::Type getVoidType() const;

      /// Get the MLIR type corresponding to void*.
      mlir::Type getVoidPtrType() const;

      /// Get the LLVM function with the given name, or declare it inside the
      /// module if not present.
      mlir::LLVM::LLVMFuncOp getOrDeclareExternalFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          llvm::StringRef name,
          mlir::LLVM::LLVMFunctionType type) const;

      /// Create the instructions to allocate some data with a given type.
      mlir::Value alloc(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type type) const;

      /// Create the instructions to deallocate some data.
      void dealloc(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Value ptr) const;

      /// Get the type of the runtime data structure.
      mlir::LLVM::LLVMStructType getRuntimeDataStructType(
          mlir::MLIRContext* context,
          mlir::modelica::ModelOp modelOp) const;

      /// Load the data structure from the opaque pointer that is passed around
      /// the simulation functions.
      mlir::Value loadDataFromOpaquePtr(
          mlir::OpBuilder& builder,
          mlir::Value ptr,
          mlir::modelica::ModelOp modelOp) const;

      /// Store the data structure within the memory addressed by the opaque
      /// pointer that is passed around the simulation functions.
      void setRuntimeData(
          mlir::OpBuilder& builder,
          mlir::Value opaquePtr,
          mlir::modelica::ModelOp modelOp,
          mlir::Value data) const;

      /// Extract a value from the data structure shared between the various
      /// simulation functions.
      mlir::Value extractValue(
          mlir::OpBuilder& builder,
          mlir::Value structValue,
          mlir::Type type,
          unsigned int position) const;

      /// Extract the solver data pointer from the data structure shared
      /// between the various simulation functions.
      mlir::Value extractSolverDataPtr(
          mlir::OpBuilder& builder,
          mlir::Value structValue,
          mlir::Type solverDataType) const;

      /// Extract a variable from the data structure shared between the various
      /// simulation functions.
      mlir::Value extractVariable(
          mlir::OpBuilder& builder,
          mlir::Value structValue,
          mlir::Type type,
          unsigned int varIndex) const;

      mlir::Value getOrCreateGlobalString(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp module,
          mlir::StringRef name,
          mlir::StringRef value) const;

    private:
      mlir::LogicalResult createGetVariablePrintableRangeBoundariesFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp,
          const DerivativesMap& derivativesMap,
          llvm::StringRef functionName,
          std::function<int64_t(const modeling::Range&)> boundaryGetterCallback) const;

      mlir::LogicalResult createGetPrintableIndexSetBoundariesFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp module,
          llvm::StringRef functionName,
          const modeling::IndexSet& indexSet,
          std::function<int64_t(const modeling::Range&)> boundaryGetterCallback,
          std::map<unsigned int, std::map<modeling::MultidimensionalRange, mlir::func::FuncOp>>& rangeBoundaryFuncOps,
          llvm::StringRef baseRangeFunctionName,
          size_t& rangeFunctionsCounter) const;

      mlir::func::FuncOp createGetPrintableMultidimensionalRangeBoundariesFunction(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp module,
          llvm::StringRef functionName,
          const modeling::MultidimensionalRange& ranges,
          std::function<int64_t(const modeling::Range&)> boundaryGetterCallback) const;

      mlir::func::FuncOp createScalarVariableGetter(
          mlir::OpBuilder& builder,
          mlir::Location loc,
          mlir::ModuleOp module,
          llvm::StringRef functionName,
          mlir::modelica::ArrayType arrayType) const;

    protected:
      mlir::LLVMTypeConverter* typeConverter;

    private:
      VariableFilter* variablesFilter;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_SOLVERS_SOLVER_H
