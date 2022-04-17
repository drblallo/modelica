#ifndef MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_MODELCONVERTER_H
#define MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_MODELCONVERTER_H

#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"
#include "marco/Codegen/Transforms/ModelSolving/ExternalSolvers/ExternalSolver.h"
#include "marco/Codegen/Transforms/ModelSolving/Scheduling.h"
#include "marco/Codegen/Transforms/ModelSolving/ModelSolving.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "llvm/ADT/StringRef.h"
#include <map>
#include <set>

namespace marco::codegen
{
  class ModelConverter
  {
    public:
      // Name for the functions of the simulation
      static constexpr llvm::StringLiteral mainFunctionName = "main";
      static constexpr llvm::StringLiteral initFunctionName = "init";
      static constexpr llvm::StringLiteral updateNonStateVariablesFunctionName = "updateNonStateVariables";
      static constexpr llvm::StringLiteral updateStateVariablesFunctionName = "updateStateVariables";
      static constexpr llvm::StringLiteral incrementTimeFunctionName = "incrementTime";
      static constexpr llvm::StringLiteral printHeaderFunctionName = "printHeader";
      static constexpr llvm::StringLiteral printFunctionName = "print";
      static constexpr llvm::StringLiteral deinitFunctionName = "deinit";
      static constexpr llvm::StringLiteral runFunctionName = "runSimulation";

    private:
      static constexpr size_t externalSolversPosition = 0;
      static constexpr size_t timeVariablePosition = 1;
      static constexpr size_t variablesOffset = 2;

      // The derivatives map keeps track of whether a variable is the derivative
      // of another one. Each variable is identified by its position within the
      // list of the "body" region arguments.

      using DerivativesPositionsMap = std::map<size_t, size_t>;

      struct ConversionInfo
      {
        std::set<std::unique_ptr<Equation>> explicitEquations;
        std::map<ScheduledEquation*, Equation*> explicitEquationsMap;
        std::set<ScheduledEquation*> implicitEquations;
        std::set<ScheduledEquation*> cyclicEquations;
      };

    public:
      ModelConverter(ModelSolvingOptions options, mlir::LLVMTypeConverter& typeConverter);

      /// Convert a scheduled model into the algorithmic functions that compose the simulation.
      /// The usage of such functions is delegated to the runtime library, which is statically
      /// linked with the code generated by the compiler. This decoupling allows to relieve the
      /// code generation phase from the generation of functions that are independent from the
      /// model being processed.
      mlir::LogicalResult convert(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const mlir::BlockAndValueMapping& derivatives) const;

    private:
      /// Get the MLIR type corresponding to void*.
      mlir::Type getVoidPtrType() const;

      mlir::LLVM::LLVMFuncOp lookupOrCreateHeapAllocFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      mlir::LLVM::LLVMFuncOp lookupOrCreateHeapFreeFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      /// Allocate some data on the heap.
      mlir::Value alloc(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Location loc,
          mlir::Type type) const;

      /// Create the main function, which is called when the executable of the simulation is run.
      /// In order to keep the code generation simpler, the real implementation of the function
      /// managing the simulation lives within the runtime library and the main just consists in
      /// a call to such function.
      mlir::LogicalResult createMainFunction(
          mlir::OpBuilder& builder, const Model<ScheduledEquationsBlock>& model) const;

      mlir::LLVM::LLVMStructType getRuntimeDataStructType(
          mlir::MLIRContext* context, const ExternalSolvers& externalSolvers, mlir::TypeRange varTypes) const;

      /// Load the data structure from the opaque pointer that is passed around the
      /// simulation functions.
      ///
      /// @param builder	    operation builder
      /// @param ptr 	        opaque pointer
      /// @param runtimeData  type of the runtime data structure
      /// @return data structure containing the variables
      mlir::Value loadDataFromOpaquePtr(
          mlir::OpBuilder& builder, mlir::Value ptr, mlir::LLVM::LLVMStructType runtimeData) const;

      /// Extract a value from the data structure shared between the various
      /// simulation main functions.
      ///
      /// @param builder 			  operation builder
      /// @param typeConverter  type converter
      /// @param structValue 	  data structure
      /// @param type 				  value type
      /// @param position 		  value position
      /// @return extracted value
      mlir::Value extractValue(
          mlir::OpBuilder& builder, mlir::Value structValue, mlir::Type type, unsigned int position) const;

      /// Bufferize the variables and convert the subsequent load/store operations to operate on the
      /// allocated memory buffer.
      mlir::Value convertMember(mlir::OpBuilder& builder, mlir::modelica::MemberCreateOp op) const;

      /// Create the initialization function that allocates the variables and
      /// stores them into an appropriate data structure to be passed to the other
      /// simulation functions.
      mlir::LogicalResult createInitFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          ExternalSolvers& externalSolvers,
          const mlir::BlockAndValueMapping& derivatives) const;

      /// Create a function to be called when the simulation has finished and the
      /// variables together with its data structure are not required anymore and
      /// thus can be deallocated.
      mlir::LogicalResult createDeinitFunction(
          mlir::OpBuilder& builder, mlir::modelica::ModelOp modelOp, ExternalSolvers& externalSolvers) const;

      mlir::FuncOp createEquationFunction(
          mlir::OpBuilder& builder,
          const ScheduledEquation& equation,
          llvm::StringRef equationFunctionName,
          mlir::FuncOp templateFunction,
          std::multimap<mlir::FuncOp, mlir::CallOp>& equationTemplateCalls,
          mlir::TypeRange varsTypes) const;

      mlir::LogicalResult createUpdateNonStateVariablesFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          const ConversionInfo& conversionInfo,
          ExternalSolvers& externalSolvers) const;

      /// Create the functions that calculates the values that the state variables will have
      /// in the next iteration.
      mlir::LogicalResult createUpdateStateVariablesFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp modelOp,
          const mlir::BlockAndValueMapping& derivatives,
          const DerivativesPositionsMap& derivativesPositionMap,
          ExternalSolvers& externalSolvers) const;

      mlir::LogicalResult createIncrementTimeFunction(
          mlir::OpBuilder& builder,
          const Model<ScheduledEquationsBlock>& model,
          ExternalSolvers& externalSolvers) const;

      void printSeparator(mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      void printNewline(mlir::OpBuilder& builder, mlir::ModuleOp module) const;

      mlir::Value getOrCreateGlobalString(
          mlir::Location loc,
          mlir::OpBuilder& builder,
          mlir::StringRef name,
          mlir::StringRef value,
          mlir::ModuleOp module) const;

      mlir::LLVM::LLVMFuncOp getOrInsertPrintNameFunction(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module) const;

      void printVariableName(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value name,
          mlir::Value value,
          VariableFilter::Filter filter,
          bool shouldPreprendSeparator = true) const;

      void printScalarVariableName(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value name,
          bool shouldPrependSeparator) const;

      void printArrayVariableName(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value name,
          mlir::Value value,
          VariableFilter::Filter filter,
          bool shouldPrependSeparator) const;

      mlir::LogicalResult createPrintHeaderFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp op,
          DerivativesPositionsMap& derivativesPositions,
          ExternalSolvers& externalSolvers) const;

      void printVariable(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value var,
          VariableFilter::Filter filter,
          bool shouldPrependSeparator = true) const;

      void printScalarVariable(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value var,
          bool shouldPrependSeparator = true) const;

      void printArrayVariable(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::Value var,
          VariableFilter::Filter filter,
          bool shouldPrependSeparator = true) const;

      void printElement(mlir::OpBuilder& builder, mlir::ModuleOp module, mlir::Value value) const;

      mlir::LogicalResult createPrintFunction(
          mlir::OpBuilder& builder,
          mlir::modelica::ModelOp op,
          DerivativesPositionsMap& derivativesPositions,
          ExternalSolvers& externalSolvers) const;

      mlir::LogicalResult createPrintFunctionBody(
          mlir::OpBuilder& builder,
          mlir::ModuleOp module,
          mlir::modelica::ModelOp op,
          mlir::TypeRange varTypes,
          DerivativesPositionsMap& derivativesPositions,
          ExternalSolvers& externalSolvers,
          llvm::StringRef functionName,
          std::function<mlir::LogicalResult(llvm::StringRef, mlir::Value, VariableFilter::Filter, mlir::ModuleOp, size_t)> elementCallback) const;

    private:
      ModelSolvingOptions options;
      mlir::LLVMTypeConverter* typeConverter;
  };
}

#endif // MARCO_CODEGEN_TRANSFORMS_MODELSOLVING_MODELCONVERTER_H
