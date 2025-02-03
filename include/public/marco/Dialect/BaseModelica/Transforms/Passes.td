#ifndef MARCO_DIALECT_SIMULATION_TRANSFORMS_PASSES_TD
#define MARCO_DIALECT_SIMULATION_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

//===---------------------------------------------------------------------===//
// Normalization
//===---------------------------------------------------------------------===//

def AutomaticDifferentiationPass : Pass<"auto-diff", "mlir::ModuleOp">
{
    let summary = "Automatic differentiation of functions";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createAutomaticDifferentiationPass()";
}

def BindingEquationConversionPass
    : Pass<"convert-binding-equations", "mlir::ModuleOp">
{
    let summary = "Convert binding equations.";

    let description = [{
        Convert the binding equations into equations or start operations.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createBindingEquationConversionPass()";
}

def DerivativeChainRulePass
    : Pass<"der-chain-rule", "mlir::ModuleOp">
{
    let summary = "Apply the chain rule to derivative operations";

    let description = [{
        Apply the chain rule to the derivative operations.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createDerivativeChainRulePass()";
}

def EquationAccessSplitPass
    : Pass<"split-overlapping-accesses", "mlir::ModuleOp">
{
    let summary = "Split the overlapping read / write accesses.";

    let description = [{
        Split the indices of the equations so that no access can be a read and
        a write at the same time.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor = "mlir::bmodelica::createEquationAccessSplitPass()";
}

def EquationInductionsExplicitationPass
    : Pass<"explicitate-equation-inductions", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor =
        "mlir::bmodelica::createEquationInductionsExplicitationPass()";
}

def EquationSidesSplitPass : Pass<"split-equation-sides", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor = "mlir::bmodelica::createEquationSidesSplitPass()";
}

def EquationTemplatesCreationPass
    : Pass<"create-equation-templates", "mlir::ModuleOp">
{
    let summary = "Create the equation templates and their instances.";

    let description = [{
        Create the equation templates and their instances from initial and
        non-initial equations.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor = "mlir::bmodelica::createEquationTemplatesCreationPass()";
}

def ExplicitCastInsertionPass
    : Pass<"explicit-cast-insertion", "mlir::ModuleOp">
{
    let summary = "Explicit cast insertion";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::scf::SCFDialect"
    ];

    let constructor = "mlir::bmodelica::createExplicitCastInsertionPass()";
}

def ExplicitInitialEquationsInsertionPass
    : Pass<"insert-explicit-initial-equations", "mlir::ModuleOp">
{
    let summary = "Explicitate all the initial equations.";

    let description = [{
        Create the initial equations from the fixed start values and from the
        already existing non-initial equations.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor =
        "mlir::bmodelica::createExplicitInitialEquationsInsertionPass()";
}

def ExplicitStartValueInsertionPass
    : Pass<"insert-missing-start-values", "mlir::ModuleOp">
{
    let summary = "Add the missing start values.";

    let description = [{
        Create the start values for variables that do not have one.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor =
        "mlir::bmodelica::createExplicitStartValueInsertionPass()";
}

def FunctionDefaultValuesConversionPass
    : Pass<"function-default-values-conversion", "mlir::ModuleOp">
{
    let summary = "Fill the function calls and the output variables with the default values.";

    let description = [{
        Explicitate all the arguments of function calls and convert the default
        output values to assdignments to be performed as soon as the function
        starts its execution.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor =
        "mlir::bmodelica::createFunctionDefaultValuesConversionPass()";
}

def FunctionInliningPass : Pass<"inline-functions", "mlir::ModuleOp">
{
    let summary = "Inline Modelica functions.";

    let description = [{
        Inline modelica functions.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor = "mlir::bmodelica::createFunctionInliningPass()";
}

def FunctionScalarizationPass : Pass<"scalarize", "mlir::ModuleOp">
{
    let summary = "Scalarize";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::cf::ControlFlowDialect",
        "mlir::linalg::LinalgDialect",
        "mlir::tensor::TensorDialect"
    ];

    let options = [
        Option<"assertions", "assertions", "bool", "true",
               "Enable the runtime assertions">,
    ];

  let constructor = "mlir::bmodelica::createFunctionScalarizationPass()";
}

def FunctionUnwrapPass : Pass<"function-unwrap", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createFunctionUnwrapPass()";
}

def ModelAlgorithmConversionPass
    : Pass<"convert-model-algorithms", "mlir::ModuleOp">
{
    let summary = "Convert algorithms inside models.";

    let description = [{
        Convert the algorithms inside models.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor = "mlir::bmodelica::createModelAlgorithmConversionPass()";
}

def RangeBoundariesInferencePass : Pass<"infer-range-boundaries", "mlir::ModuleOp">
{
    let summary = "Infer the boundaries of unbounded ranges.";

    let description = [{
        Infer the boundaries of unbounded ranges.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::tensor::TensorDialect"
    ];

    let constructor = "mlir::bmodelica::createRangeBoundariesInferencePass()";
}

def RecordInliningPass : Pass<"inline-records", "mlir::ModuleOp">
{
    let summary = "Instantiate a model.";

    let description = [{
        Instantiate a model by removing all the object-oriented features.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createRecordInliningPass()";
}

def ViewAccessFoldingPass : Pass<"fold-view-accesses", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor = "mlir::bmodelica::createViewAccessFoldingPass()";
}

//===---------------------------------------------------------------------===//
// Optimizations
//===---------------------------------------------------------------------===//

def EquationFunctionLoopHoistingPass
    : Pass<"equation-function-loop-hoist", "EquationFunctionOp">
{
    let summary = "Hoist operation out of loops inside equation functions";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createEquationFunctionLoopHoistingPass()";
}

def CallCSEPass
    : Pass<"call-cse", "mlir::ModuleOp">
{
    let summary = "Move equal function calls to dedicated equation.";

    let description = [{
        Move equal function calls to dedicated equation.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let statistics = [
        Statistic<"newCSEVariables", "new-cse-variables", "How many CSE variables have been created">,
        Statistic<"replacedCalls", "replaced-calls", "How many calls were replaced by a CSE variable usage">
    ];

    let constructor = "mlir::bmodelica::createCallCSEPass()";
}

def ReadOnlyVariablesPropagationPass
    : Pass<"propagate-read-only-variables", "mlir::ModuleOp">
{
    let summary = "Propagate parameters and constants.";

    let description = [{
        Propagate the expression used to compute read-only variables (that is,
        parameters and constants).
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let options = [
        Option<"modelName", "model-name", "std::string", "",
               "Set the model onto which perform the transformation">,
        Option<"ignoredVariables", "ignored-variables", "std::string", "",
               "List of comma-separated variables that must not be propagated">
    ];

    let constructor = "mlir::bmodelica::createReadOnlyVariablesPropagationPass()";
}

def ScheduleParallelizationPass
    : Pass<"schedule-parallelization", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let options = [
        Option<"maxParallelBlocks", "max-parallel-blocks", "int64_t", "-1",
               "Maximum number of parallel blocks">
    ];

    let constructor =
        "mlir::bmodelica::createScheduleParallelizationPass()";
}

def SingleValuedInductionEliminationPass
    : Pass<"single-valued-induction-elimination", "mlir::ModuleOp">
{
    let summary = "Remove the induction variables that obtain a single value.";

    let description = [{
        The pass visits all the equation instances and for each of them erases
        the induction variables iterating only on one single value.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor =
        "mlir::bmodelica::createSingleValuedInductionEliminationPass()";
}

def VariablesPromotionPass
    : Pass<"promote-variables-to-parameters", "mlir::ModuleOp">
{
    let summary = "Promote to variables to parameters.";

    let description = [{
        Promote to parameters the variables that depend only on other parameters
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createVariablesPromotionPass()";
}

def VariablesPruningPass : Pass<"variables-pruning", "mlir::ModuleOp">
{
    let summary = "Remove the unneeded variables.";

    let description = [{
        Remove the variables and the associated matched equations that are not
        needed to compute the output variables.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let options = [
        Option<"outputVariables",
               "output-variables", "std::string", "",
               "Comma separated list of output variables">
    ];

    let constructor = "mlir::bmodelica::createVariablesPruningPass()";
}

def ResultRematerializationPass : Pass<"result-rematerialization", "mlir::ModuleOp">
{
    let summary  = "Recompute results when their operands are likely to reside in cache.";

    let description = [{
      Recompute results when their operands are likely to reside in cache.

      TBD: This is done by building a graph to balance proximity and equation node degrees.
      TODO(Tor): Improve the description
    }];

    let dependentDialects = [
      "mlir::bmodelica::BaseModelicaDialect"
    ];


    let constructor = "mlir::bmodelica::createResultRematerializationPass()";
}

//===---------------------------------------------------------------------===//
// Model solving
//===---------------------------------------------------------------------===//

def DerivativesMaterializationPass
    : Pass<"derivatives-materialization", "mlir::ModuleOp">
{
    let summary = "Create the variables for the derivatives.";

    let description = [{
        Create the derivative variable and remove the derivative operations.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor =
        "mlir::bmodelica::createDerivativesMaterializationPass()";
}

def MatchingPass : Pass<"match", "mlir::ModuleOp">
{
    let summary = "Perform the matching process";

    let description = [{
        Perform the matching process.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let options = [
        Option<"enableSimplificationAlgorithm",
               "enable-simplification-algorithm", "bool", "true",
               "Enable the usage of the simplification algorithm">
    ];

    let constructor = "mlir::bmodelica::createMatchingPass()";
}

def SCCAbsenceVerificationPass : Pass<"scc-absence-verification", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createSCCAbsenceVerificationPass()";
}

def SCCDetectionPass : Pass<"detect-scc", "mlir::ModuleOp">
{
    let summary = "Compute the SCCs of a model";

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createSCCDetectionPass()";
}

def SchedulingPass : Pass<"schedule", "mlir::ModuleOp">
{
    let summary = "Perform the scheduling process";

    let description = [{
        Perform the scheduling process.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createSchedulingPass()";
}

def InitialConditionsSolvingPass
    : Pass<"solve-initial-conditions", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createInitialConditionsSolvingPass()";
}

//===---------------------------------------------------------------------===//
// Numerical integration algorithms
//===---------------------------------------------------------------------===//

def EulerForwardPass : Pass<"euler-forward", "mlir::ModuleOp">
{
    let summary = "Apply the explicit Euler integration method";

    let description = [{
        Apply the explicit Euler integration method.
    }];

    let dependentDialects = [
        "mlir::affine::AffineDialect",
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::runtime::RuntimeDialect"
    ];

    let options = [
        Option<"rangedStateUpdateFunctions", "ranged-state-update-functions", "bool",
               "true", "Create ranged functions for state updates">
    ];

    let constructor = "mlir::bmodelica::createEulerForwardPass()";
}

def RungeKuttaPass : Pass<"runge-kutta", "mlir::ModuleOp">
{
    let summary = "Apply the implicit Euler integration method";

    let description = [{
        Apply the implicit Euler integration method.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::runtime::RuntimeDialect"
    ];

    let options = [
        Option<"variant", "variant", "std::string", "", "Runge-Kutta variant">
    ];

    let constructor = "mlir::bmodelica::createRungeKuttaPass()";
}

def IDAPass : Pass<"ida", "mlir::ModuleOp">
{
    let summary = "Solve the model with IDA";

    let description = [{
        Apply the BDF integration algorithm by using the IDA external solver.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::ida::IDADialect",
        "mlir::runtime::RuntimeDialect"
    ];

    let options = [
        Option<"reducedSystem", "reduced-system", "bool", "true",
               "Compute a reduced system for IDA">,
        Option<"reducedDerivatives", "reduced-derivatives", "bool", "true",
               "Determine the derivatives that are always equal to zero">,
        Option<"jacobianOneSweep", "jacobian-one-sweep", "bool", "true",
               "Enable one-sweep computation of derivatives with respect to state and derivative variables">,
        Option<"debugInformation", "debug-information", "bool", "false",
               "Insert the information for runtime debugging">
    ];

    let constructor = "mlir::bmodelica::createIDAPass()";
}

//===---------------------------------------------------------------------===//
// Lowering
//===---------------------------------------------------------------------===//

def EquationExplicitationPass
    : Pass<"equation-explicitation", "mlir::ModuleOp">
{
    let summary = "Convert the explicitable equations into functions";

    let description = [{
        Convert the explicitable equations into functions.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::arith::ArithDialect",
        "mlir::scf::SCFDialect"
    ];

    let constructor = "mlir::bmodelica::createEquationExplicitationPass()";
}

def SCCSolvingBySubstitutionPass
    : Pass<"scc-solving-substitution", "mlir::ModuleOp">
{
    let summary = "Solve the cycles of matched model by substitution";

    let description = [{
        Solve the cycles of matched model by substitution.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let options = [
        Option<"maxIterations", "max-scc-iterations", "int64_t", "100",
               "Set the maximum number of iterations">,
        Option<"maxEquationsInSCC", "max-equations-in-scc", "int64_t", "5",
               "Set the number of equations above which an SCC is ignored">,
    ];

    let constructor = "mlir::bmodelica::createSCCSolvingBySubstitutionPass()";
}

def SCCSolvingWithKINSOLPass : Pass<"scc-solving-kinsol", "mlir::ModuleOp">
{
    let summary = "Solve the cycles of matched model using KINSOL";

    let description = [{
        Solve the cycles of matched model using KINSOL.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::affine::AffineDialect",
        "mlir::arith::ArithDialect",
        "mlir::kinsol::KINSOLDialect",
        "mlir::runtime::RuntimeDialect"
    ];

    let options = [
        Option<"reducedDerivatives", "reduced-derivatives", "bool", "true",
               "Determine the derivatives that are always equal to zero">,
        Option<"jacobianOneSweep", "jacobian-one-sweep", "bool", "true",
               "Enable one-sweep computation of derivatives with respect to state and derivative variables">,
        Option<"debugInformation", "debug-information", "bool", "false",
               "Insert the information for runtime debugging">
    ];

    let constructor = "mlir::bmodelica::createSCCSolvingWithKINSOLPass()";
}

def SchedulersInstantiationPass
    : Pass<"schedulers-instantiation", "mlir::ModuleOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
        "mlir::runtime::RuntimeDialect"
    ];

    let constructor = "mlir::bmodelica::createSchedulersInstantiationPass()";
}

//===---------------------------------------------------------------------===//
// Debug passes
//===---------------------------------------------------------------------===//

def PrintModelInfoPass : Pass<"print-model-info", "mlir::ModuleOp">
{
    let summary = "Print model information";

    let description = [{
        Print model information on stderr.
    }];

    let constructor = "mlir::bmodelica::createPrintModelInfoPass()";
}

//===---------------------------------------------------------------------===//
// Test passes
//===---------------------------------------------------------------------===//

def AccessReplacementTestPass
    : Pass<"test-access-replacement", "ModelOp">
{
    let summary = "Test the access replacement.";

    let description = [{
        Test the replacement of accesses within equations.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect",
    ];

    let constructor = "mlir::bmodelica::createAccessReplacementTestPass()";
}

def DivOpDistributionPass : Pass<"distribute-div", "mlir::ModuleOp">
{
    let summary = "Distribute the division operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createDivOpDistributionPass()";
}

def EquationExplicitationTestPass
    : Pass<"test-equation-explicitation", "ModelOp">
{
    let summary = "Test the explicitation of equations";

    let description = [{
        The pass is used to test the explicitation process of equations.
        It is not designed to be used within a compilation pipeline.
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createEquationExplicitationTestPass()";
}

def ModelDebugCanonicalizationPass : Pass<"canonicalize-model-for-debug", "ModelOp">
{
    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createModelDebugCanonicalizationPass()";
}

def MulOpDistributionPass : Pass<"distribute-mul", "mlir::ModuleOp">
{
    let summary = "Distribute the multiplication operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createMulOpDistributionPass()";
}

def NegateOpDistributionPass : Pass<"distribute-neg", "mlir::ModuleOp">
{
    let summary = "Distribute the negation operations";

    let description = [{
        // TODO
    }];

    let dependentDialects = [
        "mlir::bmodelica::BaseModelicaDialect"
    ];

    let constructor = "mlir::bmodelica::createNegateOpDistributionPass()";
}

#endif // MARCO_DIALECT_SIMULATION_TRANSFORMS_PASSES_TD
