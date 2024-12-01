set(SOURCES
    CodegenOptions.cpp
    CompilerInstance.cpp
    CompilerInvocation.cpp
    DiagnosticHandler.cpp
    ExecuteCompilerInvocation.cpp
    FrontendAction.cpp
    FrontendActions.cpp
    FrontendOptions.cpp
    LanguageOptions.cpp
    SimulationOptions.cpp)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(MLIR_LIBS
    ${dialect_libs}
    ${extension_libs}
    ${conversion_libs}
    MLIRExecutionEngine
    MLIRLLVMToLLVMIRTranslation
    MLIROpenMPToLLVMIRTranslation
    MLIRTargetLLVMIRImport)

marco_add_library(frontend
    ${SOURCES}

    LINK_LIBS
    marco::ast
    marco::parser
    marco::codegen
    marco::IO
    marco::Lowering
    marco::variableFilter
    ${MLIR_LIBS}

    LINK_COMPONENTS
    Option
    Support)

clang_target_link_libraries(frontend
    PRIVATE
    clangBasic
    clangDriver)

mlir_check_all_link_libraries(frontend)
