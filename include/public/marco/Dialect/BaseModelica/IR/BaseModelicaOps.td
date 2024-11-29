#ifndef MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAOPS_TD
#define MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAOPS_TD

include "marco/Dialect/BaseModelica/IR/BaseModelica.td"
include "marco/Dialect/BaseModelica/IR/BaseModelicaTypes.td"
include "marco/Dialect/BaseModelica/IR/BaseModelicaOpInterfaces.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/RegionKindInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"

//===---------------------------------------------------------------------===//
// Type constraints
//===---------------------------------------------------------------------===//

// BaseModelica Boolean type.
def BaseModelicaBoolean :
    Type<CPred<"$_self.isa<::mlir::bmodelica::BooleanType>()">,
         "boolean", "::mlir::bmodelica::BooleanType">;

// BaseModelica Boolean attribute.
def BaseModelicaBooleanAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::BoolAttr>()">,
           "boolean attribute">
{
    let storageType = [{ ::mlir::bmodelica::BoolAttr }];
    let returnType = [{ bool }];
    let valueType = BaseModelicaBoolean;

    let constBuilderCall =
        "::mlir::bmodelica::BooleanAttr::get($_builder.getContext(), $0)";
}

// BaseModelica Integer type.
def BaseModelicaInteger :
    Type<CPred<"$_self.isa<::mlir::bmodelica::IntegerType>()">,
         "integer", "::mlir::bmodelica::IntegerType">;

// BaseModelica Integer attribute.
def BaseModelicaIntegerAttr :
    Attr<CPred<"$_self.isa<::mlir::bmodelica::IntegerAttr>()">,
         "integer attribute">
{
    let storageType = [{ ::mlir::bmodelica::IntegerAttr }];
    let returnType = [{ int64_t }];
    let valueType = BaseModelicaInteger;

    let constBuilderCall =
        "::mlir::bmodelica::IntegerAttr::get($_builder.getContext(), $0)";
}

// BaseModelica Real type.
def BaseModelicaReal :
    Type<CPred<"$_self.isa<::mlir::bmodelica::RealType>()">,
         "real", "::mlir::bmodelica::RealType">;

// BaseModelica Real attribute.
def BaseModelicaRealAttr :
    Attr<CPred<"$_self.isa<::mlir::bmodelica::RealAttr>()">,
         "real attribute">
{
    let storageType = [{ ::mlir::bmodelica::RealAttr }];
    let returnType = [{ double }];
    let valueType = BaseModelicaReal;

    let constBuilderCall =
        "::mlir::bmodelica::RealAttr::get($_builder.getContext(), $0)";
}

// BaseModelica Array type.
def BaseModelicaArray :
    Type<CPred<"$_self.isa<::mlir::bmodelica::ArrayType>()">,
         "array", "::mlir::bmodelica::ArrayType">;

// BaseModelica Record type.
def BaseModelicaRecord :
    Type<CPred<"$_self.isa<::mlir::bmodelica::RecordType>()">,
         "record", "::mlir::bmodelica::RecordType">;

// BaseModelica scalar type.
def AnyBaseModelicaScalar :
    AnyTypeOf<[BaseModelicaBoolean, BaseModelicaInteger, BaseModelicaReal]>;

// BaseModelica-compatible scalar type.
def AnyBaseModelicaCompatibleScalar :
    AnyTypeOf<[AnyBaseModelicaScalar, AnyInteger, AnyFloat, Index]>;

// Whether a type is an ArrayType.
def IsBaseModelicaArrayTypePred :
    CPred<"$_self.isa<::mlir::bmodelica::ArrayType>()">;

// Array with given allowed element types.
class BaseModelicaArrayOf<list<Type> allowedTypes> :
    ShapedContainerType<
        allowedTypes, IsBaseModelicaArrayTypePred, "array",
        "::mlir::bmodelica::ArrayType">;

def AnyBaseModelicaArray : BaseModelicaArrayOf<[AnyType]>;

// Whether a type is a VariableType.
def IsVariableTypePred : CPred<"$_self.isa<::mlir::bmodelica::VariableType>()">;

class VariableOf<list<Type> allowedTypes> :
    ContainerType<AnyTypeOf<allowedTypes>,
                  IsVariableTypePred,
                  "$_self.cast<::mlir::bmodelica::VariableType>().getElementType()",
                  "variable", "::mlir::bmodelica::VariableType">;

def AnyVariable : VariableOf<[AnyType]>;

// Equation type.
def BaseModelicaEquation :
    Type<CPred<"$_self.isa<::mlir::bmodelica::EquationType>()">,
         "equation", "::mlir::bmodelica::EquationType">;

// Range type.
def BaseModelicaRange :
    Type<CPred<"$_self.isa<::mlir::bmodelica::RangeType>()">,
         "range", "::mlir::bmodelica::RangeType">;

// MultidimensionalRange attribute.
def MultidimensionalRangeAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::MultidimensionalRangeAttr>()">,
                 "Multidimensional range attribute">
{
    let storageType = [{ ::mlir::bmodelica::MultidimensionalRangeAttr }];
    let returnType = [{ ::mlir::bmodelica::MultidimensionalRangeAttr }];
    let convertFromStorage = "$_self";
}

// Function derivative attribute.
def FunctionDerivativeAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::FunctionDerivativeAttr>()">,
           "Function derivative attribute">
{
    let storageType = [{ ::mlir::bmodelica::FunctionDerivativeAttr }];
    let returnType = [{ ::mlir::bmodelica::FunctionDerivativeAttr }];
    let convertFromStorage = "$_self";
}

def VarDerivativeAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::VarDerivativeAttr>()">,
           "Variable derivative attribute">;

// Equation path attribute.
def EquationPathAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::EquationPathAttr>()">,
           "Equation path attribute">
{
    let storageType = [{ ::mlir::bmodelica::EquationPathAttr }];
    let returnType = [{ ::mlir::bmodelica::EquationPathAttr }];
    let convertFromStorage = "$_self";
}

// Equation scheduling direction attribute.
def EquationScheduleDirectionAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::EquationScheduleDirectionAttr>()">,
           "Equation scheduling direction attribute">
{
    let storageType = [{ ::mlir::bmodelica::EquationScheduleDirectionAttr }];
    let returnType = [{ ::mlir::bmodelica::EquationScheduleDirectionAttr }];
    let convertFromStorage = "$_self";
}

// Equation view attribute.
def EquationViewAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::EquationViewAttr>()">,
                 "Equation view attribute">
{
    let storageType = [{ ::mlir::bmodelica::EquationViewAttr }];
    let returnType = [{ ::mlir::bmodelica::EquationViewAttr }];
    let convertFromStorage = "$_self";
}

// Equation match attribute.
def EquationMatchAttr
    : Attr<CPred<"$_self.isa<::mlir::bmodelica::EquationMatchAttr>()">,
                 "Equation view attribute">
{
    let storageType = [{ ::mlir::bmodelica::EquationMatchAttr }];
    let returnType = [{ ::mlir::bmodelica::EquationMatchAttr }];
    let convertFromStorage = "$_self";
}

//===---------------------------------------------------------------------===//
// Properties
//===---------------------------------------------------------------------===//

def DerivativesMapProperty : Property<"DerivativesMap">
{
    let convertToAttribute = "$_storage.asAttribute($_ctxt)";

    let convertFromAttribute =
        "return DerivativesMap::setFromAttr($_storage, $_attr, $_diag);";

    let hashProperty = "$_storage.hash();";
}

def VariablesListProperty : Property<"VariablesList">
{
    let convertToAttribute = [{
        getPropertiesAsAttribute($_ctxt, $_storage)
    }];

    let convertFromAttribute = [{
        return setPropertiesFromAttribute($_storage, $_attr, $_diag);
    }];

    let hashProperty = [{
        computeHash($_storage);
    }];
}

//===---------------------------------------------------------------------===//
// Traits
//===---------------------------------------------------------------------===//

class NonEmptyVariadic<Type type> : Variadic<type>
{
    let minSize = 1;
}

//===---------------------------------------------------------------------===//
// BaseModelica operation definitions
//===---------------------------------------------------------------------===//

class BaseModelica_Op<string mnemonic, list<Trait> traits = []> :
    Op<BaseModelica_Dialect, mnemonic, traits>;

//===---------------------------------------------------------------------===//
// Range operations
//===---------------------------------------------------------------------===//

def RangeOp : BaseModelica_Op<"range",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Range";

    let description = [{
        The operation represents a mono-dimensional range, with a lower bound,
        an upper bound and a step.

        Example:

        ```mlir
        %0 = modelica.range %0, %1, %2 : (index, index, index) -> !modelica<range index>
        ```
    }];

    let arguments = (ins
        AnyType:$lowerBound,
        AnyType:$upperBound,
        AnyType:$step);

    let results = (outs BaseModelicaRange:$result);
    let assemblyFormat = "$lowerBound `,` $upperBound `,` $step attr-dict `:` `(` type($lowerBound) `,` type($upperBound) `,` type($step) `)` `->` type($result)";
    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def UnboundedRangeOp : BaseModelica_Op<"unbounded_range", [Pure]>
{
    let summary = "Unbounded range";

    let description = [{
        The operation represents a mono-dimensional range with unbounded
        bounds.

        Example:

        ```mlir
        %0 = modelica.unbounded_range : !modelica<range index>
        ```
    }];

    let arguments = (ins);
    let results = (outs BaseModelicaRange:$result);
    let assemblyFormat = "attr-dict `:` type($result)";

    let builders = [
        OpBuilder<(ins), [{
            auto rangeType = RangeType::get(
                $_builder.getContext(), $_builder.getIndexType());

            build($_builder, $_state, rangeType);
        }]>
    ];
}

def RangeBeginOp : BaseModelica_Op<"range_begin", []>
{
    let summary = "Get the begin value of a range";

    let description = [{
        Get the begin value of a range.

        Example:

        ```mlir
        %0 = ... : !modelica<range index>
        %1 = modelica.range_begin %0 : !modelica<range index> -> index
        ```
    }];

    let arguments = (ins BaseModelicaRange:$range);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$range attr-dict `:` type($range) `->` type($result)";

    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "mlir::Value":$range), [{
            auto inductionType =
                range.getType().cast<RangeType>().getInductionType();

            build($_builder, $_state, inductionType, range);
        }]>
    ];
}

def RangeEndOp : BaseModelica_Op<"range_end", []>
{
    let summary = "Get the end value of a range";

    let description = [{
        Get the end value of a range.

        Example:

        ```mlir
        %0 = ... : !modelica<range index>
        %1 = modelica.range_end %0 : !modelica<range index> -> index
        ```
    }];

    let arguments = (ins BaseModelicaRange:$range);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$range attr-dict `:` type($range) `->` type($result)";

    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "mlir::Value":$range), [{
            auto inductionType =
                range.getType().cast<RangeType>().getInductionType();

            build($_builder, $_state, inductionType, range);
        }]>
    ];
}

def RangeStepOp : BaseModelica_Op<"range_step", []>
{
    let summary = "Get the step value of a range";

    let description = [{
        Get the step value of a range.

        Example:

        ```mlir
        %0 = ... : !modelica<range index>
        %1 = modelica.range_step %0 : !modelica<range index> -> index
        ```
    }];

    let arguments = (ins BaseModelicaRange:$range);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$range attr-dict `:` type($range) `->` type($result)";

    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "mlir::Value":$range), [{
            auto inductionType =
                range.getType().cast<RangeType>().getInductionType();

            build($_builder, $_state, inductionType, range);
        }]>
    ];
}

def RangeSizeOp : BaseModelica_Op<"range_size", []>
{
    let summary = "Get the step value of a range";

    let description = [{
        Get the number of elements of a range.

        Example:

        ```mlir
        %0 = ... : !modelica<range index>
        %1 = modelica.range_size %0 : !modelica<range index> -> index
        ```
    }];

    let arguments = (ins BaseModelicaRange:$range);
    let results = (outs Index:$result);
    let assemblyFormat = "$range attr-dict `:` type($range)";
    let hasFolder = 1;
}

//===---------------------------------------------------------------------===//
// Tensor operations
//===---------------------------------------------------------------------===//

def Modelica_TensorFromElementsOp : BaseModelica_Op<"tensor_from_elements",
    [NoMemoryEffect]>
{
    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyTensor:$tensor);

    let assemblyFormat = [{
        $values attr-dict `:` type($values) `->` type($tensor)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;
}

def Modelica_TensorBroadcastOp : BaseModelica_Op<"tensor_broadcast",
    [NoMemoryEffect]>
{
    let arguments = (ins
        AnyType:$value,
        Variadic<Index>:$dynamicDimensions);

    let results = (outs AnyTensor:$tensor);

    let assemblyFormat = [{
        $value (`,` $dynamicDimensions^)? attr-dict `:` type($value) `->` type($tensor)
    }];
}

def Modelica_TensorViewOp : BaseModelica_Op<"tensor_view",
    [ViewLikeOpInterface,
    NoMemoryEffect]>
{
    let arguments = (ins
        AnyTensor:$source,
        Variadic<AnyTypeOf<[AnyBaseModelicaCompatibleScalar, BaseModelicaRange]>>:$subscriptions);

    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $source `[` $subscriptions `]` attr-dict `:` type($source) (`,` type($subscriptions)^)? `->` type($result)
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "Value":$source, "ValueRange":$subscriptions)>
    ];

    let extraClassDeclaration = [{
        static mlir::TensorType inferResultType(
            mlir::TensorType source, mlir::ValueRange subscriptions);

        /// @name ViewLikeOpInterface methods
        /// {

        Value getViewSource()
        {
            return getSource();
        }

        /// }
    }];
}

def Modelica_TensorExtractOp : BaseModelica_Op<"tensor_extract",
    [TypesMatchWith<"result type matches element type of 'tensor'",
                     "tensor", "result",
                     "$_self.cast<mlir::TensorType>().getElementType()">,
    NoMemoryEffect]>
{
    let arguments = (ins
        AnyTensor:$tensor,
        Variadic<AnyTypeOf<[Index, BaseModelicaInteger]>>:$indices);

    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

def Modelica_TensorInsertOp : BaseModelica_Op<"tensor_insert",
    [TypesMatchWith<"result type matches type of 'destination'",
                    "destination", "result", "$_self">]>
{
    let summary = "Insert a value into a tensor.";

    let description = [{
        Insert a value into a tensor and return the modified data.
    }];

    let arguments = (ins
        AnyType:$value,
        AnyTensor:$destination,
        Variadic<AnyType>:$indices);

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $value `,` $destination `[` $indices `]` attr-dict `:` type($value) `,` type($destination) (`,` type($indices)^)? `->` type($result)
    }];
}

def Modelica_TensorInsertSliceOp : BaseModelica_Op<"tensor_insert_slice",
    [TypesMatchWith<"result type matches type of 'destination'",
                    "destination", "result", "$_self">]>
{
    let summary = "Insert a tensor into another one.";

    let description = [{
        Insert a tensor into another one and return the modified data.
    }];

    let arguments = (ins
        AnyType:$value,
        AnyTensor:$destination,
        Variadic<AnyType>:$subscriptions);

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $value `,` $destination `[` $subscriptions `]` attr-dict `:` type($value) `,` type($destination) (`,` type($subscriptions)^)? `->` type($result)
    }];
}

//===---------------------------------------------------------------------===//
// Array operations
//===---------------------------------------------------------------------===//

def Modelica_AllocaOp : BaseModelica_Op<"alloca",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Allocate an array on the stack.";

    let description = [{
        Allocate an array on the stack.
        The operation takes a variadic number of arguments, each of one
        representing the size of a dynamic dimension within the desired array
        type. Note that the amount of operands must match the number of dynamic
        dimensions and they all must be of `index` type.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = modelica.alloca %0 : !modelica.array<3x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyBaseModelicaArray:$array);

    let assemblyFormat = [{
        $dynamicSizes attr-dict `:` type($array)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$dynamicSizes), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(dynamicSizes);
        }]>
    ];

    let extraClassDeclaration = [{
        ArrayType getArrayType()
        {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_AllocOp : BaseModelica_Op<"alloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Allocate an array on the heap.";

    let description = [{
        Allocate an array on the heap.
        The operation takes a variadic number of arguments, each of one
        representing the size of a dynamic dimension within the desired array
        type. Note that the amount of operands must match the number of dynamic
        dimensions and they all must be of `index` type.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = modelica.alloca %0 : !modelica.array<3x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<Index>:$dynamicSizes);
    let results = (outs AnyBaseModelicaArray:$array);

    let assemblyFormat = "($dynamicSizes^)? attr-dict `:` type($array)";

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$dynamicSizes), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(dynamicSizes);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the result array type.
        ArrayType getArrayType()
        {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_ArrayFromElementsOp : BaseModelica_Op<"array_from_elements",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Create an array with the given elements.";

    let description = [{
        Alloc an array with the given elements.
        The amount of operands must match the total number of values of the
        array.

        Example:

        ```mlir
        %0 = constant #modelica<int 1>
        %1 = constant #modelica<real 2.0>
        %2 = constant #modelica<real 3.0>
        %3 = constant #modelica<int 4>
        %4 = constant #modelica<int 5>
        %5 = constant #modelica<int 6>
        %6 = modelica.array_from_elements %0, %1, %2, %3, %4, %5 : !modelica.int, !modelica.real, !modelica.real, !modelica.int, !modelica.int, !modelica.int -> !modelica.array<3x2x!modelica.real>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyBaseModelicaArray:$array);

    let assemblyFormat = [{
        $values attr-dict `:` type($values) `->` type($array)
    }];

    let hasVerifier = 1;
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "ArrayType":$arrayType, "ValueRange":$values), [{
            $_state.types.push_back(arrayType);
            $_state.addOperands(values);
        }]>
    ];

    let extraClassDeclaration = [{
        /// Get the result array type.
        ArrayType getArrayType()
        {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def Modelica_ArrayBroadcastOp : BaseModelica_Op<"array_broadcast",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Create an array with a given value in all positions.";

    let description = [{
        Alloc an array with all the elements equal to a given one.
        Dynamically sized arrays can be created by passing also the dynamic
        dimensions.

        Example with static dimensions:

        ```mlir
        %0 = ... : !modelica.int
        %1 = modelica.array_broadcast %0 : !modelica.int -> !modelica.array<3x2x!modelica.real>
        ```

        Example with dynamic dimensions:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : index
        %2 = ... : index
        %3 = modelica.array_broadcast %0, %1, %2 : !modelica.int -> !modelica.array<?x2x?x!modelica.real>
        ```
    }];

    let arguments = (ins
        AnyBaseModelicaCompatibleScalar:$value,
        Variadic<Index>:$dynamicDimensions);

    let results = (outs AnyBaseModelicaArray:$array);

    let assemblyFormat = [{
        $value (`,` $dynamicDimensions^)? attr-dict `:` type($value) `->` type($array)
    }];

    let extraClassDeclaration = [{
        /// Get the result array type.
        ArrayType getArrayType()
        {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def BaseModelica_FreeOp : BaseModelica_Op<"free",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Deallocate an heap-allocated array.";

    let description = [{
        Deallocate an heap-allocated array.
        The array must have been created using the `modelica.alloc` operation.

        Example:

        ```mlir
        %array = ... : !modelica.array<?x!modelica.int>
        modelica.free %array : !modelica.array<?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyBaseModelicaArray:$array);
    let results = (outs);
    let assemblyFormat = "$array attr-dict `:` type($array)";
}

def Modelica_ArrayCastOp : BaseModelica_Op<"array_cast",
    [ViewLikeOpInterface]>
{
    let summary = "Cast operation for array types.";

    let description = [{
        This operation should be used for only one purpose, that is to
        generalize the dimensions of an array. This can be done by either
        converting some dimensions to unknown ones or by converting the whole
        array to an unsized one.
        The operation is NOT intended to cast the element type to a different
        one or to specialize the shape of an array with dynamic dimensions.

        Correct usage examples:

        ```mlir
        %0 = ... : !modelica.array<3x2x!modelica.int>
        %1 = modelica.array_cast %0 : !modelica.array<3x2x!modelica.int> -> !modelica.array<?x?x!modelica.int>
        %2 = modelica.array_cast %0 : !modelica.array<3x2x!modelica.int> -> !modelica.array<*x!modelica.int>
        ```

        Wrong usage examples:

        ```mlir
        %0 = ... : !modelica.array<3x2x!modelica.int>
        %1 = modelica.array_cast %0 : !modelica.array<3x2x!modelica.int> -> !modelica.array<3x2x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$source);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $source attr-dict `:` type($source) `->` type($result)
    }];

    let extraClassDeclaration = [{
        /// @name ViewLikeOpInterface methods
        /// {

        Value getViewSource()
        {
            return getSource();
        }

        /// }
    }];
}

def Modelica_DimOp : BaseModelica_Op<"dim", [NoMemoryEffect]>
{
    let summary = "Get the size of an array dimension.";

    let description = [{
        Get the size of an array dimension. The operation is trivial in case of
        statically shaped arrays, while it is more useful in case of dimensions
        that are unknown at compile-time.
        The result value has `index` type.

        Example:

        ```mlir
        %array = ... : !modelica.array<?x?x?x!modelica.int>
        %c2 = constant 2 : index
        %dim = modelica.dim %array, %c2 : !modelica.array<?x?x?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyBaseModelicaArray:$array, Index:$dimension);
    let results = (outs Index:$result);
    let assemblyFormat = "$array `,` $dimension attr-dict `:` type($array)";
    let hasCanonicalizer = 1;
}

def BaseModelica_LoadOp : BaseModelica_Op<"load",
    [TypesMatchWith<"result type matches element type of 'array'",
                     "array", "result",
                     "$_self.cast<ArrayType>().getElementType()">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Load a value from an array.";

    let description = [{
        Load a value from an array location given by indices.
        The number of arguments provided within brackets must match the rank of
        the array.

        Example:

        ```mlir
        $array = ... : !modelica.array<?x?x!modelica.int>
        %0 = constant 3 : index
        %1 = constant 5 : index
        %value = modelica.load %array[%0, %1] : !modelica.array<?x?x!modelica.int>
        ```
    }];

    let arguments = (ins
        AnyBaseModelicaArray:$array,
        Variadic<AnyTypeOf<[Index, BaseModelicaInteger]>>:$indices);

    let results = (outs AnyType:$result);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        /// Get the array type.
        ArrayType getArrayType()
        {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def BaseModelica_StoreOp : BaseModelica_Op<"store",
    [TypesMatchWith<"type of 'value' matches element type of 'array'",
                    "array", "value",
                    "$_self.cast<ArrayType>().getElementType()">,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Store a value into an array.";

    let description = [{
        Store a value to an array location given by indices.
        The value stored must have the same type as the elemental type of the
        array. The number of arguments provided within brackets must match the
        rank of the array.

        Example:

        ```mlir
        %array = ... : !modelica.array<?x?x!modelica.int>
        %0 = constant 3 : index
        %1 = constant 5 : index
        %value = ... : !modelica.int
        modelica.store %array[%0, %1], %value : !modelica.array<?x?x!modelica.int>
        ```
    }];

    let arguments = (ins
        AnyType:$value,
        AnyBaseModelicaArray:$array,
        Variadic<AnyTypeOf<[Index, BaseModelicaInteger]>>:$indices);

    let results = (outs);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        /// Get the array type.
        ArrayType getArrayType()
        {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def BaseModelica_SubscriptionOp : BaseModelica_Op<"subscription",
    [ViewLikeOpInterface,
    NoMemoryEffect]>
{
    let summary = "Create a view over an array.";

    let description = [{
        Create a view over an array.
        The values of the original arrays are not copied and all the modifications
        performed on the subview do reflect on the original array.

        Example:

        ```mlir
        %array = ... : !modelica.array<3x5x4x!modelica.int>
        %c1 : constant 1 : index
        %c2 : constant 2 : index
        %view = modelica.subscription %array[%c1, %c2] : !modelica.array<3x5x4x!modelica.int>
        // %view has type !modelica.array<4x!modelica.int>
        ```
    }];

    let arguments = (ins
        AnyBaseModelicaArray:$source,
        Variadic<AnyTypeOf<[AnyBaseModelicaCompatibleScalar, BaseModelicaRange]>>:$indices);

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $source `[` $indices `]` attr-dict `:` type($source) (`,` type($indices)^)? `->` type($result)
    }];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "Value":$source, "ValueRange":$indices)>
    ];

    let extraClassDeclaration = [{
        static ArrayType inferResultType(
            ArrayType source, mlir::ValueRange indices);

        ArrayType getSourceArrayType()
        {
            return getSource().getType().cast<ArrayType>();
        }

        ArrayType getResultArrayType()
        {
            return getResult().getType().cast<ArrayType>();
        }

        /// @name ViewLikeOpInterface methods
        /// {

        Value getViewSource()
        {
            return getSource();
        }

        /// }
    }];
}

def BaseModelica_ArrayFillOp : BaseModelica_Op<"array_fill",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Fill an array with a value.";

    let description = [{
        Set all the elements of an array to a given value.

        Example:

        ```mlir
        %value = ... : !modelica.int
        %array = ... : !modelica.array<?x!modelica.real>
        modelica.array_fill %array, %value : !modelica.array<?x!modelica.real>, !modelica.int
        ```
    }];

    let arguments = (ins AnyBaseModelicaArray:$array, AnyBaseModelicaCompatibleScalar:$value);
    let results = (outs);

    let assemblyFormat = [{
        $array `,` $value attr-dict `:` type($array) `,` type($value)
    }];

    let extraClassDeclaration = [{
        /// Get the array type.
        ArrayType getArrayType()
        {
            return getArray().getType().cast<ArrayType>();
        }
    }];
}

def BaseModelica_ArrayCopyOp : BaseModelica_Op<"array_copy",
    [SameOperandsShape,
     DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Copy the elements from an array to another.";

    let description = [{
        The operation copies the elements from the source to the destination
        array.

        Example:

        ```mlir
        %source = ... : !modelica.array<?x!modelica.int>
        %destination = ... : !modelica.array<?x!modelica.real>
        modelica.array_copy %destination, %source : !modelica.array<?x!modelica.int>, !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins BaseModelicaArray:$source, BaseModelicaArray:$destination);
    let results = (outs);

    let assemblyFormat = [{
        $source `,` $destination attr-dict `:` type($source) `,` type($destination)
    }];
}

//===---------------------------------------------------------------------===//
// Tensor-Array bridge operations
//===---------------------------------------------------------------------===//

def Modelica_TensorToArrayOp : BaseModelica_Op<"tensor_to_array", [NoMemoryEffect]>
{
    let summary = "Convert a tensor to an array.";

    let arguments = (ins AnyTensor:$tensor);
    let results = (outs AnyBaseModelicaArray:$array);

    let assemblyFormat = [{
        $tensor attr-dict `:` type($tensor) `->` type($array)
    }];
}

def Modelica_ArrayToTensorOp : BaseModelica_Op<"array_to_tensor",
    [NoMemoryEffect]>
{
    let summary = "Convert an array to a tensor.";

    let arguments = (ins AnyBaseModelicaArray:$array);
    let results = (outs AnyTensor:$tensor);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($tensor)
    }];

    let extraClassDeclaration = [{
        /*
        mlir::LogicalResult bufferize(
            mlir::RewriterBase &rewriter,
            const mlir::bufferization::BufferizationOptions &options) const
        {
          return success();
        }
        */
    }];
}

//===---------------------------------------------------------------------===//
// Variable operations
//===---------------------------------------------------------------------===//

def BaseModelica_VariableOp : BaseModelica_Op<"variable",
    [Symbol,
    ParentOneOf<["ModelOp", "FunctionOp", "RecordOp"]>]>
{
    let summary = "Declare a variable of a class.";

    let description = [{
        The operation is used to declare a variable of a class.
        It takes the name of the symbol, its type and provides a region
        allowing to specify contraints for fixed but dinamically sized
        dimensions.
        The variable value can be then read and written by means of the
        `modelica.variable_get` and `modelica.variable_set` operations.
        Notice that, in case of array variables, the array must be retrieved
        using `modelica.variable_get` if wanting to operate on some specific
        element instead of replacing the whole array.

        Example:

        ```mlir
        modelica.variable @x : !modelica.variable<?x5x?x?x!modelica.int> [fixed, unbounded, fixed] {
            %0 = ... : index
            %1 = ... : index
            modelica.yield %0, %1 : index, index
        } attributes {...}
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        StrArrayAttr:$dimensionsConstraints
    );

    let results = (outs);
    let regions = (region AnyRegion:$constraintsRegion);
    let hasCustomAssemblyFormat = 1;
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$name, "VariableType":$variableType)>
    ];

    let extraClassDeclaration = [{
        static constexpr llvm::StringLiteral kDimensionConstraintUnbounded = "unbounded";
        static constexpr llvm::StringLiteral kDimensionConstraintFixed = "fixed";

        VariableType getVariableType();

        /// Get whether the variable has the 'input' modifier.
        bool isInput();

        /// Get whether the variable has the 'output' modifier.
        bool isOutput();

        /// Get whether the variable has the 'discrete' modifier.
        bool isDiscrete();

        /// Get whether the variable has the 'parameter' modifier.
        bool isParameter();

        /// Get whether the variable has the 'constant' modifier.
        bool isConstant();

        /// Get whether the parameter is a parameter or a constant.
        bool isReadOnly();

        /// Get the number of unbounded dimensions among the unknown ones.
        size_t getNumOfUnboundedDimensions();

        /// Get the number of fixed dimensions among the unknown ones.
        size_t getNumOfFixedDimensions();

        /// Get the indices of the variable, if the variable is an array.
        IndexSet getIndices();
    }];
}

def BaseModelica_VariableGetOp : BaseModelica_Op<"variable_get",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    NoMemoryEffect]>
{
    let summary = "Load the data held by a class variable.";

    let description = [{
        Load the data held by variable of a class.
        Allows to shift from symbol semantics to SSA semantics.

        Example with scalar variable:

        ```mlir
        modelica.variable @x : !modelica.variable<!modelica.real, parameter>
        %0 = modelica.variable_get @x : !modelica.real
        ```

        Example with array variable:

        ```mlir
        modelica.variable @x : !modelica.variable<2x3x!modelica.real, parameter>
        %0 = modelica.variable_get @x : !modelica.array<2x3x!modelica.real>
        ```
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$variable attr-dict `:` type($result)";

    let builders = [
        OpBuilder<(ins "VariableOp":$variableOp)>
    ];
}

def Modelica_VariableSetOp : BaseModelica_Op<"variable_set",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Store data into a class variable.";

    let description = [{
        Store data into the variable of a class.

        Example with scalar variable:

        ```mlir
        modelica.variable @x : !modelica.variable<!modelica.real, parameter>
        %0 = modelica.constant #modelica<real 0.0>
        modelica.variable_store @x, %0 : !modelica.real
        ```

        Example with array variable:

        ```mlir
        modelica.variable @x : !modelica.variable<2x3x!modelica.real, parameter>
        %0 = ... : !modelica.array<2x3x!modelica.real>
        modelica.variable_store @x, %0 : !modelica.array<2x3x!modelica.real>
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$variable,
        Variadic<AnyType>:$indices,
        AnyType:$value);

    let results = (outs);
    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "VariableOp":$variableOp, "mlir::Value":$value)>,
        OpBuilder<(ins "VariableOp":$variableOp, "mlir::ValueRange":$indices, "mlir::Value":$value)>
    ];
}

def Modelica_VariableComponentSetOp : BaseModelica_Op<"variable_component_set", []>
{
    let arguments = (ins
        FlatSymbolRefArrayAttr:$path,
        Variadic<AnyType>:$subscriptions,
        I64ArrayAttr:$subscriptionsAmounts,
        AnyType:$value);

    let results = (outs);
    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        mlir::ValueRange getComponentSubscripts(size_t componentIndex);
    }];
}

def Modelica_ComponentGetOp : BaseModelica_Op<"component_get",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Get a component of a structured variable.";

    let description = [{
        Get a component of a structured variable.

        Example:

        ```mlir
        modelica.record @R {
            modelica.variable @x : !modelica.variable<!modelica.real>
        }

        modelica.variable @r : !modelica<record @R>

        %0 = modelica.variable_get @r : !modelica<record @R>
        %1 = modelica.component_get %0, @x, !modelica<record @R>, !modelica.real
    }];

    let arguments = (ins
        AnyTypeOf<[BaseModelicaRecord, TensorOf<[BaseModelicaRecord]>]>:$variable,
        SymbolNameAttr:$componentName);

    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $variable `,` $componentName attr-dict `:` type($variable) `->` type($result)
    }];
}

def Modelica_GlobalVariableOp : BaseModelica_Op<"global_variable",
    [Symbol,
    HasParent<"mlir::ModuleOp">]>
{
    let summary = "Declare a global variable.";

    let description = [{
        Declare a global variable inside a module.
        Its type must be an ArrayType.

        Example:

        ```mlir
        module {
            modelica.global_variable @x : !modelica.array<2x3x!modelica.real>
        }
        ```
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttr:$type,
        OptionalAttr<AnyAttr>:$initial_value
    );

    let results = (outs);
    let assemblyFormat = "$sym_name attr-dict `:` $type";

    let builders = [
        OpBuilder<(ins "mlir::StringAttr":$name, "mlir::TypeAttr":$type)>,
        OpBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$type)>
    ];
}

def BaseModelica_GlobalVariableGetOp : BaseModelica_Op<"global_variable_get",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Get the data held by a global variable.";

    let description = [{
        Get the data held by a global variable.
        Allows to shift from symbol semantics to SSA semantics.
        Notice that there is no `GlobalVariableSetOp` counterpart for storing a
        value. Global variables always have array type and thus are directly
        modifiable once its reference is obtained through this operation, even
        in case of scalar variables.

        Example with scalar variable:

        ```mlir
        modelica.global_variable @x : !modelica.array<!modelica.real>
        %0 = modelica.global_variable_get @x : !modelica.array<!modelica.real>
        ```

        Example with array variable:

        ```mlir
        modelica.global_variable @x : !modelica.array<2x3x!modelica.real>
        %0 = modelica.global_variable_get @x : !modelica.array<2x3x!modelica.real>
        ```
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$variable attr-dict `:` type($result)";

    let builders = [
        OpBuilder<(ins "GlobalVariableOp":$globalVariableOp)>
    ];
}

def BaseModelica_QualifiedVariableGetOp : BaseModelica_Op<"qualified_variable_get",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Get the data held by a class variable.";

    let description = [{
        Get the data held by variable of a class.
        Similar to `VariableGetOp`, but with the lookup being performed
        starting from the module, thus enabling the referral to class variables
        from outside the class body.
    }];

    let arguments = (ins SymbolRefAttr:$variable);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$variable attr-dict `:` type($result)";

    let builders = [
        OpBuilder<(ins "VariableOp":$variableOp)>
    ];

    let extraClassDeclaration = [{
        VariableOp getVariableOp();

        VariableOp getVariableOp(
            mlir::SymbolTableCollection& symbolTableCollection);
    }];
}

def BaseModelica_QualifiedVariableSetOp : BaseModelica_Op<"qualified_variable_set",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Store data into a class variable.";

    let description = [{
        Store data into the variable of a class.
        Similar to `VariableSetOp`, but with the lookup being performed
        starting from the module, thus enabling the referral to class variables
        from outside the class body.
    }];

    let arguments = (ins
        SymbolRefAttr:$variable,
        Variadic<AnyType>:$indices,
        AnyType:$value);

    let results = (outs);
    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "VariableOp":$variableOp, "mlir::Value":$value)>,
        OpBuilder<(ins "VariableOp":$variableOp, "mlir::ValueRange":$indices, "mlir::Value":$value)>
    ];

    let extraClassDeclaration = [{
        VariableOp getVariableOp();

        VariableOp getVariableOp(
            mlir::SymbolTableCollection& symbolTableCollection);
    }];
}

def BaseModelica_PoolVariableGetOp : BaseModelica_Op<"pool_variable_get", [NoMemoryEffect]>
{
    let summary = "Get the data held by a variable allocated in a memory pool.";

    let description = [{
        Get the data held by a variable allocated in a memory pool.

        Example:

        ```mlir
        %0 = modelica.pool_variable_get %0, %1 : !modelica.array<2x3x!modelica.real>
        ```
    }];

    let arguments = (ins Index:$pool, Index:$id);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$pool `,` $id attr-dict `:` type($result)";
}

//===---------------------------------------------------------------------===//
// Math operations
//===---------------------------------------------------------------------===//

def BaseModelica_ConstantOp : BaseModelica_Op<"constant",
    [ConstantLike,
    Pure,
    AllTypesMatch<["value", "result"]>]>
{
    let summary = "Constant.";

    let description = [{
        The operation turns a literal into an SSA value. The data is attached to the
        operation as an attribute.

        Example:

        ```mlir
        %0 = modelica.constant #modelica<int 3>
        ```
    }];

    let arguments = (ins TypedAttrInterface:$value);
    let results = (outs AnyType:$result);

    let assemblyFormat = "$value attr-dict";
    let hasFolder = 1;

    let builders = [
        OpBuilder<(ins "mlir::Attribute":$value), [{
            auto typedAttr = value.cast<mlir::TypedAttr>();
            build($_builder, $_state, typedAttr.getType(), typedAttr);
        }]>
    ];
}

def BaseModelica_NegateOp : BaseModelica_Op<"neg",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Negation operation.";

    let description = [{
        The operation takes one argument and returns the value with
        the opposite sign.
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_AddOp : BaseModelica_Op<"add",
    [Commutative,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    AlwaysSpeculatable,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Addition.";

    let description = [{
        The operation computes the sum of two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the BaseModelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.add %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_AddEWOp : BaseModelica_Op<"add_ew",
    [Commutative,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    AlwaysSpeculatable,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Element-wise addition.";

    let description = [{
        The operation computes the element-wise sum of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.real
        %2 = modelica.add_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.real) -> !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_SubOp : BaseModelica_Op<"sub",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    AlwaysSpeculatable,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Subtraction.";

    let description = [{
        The operation computes the difference between two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the BaseModelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.sub %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_SubEWOp : BaseModelica_Op<"sub_ew",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    AlwaysSpeculatable,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Element-wise subtraction.";

    let description = [{
        The operation computes the element-wise difference of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.real
        %2 = modelica.sub_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.real) -> !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_MulOp : BaseModelica_Op<"mul",
    [Commutative,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    AlwaysSpeculatable,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Multiplication.";

    let description = [{
        The operation computes the product of two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the BaseModelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.int>
        %1 = ... : !modelica.array<3x!modelica.int>
        %2 = modelica.mul %0, %1 : (!modelica.array<3x!modelica.int>, !modelica.array<3x!modelica.int>) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);

        bool isScalarProduct();
        bool isCrossProduct();
        bool isVectorMatrixProduct();
        bool isMatrixVectorProduct();
        bool isMatrixProduct();
    }];
}

def BaseModelica_MulEWOp : BaseModelica_Op<"mul_ew",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    AlwaysSpeculatable,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Element-wise multiplication.";

    let description = [{
        The operation computes the element-wise product of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.int
        %2 = modelica.mul_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.int) -> !modelica.array<?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_DivOp : BaseModelica_Op<"div",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Division.";

    let description = [{
        The operation computes the division between two values.
        It takes two operands and returns one result. Array operands are also allowed,
        but they must respect the BaseModelica specification.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.div %0, %1 : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_DivEWOp : BaseModelica_Op<"div_ew",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect,
    DeclareOpInterfaceMethods<DistributableOpInterface>,
    DeclareOpInterfaceMethods<NegateOpDistributionInterface>,
    DeclareOpInterfaceMethods<MulOpDistributionInterface>,
    DeclareOpInterfaceMethods<DivOpDistributionInterface>]>
{
    let summary = "Element-wise division.";

    let description = [{
        The operation computes the element-wise division of two values.
        It takes two operands and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.int>
        %1 = ... : !modelica.int
        %2 = modelica.div_ew %0, %1 : (!modelica.array<?x!modelica.int>, !modelica.int) -> !modelica.array<?x!modelica.int>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_PowOp : BaseModelica_Op<"pow",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect]>
{
    let summary = "Pow.";

    let description = [{
        The operation computes the power of a base to a given exponent.
        It takes to operands (base and exponent) and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = ... : !modelica.real
        %2 = modelica.pow %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` type($base) `,` type($exponent) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_PowEWOp : BaseModelica_Op<"pow_ew",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect]>
{
    let summary = "Element-wise pow.";

    let description = [{
        The operation computes the element-wise power of a (optionally array) base to a
        given exponent.
        It takes to operands (base and exponent) and returns one result.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x!modelica.real>
        %1 = ... : !modelica.real
        %2 = modelica.pow %0, %1 : (!modelica.array<?x!modelica.real>, !modelica.real) -> !modelica.array<?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$base, AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $base `,` $exponent attr-dict `:` `(` type($base) `,` type($exponent) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

//===---------------------------------------------------------------------===//
// Comparison operations
//===---------------------------------------------------------------------===//

// Base class for comparison operations.
class BaseModelica_ComparisonOp<string mnemonic, list<Trait> traits = []> :
    BaseModelica_Op<mnemonic, traits>
{
    let arguments = (ins
        AnyBaseModelicaCompatibleScalar:$lhs,
        AnyBaseModelicaCompatibleScalar:$rhs);

    let results = (outs AnyBaseModelicaCompatibleScalar:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];
}

def BaseModelica_EqOp : BaseModelica_ComparisonOp<"eq",
    [Commutative,
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Check if two values are equal.";

    let description = [{
        The operation checks whether two scalar values are equal.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.eq %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_NotEqOp : BaseModelica_ComparisonOp<"neq",
    [Commutative,
    Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Check if two values are not equal.";

    let description = [{
        The operation checks whether two scalar values are not equal.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.neq %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_GtOp : BaseModelica_ComparisonOp<"gt",
    [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Check if a value is greater than another one.";

    let description = [{
        The  operation checks whether a scalar value is greater than another one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.gt %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_GteOp : BaseModelica_ComparisonOp<"gte",
    [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Check if a value is greater or equal than another one.";

    let description = [{
        The  operation checks whether a scalar value is greater or equal to another one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.gte %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_LtOp : BaseModelica_ComparisonOp<"lt",
    [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Check if a value is less than another one.";

    let description = [{
        The operation checks whether a scalar value is less than another one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.lt %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_LteOp : BaseModelica_ComparisonOp<"lte",
    [Pure,
    DeclareOpInterfaceMethods<InferTypeOpInterface>]>
{
    let summary = "Check if a value is less or equal than another one.";

    let description = [{
        The operation checks whether a scalar value is less or equal to another one.

        Example:
        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.real
        %2 = modelica.lte %0, %1 : (!modelica.int, !modelica.real) -> !modelica.bool
        ```
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

//===---------------------------------------------------------------------===//
// Logic operations
//===---------------------------------------------------------------------===//

def BaseModelica_NotOp : BaseModelica_Op<"not",
    [AlwaysSpeculatable,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect]>
{
    let summary = "Compute the logical NOT of a boolean value.";

    let description = [{
        The operation computes the logical NOT of a boolean value.
        It takes one operand and returns one result. The type of the operand must be a
        BaseModelica boolean or an array composed by booleans. In case of array operand, its
        size must be equal to the one of the result type. Runtime checks can also be
        enabled to ensure shapes correctness when dynamic dimensions are present.

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.bool
        %1 = modelica.not %0 : !modelica.bool -> !modelica.bool
        ```

        Example with array values:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.bool>
        %1 = modelica.not %0, %1 : !modelica.array<3x!modelica.bool> -> !modelica.array<3x!modelica.bool>
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_AndOp : BaseModelica_Op<"and",
    [AlwaysSpeculatable,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect]>
{
    let summary = "Compute the logical AND between two boolean values.";

    let description = [{
        The operation computes the logical AND between two boolean values.
        It takes two operands and returns one result. The type of the operands must be
        BaseModelica booleans or arrays composed by booleans. In case of array operands, their
        size must be equal among themselves and also with respect to the result type.
        Runtime checks can also be enabled to ensure shapes correctness when dynamic
        dimensions are present.

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.bool
        %1 = ... : !modelica.bool
        %2 modelica.and %0, %1 : (!modelica.bool, !modelica.bool) -> !modelica.bool
        ```

        Example with array values:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.bool>
        %1 = ... : !modelica.array<3x!modelica.bool>
        %2 = modelica.and %0, %1 : (!modelica.array<3x!modelica.bool>, !modelica.array<3x!modelica.bool>) -> !modelica.array<3x!modelica.bool>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_OrOp : BaseModelica_Op<"or",
    [AlwaysSpeculatable,
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    NoMemoryEffect]>
{
    let summary = "Compute the logical OR between two boolean values.";

    let description = [{
        The operation computes the logical OR between two boolean values.
        It takes two operands and returns one result. The type of the operands must be
        BaseModelica booleans or arrays composed by booleans. In case of array operands, their
        size must be equal among themselves and also with respect to the result type.
        Runtime checks can also be enabled to ensure shapes correctness when dynamic
        dimensions are present.

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.bool
        %1 = ... : !modelica.bool
        %2 = modelica.or %0, %1 : (!modelica.bool, !modelica.bool) -> !modelica.bool
        ```

        Example with array values:

        ```mlir
        %0 = ... : !modelica.array<3x!modelica.bool>
        %1 = ... : !modelica.array<3x!modelica.bool>
        %2 = modelica.or %0, %1 : (!modelica.array<3x!modelica.bool>, !modelica.array<3x!modelica.bool>) -> !modelica.array<3x!modelica.bool>
        ```
    }];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $lhs `,` $rhs attr-dict `:` `(` type($lhs) `,` type($rhs) `)` `->` type($result)
    }];

    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_SelectOp : BaseModelica_Op<"select",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    SameVariadicOperandSize,
    AlwaysSpeculatable]>
{
    let summary = "Ternary operator";

    let description = [{
        The operation is used to select among two group of values according to a
        condition.
    }];

    let arguments = (ins
        AnyType:$condition,
        Variadic<AnyType>:$trueValues,
        Variadic<AnyType>:$falseValues);

    let results = (outs Variadic<AnyType>:$results);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

//===---------------------------------------------------------------------===//
// Built-in operations
//===---------------------------------------------------------------------===//

def BaseModelica_AbsOp : BaseModelica_Op<"abs",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Compute the absolute value.";

    let description = [{
        The operation computes the absolute value of another one.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.abs %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_AcosOp : BaseModelica_Op<"acos",
    [NoMemoryEffect]>
{
    let summary = "Trigonometric inverse cosine function.";

    let description = [{
        The operation computes the inverse cosine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.acos %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_AsinOp : BaseModelica_Op<"asin",
    [NoMemoryEffect]>
{
    let summary = "Trigonometric inverse sine function";

    let description = [{
        The operation computes the inverse sine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.asin %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_AtanOp : BaseModelica_Op<"atan",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Trigonometric inverse tangent function.";

    let description = [{
        The operation computes the inverse tangent of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.atan %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_Atan2Op : BaseModelica_Op<"atan2",
    [NoMemoryEffect]>
{
    let summary = "Four quadrant inverse tangent function.";

    let description = [{
        The operation takes two values y and x and returns the inverse tangent of y/x
        Usually y and x provided in such a form that if v = atan2(y, x), then y = sin(v) and x = cos(v).

        Example:

        ```mlir
        %y = ... : !modelica.real
        %x = ... : !modelica.real
        %v = modelica.atan2 %y, %x : (!modelica.real, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$y, AnyType:$x);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $y `,` $x attr-dict `:` `(` type($y) `,` type($x) `)` `->` type($result)
    }];
}

def BaseModelica_CeilOp : BaseModelica_Op<"ceil",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Ceil function.";

    let description = [{
        The function takes an operand x and returns the smallest integer not less than x.

        Example:

        ```mlir
        %x = ... : !modelica.real
        %0 = modelica.ceil %x : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_CosOp : BaseModelica_Op<"cos",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Trigonometric cosine function.";

    let description = [{
        The operation computes the cosine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.cos %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_CoshOp : BaseModelica_Op<"cosh",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Hyperbolic cosine function.";

    let description = [{
        The operation computes the hyperbolic cosine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.cosh %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_DiagonalOp : BaseModelica_Op<"diagonal",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get a diagonal matrix with the given values";

    let description = [{
        Given an array of values, the operation returns a matrix
        having those elements on the main diagonal.

        Example:

        ```mlir
        $0 = ... : tensor<3x!modelica.real>
        %1 = modelica.diagonal %0 : tensor<3x!modelica.real> -> tensor<3x3x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$values);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $values attr-dict `:` type($values) `->` type($result)
    }];
}

def Modelica_DivTruncOp : BaseModelica_Op<"div_trunc", []>
{
    let summary = "Get the integer part of a quotient.";

    let description = [{
        The operation takes two arguments `x` and `y` and returns
        the algebraic quotient `x / y` with any fractional part discarded.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.div_trunc %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$x, AnyType:$y);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $x `,` $y attr-dict `:` `(` type($x) `,` type($y) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_ExpOp : BaseModelica_Op<"exp",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Base 'e' exponential function.";

    let description = [{
        The function takes an operand x and returns the e^x value.

        Example:

        ```mlir
        %exponent = ... : !modelica.real
        %0 = modelica.exp %exponent : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$exponent);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $exponent attr-dict `:` type($exponent) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_FillOp : BaseModelica_Op<"fill", [NoMemoryEffect]>
{
    let summary = "Create a tensor with a given value in all positions.";

    let description = [{
        Create a tensor with all the elements equal to a given one.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = modelica.fill %0 : !modelica.int -> tensor<3x2x!modelica.real>
        ```
    }];

    let arguments = (ins AnyBaseModelicaCompatibleScalar:$value);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $value attr-dict `:` type($value) `->` type($result)
    }];
}

def BaseModelica_FloorOp : BaseModelica_Op<"floor",
   [AlwaysSpeculatable,
   NoMemoryEffect]>
{
    let summary = "Floor function.";

    let description = [{
        The function takes an operand x and returns the largest integer not greater than x.

        Example:

        ```mlir
        %x = ... : !modelica.real
        %0 = modelica.floor %x : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_IdentityOp : BaseModelica_Op<"identity",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Returns an identity matrix with a given dimensions size.";

    let description = [{
        The operation takes one operand, representing the matrix size,
        and returns an identity matrix.

        Example:

        ```mlir
        %dim = constant 3 : index
        %identity = modelica.identity %dim : index -> tensor<?x?x!modelica.real>
        ```
    }];

    let arguments = (ins AnyType:$size);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $size attr-dict `:` type($size) `->` type($result)
    }];
}

def BaseModelica_IntegerOp : BaseModelica_Op<"integer",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Integer function.";

    let description = [{
        The function takes an operand x and returns the largest integer not greater than x.

        Example:

        ```mlir
        %x = ... : !modelica.real
        %0 = modelica.integer %x : !modelica.real -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_LinspaceOp : BaseModelica_Op<"linspace",
    [NoMemoryEffect]>
{
    let summary = "Get an array with equally spaced values.";

    let description = [{
        The operation creates a tensor with equally spaces values.
        It takes three operands representing the first value that will be present
        in the array, the last value, and the number of elements that will populate
        the tensor. All the elements of the resulting array will be equally spaced
        according to the desired amount of values.
    }];

    let arguments = (ins AnyType:$begin, AnyType:$end, AnyType:$amount);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $begin `,` $end `,` $amount attr-dict `:` `(` type($begin) `,` type($end) `,` type($amount) `)` `->` type($result)
    }];
}

def Modelica_LogOp : BaseModelica_Op<"log",
    [NoMemoryEffect]>
{
    let summary = "Natural (base e) logarithm function.";

    let description = [{
        The operation computes the natural (base 'e') logarithm of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.log %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_Log10Op : BaseModelica_Op<"log10",
    [NoMemoryEffect]>
{
    let summary = "Base 10 logarithm function.";

    let description = [{
        The operation computes the base 10 logarithm of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.log10 %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_MaxOp : BaseModelica_Op<"max",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get the greatest value among an array or two scalar values.";

    let description = [{
        Get the greatest value among an array or two scalar values.

        Example with array value:

        ```mlir
        %0 = ... : !modelica.array<5x!modelica.int>
        %1 = modelica.max %0 : !modelica.array<5x!modelica.int> -> !modelica.int
        ```

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.max %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$first, Optional<AnyType>:$second);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins "Type":$result, "ValueRange":$values), [{
            $_state.addTypes(result);
            $_state.addOperands(values);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_MinOp : BaseModelica_Op<"min",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get the smallest value among an array or two scalar values.";

    let description = [{
        Get the smallest value among an array or two scalar values.

        Example with array value:

        ```mlir
        %0 = ... : !modelica.array<5x!modelica.int>
        %1 = modelica.min %0 : tensor<5x!modelica.int> -> !modelica.int
        ```

        Example with scalar values:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.min %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$first, Optional<AnyType>:$second);
    let results = (outs AnyType:$result);

    let builders = [
        OpBuilder<(ins "Type":$result, "ValueRange":$values), [{
            $_state.addTypes(result);
            $_state.addOperands(values);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;
    let hasFolder = 1;

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def Modelica_ModOp : BaseModelica_Op<"mod", []>
{
    let summary = "Get the modulus of a division of two numbers.";

    let description = [{
        The operation takes two arguments `x` and `y` and returns the integer
        modulus of `x / y`, i.e., `mod(x, y) = x - floor(x / y) * y`.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = ... : !modelica.int
        %2 = modelica.mod %0, %1 : (!modelica.int, !modelica.int) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$x, AnyType:$y);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $x `,` $y attr-dict `:` `(` type($x) `,` type($y) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_NDimsOp : BaseModelica_Op<"ndims", [Pure]>
{
    let summary = "Get the number of dimensions of an array.";

    let description = [{
        The operation takes an array as argument and returns
        the number of its dimensions.

        Example:

        ```mlir
        %0 = ... : tensor<4x?x5x!modelica.int>
        %1 = modelica.ndims %0 : tensor<4x?x5x!modelica.int> -> !modelica.int
        // %1 will have value 3.
        ```
    }];

    let arguments = (ins AnyTensor:$array);
    let results = (outs AnyBaseModelicaCompatibleScalar:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];
}

def BaseModelica_OnesOp : BaseModelica_Op<"ones",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get an array with all the elements set to '1'";

    let description = [{
        Returns an array with "1" elements.
        The dimensions of the array are specified as input.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = constant 3 : index
        %2 = constant 4 : index

        %array = modelica.ones %0, %1, %2 : (index, index, index) -> tensor<?x?x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$sizes);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $sizes attr-dict `:` `(` type($sizes) `)` `->` type($result)
    }];
}

def BaseModelica_ProductOp : BaseModelica_Op<"product",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get the product of the values of an array.";

    let description = [{
        The operation takes an array and computes the product of all its values.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.real>
        %1 = modelica.product %0 : !modelica.array<?x?x!modelica.real> -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$array);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];
}

def Modelica_RemOp : BaseModelica_Op<"rem", []>
{
    let summary = "Get the modulus of a division of two numbers.";

    let description = [{
        The operation takes two arguments `x` and `y` and returns the integer
        remainder of `x / y`, such that `div(x,y) * y + rem(x, y) = x`.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = ... : !modelica.real
        %2 = modelica.rem %0, %1 : (!modelica.real, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$x, AnyType:$y);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $x `,` $y attr-dict `:` `(` type($x) `,` type($y) `)` `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_SignOp : BaseModelica_Op<"sign",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get an integer representing the sign of a value.";

    let description = [{
        The operation takes an operand and returns 1 if its value is greater
        than zero, -1 if less than zero or 0 if equal to zero.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.sign %0 : !modelica.real -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_SinOp : BaseModelica_Op<"sin",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Trigonometric sine function.";

    let description = [{
        The operation computes the sine of a value.
        It takes one operand and returns one result. If the operand is an array,
        then the BaseModelica vectorization rules are applied and an array with the
        same dimensions is returned.
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_SinhOp : BaseModelica_Op<"sinh",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Hyperbolic sine function.";

    let description = [{
        The operation computes the hyperbolic sine of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.sinh %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_SizeOp : BaseModelica_Op<"size",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get the dimensions of an array.";

    let description = [{
        The operation can accept either one or two arguments.

        In the first case the operand represents the array of interest and
        the result consists in an array containing the size of its dimensions.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.int>
        %1 = modelica.size %0 : !modelica.array<?x?x!modelica.int> -> !modelica.array<2x!modelica.int>
        ```

        In the second case the second operand also specifies the (zero-based)
        dimension of interest, and thus the operation returns just a scalar
        value corresponding to the size of that dimension.

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.int>
        %1 = constant 1 : index
        %2 = modelica.size %0, %1 : (!modelica.array<?x?x!modelica.int>, index) -> !modelica.int
        ```
    }];

    let arguments = (ins AnyType:$array, Optional<AnyType>:$dimension);
    let results = (outs AnyType:$result);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        bool hasDimension() {
            return getNumOperands() == 2;
        }
    }];
}

def Modelica_SqrtOp : BaseModelica_Op<"sqrt",
    [NoMemoryEffect]>
{
    let summary = "Square root.";

    let description = [{
        The operation computes the square root of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.sqrt %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_SumOp : BaseModelica_Op<"sum",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get the sum of the values of an array.";

    let description = [{
        The operation takes an array and computes the sum of all its values.

        Example:

        ```mlir
        %0 = ... : !modelica.array<?x?x!modelica.real>
        %1 = modelica.sum %0 : !modelica.array<?x?x!modelica.real> -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$array);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $array attr-dict `:` type($array) `->` type($result)
    }];
}

def BaseModelica_SymmetricOp : BaseModelica_Op<"symmetric",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get a symmetric matrix.";

    let description = [{
        The operation takes a matrix and returns a new symmetric ones that is built
        by considering only the elements above the source diagonal and mirroring
        them into the bottom part.

        Example:

        ```
        %0 = ... : !modelica.array<3x3x!modelica.int>
        %1 = modelica.symmetric %0 : !modelica.array<3x3x!modelica.int> -> !modelica.array<3x3x!modelica.int>

        //      [ 1 2 3 ]           [ 1 2 3 ]
        // %0 = [ 4 5 6 ]  ->  %1 = [ 2 5 6 ]
        //      [ 7 8 9 ]           [ 3 6 9 ]
        ```
    }];

    let arguments = (ins AnyTensor:$matrix);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $matrix attr-dict `:` type($matrix) `->` type($result)
    }];
}

def Modelica_TanOp : BaseModelica_Op<"tan",
    [NoMemoryEffect]>
{
    let summary = "Trigonometric tangent function.";

    let description = [{
        The operation computes the tangent of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.tan %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def BaseModelica_TanhOp : BaseModelica_Op<"tanh",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Hyperbolic tangent function.";

    let description = [{
        The operation computes the hyperbolic tangent of a value.

        Example:

        ```mlir
        %0 = ... : !modelica.real
        %1 = modelica.tanh %0 : !modelica.real -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_TransposeOp : BaseModelica_Op<"transpose",
    [DeclareOpInterfaceMethods<InferTypeOpInterface>,
    AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Matrix transposition.";

    let description = [{
        The operation takes a matrix and computes its transpose.

        Example:

        ```mlir
        %0 = ... : !modelica.array<3x2x!modelica.int>
        %1 = modelica.transpose : !modelica.array<3x2x!modelica.int> -> !modelica.array<2x3x!modelica.int>

        //      [ 1 2 ]
        // %0 = [ 3 4 ]  ->  %1 = [ 1 3 5 ]
        //      [ 5 6 ]           [ 2 4 6 ]
        ```
    }];

    let arguments = (ins AnyTensor:$matrix);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $matrix attr-dict `:` type($matrix) `->` type($result)
    }];

    let extraClassDeclaration = [{
        static bool isCompatibleReturnTypes(
            ::mlir::TypeRange lhs, ::mlir::TypeRange rhs);
    }];
}

def BaseModelica_ZerosOp : BaseModelica_Op<"zeros",
    [AlwaysSpeculatable,
    NoMemoryEffect]>
{
    let summary = "Get an array with all the elements set to '0'";

    let description = [{
        Returns an array with "0" elements.
        The dimensions of the array are specified as input.

        Example:

        ```mlir
        %0 = constant 5 : index
        %1 = constant 3 : index
        %2 = constant 4 : index

        %array = modelica.zeros %0, %1, %2 : (index, index, index) -> !modelica.array<?x?x?x!modelica.int>
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$sizes);
    let results = (outs AnyTensor:$result);

    let assemblyFormat = [{
        $sizes attr-dict `:` `(` type($sizes) `)` `->` type($result)
    }];
}

def Modelica_ReductionOp : BaseModelica_Op<"reduction", [SingleBlock]>
{
    let summary = "Iteration space reduction.";

    let description = [{
        Apply a function on a given iteration space.

        Example:

        ```mlir
        %0 = ... : !modelica.range<index>
        %1 = ... : !modelica.range<index>

        %2 = modelica.reduction max iterables = [%0, %1] inductions = [%i: index, %j: index] {
            %3 = ... : !modelica.real
            modelica.yield %3 : !modelica.real
        } : (!modelica.range<index>, !modelica.range<index>) -> !modelica.real
        ```
    }];

    let arguments = (ins
        StrAttr:$action,
        Variadic<BaseModelicaRange>:$iterables);

    let results = (outs AnyType:$result);

    let regions = (region
        SizedRegion<1>:$expressionRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        mlir::Block* createExpressionBlock(mlir::OpBuilder& builder);

        auto getInductions()
        {
            return getExpressionRegion().getArguments();
        }
    }];
}

//===---------------------------------------------------------------------===//
// Modeling operations
//===---------------------------------------------------------------------===//

def BaseModelica_PackageOp : BaseModelica_Op<"package",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica package.";

    let description = [{
        The operation represents a BaseModelica package.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    // Ensure that the block inside the region exists.
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [OpBuilder<(ins "::llvm::StringRef":$name)>];

    let extraClassDeclaration = [{
        Block* bodyBlock();

        auto getVariables()
        {
            return getOps<VariableOp>();
        }
    }];
}

def BaseModelica_ModelOp : BaseModelica_Op<"model",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica model.";

    let description = [{
        The operation represents a BaseModelica model.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        DerivativesMapProperty:$derivativesMap);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
        $sym_name custom<ModelDerivativesMap>($derivativesMap)
        attr-dict-with-keyword $bodyRegion
    }];

    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        static void getCleaningPatterns(
            mlir::RewritePatternSet& patterns,
            mlir::MLIRContext* context);

        auto getVariables()
        {
            return getOps<VariableOp>();
        }

        /// Collect all the variables.
        void collectVariables(llvm::SmallVectorImpl<VariableOp>& variables);

        /// Collect all the initial equations of a model.
        template<typename Equation>
        void collectInitialEquations(llvm::SmallVectorImpl<Equation>& equations)
        {
            for (InitialOp initialOp : getOps<InitialOp>()) {
                initialOp.collectEquations(equations);
            }
        }

        /// Collect all the non-initial equations of a model.
        template<typename Equation>
        void collectMainEquations(llvm::SmallVectorImpl<Equation>& equations)
        {
            for (DynamicOp dynamicOp : getOps<DynamicOp>()) {
                dynamicOp.collectEquations(equations);
            }
        }

        /// Collect all the initial algorithms of a model.
        void collectInitialAlgorithms(
            llvm::SmallVectorImpl<AlgorithmOp>& algorithms);

        /// Collect all the non-initial algorithms of a model.
        void collectMainAlgorithms(
            llvm::SmallVectorImpl<AlgorithmOp>& algorithms);

        /// Collect all the initial SCCs of a model.
        void collectInitialSCCs(llvm::SmallVectorImpl<SCCOp>& SCCs);

        /// Collect all the non-initial SCCs of a model.
        void collectMainSCCs(llvm::SmallVectorImpl<SCCOp>& SCCs);

        /// Collect all the initial and non-initial SCC groups of a model.
        void collectSCCGroups(
            llvm::SmallVectorImpl<SCCGroupOp>& initialSCCGroups,
            llvm::SmallVectorImpl<SCCGroupOp>& SCCGroups);
    }];
}

def BaseModelica_RecordOp : BaseModelica_Op<"record",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica record.";

    let description = [{
        The operation represents a BaseModelica record.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";

    let extraClassDeclaration = [{
        auto getVariables()
        {
            return getOps<VariableOp>();
        }
    }];
}

def BaseModelica_RecordCreateOp : BaseModelica_Op<"record_create", []>
{
    let summary = "Create record instance.";

    let description = [{
        Create an instance of a record.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $values attr-dict `:` type($values) `->` type($result)
    }];
}

def BaseModelica_OperatorRecordOp : BaseModelica_Op<"operator_record",
    [IsolatedFromAbove,
    NoRegionArguments,
    Symbol,
    SymbolTable,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica operator record.";

    let description = [{
        The operation represents a BaseModelica operator record.
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    // Ensure that the block inside the region exists.
    let hasCustomAssemblyFormat = 1;
    let skipDefaultBuilders = 1;

    let builders = [OpBuilder<(ins "::llvm::StringRef":$name)>];

    let extraClassDeclaration = [{
        Block* bodyBlock();

        auto getVariables()
        {
            return getOps<VariableOp>();
        }
    }];
}

def BaseModelica_StartOp : BaseModelica_Op<"start",
    [ParentOneOf<["ModelOp"]>,
    SingleBlockImplicitTerminator<"YieldOp">]>
{
    let summary = "Start attribute of a variable";

    let description = [{
        The operation describes how to compute the initial value of a variable.
    }];

    let arguments = (ins
        SymbolRefAttr:$variable,
        BoolAttr:$fixed,
        BoolAttr:$each,
        DefaultValuedAttr<BoolAttr, "false">:$implicit);

    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let assemblyFormat = [{
        $variable $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        public:
            VariableOp getVariableOp(mlir::SymbolTableCollection& symbolTable);

            mlir::LogicalResult getAccesses(
                llvm::SmallVectorImpl<VariableAccess>& result,
                mlir::SymbolTableCollection& symbolTable);

        private:
            mlir::LogicalResult searchAccesses(
                llvm::SmallVectorImpl<VariableAccess>& accesses,
                mlir::SymbolTableCollection& symbolTable,
                llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
                mlir::Value value,
                EquationPath path);

        public:
    }];
}

def BaseModelica_DefaultOp : BaseModelica_Op<"default",
    [ParentOneOf<["FunctionOp"]>,
    SingleBlockImplicitTerminator<"YieldOp">]>
{
    let summary = "Default attribute of a variable.";

    let description = [{
        The operation describes how to compute the default value of a variable.
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let assemblyFormat = [{
        $variable $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        VariableOp getVariableOp(mlir::SymbolTableCollection& symbolTable);
    }];
}

def Modelica_DerOp : BaseModelica_Op<"der", []>
{
    let summary = "Derivative.";

    let description = [{
        The derivative operation is used to obtain the derivative of a variable.
    }];

    let arguments = (ins AnyType:$operand);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $operand attr-dict `:` type($operand) `->` type($result)
    }];
}

def Modelica_TimeOp : BaseModelica_Op<"time", []>
{
    let summary = "Time.";

    let description = [{
        The operation represents the time of the simulation.
    }];

    let arguments = (ins);
    let results = (outs BaseModelicaReal);

    let assemblyFormat = [{
        attr-dict `:` type(results)
    }];

    let builders = [
        OpBuilder<(ins), [{
            auto realType = RealType::get($_builder.getContext());
            build($_builder, $_state, realType);
        }]>
    ];
}

def BaseModelica_BindingEquationOp : BaseModelica_Op<"binding_equation",
    [HasParent<"ModelOp">]>
{
    let summary = "Binding equation.";

    let description = [{
        The operation represents an implicit equation whose expression is
        written in form of assignment during the declaration of a variable.
    }];

    let arguments = (ins SymbolNameAttr:$variable);
    let results = (outs);

    let regions = (region
        AnyRegion:$bodyRegion
    );

    let assemblyFormat = [{
        $variable $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        VariableOp getVariableOp(mlir::SymbolTableCollection& symbolTable);
    }];
}

def BaseModelica_ForEquationOp : BaseModelica_Op<"for_equation",
    [ParentOneOf<["DynamicOp", "InitialOp", "ForEquationOp"]>,
    NoTerminator]>
{
    let summary = "Loop defining more equation through a for loop.";

    let description = [{
        The for equation loop is an operation used to declare multiple equations whose
        structure just differ by the indices they depend on. In this sense, the loop is
        not the standard algorithmic loop developers are used to, but rather a modeling
        construct that changes the model structure.
        The iteration starts from the `from` value (included) and ends with the `to`
        value (excluded), with a increment of 'step'.
        Multiple for loops can be nested into each other.

        Example:

        ```mlir
        modelica.for_equation {from = 3, to = 9, step = 2} {
            modelica.equation {
                ...
            }
        }
        ```
    }];

    let arguments = (ins IndexAttr:$from, IndexAttr:$to, IndexAttr:$step);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "long":$from, "long":$to, "long":$step)>
    ];

    let extraClassDeclaration = [{
        Block* bodyBlock();
        Value induction();
    }];
}

def BaseModelica_EquationTemplateOp : BaseModelica_Op<"equation_template",
    [IsolatedFromAbove,
    ParentOneOf<["ModelOp"]>,
    SingleBlock,
    NoMemoryEffect]>
{
    let summary = "Equation template.";

    let description = [{
        A generic template for an equation or an initial equation.
    }];

    let arguments = (ins);
    let results = (outs EquationType:$result);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;
    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins), [{
            auto equationType = EquationType::get($_builder.getContext());
            build($_builder, $_state, equationType);
        }]>
    ];

    let extraClassDeclaration = [{
        public:
            // Create the body block of the operation.
            mlir::Block* createBody(unsigned int numOfInductions);

            void printInline(llvm::raw_ostream& os);

            /// Get the induction variables.
            mlir::ValueRange getInductionVariables();

            /// Get the induction variables available at a certain path.
            /// This list may differ from the one given by the equation itself,
            /// due to operation introducing additional ones (e.g. for
            /// reduction purposes).
            llvm::SmallVector<mlir::Value> getInductionVariablesAtPath(
                const EquationPath& path);

            /// Get the accesses to variables performed by the equation.
            mlir::LogicalResult getAccesses(
                llvm::SmallVectorImpl<VariableAccess>& result,
                mlir::SymbolTableCollection& symbolTable);

            /// Get the write accesses.
            mlir::LogicalResult getWriteAccesses(
                llvm::SmallVectorImpl<VariableAccess>& result,
                const IndexSet& equationIndices,
                llvm::ArrayRef<VariableAccess> accesses,
                const VariableAccess& matchedAccess);

            /// Get the read accesses.
            mlir::LogicalResult getReadAccesses(
                llvm::SmallVectorImpl<VariableAccess>& result,
                const IndexSet& equationIndices,
                llvm::ArrayRef<VariableAccess> accesses,
                const VariableAccess& matchedAccess);

            /// Get the IR value at a given path.
            mlir::Value getValueAtPath(const EquationPath& path);

            /// Get the access at a given path.
            std::optional<VariableAccess> getAccessAtPath(
                mlir::SymbolTableCollection& symbolTable,
                const EquationPath& path);

            IndexSet applyAccessFunction(
                const AccessFunction& accessFunction,
                std::optional<MultidimensionalRange> equationIndices,
                const EquationPath& path);

            mlir::LogicalResult cloneWithReplacedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<
                    std::reference_wrapper<const IndexSet>> equationIndices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results);

            std::unique_ptr<AccessFunction> getReplacementTransformationAccess(
                const AccessFunction& destinationAccess,
                const AccessFunction& sourceAccess);

            mlir::LogicalResult explicitate(
                mlir::RewriterBase& rewriter,
                mlir::SymbolTableCollection& symbolTableCollection,
                std::optional<MultidimensionalRange> equationIndices,
                const EquationPath& path);

            EquationTemplateOp cloneAndExplicitate(
                mlir::RewriterBase& rewriter,
                mlir::SymbolTableCollection& symbolTableCollection,
                std::optional<MultidimensionalRange> equationIndices,
                const EquationPath& path);

        private:
            llvm::DenseMap<mlir::Value, unsigned int> getInductionsPositionMap();

            mlir::LogicalResult searchAccesses(
                llvm::SmallVectorImpl<VariableAccess>& accesses,
                mlir::SymbolTableCollection& symbolTableCollection,
                llvm::DenseMap<mlir::Value, unsigned int>& inductionsPositionMap,
                mlir::Value value,
                EquationPath path);

            mlir::LogicalResult cloneWithReplacedVectorizedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<
                    std::reference_wrapper<const IndexSet>> equationIndices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                llvm::ArrayRef<mlir::Value> additionalSubscriptions,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results,
                IndexSet& remainingEquationIndices);

            mlir::LogicalResult cloneWithReplacedVectorizedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<
                    std::reference_wrapper<const IndexSet>> equationIndices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                const AccessFunction& transformation,
                llvm::ArrayRef<mlir::Value> additionalSubscriptions,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results,
                IndexSet& remainingEquationIndices);

            mlir::LogicalResult cloneWithReplacedVectorizedAccess(
                mlir::RewriterBase& rewriter,
                std::optional<std::reference_wrapper<
                    const MultidimensionalRange>> indices,
                const VariableAccess& access,
                EquationTemplateOp replacementEquation,
                const VariableAccess& replacementAccess,
                const AccessFunction& transformation,
                llvm::ArrayRef<mlir::Value> additionalSubscriptions,
                llvm::SmallVectorImpl<
                    std::pair<IndexSet, EquationTemplateOp>>& results,
                IndexSet& remainingEquationIndices);

            mlir::LogicalResult mapInductionVariables(
                mlir::OpBuilder& builder,
                mlir::Location loc,
                mlir::IRMapping& mapping,
                EquationTemplateOp source,
                EquationTemplateOp destination,
                const EquationPath& destinationPath,
                const AccessFunction& transformation);

            mlir::LogicalResult explicitateLeaf(
                mlir::RewriterBase& rewriter,
                size_t argumentIndex,
                EquationPath::EquationSide side);

            mlir::LogicalResult groupLeftHandSide(
                mlir::RewriterBase& rewriter,
                mlir::SymbolTableCollection& symbolTableCollection,
                std::optional<MultidimensionalRange> equationRanges,
                const VariableAccess& requestedAccess);

            bool checkAccessEquivalence(
                const IndexSet& equationIndices,
                const VariableAccess& firstAccess,
                const VariableAccess& secondAccess);

            mlir::LogicalResult collectSummedValues(
                llvm::SmallVectorImpl<std::pair<mlir::Value, EquationPath>>& result,
                mlir::Value root,
                EquationPath path);

            std::optional<std::pair<unsigned int, mlir::Value>>
            getMultiplyingFactor(
                mlir::OpBuilder& builder,
                mlir::SymbolTableCollection& symbolTableCollection,
                llvm::DenseMap<
                    mlir::Value, unsigned int>& inductionsPositionMap,
                const IndexSet& equationIndices,
                mlir::Value value,
                llvm::StringRef variable,
                const IndexSet& variableIndices,
                EquationPath path);

        public:
    }];
}

def BaseModelica_EquationFunctionOp : BaseModelica_Op<"equation_function",
    [HasParent<"mlir::ModuleOp">,
    IsolatedFromAbove,
    AffineScope,
    AutomaticAllocationScope,
    Symbol,
    FunctionOpInterface,
    CallableOpInterface]>
{
    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<StrAttr>:$sym_visibility,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name,
            "uint64_t":$numOfInductions,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
            CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
    ];

    let extraClassDeclaration = [{
        /// @name CallableOpInterface methods
        /// {

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name FunctionOpInterface methods
        /// {

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name SymbolOpInterface methods
        /// {

        bool isDeclaration()
        {
            return isExternal();
        }

        /// }

        mlir::Value getLowerBound(uint64_t induction);
        mlir::Value getUpperBound(uint64_t induction);
    }];
}

def BaseModelica_EquationCallOp : BaseModelica_Op<"equation_call", []>
{
    let arguments = (ins
        SymbolNameAttr:$callee,
        OptionalAttr<MultidimensionalRangeAttr>:$indices,
        DefaultValuedAttr<BoolAttr, "false">:$parallelizable);

    let results = (outs);
    let assemblyFormat = "$callee attr-dict";
}

def BaseModelica_InitialOp : BaseModelica_Op<"initial",
    [ParentOneOf<["ModelOp", "ScheduleOp"]>,
    SingleBlock,
    NoTerminator]>
{
    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$bodyRegion attr-dict";
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        template<typename Equation>
        void collectEquations(llvm::SmallVectorImpl<Equation>& equations)
        {
            walk([&](Equation equation) {
                equations.push_back(equation);
            });
        }

        void collectSCCs(llvm::SmallVectorImpl<SCCOp>& SCCs);

        void collectAlgorithms(llvm::SmallVectorImpl<AlgorithmOp>& algorithms);
    }];
}

def BaseModelica_DynamicOp : BaseModelica_Op<"dynamic",
    [ParentOneOf<["ModelOp", "ScheduleOp"]>,
    SingleBlock,
    NoTerminator]>
{
    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$bodyRegion attr-dict";
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        template<typename Equation>
        void collectEquations(llvm::SmallVectorImpl<Equation>& equations)
        {
            walk([&](Equation equation) {
                equations.push_back(equation);
            });
        }

        void collectSCCs(llvm::SmallVectorImpl<SCCOp>& SCCs);

        void collectAlgorithms(llvm::SmallVectorImpl<AlgorithmOp>& algorithms);
    }];
}

def BaseModelica_AbstractEquationOp : BaseModelica_Op<"abstract_equation",
    [ParentOneOf<["SCCOp"]>,
    SingleBlock,
    NoTerminator]>
{
    let arguments = (ins
        VariablesListProperty:$writtenVariables,
        VariablesListProperty:$readVariables);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
        custom<AbstractEquationWrittenVars>($writtenVariables) `,`
        custom<AbstractEquationReadVars>($readVariables)
        $bodyRegion attr-dict
    }];
}

def BaseModelica_StartEquationInstanceOp : BaseModelica_Op<"start_equation_instance",
    [ParentOneOf<["InitialOp"]>,
    DeclareOpInterfaceMethods<EquationInstanceInterface>]>
{
    let arguments = (ins
        BaseModelicaEquation:$base,
        OptionalAttr<MultidimensionalRangeAttr>:$indices);

    let results = (outs);

    let assemblyFormat = [{
        $base attr-dict `:` type($base)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "EquationTemplateOp":$equationTemplate)>
    ];

    let extraClassDeclaration = [{
        /// Get the equation template.
        EquationTemplateOp getTemplate();

        /// @name Forwarded methods.
        /// {

        void printInline(llvm::raw_ostream& os);

        mlir::ValueRange getInductionVariables();

        mlir::LogicalResult getAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTable);

        std::optional<VariableAccess> getAccessAtPath(
            mlir::SymbolTableCollection& symbolTable,
            const EquationPath& path);

        /// }

        std::optional<VariableAccess> getWriteAccess(
            mlir::SymbolTableCollection& symbolTableCollection);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);
    }];
}

def BaseModelica_EquationInstanceOp : BaseModelica_Op<"equation_instance",
    [ParentOneOf<["InitialOp", "DynamicOp"]>,
    DeclareOpInterfaceMethods<EquationInstanceInterface>]>
{
    let arguments = (ins
        BaseModelicaEquation:$base,
        OptionalAttr<MultidimensionalRangeAttr>:$indices);

    let results = (outs);

    let assemblyFormat = [{
        $base attr-dict `:` type($base)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "EquationTemplateOp":$equationTemplate)>,
    ];

    let extraClassDeclaration = [{
        /// Get the equation template.
        EquationTemplateOp getTemplate();

        /// @name Forwarded methods.
        /// {

        void printInline(llvm::raw_ostream& os);

        mlir::ValueRange getInductionVariables();

        mlir::LogicalResult getAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTable);

        std::optional<VariableAccess> getAccessAtPath(
            mlir::SymbolTableCollection& symbolTable,
            const EquationPath& path);

        /// }

        mlir::LogicalResult cloneWithReplacedAccess(
            mlir::RewriterBase& rewriter,
            std::optional<
                std::reference_wrapper<const IndexSet>> equationIndices,
            const VariableAccess& access,
            EquationTemplateOp replacementEquation,
            const VariableAccess& replacementAccess,
            llvm::SmallVectorImpl<EquationInstanceOp>& results);
    }];
}

def BaseModelica_MatchedEquationInstanceOp : BaseModelica_Op<"matched_equation_instance",
    [ParentOneOf<["InitialOp", "DynamicOp", "SCCOp"]>,
    DeclareOpInterfaceMethods<EquationInstanceInterface>]>
{
    let arguments = (ins
        BaseModelicaEquation:$base,
        OptionalAttr<MultidimensionalRangeAttr>:$indices,
        EquationPathAttr:$path);

    let results = (outs);

    let assemblyFormat = [{
        $base attr-dict `:` type($base)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "EquationTemplateOp":$equationTemplate,
                       "EquationPathAttr":$path)>
    ];

    let extraClassDeclaration = [{
        /// Get the equation template.
        EquationTemplateOp getTemplate();

        /// @name Forwarded methods.
        /// {

        void printInline(llvm::raw_ostream& os);

        mlir::ValueRange getInductionVariables();

        mlir::LogicalResult getAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTable);

        std::optional<VariableAccess> getAccessAtPath(
            mlir::SymbolTableCollection& symbolTable,
            const EquationPath& path);

        /// }

        std::optional<VariableAccess> getMatchedAccess(
            mlir::SymbolTableCollection& symbolTableCollection);

       mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult explicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        MatchedEquationInstanceOp cloneAndExplicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        mlir::LogicalResult cloneWithReplacedAccess(
            mlir::RewriterBase& rewriter,
            std::optional<
                std::reference_wrapper<const IndexSet>> equationIndices,
            const VariableAccess& access,
            EquationTemplateOp replacementEquation,
            const VariableAccess& replacementAccess,
            llvm::SmallVectorImpl<MatchedEquationInstanceOp>& results);
    }];
}

def BaseModelica_SCCGroupOp : BaseModelica_Op<"scc_group",
    [ParentOneOf<["ModelOp", "ScheduleOp", "InitialOp", "DynamicOp"]>,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    SingleBlock,
    NoTerminator]>
{
    let arguments = (ins);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = [{
        $bodyRegion attr-dict
    }];

    let extraClassDeclaration = [{
        void collectSCCs(llvm::SmallVectorImpl<SCCOp>& SCCs);
    }];
}

def BaseModelica_SCCOp : BaseModelica_Op<"scc",
    [ParentOneOf<["InitialOp", "DynamicOp", "ScheduleOp", "SCCGroupOp"]>,
    DeclareOpInterfaceMethods<RegionKindInterface>,
    SingleBlock,
    NoTerminator]>
{
    let arguments = (ins
        DefaultValuedAttr<BoolAttr, "false">:$cycle);

    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "$bodyRegion attr-dict";
    let hasCanonicalizer = 1;

    let extraClassDeclaration = [{
        template<typename Equation>
        void collectEquations(llvm::SmallVectorImpl<Equation>& equations)
        {
            walk([&](Equation equation) {
                equations.push_back(equation);
            });
        }
    }];
}

def BaseModelica_ScheduledEquationInstanceOp : BaseModelica_Op<"scheduled_equation_instance",
    [ParentOneOf<["SCCOp"]>,
    DeclareOpInterfaceMethods<EquationInstanceInterface>]>
{
    let arguments = (ins
        BaseModelicaEquation:$base,
        OptionalAttr<MultidimensionalRangeAttr>:$indices,
        EquationPathAttr:$path,
        TypedArrayAttrBase<
            EquationScheduleDirectionAttr,
            "Schedule iteration directions">:$iteration_directions);

    let results = (outs);

    let assemblyFormat = [{
        $base attr-dict `:` type($base)
    }];

    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "EquationTemplateOp":$equationTemplate,
                       "EquationPathAttr":$path,
                       "mlir::ArrayAttr":$iterationDirections)>
    ];

    let extraClassDeclaration = [{
        /// Get the equation template.
        EquationTemplateOp getTemplate();

        /// @name Forwarded methods.
        /// {

        void printInline(llvm::raw_ostream& os);

        mlir::ValueRange getInductionVariables();

        mlir::LogicalResult getAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTable);

        std::optional<VariableAccess> getAccessAtPath(
            mlir::SymbolTableCollection& symbolTable,
            const EquationPath& path);

        /// }

        std::optional<VariableAccess> getMatchedAccess(
            mlir::SymbolTableCollection& symbolTableCollection);

       mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getWriteAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult getReadAccesses(
            llvm::SmallVectorImpl<VariableAccess>& result,
            mlir::SymbolTableCollection& symbolTableCollection,
            const IndexSet& equationIndices,
            llvm::ArrayRef<VariableAccess> accesses);

        mlir::LogicalResult explicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        ScheduledEquationInstanceOp cloneAndExplicitate(
            mlir::RewriterBase& rewriter,
            mlir::SymbolTableCollection& symbolTableCollection);

        mlir::LogicalResult cloneWithReplacedAccess(
            mlir::RewriterBase& rewriter,
            std::optional<
                std::reference_wrapper<const IndexSet>> equationIndices,
            const VariableAccess& access,
            EquationTemplateOp replacementEquation,
            const VariableAccess& replacementAccess,
            llvm::SmallVectorImpl<ScheduledEquationInstanceOp>& results);
    }];
}

def BaseModelica_EquationOp : BaseModelica_Op<"equation",
    [ParentOneOf<["DynamicOp", "InitialOp", "ForEquationOp"]>,
    SingleBlock]>
{
    let summary = "Equation.";

    let description = [{
        The operation represents a BaseModelica equation.
    }];

    let arguments = (ins);
    let results = (outs);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
}

def BaseModelica_EquationSideOp : BaseModelica_Op<"equation_side",
    [ParentOneOf<["EquationTemplateOp", "EquationOp"]>]>
{
    let summary = "Equation side.";

    let description = [{
        The operation represents one of the sides (left or right)
        of an equation.
    }];

    let arguments = (ins NonEmptyVariadic<AnyType>:$values);
    let results = (outs AnyTuple);

    let hasCustomAssemblyFormat = 1;

    let hasCanonicalizer = 1;

    let builders = [
        OpBuilder<(ins "ValueRange":$values), [{
            auto tupleType = $_builder.getTupleType(values.getTypes());
            build($_builder, $_state, tupleType, values);
        }]>
    ];
}

def BaseModelica_EquationSidesOp : BaseModelica_Op<"equation_sides",
    [ParentOneOf<["EquationTemplateOp", "EquationOp"]>,
    Terminator]>
{
    let summary = "Equation terminator.";

    let description = [{
        The operation represents the equality between two equation sides of an equation.
        It is intended to be used as terminator for the body of an equation.

        Example:

        ```mlir
        modelica.equation {
            %0 = modelica.equation_side ... : tuple<!modelica.real>
            %1 = modelica.equation_side ... : tuple<!modelica.real>
            modelica.equation_sides %0, %1 : tuple<!modelica.real>, tuple<!modelica.real>
        }
        ```
    }];

    let arguments = (ins AnyTuple:$lhs, AnyTuple:$rhs);
    let results = (outs);
    let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs)";
    let hasVerifier = 1;

    let extraClassDeclaration = [{
        mlir::ValueRange getLhsValues() {
            auto sideOp = getLhs().getDefiningOp();
            assert(sideOp != nullptr && mlir::isa<EquationSideOp>(sideOp));
            return mlir::cast<EquationSideOp>(sideOp).getValues();
        }

        mlir::ValueRange getRhsValues() {
            auto sideOp = getRhs().getDefiningOp();
            assert(sideOp != nullptr && mlir::isa<EquationSideOp>(sideOp));
            return mlir::cast<EquationSideOp>(sideOp).getValues();
        }
    }];
}

def BaseModelica_AlgorithmOp : BaseModelica_Op<"algorithm",
    [IsolatedFromAbove,
    ParentOneOf<["DynamicOp", "InitialOp", "FunctionOp"]>,
    SingleBlock,
    NoTerminator]>
{
    let summary = "Algorithm.";

    let description = [{
        The operation represents a BaseModelica algorithm.
    }];

    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$bodyRegion
    );

    let assemblyFormat = "attr-dict-with-keyword $bodyRegion";
}

def BaseModelica_FunctionOp : BaseModelica_Op<"function",
    [IsolatedFromAbove,
    NoRegionArguments,
    AutomaticAllocationScope,
    Symbol,
    SymbolTable,
    SingleBlock,
    NoTerminator,
    DeclareOpInterfaceMethods<ClassInterface>]>
{
    let summary = "Modelica function.";

    let description = [{
        The operation represents a BaseModelica function.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        OptionalAttr<FunctionDerivativeAttr>:$derivative,
        DefaultValuedAttr<I64Attr, "0">:$timeDerivativeOrder);

    let regions = (region AnyRegion:$bodyRegion);
    let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";

    let builders = [
        OpBuilder<(ins "llvm::StringRef":$name)>
    ];

    let extraClassDeclaration = [{
        auto getVariables()
        {
            return getOps<VariableOp>();
        }

        auto getDefaultValues()
        {
            return getOps<DefaultOp>();
        }

        bool shouldBeInlined();

        /// Returns the argument types of this function.
        llvm::SmallVector<Type> getArgumentTypes();

        /// Returns the result types of this function.
        llvm::SmallVector<Type> getResultTypes();

        mlir::FunctionType getFunctionType();

        /// @name SymbolOpInterface Methods
        /// {

        bool isDeclaration()
        {
            return false;
        }

        /// }
    }];
}

def BaseModelica_DerFunctionOp : BaseModelica_Op<"der_function",
    [IsolatedFromAbove, Symbol]>
{
    let summary = "Modelica derivative function.";

    let description = [{
        The operation represents the derivative of a BaseModelica function.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        SymbolRefAttr:$derivedFunction,
        StrArrayAttr:$independentVars);

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        /// @name SymbolOpInterface Methods
        /// {

        bool isDeclaration() {
            return false;
        }

        /// }
    }];
}

def BaseModelica_RawFunctionOp : BaseModelica_Op<"raw_function",
    [IsolatedFromAbove,
    AutomaticAllocationScope,
    Symbol,
    FunctionOpInterface,
    CallableOpInterface]>
{
    let summary = "Raw function.";

    let description = [{
        A raw BaseModelica function consists in a BaseModelica function that has been converted
        to explicit control flow. It is intendend to be used only internally during the
        lowering process.
    }];

    let arguments = (ins
        SymbolNameAttr:$sym_name,
        TypeAttrOf<FunctionType>:$function_type,
        OptionalAttr<StrAttr>:$sym_visibility,
        OptionalAttr<DictArrayAttr>:$arg_attrs,
        OptionalAttr<DictArrayAttr>:$res_attrs);

    let regions = (region AnyRegion:$body);

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins
            "llvm::StringRef":$name, "mlir::FunctionType":$type,
            CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
            CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
    ];

    let extraClassDeclaration = [{
        bool shouldBeInlined();

        static RawFunctionOp create(
            mlir::Location location, llvm::StringRef name,
            mlir::FunctionType type,
            llvm::ArrayRef<mlir::NamedAttribute> attrs = {});

        static RawFunctionOp create(
            mlir::Location location, llvm::StringRef name,
            mlir::FunctionType type,
            mlir::Operation::dialect_attr_range attrs);

        static RawFunctionOp create(
            mlir::Location location, llvm::StringRef name,
            mlir::FunctionType type,
            llvm::ArrayRef<mlir::NamedAttribute> attrs,
            llvm::ArrayRef<mlir::DictionaryAttr> argAttrs);

        /// Create a deep copy of this function and all of its blocks, remapping any
        /// operands that use values outside of the function using the map that is
        /// provided (leaving them alone if no entry is present). If the mapper
        /// contains entries for function arguments, these arguments are not
        /// included in the new function. Replaces references to cloned sub-values
        /// with the corresponding value that is copied, and adds those mappings to
        /// the mapper.
        RawFunctionOp clone(mlir::IRMapping &mapper);
        RawFunctionOp clone();

        /// Clone the internal blocks and attributes from this function into dest.
        /// Any cloned blocks are appended to the back of dest. This function
        /// asserts that the attributes of the current function and dest are
        /// compatible.
        void cloneInto(RawFunctionOp dest, mlir::IRMapping &mapper);

        /// @name CallableOpInterface methods
        /// {

        /// Returns the region on the current operation that is callable. This may
        /// return null in the case of an external callable object, e.g. an external
        /// function.
        mlir::Region* getCallableRegion()
        {
            return isExternal() ? nullptr : &getBody();
        }

        /// Returns the results types that the callable region produces when
        /// executed.
        llvm::ArrayRef<mlir::Type> getCallableResults()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name FunctionOpInterface methods
        /// {

        /// Returns the argument types of this function.
        llvm::ArrayRef<mlir::Type> getArgumentTypes()
        {
            return getFunctionType().getInputs();
        }

        /// Returns the result types of this function.
        llvm::ArrayRef<mlir::Type> getResultTypes()
        {
            return getFunctionType().getResults();
        }

        /// }
        /// @name SymbolOpInterface methods
        /// {

        bool isDeclaration() {
            return isExternal();
        }

        /// }
    }];
}

def RawReturnOp : BaseModelica_Op<"raw_return",
    [NoMemoryEffect,
    HasParent<"RawFunctionOp">,
    ReturnLike, Terminator]>
{
    let summary = "Raw function return operation.";

    let description = [{
        The `modelica.return` operation represents a return operation within a raw
        function. The operation takes variable number of operands and produces no
        results. The operand number and types must match the signature of the
        function that contains the operation.

        Example:

        ```mlir
        modelica.raw_func @foo() : (!modelica.int, !modelica.real) {
            ...
            modelica.raw_return %0, %1 : !modelica.int, !modelica.real
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$operands);
    let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
    let hasVerifier = 1;

    let builders = [OpBuilder<(ins), [{
        build($_builder, $_state, std::nullopt);
    }]>];
}

def Modelica_RawVariableOp : BaseModelica_Op<"raw_variable",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Declare a raw variable of a class.";

    let description = [{
        Declare a raw variable within a class.
        This is used to abandon the BaseModelica-style declaration of variables and
        progressively move towards more traditional argument and return-based
        functions.
    }];

    let arguments = (ins
        StrAttr:$name,
        DefaultValuedAttr<StrArrayAttr, "{}">:$dimensionsConstraints,
        Variadic<Index>:$dynamicSizes,
        UnitAttr:$output,
        DefaultValuedAttr<BoolAttr, "true">:$heap);

    let results = (outs AnyTypeOf<[AnyTensor, AnyMemRef]>:$variable);

    let assemblyFormat = "$dynamicSizes attr-dict `:` type($variable)";

    let extraClassDeclaration = [{
        static constexpr llvm::StringLiteral kDimensionConstraintUnbounded = "unbounded";
        static constexpr llvm::StringLiteral kDimensionConstraintFixed = "fixed";

        static bool isScalarVariable(mlir::Type variableType);
        static bool isStaticArrayVariable(mlir::Type variableType);
        static bool isDynamicArrayVariable(mlir::Type variableType);

        bool isScalarVariable();
        bool isStaticArrayVariable();
        bool isDynamicArrayVariable();

        bool isProtected();

        bool isOutput();
    }];
}

def Modelica_RawVariableDeallocOp : BaseModelica_Op<"raw_variable_dealloc",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Deallocate the data held by a raw variable.";

    let description = [{
        Deallocate the data held by a raw variable.
    }];

    let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$variable);
    let results = (outs);
    let assemblyFormat = "$variable attr-dict `:` type($variable)";

    let extraClassDeclaration = [{
        bool isScalarVariable();
        bool isStaticArrayVariable();
        bool isDynamicArrayVariable();
    }];
}

def Modelica_RawVariableGetOp : BaseModelica_Op<"raw_variable_get",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Load the data held by a raw variable.";

    let description = [{
        Load the data held by a raw variable.
    }];

    let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$variable);
    let results = (outs AnyType:$result);
    let assemblyFormat = "$variable attr-dict `:` type($variable) `->` type($result)";

    let builders = [
        OpBuilder<(ins "RawVariableOp":$variableOp), [{
            mlir::Type variableType = variableOp.getVariable().getType();
            mlir::Type resultType = computeResultType(variableType);
            build($_builder, $_state, resultType, variableOp);
        }]>
    ];

    let extraClassDeclaration = [{
        static mlir::Type computeResultType(mlir::Type rawVariableType);

        bool isScalarVariable();
        bool isStaticArrayVariable();
        bool isDynamicArrayVariable();
    }];
}

def Modelica_RawVariableSetOp : BaseModelica_Op<"raw_variable_set",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Store data into a raw variable.";

    let description = [{
        Store data into a raw variable of a class.
    }];

    let arguments = (ins AnyTypeOf<[AnyTensor, AnyMemRef]>:$variable, AnyType:$value);
    let results = (outs);
    let assemblyFormat = "$variable `,` $value attr-dict `:` type($variable) `,` type($value)";

    let extraClassDeclaration = [{
        bool isScalarVariable();
        bool isStaticArrayVariable();
        bool isDynamicArrayVariable();
    }];
}

def BaseModelica_CallOp : BaseModelica_Op<"call",
    [CallOpInterface,
    DeclareOpInterfaceMethods<SymbolUserOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]>
{
    let summary = "Call operation.";

    let description = [{
        The operation represents a direct call to a user-defined BaseModelica function
        that is within the same symbol scope as the call. The operands and result
        types of the call must match the specified function type. The callee is
        encoded as a symbol reference attribute named "callee".

        Example:

        ```mlir
        %2 = modelica.call @foo(%0, %1) : (!modelica.int, !modelica.real) -> !modelica.real
        ```
    }];

    let arguments = (ins
        SymbolRefAttr:$callee,
        Variadic<AnyType>:$args,
        OptionalAttr<FlatSymbolRefArrayAttr>:$argNames);

    let results = (outs Variadic<AnyType>);

    let assemblyFormat = [{
        $callee `(` $args `)` ($argNames^)? attr-dict `:` functional-type($args, results)
    }];

    let skipDefaultBuilders = 1;

    let builders = [
        OpBuilder<(ins
            "FunctionOp":$callee,
            CArg<"mlir::ValueRange", "{}">:$args,
            CArg<"std::optional<mlir::ArrayAttr>", "std::nullopt">:$argNames)>,
        OpBuilder<(ins
            "RawFunctionOp":$callee,
            CArg<"mlir::ValueRange", "{}">:$args)>,
        OpBuilder<(ins
            "EquationFunctionOp":$callee,
            CArg<"mlir::ValueRange", "{}">:$args)>,
        OpBuilder<(ins
            "mlir::SymbolRefAttr":$callee,
            "mlir::TypeRange":$results,
            CArg<"mlir::ValueRange", "{}">:$args,
            CArg<"std::optional<mlir::ArrayAttr>", "std::nullopt">:$argNames)>
    ];

    let extraClassDeclaration = [{
        /// @name CallOpInterface
        /// {

        mlir::FunctionType getCalleeType();

        /// Get the argument operands to the called function.
        operand_range getArgOperands()
        {
            return { arg_operand_begin(), arg_operand_end() };
        }

        MutableOperandRange getArgOperandsMutable()
        {
            return getArgsMutable();
        }

        operand_iterator arg_operand_begin()
        {
            return operand_begin();
        }

        operand_iterator arg_operand_end()
        {
            return operand_end();
        }

        /// Return the callee of this operation.
        mlir::CallInterfaceCallable getCallableForCallee()
        {
            return (*this)->getAttrOfType<SymbolRefAttr>("callee");
        }

        /// Set the callee for this operation.
        void setCalleeFromCallable(CallInterfaceCallable callee)
        {
            (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
        }

        /// }

        mlir::Operation* getFunction(
            mlir::ModuleOp moduleOp, mlir::SymbolTableCollection& symbolTable);
    }];
}

def BaseModelica_ScheduleOp : BaseModelica_Op<"schedule",
    [ParentOneOf<["ModelOp"]>,
    Symbol,
    SingleBlock,
    NoTerminator]>
{
    let summary = "List of assignments derived from equations.";

    let description = [{
        A schedule is used to list the assignments used to update the variables
        of a model.
        Its content varies according to the point within the compilation
        pipeline in which the operation is used. It may contain yet unscheduled
        SCCs of matched equation, their scheduled version or even more opaque
        representations.

        Examples:

        ```mlir
        modelica.schedule @schedule_0 {
            modelica.scc {
                modelica.matched_equation_instance %template_0 : !modelica.equation
            }
            modelica.scc {
                modelica.matched_equation_instance %template_1 : !modelica.equation
            }
        }

        modelica.schedule @schedule_1 {
            modelica.scc {
                modelica.scheduled_equation_instance %template_1 {...} : !modelica.equation
            }
            modelica.scc {
                modelica.scheduled_equation_instance %template_0 {...} : !modelica.equation
            }
        }
        ```
    }];

    let arguments = (ins SymbolNameAttr:$sym_name);
    let results = (outs);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$sym_name attr-dict-with-keyword $bodyRegion";

    let extraClassDeclaration = [{
        /// Collect the SCC groups.
        void collectSCCGroups(llvm::SmallVectorImpl<SCCGroupOp>& SCCGroups);

        /// Collect the SCCs.
        void collectSCCs(llvm::SmallVectorImpl<SCCOp>& SCCs);
    }];
}

def BaseModelica_ScheduleBlockOp : BaseModelica_Op<"schedule_block",
    [ParentOneOf<["InitialOp", "DynamicOp", "ParallelScheduleBlocksOp"]>,
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove]>
{
    let summary = "Opaque block of code writing and reading some variables.";

    let description = [{
        The operation represents an opaque block of code writing and reading to
        some variables. It is used as a common ground for the operations that
        can live inside a schedule, and on which some generic optimizations may
        be carried out.
        The `parallelizable` attribute indicates whether the block may be
        grouped with some other parallelizable ones.
    }];

    let arguments = (ins
        DefaultValuedAttr<BoolAttr, "false">:$parallelizable,
        VariablesListProperty:$writtenVariables,
        VariablesListProperty:$readVariables);

    let results = (outs);
    let regions = (region SizedRegion<1>:$bodyRegion);

    let assemblyFormat = [{
        custom<ScheduleBlockWrittenVars>($writtenVariables) `,`
        custom<ScheduleBlockReadVars>($readVariables)
        $bodyRegion attr-dict
    }];
}

def BaseModelica_ParallelScheduleBlocksOp : BaseModelica_Op<"parallel_schedule_blocks",
    [ParentOneOf<["InitialOp", "DynamicOp"]>,
    SingleBlock,
    NoTerminator,
    IsolatedFromAbove]>
{
    let summary = "Group of parallel schedule blocks.";

    let description = [{
        The operation represents a group of schedule blocks that can be
        executed independently.

        Example:

        ```mlir
        modelica.parallel_schedule_blocks {
            modelica.schedule_block {
                ...
            }
            modelica.schedule_block {
                ...
            }
        }

        modelica.run_schedule @schedule
        ```
    }];

    let arguments = (ins);
    let results = (outs);
    let regions = (region SizedRegion<1>:$bodyRegion);
    let assemblyFormat = "$bodyRegion attr-dict";
}

def BaseModelica_RunScheduleOp : BaseModelica_Op<"run_schedule",
    [DeclareOpInterfaceMethods<SymbolUserOpInterface>]>
{
    let summary = "Run a schedule.";

    let description = [{
        Run a schedule, similarly to how a call to a function would behave.
        The schedule is referenced starting from the parent module.

        Example:

        ```mlir
        modelica.schedule @schedule {
            ...
        }

        modelica.run_schedule @schedule
        ```
    }];

    let arguments = (ins SymbolRefAttr:$schedule);
    let results = (outs);
    let assemblyFormat = "$schedule attr-dict";

    let builders = [
        OpBuilder<(ins "ScheduleOp":$scheduleOp)>
    ];

    let extraClassDeclaration = [{
        ScheduleOp getScheduleOp();

        ScheduleOp getScheduleOp(
            ::mlir::SymbolTableCollection& symbolTableCollection);
    }];
}

//===---------------------------------------------------------------------===//
// Control flow operations
//===---------------------------------------------------------------------===//

def BaseModelica_BreakOp : BaseModelica_Op<"break", [Terminator]>
{
    let summary = "Loop breaking operation.";

    let description = [{
        The `modelica.break` operation terminates the execution of the closest
        loop that contains the instruction.

        Example:

        ```mlir
        modelica.while {
            ...
        } do {
            %condition = ... : !modelica.bool

            modelica.if %condition {
                modelica.break
            }

            modelica.yield
        }
        ```
    }];

    let arguments = (ins);
    let results = (outs);
    let assemblyFormat = "attr-dict";
}

def BaseModelica_ReturnOp : BaseModelica_Op<"return", [Terminator]>
{
    let summary = "Early termination instruction for functions.";

    let description = [{
        The `modelica.return` operation terminates the execution of the whole
        function, making it returning the values as computed so far.

        Example:

        ```mlir
        modelica.function @foo ... {
            %condition = ... : !modelica.bool

            modelica.if %condition {
                modelica.return
            }
            ...
        }
        ```
    }];

    let arguments = (ins);
    let results = (outs);
    let assemblyFormat = "attr-dict";
}

def Modelica_IfOp : BaseModelica_Op<"if", [NoTerminator]>
{
    let summary = "Algorithmic if-else construct.";

    let description = [{
        The operation takes a value representing the condition and executes the
        code in the "then" region; if not, it executes the code in the optional
        "else" region.
        The value used as condition must be convertible to `!modelica.bool`.

        Example:

        ```mlir
        %0 = ... : !modelica.bool

        if (%0 : !modelica.bool) {
            ...
        } else {
            ...
        }
        ```
    }];

    let arguments = (ins AnyType:$condition);
    let results = (outs);

    let regions = (region
      SizedRegion<1>:$thenRegion,
      AnyRegion:$elseRegion
    );

    let hasCustomAssemblyFormat = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value":$condition, "bool":$withElseRegion)>
    ];

    let extraClassDeclaration = [{
        ::mlir::Block* thenBlock();
        ::mlir::Block* elseBlock();
    }];
}

def Modelica_ForOp : BaseModelica_Op<"for",
    [DeclareOpInterfaceMethods<LoopLikeOpInterface>]>
{
    let summary = "For algorithmic loop.";

    let description = [{
        The operation represents an algorithmic for loop.
        It consists in three regions:
            1. A condition region that determines whether the iterations should go on.
            2. The body region to be executed at each iteration.
            3. A post-iteration region that is executed after each execution of the body
               region.

        Example:

        ```mlir
        modelica.for condition {
            %0 = ... : !modelica.bool
            modelica.condition (%0 : !modelica.bool)
        } body {
            modelica.yield
        } step {
            modelica.yield
        }
        ```
    }];

    let arguments = (ins Variadic<AnyType>:$args);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$bodyRegion,
        SizedRegion<1>:$stepRegion
    );

    let hasCustomAssemblyFormat = 1;

    let extraClassDeclaration = [{
        ::mlir::Block* conditionBlock();
        ::mlir::Block* bodyBlock();
        ::mlir::Block* stepBlock();
    }];
}

def BaseModelica_WhileOp : BaseModelica_Op<"while",
    [NoTerminator,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]>
{
    let summary = "While algorithmic loop.";

    let description = [{
        The operation represents an algorithmic while loop.
        It consists in two regions:
            1. A condition region that determines whether the iterations should go on.
            2. The body region to be executed at each iteration.

        Example:

        ```mlir
        modelica.while {
            %0 = ... : !modelica.bool
            modelica.condition (%0 : !modelica.bool)
        } do {
            ...
        }
        ```
    }];

    let arguments = (ins);
    let results = (outs);

    let regions = (region
        SizedRegion<1>:$conditionRegion,
        SizedRegion<1>:$bodyRegion
    );

    let hasCustomAssemblyFormat = 1;
}

def BaseModelica_ConditionOp : BaseModelica_Op<"condition",
    [Terminator,
    ParentOneOf<["ForOp", "WhileOp"]>]>
{
    let summary = "Iteration condition.";

    let description = [{
        The operation represents the condition to be evaluated at each
        iteration of the parent loop (which can be either a `modelica.for` or
        `modelica.while`).
    }];

    let arguments = (ins
        AnyType:$condition,
        Variadic<AnyType>:$values);

    let results = (outs);

    let assemblyFormat = [{
        `(` $condition `:` type($condition) `)` attr-dict ($values^ `:` type($values))?
    }];

    let builders = [
        OpBuilder<(ins "::mlir::Value":$condition), [{
            build($_builder, $_state, condition, std::nullopt);
        }]>
    ];
}

//===---------------------------------------------------------------------===//
// Utility operations
//===---------------------------------------------------------------------===//

def BaseModelica_YieldOp : BaseModelica_Op<"yield", [Terminator]>
{
    let summary = "Yield values to the parent operation.";

    let description = [{
        The operation yields zero or more SSA values from an op region and
        terminates the region. The semantics of how the yielded values are used
        is defined by the parent operation.
    }];

    let arguments = (ins Variadic<AnyType>:$values);
    let results = (outs);
    let assemblyFormat = "attr-dict ($values^ `:` type($values))?";
    let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];
}

def Modelica_CastOp : BaseModelica_Op<"cast", [NoMemoryEffect]>
{
    let summary = "Cast a scalar value to another type.";

    let description = [{
        Cast a value from a BaseModelica-compatible type to another
        BaseModelica-compatible type. The operation does allow only for scalars to
        be casted.

        Example:

        ```mlir
        %0 = ... : !modelica.int
        %1 = modelica.cast %0 : !modelica.int -> !modelica.real
        ```
    }];

    let arguments = (ins AnyType:$value);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $value attr-dict `:` type($value) `->` type($result)
    }];

    let hasFolder = 1;
}

def Modelica_PrintOp : BaseModelica_Op<"print", []>
{
    let summary = "Print a value to the standard output.";

    let description = [{
        The operation prints a value to the standard output.
        It is thought for debugging purpose, especially for tests.
    }];

    let arguments = (ins AnyType:$value);
    let results = (outs);
    let assemblyFormat = "$value attr-dict `:` type($value)";
}

#endif // MARCO_DIALECT_BASEMODELICA_IR_BASEMODELICAOPS_TD
