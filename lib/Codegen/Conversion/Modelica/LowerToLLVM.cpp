#include "marco/Codegen/Conversion/Modelica/LowerToLLVM.h"
#include "marco/Codegen/Conversion/Modelica/TypeConverter.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "marco/Codegen/ArrayDescriptor.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/OpenMPToLLVM/ConvertOpenMPToLLVM.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Support/MathExtras.h"

#include "marco/Codegen/Conversion/PassDetail.h"

#include "llvm/Support/Debug.h"

using namespace ::marco::codegen;
using namespace ::mlir::modelica;

namespace
{
  /// Generic conversion pattern that provides some utility functions.
  template<typename FromOp>
  class ModelicaOpConversion : public mlir::ConvertOpToLLVMPattern<FromOp>
  {
    protected:
      using mlir::ConvertOpToLLVMPattern<FromOp>::ConvertOpToLLVMPattern;

    public:
      mlir::Type convertType(mlir::Type type) const
      {
        return this->getTypeConverter()->convertType(type);
      }

      mlir::Value materializeTargetConversion(mlir::OpBuilder& builder, mlir::Value value) const
      {
        mlir::Type type = this->getTypeConverter()->convertType(value.getType());
        return this->getTypeConverter()->materializeTargetConversion(builder, value.getLoc(), type, value);
      }
  };

  template<typename FromOp>
  struct AllocLikeOpLowering : public ModelicaOpConversion<FromOp>
  {
    using ModelicaOpConversion<FromOp>::ModelicaOpConversion;
    using OpAdaptor = typename ModelicaOpConversion<FromOp>::OpAdaptor;

    protected:
      virtual ArrayType getResultType(FromOp op) const = 0;
      virtual mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, FromOp op, mlir::Value sizeBytes) const = 0;

    public:
      mlir::LogicalResult matchAndRewrite(FromOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
      {
        auto loc = op->getLoc();
        auto arrayType = getResultType(op);
        auto typeConverter = this->getTypeConverter();
        mlir::Type indexType = this->convertType(rewriter.getIndexType());

        // Create the descriptor
        auto descriptor = ArrayDescriptor::undef(rewriter, typeConverter, loc, this->convertType(arrayType));
        mlir::Type sizeType = descriptor.getSizeType();

        // Save the rank into the descriptor
        mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(
            loc, descriptor.getRankType(), rewriter.getIntegerAttr(descriptor.getRankType(), arrayType.getRank()));

        descriptor.setRank(rewriter, loc, rank);

        // Determine the total size of the array in bytes
        auto shape = arrayType.getShape();
        llvm::SmallVector<mlir::Value, 3> sizes;

        // Multi-dimensional arrays must be flattened into a 1-dimensional one.
        // For example, v[s1][s2][s3] becomes v[s1 * s2 * s3] and the access rule
        // is such that v[i][j][k] = v[(i * s1 + j) * s2 + k].

        mlir::Value totalSize = rewriter.create<mlir::LLVM::ConstantOp>(loc, sizeType, rewriter.getIntegerAttr(sizeType, 1));

        for (size_t i = 0, dynamicDimensions = 0, end = shape.size(); i < end; ++i) {
          long dimension = shape[i];

          if (dimension == ArrayType::kDynamicSize) {
            mlir::Value size = adaptor.getOperands()[dynamicDimensions++];
            sizes.push_back(size);
          } else {
            mlir::Value size = rewriter.create<mlir::LLVM::ConstantOp>(loc, sizeType, rewriter.getIntegerAttr(sizeType, dimension));
            sizes.push_back(size);
          }

          totalSize = rewriter.create<mlir::LLVM::MulOp>(loc, sizeType, totalSize, sizes[i]);
        }

        // Determine the buffer size in bytes
        mlir::Type elementPtrType = mlir::LLVM::LLVMPointerType::get(this->convertType(arrayType.getElementType()));
        mlir::Value nullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, elementPtrType);
        mlir::Value gepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, elementPtrType, nullPtr, totalSize);
        mlir::Value sizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, indexType, gepPtr);

        // Allocate the underlying buffer and store the pointer into the descriptor
        mlir::Value buffer = allocateBuffer(rewriter, loc, op, sizeBytes);
        descriptor.setPtr(rewriter, loc, buffer);

        // Store the sizes into the descriptor
        for (auto size : llvm::enumerate(sizes)) {
          descriptor.setSize(rewriter, loc, size.index(), size.value());
        }

        rewriter.replaceOp(op, *descriptor);
        return mlir::success();
      }
  };

  class AllocaOpLowering : public AllocLikeOpLowering<AllocaOp>
  {
    using AllocLikeOpLowering<AllocaOp>::AllocLikeOpLowering;

    ArrayType getResultType(AllocaOp op) const override
    {
      return op.getArrayType();
    }

    mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocaOp op, mlir::Value sizeBytes) const override
    {
      mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.getArrayType().getElementType()));
      return rewriter.create<mlir::LLVM::AllocaOp>(loc, bufferPtrType, sizeBytes, op->getAttrs());
    }
  };

  class AllocOpLowering : public AllocLikeOpLowering<AllocOp>
  {
    using AllocLikeOpLowering<AllocOp>::AllocLikeOpLowering;

    ArrayType getResultType(AllocOp op) const override
    {
      return op.getArrayType();
    }

    mlir::Value allocateBuffer(mlir::ConversionPatternRewriter& rewriter, mlir::Location loc, AllocOp op, mlir::Value sizeBytes) const override
    {
      // Insert the "malloc" declaration if it is not already present in the module
      auto heapAllocFunc = lookupOrCreateHeapAllocFn(rewriter, op->getParentOfType<mlir::ModuleOp>());

      // Allocate the buffer
      mlir::Type bufferPtrType = mlir::LLVM::LLVMPointerType::get(convertType(op.getArrayType().getElementType()));
      auto results = createLLVMCall(rewriter, loc, heapAllocFunc, sizeBytes, getVoidPtrType());
      return rewriter.create<mlir::LLVM::BitcastOp>(loc, bufferPtrType, results[0]);
    }

    mlir::LLVM::LLVMFuncOp lookupOrCreateHeapAllocFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      std::string name = "_MheapAlloc_pvoid_i64";

      if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return foo;
      }

      mlir::PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidPtrType(), builder.getI64Type());
      return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
    }
  };

  class FreeOpLowering: public ModelicaOpConversion<FreeOp>
  {
    using ModelicaOpConversion<FreeOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(FreeOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto typeConverter = this->getTypeConverter();

      // Insert the "free" declaration if it is not already present in the module
      auto freeFunc = lookupOrCreateHeapFreeFn(rewriter, op->getParentOfType<mlir::ModuleOp>());

      // Extract the buffer address and call the "free" function
      ArrayDescriptor descriptor(typeConverter, adaptor.getArray());
      mlir::Value address = descriptor.getPtr(rewriter, loc);
      mlir::Value casted = rewriter.create<mlir::LLVM::BitcastOp>(loc, getVoidPtrType(), address);
      rewriter.replaceOpWithNewOp<mlir::LLVM::CallOp>(op, llvm::None, freeFunc.getPersonalityAttr(), casted);

      return mlir::success();
    }

    mlir::LLVM::LLVMFuncOp lookupOrCreateHeapFreeFn(mlir::OpBuilder& builder, mlir::ModuleOp module) const
    {
      std::string name = "_MheapFree_void_pvoid";

      if (auto foo = module.lookupSymbol<mlir::LLVM::LLVMFuncOp>(name)) {
        return foo;
      }

      mlir::PatternRewriter::InsertionGuard insertGuard(builder);
      builder.setInsertionPointToStart(module.getBody());
      auto llvmFnType = mlir::LLVM::LLVMFunctionType::get(getVoidType(), getVoidPtrType());
      return builder.create<mlir::LLVM::LLVMFuncOp>(module->getLoc(), name, llvmFnType);
    }
  };

  class DimOpLowering: public ModelicaOpConversion<DimOp>
  {
    using ModelicaOpConversion<DimOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(DimOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();

      // The actual size of each dimension is stored in the memory description
      // structure.
      ArrayDescriptor descriptor(this->getTypeConverter(), adaptor.getArray());
      mlir::Value size = descriptor.getSize(rewriter, loc, adaptor.getDimension());

      rewriter.replaceOp(op, size);
      return mlir::success();
    }
  };

  class SubscriptOpLowering : public ModelicaOpConversion<SubscriptionOp>
  {
    using ModelicaOpConversion<SubscriptionOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(SubscriptionOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op.getLoc();
      auto typeConverter = this->getTypeConverter();
      mlir::Type indexType = convertType(rewriter.getIndexType());

      auto sourceArrayType = op.getSourceArrayType();
      auto resultArrayType = op.getResultArrayType();

      ArrayDescriptor sourceDescriptor(typeConverter, adaptor.getSource());
      ArrayDescriptor result = ArrayDescriptor::undef(rewriter, typeConverter, loc, convertType(resultArrayType));

      mlir::Value index = adaptor.getIndices()[0];

      for (size_t i = 1, e = sourceArrayType.getRank(); i < e; ++i) {
        mlir::Value size = sourceDescriptor.getSize(rewriter, loc, i);
        index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);

        if (i < adaptor.getIndices().size()) {
          mlir::Value offset = adaptor.getIndices()[i];
          index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, offset);
        }
      }

      mlir::Value base = sourceDescriptor.getPtr(rewriter, loc);
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);
      result.setPtr(rewriter, loc, ptr);

      mlir::Type rankType = result.getRankType();
      mlir::Value rank = rewriter.create<mlir::LLVM::ConstantOp>(loc, rankType, rewriter.getIntegerAttr(rankType, resultArrayType.getRank()));
      result.setRank(rewriter, loc, rank);

      for (size_t i = sourceArrayType.getRank() - resultArrayType.getRank(), e = sourceArrayType.getRank(), j = 0; i < e; ++i, ++j) {
        result.setSize(rewriter, loc, j, sourceDescriptor.getSize(rewriter, loc, i));
      }

      rewriter.replaceOp(op, *result);
      return mlir::success();
    }
  };

  class LoadOpLowering: public ModelicaOpConversion<LoadOp>
  {
    using ModelicaOpConversion<LoadOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(LoadOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto typeConverter = this->getTypeConverter();
      auto loc = op->getLoc();
      auto indexes = adaptor.getIndices();

      assert(op.getArrayType().getRank() == indexes.size() && "Wrong indexes amount");

      // Determine the address into which the value has to be stored.
      ArrayDescriptor descriptor(typeConverter, adaptor.getArray());
      auto indexType = convertType(rewriter.getIndexType());

      auto indexFn = [&]() -> mlir::Value {
        if (indexes.empty()) {
          return rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0));
        }

        return indexes[0];
      };

      mlir::Value index = indexFn();

      for (size_t i = 1, e = indexes.size(); i < e; ++i) {
        mlir::Value size = descriptor.getSize(rewriter, loc, i);
        index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);
        index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, indexes[i]);
      }

      mlir::Value base = descriptor.getPtr(rewriter, loc);
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);

      // Load the value
      rewriter.replaceOpWithNewOp<mlir::LLVM::LoadOp>(op, ptr);

      return mlir::success();
    }
  };

  class StoreOpLowering: public ModelicaOpConversion<StoreOp>
  {
    using ModelicaOpConversion<StoreOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(StoreOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      auto indexes = adaptor.getIndices();

      assert(op.getArrayType().getRank() == indexes.size() && "Wrong indexes amount");

      // Determine the address into which the value has to be stored.
      ArrayDescriptor memoryDescriptor(this->getTypeConverter(), adaptor.getArray());

      auto indexType = convertType(rewriter.getIndexType());
      mlir::Value index = indexes.empty() ? rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIndexAttr(0)) : indexes[0];

      for (size_t i = 1, e = indexes.size(); i < e; ++i) {
        mlir::Value size = memoryDescriptor.getSize(rewriter, loc, i);
        index = rewriter.create<mlir::LLVM::MulOp>(loc, indexType, index, size);
        index = rewriter.create<mlir::LLVM::AddOp>(loc, indexType, index, indexes[i]);
      }

      mlir::Value base = memoryDescriptor.getPtr(rewriter, loc);
      mlir::Value ptr = rewriter.create<mlir::LLVM::GEPOp>(loc, base.getType(), base, index);

      // Store the value
      rewriter.replaceOpWithNewOp<mlir::LLVM::StoreOp>(op, adaptor.getValue(), ptr);

      return mlir::success();
    }
  };

  class CastOpIndexLowering: public ModelicaOpConversion<CastOp>
  {
    using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!op.getValue().getType().isa<mlir::IndexType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not an IndexType");
      }

      auto loc = op.getLoc();

      auto source = op.getValue().getType().cast<mlir::IndexType>();
      mlir::Type destination = op.getResult().getType();

      if (source == destination) {
        rewriter.replaceOp(op, op.getValue());
        return mlir::success();
      }

      if (destination.isa<IntegerType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, convertType(destination), op.getValue());
        return mlir::success();
      }

      if (destination.isa<RealType>()) {
        mlir::Value value = rewriter.create<mlir::arith::IndexCastOp>(loc, convertType(IntegerType::get(rewriter.getContext())), op.getValue());
        value = materializeTargetConversion(rewriter, value);
        rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(op, convertType(destination), value);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
    }
  };

  class CastOpBooleanLowering: public ModelicaOpConversion<CastOp>
  {
    using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!op.getValue().getType().isa<BooleanType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not a BooleanType");
      }

      auto loc = op.getLoc();

      auto source = op.getValue().getType().cast<BooleanType>();
      mlir::Type destination = op.getResult().getType();

      if (source == destination) {
        rewriter.replaceOp(op, op.getValue());
        return mlir::success();
      }

      if (destination.isa<RealType>()) {
        mlir::Value value = adaptor.getValue();
        value = rewriter.create<mlir::LLVM::SExtOp>(loc, convertType(IntegerType::get(rewriter.getContext())), value);
        rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(op, convertType(destination), value);
        return mlir::success();
      }

      if (destination.isa<mlir::IndexType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, rewriter.getIndexType(), adaptor.getValue());
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
    }
  };

  class CastOpIntegerLowering: public ModelicaOpConversion<CastOp>
  {
    using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!op.getValue().getType().isa<IntegerType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not an IntegerType");
      }

      auto source = op.getValue().getType().cast<IntegerType>();
      mlir::Type destination = op.getResult().getType();

      if (source == destination) {
        rewriter.replaceOp(op, op.getValue());
        return mlir::success();
      }

      if (destination.isa<RealType>()) {
        mlir::Value value = adaptor.getValue();
        rewriter.replaceOpWithNewOp<mlir::LLVM::SIToFPOp>(op, convertType(destination), value);
        return mlir::success();
      }

      if (destination.isa<mlir::IndexType>()) {
        rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, rewriter.getIndexType(), adaptor.getValue());
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
    }
  };

  class CastOpRealLowering: public ModelicaOpConversion<CastOp>
  {
    using ModelicaOpConversion<CastOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(CastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      if (!op.getValue().getType().isa<RealType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not a RealType");
      }

      auto loc = op.getLoc();

      auto source = op.getValue().getType().cast<RealType>();
      mlir::Type destination = op.getResult().getType();

      if (source == destination) {
        rewriter.replaceOp(op, op.getValue());
        return mlir::success();
      }

      if (destination.isa<IntegerType>()) {
        mlir::Value value = adaptor.getValue();
        rewriter.replaceOpWithNewOp<mlir::LLVM::FPToSIOp>(op, convertType(destination), value);
        return mlir::success();
      }

      if (destination.isa<mlir::IndexType>()) {
        mlir::Value value = rewriter.create<mlir::LLVM::FPToSIOp>(loc, convertType(destination), adaptor.getValue());
        rewriter.replaceOpWithNewOp<mlir::arith::IndexCastOp>(op, rewriter.getIndexType(), value);
        return mlir::success();
      }

      return rewriter.notifyMatchFailure(op, "Unknown conversion to destination type");
    }
  };

  struct ArrayCastOpArrayToArrayLowering : public ModelicaOpConversion<ArrayCastOp>
  {
    using ModelicaOpConversion<ArrayCastOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(ArrayCastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      mlir::Type sourceType = op.getSource().getType();
      auto resultType = op.getResult().getType();

      if (!sourceType.isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not an array");
      }

      auto sourceArrayType = sourceType.cast<ArrayType>();
      ArrayDescriptor sourceDescriptor(this->getTypeConverter(), adaptor.getSource());

      if (!resultType.isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not an array");
      }

      auto resultArrayType = resultType.cast<ArrayType>();

      if (sourceArrayType.getRank() != resultArrayType.getRank()) {
        return rewriter.notifyMatchFailure(op, "The destination array type has a different rank");
      }

      for (const auto& dimension : resultArrayType.getShape()) {
        if (dimension != ArrayType::kDynamicSize) {
          return rewriter.notifyMatchFailure(op, "The destination array type has some fixed dimensions");
        }
      }

      ArrayDescriptor resultDescriptor = ArrayDescriptor::undef(rewriter, mlir::ConvertToLLVMPattern::getTypeConverter(), loc, convertType(resultType));
      resultDescriptor.setPtr(rewriter, loc, sourceDescriptor.getPtr(rewriter, loc));
      resultDescriptor.setRank(rewriter, loc, sourceDescriptor.getRank(rewriter, loc));

      for (unsigned int i = 0; i < sourceArrayType.getRank(); ++i) {
        resultDescriptor.setSize(rewriter, loc, i, sourceDescriptor.getSize(rewriter, loc, i));
      }

      mlir::Value result = getTypeConverter()->materializeSourceConversion(rewriter, loc, resultType, *resultDescriptor);
      rewriter.replaceOp(op, result);
      return mlir::success();
    }
  };

  struct ArrayCastOpArrayToUnsizedArrayLowering : public ModelicaOpConversion<ArrayCastOp>
  {
    using ModelicaOpConversion<ArrayCastOp>::ModelicaOpConversion;

    mlir::LogicalResult matchAndRewrite(ArrayCastOp op, OpAdaptor adaptor, mlir::ConversionPatternRewriter& rewriter) const override
    {
      auto loc = op->getLoc();
      mlir::Type sourceType = op.getSource().getType();
      auto resultType = op.getResult().getType();

      if (!sourceType.isa<ArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Source is not an array");
      }

      ArrayDescriptor sourceDescriptor(this->getTypeConverter(), adaptor.getSource());

      if (!resultType.isa<UnsizedArrayType>()) {
        return rewriter.notifyMatchFailure(op, "Result is not an unsized array");
      }

      // Create the unsized array descriptor that holds the ranked one. It is allocated on the stack, because
      // the runtime library expects a pointer in order to avoid any unrolling due to calling conventions.
      // The inner descriptor (that is, the sized array descriptor) is also allocated on the stack.
      mlir::Type indexType = getTypeConverter()->getIndexType();
      mlir::Value one = rewriter.create<mlir::LLVM::ConstantOp>(loc, indexType, rewriter.getIntegerAttr(indexType, 1));

      // Allocate space on the stack for the sized array descriptor
      auto sizedArrayDescPtrType = mlir::LLVM::LLVMPointerType::get(getTypeConverter()->convertType(sourceType));
      mlir::Value sizedArrayDescNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, sizedArrayDescPtrType);
      mlir::Value sizedArrayDescGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, sizedArrayDescPtrType, sizedArrayDescNullPtr, one);
      mlir::Value sizedArrayDescSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, indexType, sizedArrayDescGepPtr);
      mlir::Value sizedArrayDescOpaquePtr = rewriter.create<mlir::LLVM::AllocaOp>(loc, getVoidPtrType(), sizedArrayDescSizeBytes);
      mlir::Value sizedArrayDescPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, sizedArrayDescPtrType, sizedArrayDescOpaquePtr);

      // Allocate space on the stack for the unsized array descriptor
      auto unsizedArrayDescPtrType = getTypeConverter()->convertType(resultType).cast<mlir::LLVM::LLVMPointerType>();
      mlir::Value unsizedArrayDescNullPtr = rewriter.create<mlir::LLVM::NullOp>(loc, unsizedArrayDescPtrType);
      mlir::Value unsizedArrayDescGepPtr = rewriter.create<mlir::LLVM::GEPOp>(loc, unsizedArrayDescPtrType, unsizedArrayDescNullPtr, one);
      mlir::Value unsizedArrayDescSizeBytes = rewriter.create<mlir::LLVM::PtrToIntOp>(loc, indexType, unsizedArrayDescGepPtr);
      mlir::Value unsizedArrayDescOpaquePtr = rewriter.create<mlir::LLVM::AllocaOp>(loc, getVoidPtrType(), unsizedArrayDescSizeBytes);
      mlir::Value unsizedArrayDescPtr = rewriter.create<mlir::LLVM::BitcastOp>(loc, unsizedArrayDescPtrType, unsizedArrayDescOpaquePtr);

      // Populate the sized array descriptor
      rewriter.create<mlir::LLVM::StoreOp>(loc, *sourceDescriptor, sizedArrayDescPtr);

      // Populate the unsized array descriptor
      UnsizedArrayDescriptor unsizedArrayDescriptor = UnsizedArrayDescriptor::undef(rewriter, loc, unsizedArrayDescPtrType.getElementType());
      unsizedArrayDescriptor.setPtr(rewriter, loc, sizedArrayDescOpaquePtr);
      unsizedArrayDescriptor.setRank(rewriter, loc, sourceDescriptor.getRank(rewriter, loc));
      rewriter.create<mlir::LLVM::StoreOp>(loc, *unsizedArrayDescriptor, unsizedArrayDescPtr);

      rewriter.replaceOp(op, unsizedArrayDescPtr);
      return mlir::success();
    }
  };

  static void populateModelicaToLLVMConversionPatterns(mlir::LLVMTypeConverter& typeConverter, mlir::RewritePatternSet& patterns)
  {
    patterns.insert<
        AllocaOpLowering,
        AllocOpLowering,
        FreeOpLowering,
        DimOpLowering,
        SubscriptOpLowering,
        LoadOpLowering,
        StoreOpLowering,
        CastOpIndexLowering,
        CastOpBooleanLowering,
        CastOpIntegerLowering,
        CastOpRealLowering,
        ArrayCastOpArrayToArrayLowering,
        ArrayCastOpArrayToUnsizedArrayLowering>(typeConverter);
  }

  class LLVMLoweringPass : public ConvertModelicaToLLVMBase<LLVMLoweringPass>
  {
    public:
      LLVMLoweringPass(ModelicaToLLVMConversionOptions options, unsigned int bitWidth)
          : options(std::move(options)), bitWidth(bitWidth)
      {
      }

      void runOnOperation() override
      {
        auto module = getOperation();


        if (mlir::failed(stdToLLVMConversionPass(module))) {
          module.emitError("Error in converting to LLVM dialect");
          return signalPassFailure();
        }

        llvm::DebugFlag = true;
      }

    private:
      mlir::LogicalResult stdToLLVMConversionPass(mlir::ModuleOp module)
      {
        mlir::LowerToLLVMOptions llvmOptions(&getContext());
        TypeConverter typeConverter(&getContext(), llvmOptions, bitWidth);

        mlir::ConversionTarget target(getContext());
        target.addIllegalDialect<ModelicaDialect>();
        // TODO target.addIllegalOp<mlir::func::FuncOp>();

        target.addLegalDialect<mlir::LLVM::LLVMDialect>();
        //target.addLegalOp<mlir::UnrealizedConversionCastOp>();
        //target.addLegalOp<mlir::ModuleOp>();

        /*
        target.addDynamicallyLegalOp<
            mlir::omp::MasterOp,
            mlir::omp::ParallelOp,
            mlir::omp::WsLoopOp>([&](mlir::Operation *op) {
          return typeConverter.isLegal(&op->getRegion(0));
        });

        target.addLegalOp<
            mlir::omp::TerminatorOp,
            mlir::omp::TaskyieldOp,
            mlir::omp::FlushOp,
            mlir::omp::BarrierOp,
            mlir::omp::TaskwaitOp>();
            */

        target.markUnknownOpDynamicallyLegal([](mlir::Operation* op) {
          return true;
        });

        mlir::RewritePatternSet patterns(&getContext());
        populateModelicaToLLVMConversionPatterns(typeConverter, patterns);
        //mlir::populateOpenMPToLLVMConversionPatterns(typeConverter, patterns);

        return applyPartialConversion(module, target, std::move(patterns));
      }

    private:
      ModelicaToLLVMConversionOptions options;
      unsigned int bitWidth;
  };
}

namespace marco::codegen
{
  std::unique_ptr<mlir::Pass> createLLVMLoweringPass(ModelicaToLLVMConversionOptions options, unsigned int bitWidth)
  {
    return std::make_unique<LLVMLoweringPass>(options, bitWidth);
  }
}
