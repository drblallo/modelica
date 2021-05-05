#include <llvm/ADT/STLExtras.h>
#include <mlir/IR/DialectImplementation.h>
#include <modelica/mlirlowerer/Type.h>

using namespace modelica::codegen;

bool IntegerTypeStorage::operator==(const KeyTy& key) const
{
	return key == getBitWidth();
}

unsigned int IntegerTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(key);
}

IntegerTypeStorage* IntegerTypeStorage::construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth)
{
	auto* storage = allocator.allocate<IntegerTypeStorage>();
	return new (storage) IntegerTypeStorage(bitWidth);
}

unsigned int IntegerTypeStorage::getBitWidth() const
{
	return bitWidth;
}

IntegerTypeStorage::IntegerTypeStorage(unsigned int bitWidth) : bitWidth(bitWidth)
{
}

bool RealTypeStorage::operator==(const KeyTy& key) const
{
	return key == getBitWidth();
}

unsigned int RealTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(key);
}

RealTypeStorage* RealTypeStorage::construct(mlir::TypeStorageAllocator&allocator, unsigned int bitWidth)
{
	auto* storage = allocator.allocate<RealTypeStorage>();
	return new (storage) RealTypeStorage(bitWidth);
}

unsigned int RealTypeStorage::getBitWidth() const
{
	return bitWidth;
}

RealTypeStorage::RealTypeStorage(unsigned int bitWidth) : bitWidth(bitWidth)
{
}

llvm::hash_code hash_value(const PointerType::Shape& shape)
{
	if (shape.size()) {
		return llvm::hash_combine_range(shape.begin(), shape.end());
	}
	return llvm::hash_combine(0);
}

bool PointerTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getAllocationScope(), getElementType(), getShape()};
}

unsigned int PointerTypeStorage::hashKey(const KeyTy& key)
{
	auto shapeHash{hash_value(std::get<PointerType::Shape>(key))};
	return llvm::hash_combine(std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), shapeHash);
}

PointerTypeStorage* PointerTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key)
{
	auto *storage = allocator.allocate<PointerTypeStorage>();
	return new (storage) PointerTypeStorage{std::get<BufferAllocationScope>(key), std::get<mlir::Type>(key), std::get<PointerType::Shape>(key)};
}

BufferAllocationScope PointerTypeStorage::getAllocationScope() const
{
	return allocationScope;
}

PointerType::Shape PointerTypeStorage::getShape() const
{
	return shape;
}

mlir::Type PointerTypeStorage::getElementType() const
{
	return elementType;
}

PointerTypeStorage::PointerTypeStorage(BufferAllocationScope allocationScope, mlir::Type elementType, const Shape& shape)
		: allocationScope(allocationScope),
			elementType(elementType),
			shape(std::move(shape))
{
}

bool UnsizedPointerTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getElementType(), getRank()};
}

unsigned int UnsizedPointerTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(std::get<mlir::Type>(key), std::get<unsigned int>(key));
}

UnsizedPointerTypeStorage* UnsizedPointerTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy &key)
{
	auto *storage = allocator.allocate<UnsizedPointerTypeStorage>();
	return new (storage) UnsizedPointerTypeStorage{std::get<mlir::Type>(key), std::get<unsigned int>(key)};
}

mlir::Type UnsizedPointerTypeStorage::getElementType() const
{
	return elementType;
}

unsigned int UnsizedPointerTypeStorage::getRank() const
{
	return rank;
}

UnsizedPointerTypeStorage::UnsizedPointerTypeStorage(mlir::Type elementType, unsigned int rank)
		: elementType(elementType),
			rank(rank)
{
}

bool StructTypeStorage::operator==(const KeyTy& key) const
{
	return key == KeyTy{getElementTypes()};
}

unsigned int StructTypeStorage::hashKey(const KeyTy& key)
{
	return llvm::hash_combine(key);
}

StructTypeStorage* StructTypeStorage::construct(mlir::TypeStorageAllocator& allocator, const KeyTy& key)
{
	llvm::ArrayRef<mlir::Type> elementTypes = allocator.copyInto(key);
	return new (allocator.allocate<StructTypeStorage>()) StructTypeStorage(elementTypes);
}

llvm::ArrayRef<mlir::Type> StructTypeStorage::getElementTypes() const
{
	return elementTypes;
}

StructTypeStorage::StructTypeStorage(llvm::ArrayRef<mlir::Type> elementTypes)
		: elementTypes(elementTypes)
{
}

BooleanType BooleanType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

IntegerType IntegerType::get(mlir::MLIRContext* context, unsigned int bitWidth)
{
	return Base::get(context, bitWidth);
}

unsigned int IntegerType::getBitWidth() const
{
	return getImpl()->getBitWidth();
}

RealType RealType::get(mlir::MLIRContext* context, unsigned int bitWidth)
{
	return Base::get(context, bitWidth);
}

unsigned int RealType::getBitWidth() const
{
	return getImpl()->getBitWidth();
}

PointerType PointerType::get(mlir::MLIRContext* context, BufferAllocationScope allocationScope, mlir::Type elementType, llvm::ArrayRef<long> shape)
{
	return Base::get(context, allocationScope, elementType, Shape(shape.begin(), shape.end()));
}

BufferAllocationScope PointerType::getAllocationScope() const
{
	return getImpl()->getAllocationScope();
}

mlir::Type PointerType::getElementType() const
{
	return getImpl()->getElementType();
}

PointerType::Shape PointerType::getShape() const
{
	return getImpl()->getShape();
}

unsigned int PointerType::getRank() const
{
	return getShape().size();
}

unsigned int PointerType::getConstantDimensions() const
{
	auto dimensions = getShape();
	unsigned int count = 0;

	for (auto dimension : dimensions) {
		if (dimension > 0)
			count++;
	}

	return count;
}

unsigned int PointerType::getDynamicDimensions() const
{
	auto dimensions = getShape();
	unsigned int count = 0;

	for (auto dimension : dimensions) {
		if (dimension  == -1)
			count++;
	}

	return count;
}

long PointerType::rawSize() const
{
	long result = 1;

	for (long size : getShape())
	{
		if (size == -1)
			return -1;

		result *= size;
	}

	return result;
}

bool PointerType::hasConstantShape() const
{
	return llvm::all_of(getShape(), [](long size) {
		return size != -1;
	});
}

PointerType PointerType::slice(unsigned int subscriptsAmount)
{
	auto shape = getShape();
	assert(subscriptsAmount <= shape.size() && "Too many subscriptions");
	llvm::SmallVector<long, 3> resultShape;

	for (size_t i = subscriptsAmount, e = shape.size(); i < e; ++i)
		resultShape.push_back(shape[i]);

	return PointerType::get(getContext(), getAllocationScope(), getElementType(), resultShape);
}

PointerType PointerType::toAllocationScope(BufferAllocationScope scope)
{
	return PointerType::get(getContext(), scope, getElementType(), getShape());
}

PointerType PointerType::toUnknownAllocationScope()
{
	return toAllocationScope(unknown);
}

PointerType PointerType::toMinAllowedAllocationScope()
{
	if (getAllocationScope() == heap)
		return *this;

	if (canBeOnStack())
		return toAllocationScope(stack);

	return toAllocationScope(heap);
}

UnsizedPointerType PointerType::toUnsized()
{
	return UnsizedPointerType::get(getContext(), getElementType(), getRank());
}

bool PointerType::canBeOnStack() const
{
	return hasConstantShape();
}

UnsizedPointerType UnsizedPointerType::get(mlir::MLIRContext* context, mlir::Type elementType, unsigned int rank)
{
	return Base::get(context, elementType, rank);
}

mlir::Type UnsizedPointerType::getElementType() const
{
	return getImpl()->getElementType();
}

unsigned int UnsizedPointerType::getRank() const
{
	return getImpl()->getRank();
}

OpaquePointerType OpaquePointerType::get(mlir::MLIRContext* context)
{
	return Base::get(context);
}

StructType StructType::get(mlir::MLIRContext* context, llvm::ArrayRef<mlir::Type> elementTypes)
{
	return Base::get(context, elementTypes);
}

llvm::ArrayRef<mlir::Type> StructType::getElementTypes()
{
	return getImpl()->getElementTypes();
}

void modelica::codegen::printModelicaType(mlir::Type type, mlir::DialectAsmPrinter& printer) {
	auto& os = printer.getStream();

	if (type.isa<BooleanType>())
	{
		os << "bool";
		return;
	}

	if (type.isa<IntegerType>())
	{
		os << "int";
		return;
	}

	if (type.dyn_cast<RealType>())
	{
		os << "real";
		return;
	}

	if (auto pointerType = type.dyn_cast<PointerType>())
	{
		os << "ptr<";

		if (pointerType.getAllocationScope() == stack)
			os << "stack, ";
		else if (pointerType.getAllocationScope() == heap)
			os << "heap, ";

		auto dimensions = pointerType.getShape();

		for (const auto& dimension : dimensions)
			os << (dimension == -1 ? "?" : std::to_string(dimension)) << "x";

		printer.printType(pointerType.getElementType());
		os << ">";
		return;
	}

	if (auto pointerType = type.dyn_cast<UnsizedPointerType>())
	{
		os << "ptr<*x" << pointerType.getElementType() << ">";
		return;
	}

	if (type.isa<OpaquePointerType>())
	{
		os << "opaque_ptr";
		return;
	}

	if (auto structType = type.dyn_cast<StructType>())
	{
		os << "struct<";

		for (auto subtype : llvm::enumerate(structType.getElementTypes()))
		{
			if (subtype.index() != 0)
				os << ", ";

			os << subtype.value();
		}

		os << ">";
	}
}
