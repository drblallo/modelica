#pragma once

#include "llvm/IR/IRBuilder.h"
#include "llvm/Support/Error.h"
#include "modelica/lowerer/Simulation.hpp"

namespace modelica
{
	/**
	 * \brief create a function with internal linkage in the provided module and
	 * provided name.
	 *
	 * \return the created function, FunctionAlreadyExists if the function already
	 * existed
	 */
	llvm::Expected<llvm::Function*> makePrivateFunction(
			llvm::StringRef name, llvm::Module& m);

	/**
	 * \return the pointer to the index element inside the array pointed by
	 * arrayPtr
	 */
	llvm::Value* getArrayElementPtr(
			llvm::IRBuilder<>& bld, llvm::Value* arrayPtr, llvm::Value* index);

	/**
	 * \return the pointer to the index element inside the array pointed by
	 * arrayPtr
	 * \pre index in bounds
	 */
	llvm::Value* getArrayElementPtr(
			llvm::IRBuilder<>& bld, llvm::Value* arrayPtr, size_t index);

	/**
	 * arrayPtr[index] = value;
	 */
	void storeToArrayElement(
			llvm::IRBuilder<>& bld,
			llvm::Value* value,
			llvm::Value* arrayPtr,
			llvm::Value* index);

	/**
	 * arrayPtr[index] = value;
	 *
	 */
	void storeToArrayElement(
			llvm::IRBuilder<>& bld,
			llvm::Value* value,
			llvm::Value* arrayPtr,
			size_t index);

	/**
	 * \return arrayPtr[index]
	 */
	llvm::Value* loadArrayElement(
			llvm::IRBuilder<>& bld, llvm::Value* arrayPtr, size_t index);

	/**
	 * \return arrayPtr[index]
	 */
	llvm::Value* loadArrayElement(
			llvm::IRBuilder<>& bld, llvm::Value* arrayPtr, llvm::Value* index);
	/**
	 * creates a type from a SimType
	 * \return the created type
	 */
	[[nodiscard]] llvm::ArrayType* typeToLLVMType(
			llvm::LLVMContext& context, const SimType& type);

	/**
	 * creates a type from a builtin type
	 * \return the created type.
	 */
	[[nodiscard]] llvm::Type* builtInToLLVMType(
			llvm::LLVMContext& context, BultinSimTypes type);

	/**
	 * allocates the global var into the module
	 * and initializes it with the provided value.
	 * the variable must not have been already inserted
	 *
	 * \return an error if the allocation failed
	 */
	llvm::Error simExpToGlobalVar(
			llvm::Module& module,
			llvm::StringRef name,
			const SimType& type,
			llvm::GlobalValue::LinkageTypes linkage);

	/**
	 * creates a store of an single variable to the provided location.
	 * \return the StoreInt
	 */
	template<typename T>
	llvm::Value* makeConstantStore(
			llvm::IRBuilder<>& builder, T value, llvm::Value* location)
	{
		auto ptrType = llvm::dyn_cast<llvm::PointerType>(location->getType());
		auto underlyingType = ptrType->getContainedType(0);

		if (underlyingType == llvm::Type::getFloatTy(builder.getContext()))

			return builder.CreateStore(
					llvm::ConstantFP::get(underlyingType, value), location);
		return builder.CreateStore(
				llvm::ConstantInt::get(underlyingType, value), location);
	}

	/**
	 * arrayPtr[index] = value
	 */
	template<typename T>
	void storeConstantToArrayElement(
			llvm::IRBuilder<>& bld, T value, llvm::Value* arrayPtr, size_t index)
	{
		auto ptrToElem = getArrayElementPtr(bld, arrayPtr, index);
		makeConstantStore<T>(bld, value, ptrToElem);
	}

	/**
	 * creates a load instruction to load the old value of a particular var.
	 *
	 * \return the loadInst
	 */
	llvm::Expected<llvm::Value*> lowerReference(
			llvm::IRBuilder<>& builder, llvm::StringRef exp);

	/**
	 * \return a llvm::type rappresenting the array of types of the provided
	 * SimType.
	 */
	llvm::AllocaInst* allocaSimType(llvm::IRBuilder<>& bld, const SimType& type);

	/**
	 * Creates a for cycle that last interationsCount iterations
	 * that will be inserted in the provided builder
	 * The caller has to provide whileContent which is function that will
	 * produce the actual body of the loop.
	 *
	 * \return the exit point of the loop, that is the basic block that will
	 * always be executed at some point. The builder will look at that basic
	 * block.
	 */
	llvm::BasicBlock* createForCycle(
			llvm::Function* function,
			llvm::IRBuilder<>& builder,
			size_t iterationCount,
			std::function<void(llvm::IRBuilder<>&, llvm::Value*)> whileContent);

	using TernaryOpFunction =
			std::function<llvm::Expected<llvm::Value*>(llvm::IRBuilder<>&)>;

	/**
	 * Creates a if else branch based on the result value of condition()
	 * \pre the returned llvm::type of trueBlock() must be equal to the returned
	 * llvm::type of falseBlock() and to outType, the returned llvm::type of
	 * condition() bust be int1.
	 * \return the phi instruction that contains the result of the brach taken.
	 *
	 * builder will now point at the exit BB.
	 */
	llvm::Expected<llvm::Value*> createTernaryOp(
			llvm::Function* function,
			llvm::IRBuilder<>& builder,
			llvm::Type* outType,
			TernaryOpFunction condition,
			TernaryOpFunction trueBlock,
			TernaryOpFunction falseBlock);
}	 // namespace modelica
