#include "modelica/lowerer/Simulation.hpp"

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/IRBuilder.h"
#include "modelica/lowerer/ExpLowerer.hpp"
#include "modelica/lowerer/SimLowerer.hpp"
#include "modelica/simulation/SimErrors.hpp"

using namespace std;
using namespace modelica;
using namespace llvm;

static Expected<Function*> populateMain(
		Module& m,
		StringRef entryPointName,
		Function* init,
		Function* update,
		Function* printValues,
		unsigned simulationStop)
{
	assert(init != nullptr);				 // NOLINT
	assert(update != nullptr);			 // NOLINT
	assert(printValues != nullptr);	 // NOLINT

	auto expectedMain = makePrivateFunction(entryPointName, m);
	if (!expectedMain)
		return expectedMain;
	auto main = expectedMain.get();
	main->setLinkage(GlobalValue::LinkageTypes::ExternalLinkage);

	IRBuilder<> builder(&main->getEntryBlock());
	// call init
	builder.CreateCall(init);

	const auto forBody = [update, printValues](IRBuilder<>& builder, auto index) {
		builder.CreateCall(update);
		builder.CreateCall(printValues);
	};

	// creates a for with simulationStop iterations what invokes
	// update and print values each time
	auto loopExit = createForCycle(main, builder, simulationStop, forBody);

	// returns right after the loop
	builder.SetInsertPoint(loopExit);
	builder.CreateRet(nullptr);
	return main;
}

static void insertGlobalString(Module& m, StringRef name, StringRef content)
{
	auto str = ConstantDataArray::getString(m.getContext(), content);
	auto type = ArrayType::get(
			IntegerType::getInt8Ty(m.getContext()), content.size() + 1);
	auto global = m.getOrInsertGlobal(name, type);
	dyn_cast<GlobalVariable>(global)->setInitializer(str);
}

static Error insertGlobal(
		Module& module,
		StringRef name,
		const SimExp& exp,
		GlobalValue::LinkageTypes linkage)
{
	const auto& type = exp.getSimType();
	const string oldName = name.str() + "_old";

	if (auto e = simExpToGlobalVar(module, name, type, linkage); e)
		return e;
	if (auto e = simExpToGlobalVar(module, oldName, type, linkage); e)
		return e;
	insertGlobalString(module, name.str() + "_str", name);
	return Error::success();
}

static Error createAllGlobals(
		Module& m, StringMap<SimExp> vars, GlobalValue::LinkageTypes linkType)
{
	for (const auto& pair : vars)
		if (auto e = insertGlobal(m, pair.first(), pair.second, linkType); e)
			return e;

	return Error::success();
}

static Expected<Function*> initializeGlobals(Module& m, StringMap<SimExp> vars)
{
	auto initFunctionExpected = makePrivateFunction("init", m);
	if (!initFunctionExpected)
		return initFunctionExpected;
	auto initFunction = initFunctionExpected.get();
	IRBuilder builder(&initFunction->getEntryBlock());

	for (const auto& pair : vars)
	{
		auto val = lowerExp(builder, initFunction, pair.second);
		if (!val)
			return val.takeError();
		auto loaded = builder.CreateLoad(*val);
		builder.CreateStore(loaded, m.getGlobalVariable(pair.first(), true));
		builder.CreateStore(
				loaded, m.getGlobalVariable(pair.first().str() + "_old", true));
	}
	builder.CreateRet(nullptr);
	return initFunction;
}

static Expected<Function*> createUpdates(Module& m, StringMap<SimExp> upds)
{
	auto updateFunctionExpected = makePrivateFunction("update", m);
	if (!updateFunctionExpected)
		return updateFunctionExpected;
	auto updateFunction = updateFunctionExpected.get();
	IRBuilder bld(&updateFunction->getEntryBlock());

	for (const auto& pair : upds)
	{
		auto expFun = makePrivateFunction("update" + pair.first().str(), m);
		if (!expFun)
			return expFun;

		auto fun = expFun.get();
		bld.SetInsertPoint(&fun->getEntryBlock());

		auto val = lowerExp(bld, fun, pair.second);
		if (!val)
			return val.takeError();
		auto loaded = bld.CreateLoad(*val);
		bld.CreateStore(loaded, m.getGlobalVariable(pair.first(), true));
		bld.CreateRet(nullptr);

		bld.SetInsertPoint(&updateFunction->getEntryBlock());
		bld.CreateCall(fun);
	}
	for (const auto& pair : upds)
	{
		auto globalVal = m.getGlobalVariable(pair.first(), true);
		auto val = bld.CreateLoad(globalVal);

		bld.CreateStore(
				val, m.getGlobalVariable(pair.first().str() + "_old", true));
	}

	bld.CreateRet(nullptr);
	return updateFunction;
}

static void createPrintOfVar(
		Module& m,
		IRBuilder<>& builder,
		GlobalValue* varString,
		GlobalValue* ptrToVar)
{
	size_t index = 0;
	auto ptrToFirstElem = getArrayElementPtr(builder, ptrToVar, index);
	auto ptrToStrName = getArrayElementPtr(builder, varString, index);

	auto ptrType = ptrToVar->getType();
	auto arrayType = dyn_cast<ArrayType>(ptrType->getContainedType(0));
	auto ptrToBaseType = ptrToFirstElem->getType();
	auto baseType = ptrToBaseType->getContainedType(0);

	auto charPtrType = ptrToStrName->getType();
	auto intType = IntegerType::getInt32Ty(builder.getContext());

	SmallVector<Type*, 3> args({ charPtrType, ptrToBaseType, intType });
	auto printType =
			FunctionType::get(Type::getVoidTy(m.getContext()), args, false);

	const auto selectPrintName = [intType](Type* t) {
		if (t->isFloatTy())
			return "modelicaPrintFVector";
		if (t == intType)
			return "modelicaPrintIVector";

		return "modelicaPrintBVector";
	};

	auto callee = m.getOrInsertFunction(selectPrintName(baseType), printType);
	auto externalPrint = dyn_cast<Function>(callee.getCallee());

	auto numElements = ConstantInt::get(intType, arrayType->getNumElements());
	SmallVector<Value*, 3> argsVal({ ptrToStrName, ptrToFirstElem, numElements });
	builder.CreateCall(externalPrint, argsVal);
}

static Expected<Function*> populatePrint(Module& m, StringMap<SimExp> vars)
{
	auto printFunctionExpected = makePrivateFunction("print", m);
	if (!printFunctionExpected)
		return printFunctionExpected;
	auto printFunction = printFunctionExpected.get();
	IRBuilder bld(&printFunction->getEntryBlock());

	for (const auto& pair : vars)
	{
		auto varString = m.getGlobalVariable(pair.first().str() + "_str");
		auto ptrToVar = m.getGlobalVariable(pair.first());
		createPrintOfVar(m, bld, varString, ptrToVar);
	}
	bld.CreateRet(nullptr);
	return printFunction;
}

Error Simulation::lower()
{
	if (auto e = createAllGlobals(module, variables, getVarLinkage()); e)
		return e;

	auto initFunction = initializeGlobals(module, variables);
	if (!initFunction)
		return initFunction.takeError();

	auto updateFunction = createUpdates(module, updates);
	if (!updateFunction)
		return updateFunction.takeError();

	auto printFunction = populatePrint(module, variables);
	if (!printFunction)
		return printFunction.takeError();

	auto e = populateMain(
			module,
			entryPointName,
			*initFunction,
			*updateFunction,
			*printFunction,
			stopTime);
	if (!e)
		return e.takeError();

	return Error::success();
}

void Simulation::dump(raw_ostream& OS) const
{
	auto const dumpEquation = [&OS](const auto& couple) {
		OS << couple.first().data();
		OS << " = ";
		couple.second.dump(OS);
		OS << "\n";
	};

	OS << "Init:\n";
	for_each(begin(variables), end(variables), dumpEquation);

	OS << "Update:\n";
	for_each(begin(updates), end(updates), dumpEquation);
}

void Simulation::dumpBC(raw_ostream& OS) const
{
	WriteBitcodeToFile(module, OS);
}

void Simulation::dumpHeader(raw_ostream& OS) const
{
	OS << "#pragma once\n\n";

	for (const auto& var : variables)
	{
		const auto& type = var.second.getSimType();

		OS << "extern ";
		type.dumpCSyntax(var.first(), OS);

		OS << ";\n";
	}

	OS << "extern \"C\"{";
	OS << "void " << entryPointName << "();";
	OS << "}";
}
