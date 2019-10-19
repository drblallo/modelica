#pragma once

#include "llvm/ADT/StringMap.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "modelica/model/Assigment.hpp"
#include "modelica/model/ModExp.hpp"

namespace modelica
{
	/**
	 * The default number of iterations to be performed.
	 * Totaly arbitrary number.
	 */
	constexpr int defaultModulationIterations = 10;

	/**
	 * A Lowerer is the main container of the library. It lowers
	 * a simulation and dumps into a bc file to be later compiled.
	 *
	 * A Lowerer is made of a inizialization section and of an update section.
	 * The generated file will invoke the initialization values once and then
	 * update a certain number of time and will print the values of the vars at
	 * each update.
	 */
	class Lowerer
	{
		public:
		Lowerer(
				llvm::LLVMContext& context,
				llvm::StringMap<ModExp> vars,
				llvm::SmallVector<Assigment, 0> updates,
				std::string name = "Modelica Module",
				std::string entryPointName = "main",
				unsigned stopTime = defaultModulationIterations)
				: module(std::move(name), context),
					variables(std::move(vars)),
					updates(std::move(updates)),
					stopTime(stopTime),
					entryPointName(std::move(entryPointName)),
					varsLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage)
		{
		}

		Lowerer(
				llvm::LLVMContext& context,
				std::string name = "Modelica Module",
				std::string entryPointName = "main",
				unsigned stopTime = defaultModulationIterations)
				: module(std::move(name), context),
					stopTime(stopTime),
					entryPointName(std::move(entryPointName)),
					varsLinkage(llvm::GlobalValue::LinkageTypes::InternalLinkage)
		{
		}

		/**
		 * adds a var to the simulation that will be intialized with the provided
		 * expression. Notice that in the initialization is undefined behaviour to
		 * use references to other variables.
		 *
		 * \return true if there were no other vars with the same name already.
		 */
		[[nodiscard]] bool addVar(std::string name, ModExp exp)
		{
			if (variables.find(name) != variables.end())
				return false;
			variables.try_emplace(std::move(name), std::move(exp));
			return true;
		}

		/**
		 * Add an update expression for a particular variable.
		 * notice that if a expression is referring to a missing
		 * variable then it's lower that will fail, not addUpdate
		 *
		 */
		void addUpdate(Assigment assigment)
		{
			updates.push_back(std::move(assigment));
		}

		/**
		 * \requires lower was not already invoked on this object.
		 * Populate the module with the simulation.
		 *
		 * \return a error if there were missing references or if a type missmatch
		 * was encountered.
		 */
		llvm::Error lower();

		/**
		 * dumpds a human readable rappresentation of the simulation to OS, standard
		 * out by default
		 */
		void dump(llvm::raw_ostream& OS = llvm::outs()) const;

		/**
		 * \requires lower has been invoked and it returned success.
		 * dumps the simulation as bytecode
		 */
		void dumpBC(llvm::raw_ostream& OS) const;

		/**
		 * generates the header for this particular simulation
		 */
		void dumpHeader(llvm::raw_ostream& OS) const;

		/**
		 * \return the number of updates of the simulation
		 */
		[[nodiscard]] unsigned getStopTime() const { return stopTime; }

		/**
		 * \return the kind of linkage specified to the variables.
		 *
		 * This is usefull if you need to expose variables that are by default
		 * internal so that other programs can link to the simulation and drive it.
		 */
		[[nodiscard]] llvm::GlobalValue::LinkageTypes getVarLinkage() const
		{
			return varsLinkage;
		}

		/**
		 *
		 * Set the linkage type of the variables
		 * This is usefull if you need to expose variables that are by default
		 * internal so that other programs can link to the simulation and drive it.
		 */
		void setVarsLinkage(llvm::GlobalValue::LinkageTypes newLinkage)
		{
			varsLinkage = newLinkage;
		}

		private:
		llvm::Module module;
		llvm::StringMap<ModExp> variables;
		llvm::SmallVector<Assigment, 0> updates;
		unsigned stopTime;

		std::string entryPointName;
		llvm::GlobalValue::LinkageTypes varsLinkage;
	};
}	 // namespace modelica
