#pragma once

#include <llvm/ADT/SmallVector.h>
#include <variant>

#include "Expression.h"
#include "Model.h"
#include "Path.h"

namespace marco::codegen::model
{
	/**
	 * This class, given an equation or a blt block, finds and stores pointers
	 * to subexpressions that are variable accesses.
	 */
	class ReferenceMatcher
	{
		private:
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using iterator = Container<ExpressionPath>::iterator;
		using const_iterator = Container<ExpressionPath>::const_iterator;

		ReferenceMatcher();
		ReferenceMatcher(std::variant<Equation, BltBlock> content);

		[[nodiscard]] ExpressionPath& operator[](size_t index);
		[[nodiscard]] const ExpressionPath& operator[](size_t index) const;

		[[nodiscard]] iterator begin();
		[[nodiscard]] const_iterator begin() const;

		[[nodiscard]] iterator end();
		[[nodiscard]] const_iterator end() const;

		[[nodiscard]] size_t size() const;

		[[nodiscard]] ExpressionPath& at(size_t index);
		[[nodiscard]] const ExpressionPath& at(size_t index) const;

		[[nodiscard]] Expression getExp(size_t index) const;

		void visit(Expression exp, bool isLeft);
		void visit(Equation equation, bool ignoreMatched = false);
		void visit(std::variant<Equation, BltBlock> content, bool ignoreMatched = false);

		private:
		void removeBack();

		Container<size_t> currentPath;
		Container<ExpressionPath> vars;
	};

};	// namespace marco
