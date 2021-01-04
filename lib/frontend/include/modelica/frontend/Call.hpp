#pragma once

#include <boost/iterator/indirect_iterator.hpp>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/raw_ostream.h>
#include <modelica/utils/SourceRange.hpp>

namespace modelica
{
	class Expression;

	class Call
	{
		private:
		using UniqueExpression = std::unique_ptr<Expression>;
		template<typename T> using Container = llvm::SmallVector<T, 3>;

		public:
		using args_iterator = boost::indirect_iterator<Container<UniqueExpression>::iterator>;
		using args_const_iterator = boost::indirect_iterator<Container<UniqueExpression>::const_iterator>;

		Call(SourcePosition location, Expression function, llvm::ArrayRef<Expression> args = {});

		Call(const Call& other);
		Call(Call&& other) = default;

		Call& operator=(const Call& other);
		Call& operator=(Call&& other) = default;

		~Call() = default;

		[[nodiscard]] bool operator==(const Call& other) const;
		[[nodiscard]] bool operator!=(const Call& other) const;

		[[nodiscard]] Expression& operator[](size_t index);
		[[nodiscard]] const Expression& operator[](size_t index) const;

		void dump() const;
		void dump(llvm::raw_ostream& os, size_t indents = 0) const;

		[[nodiscard]] SourcePosition getLocation() const;

		[[nodiscard]] Expression& getFunction();
		[[nodiscard]] const Expression& getFunction() const;

		[[nodiscard]] size_t argumentsCount() const;

		[[nodiscard]] args_iterator begin();
		[[nodiscard]] args_const_iterator begin() const;

		[[nodiscard]] args_iterator end();
		[[nodiscard]] args_const_iterator end() const;

		private:
		SourcePosition location;
		UniqueExpression function;
		Container<UniqueExpression> args;
	};
}	 // namespace modelica
