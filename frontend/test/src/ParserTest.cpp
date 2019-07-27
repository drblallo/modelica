#include "gtest/gtest.h"

#include "modelica/Parser.hpp"

using namespace modelica;

TEST(ParserTest, dyn_castShouldCast)
{
	auto exp = BoolLiteralExpr(SourceRange(0, 0, 1, 1), true);
	Expr* castedExp = &exp;

	EXPECT_EQ(&exp, llvm::dyn_cast<BoolLiteralExpr>(castedExp));
}

TEST(ParserTest, literalIntegerShouldParse)
{
	auto parser = Parser("5");
	auto exp = parser.primary();
	if (!exp)
		FAIL();

	std::unique_ptr<Expr> expr = std::move(exp.get());
	Expr* pointer = &(*expr);

	EXPECT_EQ(expr->getType(), Type(BuiltinType::Integer));

	ASSERT_EQ(llvm::isa<IntLiteralExpr>(pointer), true);
	auto p = llvm::dyn_cast<IntLiteralExpr>(pointer);

	EXPECT_EQ(p->getValue(), 5);
}

TEST(ParserTest, literalStringsShouldParse)
{
	auto parser = Parser("\"Asd\"");
	auto exp = parser.primary();
	if (!exp)
		FAIL();

	std::unique_ptr<Expr> expr = std::move(exp.get());
	Expr* pointer = &(*expr);

	EXPECT_EQ(expr->getType(), Type(BuiltinType::String));

	ASSERT_EQ(llvm::isa<StringLiteralExpr>(pointer), true);
	auto p = llvm::dyn_cast<StringLiteralExpr>(pointer);

	EXPECT_EQ(p->getValue(), "Asd");
}

TEST(ParserTest, ifElseShouldNotFail)
{
	auto parser = Parser("if true then 1 elseif false then 3 else 2");

	auto exp = parser.expression();
	if (!exp)
		FAIL();

	auto pointer = exp.get().get();
	EXPECT_EQ(true, llvm::isa<IfElseExpr>(pointer));
	auto ifelse = llvm::dyn_cast<IfElseExpr>(pointer);

	auto firstExpr = ifelse->getExpression(0);
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(firstExpr));
	auto casted = llvm::dyn_cast<IntLiteralExpr>(firstExpr);
	EXPECT_EQ(1, casted->getValue());
}

TEST(ParserTest, ifElseFailures)
{
	auto parser = Parser("if true true");

	auto exp = parser.expression();
	if (exp)
		FAIL();

	EXPECT_EQ(true, exp.errorIsA<UnexpectedToken>());
}

TEST(ParserTest, logicalExpCanHave2Terms)
{
	auto parser = Parser("true or false");
	auto exp = parser.logicalExpression();
	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));

	auto casted = llvm::cast<BinaryExpr>(ptr);

	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getLeftHand()));
	EXPECT_EQ(BinaryExprOp::LogicalOr, casted->getOpCode());
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getRightHand()));
}

TEST(ParserTest, logicalExpCanHave3Terms)
{
	auto parser = Parser("true or false or true");
	auto exp = parser.logicalExpression();
	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));
}

TEST(ParserTest, logicalTermCanHave2Terms)
{
	auto parser = Parser("true and false");
	auto exp = parser.logicalTerm();
	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));

	auto casted = llvm::cast<BinaryExpr>(ptr);

	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getLeftHand()));
	EXPECT_EQ(BinaryExprOp::LogicalAnd, casted->getOpCode());
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getRightHand()));
}

TEST(ParserTest, logicalTerm)
{
	auto parser = Parser("true and false and true");
	auto exp = parser.logicalTerm();
	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));
}

TEST(ParserTest, arrayConcatExpr)
{
	auto parser = Parser("[1, 2; 3, 4]");
	auto exp = parser.primary();
	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<ArrayConcatExpr>(ptr));
}

TEST(ParserTest, rangeExpression)
{
	auto parser = Parser("1:2:3");
	auto exp = parser.simpleExpression();
	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<RangeExpr>(ptr));
}

TEST(ParserTest, logicalFatorShouldNegate)
{
	auto parser = Parser("not true");
	auto exp = parser.logicalFactor();

	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<UnaryExpr>(ptr));
	auto casted = llvm::cast<UnaryExpr>(ptr);
	EXPECT_EQ(UnaryExprOp::LogicalNot, casted->getOpCode());
}

TEST(ParserTest, relationShouldWork)
{
	auto parser = Parser("true == false");
	auto exp = parser.relation();
	if (!exp)
		FAIL();
	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));

	auto casted = llvm::cast<BinaryExpr>(ptr);

	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getLeftHand()));
	EXPECT_EQ(BinaryExprOp::Equal, casted->getOpCode());
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->getRightHand()));
}

TEST(ParserTest, arithmeticOperation)
{
	auto parser = Parser("-1 + 3");
	auto exp = parser.arithmeticExpression();
	if (!exp)
		FAIL();

	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));

	auto casted = llvm::cast<BinaryExpr>(ptr);
	EXPECT_EQ(BinaryExprOp::Sum, casted->getOpCode());
	auto lhs = casted->getLeftHand();
	EXPECT_EQ(true, llvm::isa<UnaryExpr>(lhs));
	auto castedLhs = llvm::cast<UnaryExpr>(lhs);
	EXPECT_EQ(UnaryExprOp::Minus, castedLhs->getOpCode());
}

TEST(ParserTest, term)
{
	auto parser = Parser("1 * 3");
	auto exp = parser.term();
	if (!exp)
		FAIL();

	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));

	auto casted = llvm::cast<BinaryExpr>(ptr);
	EXPECT_EQ(BinaryExprOp::Multiply, casted->getOpCode());
	auto lhs = casted->getLeftHand();
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(lhs));
}

TEST(ParserTest, forInArrayConstructor)
{
	auto parser = Parser("{2+2 for i in {1, 2, 3}}");
	auto exp = parser.primary();
	if (!exp)
		FAIL();

	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<ForInArrayConstructorExpr>(ptr));
	auto casted = llvm::cast<ForInArrayConstructorExpr>(ptr);

	EXPECT_EQ(true, llvm::isa<BinaryExpr>(casted->getEvaluationExpr()));
	EXPECT_EQ("i", casted->getDeclaredName(0));
	const Expr* e = casted->at(0);
	EXPECT_EQ(true, llvm::isa<DirectArrayConstructorExpr>(e));
}

TEST(Parser, directoArrayConstructorSingleArgument)
{
	auto parser = Parser("{1}");
	auto exp = parser.primary();
	if (!exp)
	{
		llvm::outs() << exp.takeError();
		FAIL();
	}

	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<DirectArrayConstructorExpr>(ptr));
	auto casted = llvm::cast<DirectArrayConstructorExpr>(ptr);

	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(casted->at(0)));
}

TEST(Parser, directoArrayConstructor)
{
	auto parser = Parser("{1, 2, 3}");
	auto exp = parser.expression();
	if (!exp)
		FAIL();

	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(ptr));
	EXPECT_EQ(true, llvm::isa<DirectArrayConstructorExpr>(ptr));
	auto casted = llvm::cast<DirectArrayConstructorExpr>(ptr);

	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(casted->at(0)));
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(casted->at(1)));
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(casted->at(2)));
}

TEST(ParserTest, factor)
{
	auto parser = Parser("1 ^ 3");
	auto exp = parser.factor();
	if (!exp)
		FAIL();

	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));

	auto casted = llvm::cast<BinaryExpr>(ptr);
	EXPECT_EQ(BinaryExprOp::PowerOf, casted->getOpCode());
	auto lhs = casted->getLeftHand();
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(lhs));
}

TEST(ParserTest, multipleArithmeticOperations)
{
	auto parser = Parser("-1 + 3 - 5");
	auto exp = parser.arithmeticExpression();
	if (!exp)
		FAIL();

	auto ptr = exp.get().get();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(ptr));

	auto casted = llvm::cast<BinaryExpr>(ptr);
	EXPECT_EQ(BinaryExprOp::Subtraction, casted->getOpCode());

	auto lhs = casted->getLeftHand();
	EXPECT_EQ(true, llvm::isa<BinaryExpr>(lhs));
	auto castedLhs = llvm::cast<BinaryExpr>(lhs);
	EXPECT_EQ(BinaryExprOp::Sum, castedLhs->getOpCode());

	auto lhs2 = castedLhs->getLeftHand();
	EXPECT_EQ(true, llvm::isa<UnaryExpr>(lhs2));
	auto castedLhs2 = llvm::cast<UnaryExpr>(lhs2);
	EXPECT_EQ(UnaryExprOp::Minus, castedLhs2->getOpCode());
}

TEST(ParserTest, exprCanBeMultiType)
{
	auto parser = Parser("\"a\", true, 1");
	auto list = parser.expressionList();

	if (!list)
		FAIL();

	auto ptr = list.get().get();
	EXPECT_EQ(true, llvm::isa<ExprList>(ptr));

	auto casted = llvm::cast<ExprList>(ptr);
	EXPECT_EQ(3, casted->size());

	EXPECT_EQ(true, llvm::isa<StringLiteralExpr>(casted->at(0)));
	EXPECT_EQ(true, llvm::isa<BoolLiteralExpr>(casted->at(1)));
	EXPECT_EQ(true, llvm::isa<IntLiteralExpr>(casted->at(2)));
}
