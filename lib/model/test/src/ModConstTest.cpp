#include "gtest/gtest.h"

#include "marco/model/ModCall.hpp"
#include "marco/model/ModConst.hpp"
#include "marco/model/ModExp.hpp"
#include "marco/model/Model.hpp"

using namespace marco;
using namespace std;

TEST(ModConstTest, constantVectorCanBeAdded)
{
	ModConst l(3, 4, 5);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::sum(l, r), ModConst(8, 8, 10));
}

TEST(ModConstTest, constantVectorCanSubtracted)
{
	ModConst l(3, 4, 5);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::sub(l, r), ModConst(-2, 0, 0));
}

TEST(ModConstTest, constantVectorCanMultipled)
{
	ModConst l(3, 4, 5);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::mult(l, r), ModConst(15, 16, 25));
}

TEST(ModConstTest, constantVectorCanBeDivided)
{
	ModConst l(10, 16, 25);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::divide(l, r), ModConst(2, 4, 5));
}

TEST(ModConstTest, constantVectorGreaterThan)
{
	ModConst l(10, 2, 25);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::greaterThan(l, r), ModConst(true, false, true));
}

TEST(ModConstTest, constantVectorGreaterEqual)
{
	ModConst l(10, 4, 1);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::greaterEqual(l, r), ModConst(true, true, false));
}

TEST(ModConstTest, constantVectorEqual)
{
	ModConst l(10, 4, 1);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::equal(l, r), ModConst(false, true, false));
}

TEST(ModConstTest, constantVectorDifferent)
{
	ModConst l(10, 4, 1);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::different(l, r), ModConst(true, false, true));
}

TEST(ModConstTest, constantVectorLess)
{
	ModConst l(10, 4, 1);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::lessThan(l, r), ModConst(false, false, true));
}

TEST(ModConstTest, constantVectorLessEqual)
{
	ModConst l(10, 4, 1);
	ModConst r(5, 4, 5);
	EXPECT_EQ(ModConst::lessEqual(l, r), ModConst(false, true, true));
}

TEST(ModConstTest, constantVectorElevate)
{
	ModConst l(10, 4, 1);
	ModConst r(1, 2, 3);
	EXPECT_EQ(ModConst::elevate(l, r), ModConst(10, 16, 1));
}

TEST(ModConstTest, constantVectorModule)
{
	ModConst l(10, 4, 1);
	ModConst r(1, 2, 3);
	EXPECT_EQ(ModConst::module(l, r), ModConst(0, 0, 1));
}

TEST(ModConstTest, constantVectorAddedAreCasted)
{
	ModConst l(3.0, 4.0, 5.0);
	ModConst r(5, 4, 5);

	auto sum = ModConst::sum(l, r);
	EXPECT_TRUE(sum.isA<double>());

	EXPECT_NEAR(sum.get<double>(0), 8.0f, 0.1);
}

TEST(ModConstTest, constantShouldBeMoveAsssignable)
{
	ModConst c(4);
	c = ModConst(3);
	EXPECT_EQ(c.get<long>(0), 3);
}
