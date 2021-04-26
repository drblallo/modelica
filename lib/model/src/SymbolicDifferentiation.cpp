#include "modelica/model/SymbolicDifferentiation.hpp"

#include "modelica/model/ModEquation.hpp"
#include "modelica/model/ModExp.hpp"
#include "modelica/model/ModVariable.hpp"

using namespace modelica;
using namespace std;

ModExp differentiateVariableVector(const ModExp& exp, const ModVariable& var)
{
	// TODO: Check if the induction variable is the same.
	// In that case return 1, otherwise return 0.
	return ModConst(1.0);
}

ModExp differentiateNegate(const ModExp& exp, const ModVariable& var)
{
	// If the expression is constant, return 0.
	if (exp.getLeftHand().isConstant())
		return ModConst(0.0);

	// Otherwise, return the negated derivative of the espression.
	ModExp leftDerivative = differentiate(exp.getLeftHand(), var);

	ModExp result = ModExp::negate(move(leftDerivative));
	result.tryFoldConstant();
	return result;
}

ModExp differentiateAddition(const ModExp& exp, const ModVariable& var)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If a side is constant, return the derivative of the other side.
	if (exp.getLeftHand().isConstant())
		return differentiate(exp.getRightHand(), var);
	if (exp.getRightHand().isConstant())
		return differentiate(exp.getLeftHand(), var);

	// Otherwise, return der(f(x)) + der(g(x)).
	ModExp leftHand = differentiate(exp.getLeftHand(), var);
	ModExp rightHand = differentiate(exp.getRightHand(), var);

	ModExp result = ModExp::add(move(leftHand), move(rightHand));
	result.tryFoldConstant();
	return result;
}

ModExp differentiateSubtraction(const ModExp& exp, const ModVariable& var)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If a side is constant, return the derivative of the other side.
	if (exp.getRightHand().isConstant())
		return differentiate(exp.getLeftHand(), var);

	ModExp rightHand = differentiate(exp.getRightHand(), var);

	if (exp.getLeftHand().isConstant())
	{
		ModExp result = ModExp::negate(rightHand);
		result.tryFoldConstant();
		return result;
	}

	// Otherwise, return der(f(x)) - der(g(x)).
	ModExp leftHand = differentiate(exp.getLeftHand(), var);

	ModExp result = ModExp::subtract(move(leftHand), move(rightHand));
	result.tryFoldConstant();
	return result;
}

ModExp differentiateMultiplication(const ModExp& exp, const ModVariable& var)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If a side is constant, return the derivative of the otherside multiplied by
	// the constant.
	if (exp.getLeftHand().isConstant())
	{
		ModExp rightDerivative = differentiate(exp.getRightHand(), var);
		ModExp result = ModExp::multiply(exp.getLeftHand(), move(rightDerivative));
		result.tryFoldConstant();
		return result;
	}
	if (exp.getRightHand().isConstant())
	{
		ModExp leftDerivative = differentiate(exp.getLeftHand(), var);
		ModExp result = ModExp::multiply(exp.getRightHand(), move(leftDerivative));
		result.tryFoldConstant();
		return result;
	}

	// Otherwise, return f(x)*der(g(x)) + g(x)*der(f(x)).
	ModExp leftDerivative = differentiate(exp.getLeftHand(), var);
	ModExp rightDerivative = differentiate(exp.getRightHand(), var);

	ModExp leftHand = ModExp::multiply(exp.getLeftHand(), move(rightDerivative));
	leftHand.tryFoldConstant();
	ModExp rightHand = ModExp::multiply(exp.getRightHand(), move(leftDerivative));
	rightHand.tryFoldConstant();

	ModExp result = ModExp::add(move(leftHand), move(rightHand));
	result.tryFoldConstant();
	return result;
}

ModExp differentiateDivision(const ModExp& exp, const ModVariable& var)
{
	// If both sides are constants, return 0.
	if (exp.getLeftHand().isConstant() && exp.getRightHand().isConstant())
		return ModConst(0.0);

	// If the divisor is a constant, return der(f(x)) / g(x)
	if (exp.getRightHand().isConstant())
	{
		ModExp leftDerivative = differentiate(exp.getLeftHand(), var);
		ModExp result = ModExp::divide(move(leftDerivative), exp.getRightHand());
		result.tryFoldConstant();
		return result;
	}

	// If the dividend is constant, return -f(x)*der(g(x)) / (g(x)*g(x))
	if (exp.getLeftHand().isConstant())
	{
		ModExp rightDerivative = differentiate(exp.getRightHand(), var);
		ModExp dividend =
				ModExp::multiply(exp.getLeftHand(), move(rightDerivative));
		dividend.tryFoldConstant();
		dividend = ModExp::negate(move(dividend));
		dividend.tryFoldConstant();

		ModExp divisor = ModExp::multiply(exp.getRightHand(), exp.getRightHand());
		divisor.tryFoldConstant();

		ModExp result = ModExp::divide(move(dividend), move(divisor));
		result.tryFoldConstant();
		return result;
	}

	// Otherwise return (g(x)*der(f(x)) - f(x)*der(g(x))) / (g(x) * g(x)).
	ModExp leftDerivative = differentiate(exp.getLeftHand(), var);
	ModExp rightDerivative = differentiate(exp.getRightHand(), var);

	ModExp lDividend = ModExp::multiply(exp.getRightHand(), move(leftDerivative));
	lDividend.tryFoldConstant();
	ModExp rDividend = ModExp::multiply(exp.getLeftHand(), move(rightDerivative));
	rDividend.tryFoldConstant();

	ModExp dividend = ModExp::subtract(move(lDividend), move(rDividend));
	dividend.tryFoldConstant();
	ModExp divisor = ModExp::multiply(exp.getRightHand(), exp.getRightHand());
	divisor.tryFoldConstant();

	ModExp result = ModExp::divide(move(dividend), move(divisor));
	result.tryFoldConstant();
	return result;
}

ModExp differentiateElevation(const ModExp& exp, const ModVariable& var)
{
	assert(exp.getRightHand().isConstant());

	ModConst exponent = exp.getRightHand().getConstant().as<double>();

	// If the base is constant, return 0.
	if (exp.getLeftHand().isConstant())
		return ModConst(0.0);

	// If the exponent is 1, return the derivative of the base.
	if (exponent == ModConst(1.0))
		return differentiate(exp.getLeftHand(), var);

	ModExp leftDerivative = differentiate(exp.getLeftHand(), var);

	// If the exponent is 2, return two times the derivative of the base.
	if (exponent == ModConst(2.0))
	{
		ModExp right = ModExp::multiply(exp.getLeftHand(), move(leftDerivative));
		right.tryFoldConstant();

		ModExp result = ModExp::multiply(ModConst(2.0), move(right));
		result.tryFoldConstant();
		return result;
	}

	// Otherwise return  (c * der(f(x))) * exp(f(x), (c-1)).
	ModExp elevationResidual = ModExp::elevate(
			exp.getLeftHand(), ModConst::sub(exponent, ModConst(1.0)));
	elevationResidual.tryFoldConstant();
	ModExp left = ModExp::multiply(exponent, move(leftDerivative));
	left.tryFoldConstant();

	ModExp result = ModExp::multiply(move(left), move(elevationResidual));
	result.tryFoldConstant();
	return result;
}

ModExp modelica::differentiate(const ModExp& exp, const ModVariable& var)
{
	if (exp.isConstant())
		return ModConst(0.0);

	if (exp.isReference() && exp.getReference() != var.getName())
		return ModConst(0.0);

	if (exp.isReference() && exp.getReference() == var.getName())
		return ModConst(1.0);

	assert(exp.isOperation());

	if (exp.isReferenceAccess() && exp.getReference() == var.getName())
		return differentiateVariableVector(exp, var);

	switch (exp.getKind())
	{
		case ModExpKind::negate:
			return differentiateNegate(exp, var);
		case ModExpKind::add:
			return differentiateAddition(exp, var);
		case ModExpKind::sub:
			return differentiateSubtraction(exp, var);
		case ModExpKind::mult:
			return differentiateMultiplication(exp, var);
		case ModExpKind::divide:
			return differentiateDivision(exp, var);
		case ModExpKind::elevation:
			return differentiateElevation(exp, var);
		default:
			assert(false && "Not differentiable expression");
	}

	assert(false && "Unreachable");
}

ModEquation modelica::differentiate(
		const ModEquation& equation, const ModVariable& var)
{
	ModExp leftHand = differentiate(equation.getLeft(), var);
	ModExp rightHand = differentiate(equation.getRight(), var);

	return ModEquation(move(leftHand), move(rightHand));
}
