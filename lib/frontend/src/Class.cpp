#include <modelica/frontend/Class.hpp>

using namespace llvm;
using namespace modelica;
using namespace std;

raw_ostream& modelica::operator<<(raw_ostream& stream, const ClassType& obj)
{
	return stream << toString(obj);
}

string modelica::toString(ClassType type)
{
	switch (type)
	{
		case ClassType::Block:
			return "block";
		case ClassType::Class:
			return "class";
		case ClassType::Connector:
			return "connector";
		case ClassType::Function:
			return "function";
		case ClassType::Model:
			return "model";
		case ClassType::Package:
			return "package";
		case ClassType::Operator:
			return "operator";
		case ClassType::Record:
			return "record";
		case ClassType::Type:
			return "type";
	}

	assert(false && "Unknown class type");
}

Class::Class(
		ClassType classType,
		string name,
		ArrayRef<Member> members,
		ArrayRef<Equation> equations,
		ArrayRef<ForEquation> forEquations)
		: classType(move(classType)),
			name(move(name)),
			type(Type::unknown()),
			members(iterator_range<ArrayRef<Member>::iterator>(move(members))),
			equations(iterator_range<ArrayRef<Equation>::iterator>(move(equations))),
			forEquations(
					iterator_range<ArrayRef<ForEquation>::iterator>(move(forEquations)))
{
}

void Class::dump() const { dump(outs(), 0); }

void Class::dump(raw_ostream& os, size_t indents) const
{
	os.indent(indents);
	os << classType << " " << name << "\n";

	os.indent(indents + 1);
	os << "type: ";
	type.dump(os);
	os << "\n";

	for (const auto& member : members)
		member.dump(os, indents + 1);

	for (const auto& function : functions)
		function->dump(os, indents + 1);

	for (const auto& equation : equations)
		equation.dump(os, indents + 1);

	for (const auto& equation : forEquations)
		equation.dump(os, indents + 1);

	for (const auto& algorithm : algorithms)
		algorithm.dump(os, indents + 1);
}

ClassType Class::getClassType() const { return classType; }

string& Class::getName() { return name; }

Type& Class::getType() { return type; }

const Type& Class::getType() const { return type; }

void Class::setType(Type t) { type = move(t); }

SmallVectorImpl<Member>& Class::getMembers() { return members; }

SmallVectorImpl<Equation>& Class::getEquations() { return equations; }

SmallVectorImpl<ForEquation>& Class::getForEquations() { return forEquations; }

SmallVectorImpl<Algorithm>& Class::getAlgorithms() { return algorithms; }

SmallVectorImpl<Func>& Class::getFunctions() { return functions; }

const string& Class::getName() const { return name; }

const SmallVectorImpl<Member>& Class::getMembers() const { return members; }

size_t Class::membersCount() const { return members.size(); }

const SmallVectorImpl<Equation>& Class::getEquations() const
{
	return equations;
}

const SmallVectorImpl<ForEquation>& Class::getForEquations() const
{
	return forEquations;
}

const SmallVectorImpl<Algorithm>& Class::getAlgorithms() const
{
	return algorithms;
}

const SmallVectorImpl<Func>& Class::getFunctions() const { return functions; }

void Class::addMember(Member newMember)
{
	return members.push_back(std::move(newMember));
}

void Class::eraseMember(size_t memberIndex)
{
	assert(memberIndex < members.size());
	members.erase(members.begin() + memberIndex);
}

void Class::addEquation(Equation equation)
{
	return equations.push_back(move(equation));
}

void Class::addForEquation(ForEquation equation)
{
	return forEquations.push_back(move(equation));
}

void Class::addAlgorithm(Algorithm algorithm)
{
	algorithms.push_back(move(algorithm));
}

void Class::addFunction(Class function)
{
	return functions.push_back(std::make_unique<Class>(move(function)));
}
