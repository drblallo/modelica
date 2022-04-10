#include "marco/Dialect/Modelica/Attributes.h"
#include "marco/Dialect/Modelica/ModelicaDialect.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir::modelica
{
  mlir::Attribute getAttr(mlir::Type type, long value)
  {
    if (type.isa<BooleanType>()) {
      return BooleanAttr::get(type.getContext(), value > 0);
    }

    if (type.isa<IntegerType>()) {
      return IntegerAttr::get(type.getContext(), value);
    }

    if (type.isa<RealType>()) {
      return RealAttr::get(type.getContext(), value);
    }

    llvm_unreachable("Unknown Modelica type");
    return {};
  }

  mlir::Attribute getAttr(mlir::Type type, double value)
  {
    if (type.isa<BooleanType>()) {
      return BooleanAttr::get(type.getContext(), value > 0);
    }

    if (type.isa<IntegerType>()) {
      return IntegerAttr::get(type.getContext(), value);
    }

    if (type.isa<RealType>()) {
      return RealAttr::get(type.getContext(), value);
    }

    llvm_unreachable("Unknown Modelica type");
    return {};
  }

  mlir::Attribute getZeroAttr(mlir::Type type)
  {
    if (type.isa<BooleanType>()) {
      return BooleanAttr::get(type.getContext(), false);
    }

    if (type.isa<IntegerType>()) {
      return IntegerAttr::get(type.getContext(), 0);
    }

    if (type.isa<RealType>()) {
      return RealAttr::get(type.getContext(), 0);
    }

    llvm_unreachable("Unknown Modelica type");
    return mlir::Attribute();
  }

  //===----------------------------------------------------------------------===//
  // BooleanAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute BooleanAttr::parse(
      mlir::MLIRContext* context, mlir::DialectAsmParser& parser, mlir::Type type)
  {
    bool value;

    if (parser.parseLess()) {
      return {};
    }

    if (mlir::succeeded(parser.parseOptionalKeyword("true"))) {
      value = true;
    } else {
      if (parser.parseKeyword("false")) {
        return {};
      }

      value = false;
    }

    if (parser.parseGreater()) {
      return {};
    }

    if (!type) {
      type = BooleanType::get(context);
    }

    return BooleanAttr::get(context, type, value);
  }

  void BooleanAttr::print(mlir::DialectAsmPrinter& os) const
  {
    os << "bool<" << (getValue() == 0 ? "false" : "true") << ">";
  }

  //===----------------------------------------------------------------------===//
  // IntegerAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute IntegerAttr::parse(
      mlir::MLIRContext* context, mlir::DialectAsmParser& parser, mlir::Type type)
  {
    long value;

    if (parser.parseLess() ||
        parser.parseInteger(value) ||
        parser.parseGreater()) {
      return {};
    }

    if (!type) {
      type = IntegerType::get(context);
    }

    return IntegerAttr::get(context, type, llvm::APInt(sizeof(long) * 8, value, true));
  }

  void IntegerAttr::print(mlir::DialectAsmPrinter& os) const
  {
    os << "int<" << getValue() << ">";
  }

  //===----------------------------------------------------------------------===//
  // RealAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute RealAttr::parse(
      mlir::MLIRContext* context, mlir::DialectAsmParser& parser, mlir::Type type)
  {
    double value;

    if (parser.parseLess() ||
        parser.parseFloat(value) ||
        parser.parseGreater()) {
      return {};
    }

    if (!type) {
      type = RealType::get(context);
    }

    return RealAttr::get(context, type, llvm::APFloat(value));
  }

  void RealAttr::print(mlir::DialectAsmPrinter& os) const
  {
    os << "real<" << getValue() << ">";
  }

  //===----------------------------------------------------------------------===//
  // InverseFunctionsAttr
  //===----------------------------------------------------------------------===//

  bool InverseFunctionsMap::operator==(const InverseFunctionsMap& other) const
  {
    return map == other.map;
  }

  InverseFunctionsMap::InverseFunction& InverseFunctionsMap::operator[](unsigned int arg)
  {
    return map[arg];
  }

  bool InverseFunctionsMap::empty() const
  {
    return map.empty();
  }

  InverseFunctionsMap::iterator InverseFunctionsMap::begin()
  {
    return map.begin();
  }

  InverseFunctionsMap::const_iterator InverseFunctionsMap::begin() const
  {
    return map.begin();
  }

  InverseFunctionsMap::iterator InverseFunctionsMap::end()
  {
    return map.end();
  }

  InverseFunctionsMap::const_iterator InverseFunctionsMap::end() const
  {
    return map.end();
  }

  InverseFunctionsMap InverseFunctionsMap::allocateInto(mlir::StorageUniquer::StorageAllocator& allocator)
  {
    InverseFunctionsMap result;

    for (const auto& entry : map) {
      result.map[entry.first] = std::make_pair(
          allocator.copyInto(entry.second.first),
          allocator.copyInto(entry.second.second));
    }

    return result;
  }

  bool InverseFunctionsMap::isInvertible(unsigned int argumentIndex) const
  {
    return map.find(argumentIndex) != map.end();
  }

  llvm::StringRef InverseFunctionsMap::getFunction(unsigned int argumentIndex) const
  {
    return map.find(argumentIndex)->second.first;
  }

  llvm::ArrayRef<unsigned int> InverseFunctionsMap::getArgumentsIndexes(unsigned int argumentIndex) const
  {
    return map.find(argumentIndex)->second.second;
  }

  llvm::hash_code hash_value(const InverseFunctionsMap& map) {
    return llvm::hash_combine_range(map.begin(), map.end());
  }

  mlir::Attribute InverseFunctionsAttr::parse(mlir::MLIRContext*, mlir::DialectAsmParser&, mlir::Type)
  {
    // TODO parse InverseFunctionsAttr
    llvm_unreachable("InverseFunctionsAttr parsing is not implemented");
    return mlir::Attribute();
  }

  void InverseFunctionsAttr::print(mlir::DialectAsmPrinter& os) const
  {
    os << "inverse<";
    auto inverseFunctionsMap = getInverseFunctionsMap();
    bool separator = false;

    for (auto invertibleArg : inverseFunctionsMap) {
      if (separator) {
        os << ", ";
      }

      os << invertibleArg << ": ";
      os << inverseFunctionsMap.getFunction(invertibleArg) << "(";
      bool innerSeparator = false;

      for (const auto& arg : inverseFunctionsMap.getArgumentsIndexes(invertibleArg)) {
        if (innerSeparator) {
          os << ", ";
        }

        os << arg;
        innerSeparator = true;
      }

      os << ")";
      separator = true;
    }

    os << ">";
  }

  bool InverseFunctionsAttr::isInvertible(unsigned int argumentIndex) const
  {
    return getInverseFunctionsMap().isInvertible(argumentIndex);
  }

  llvm::StringRef InverseFunctionsAttr::getFunction(unsigned int argumentIndex) const
  {
    return getInverseFunctionsMap().getFunction(argumentIndex);
  }

  llvm::ArrayRef<unsigned int> InverseFunctionsAttr::getArgumentsIndexes(unsigned int argumentIndex) const
  {
    return getInverseFunctionsMap().getArgumentsIndexes(argumentIndex);
  }

  //===----------------------------------------------------------------------===//
  // DerivativeAttr
  //===----------------------------------------------------------------------===//

  mlir::Attribute DerivativeAttr::parse(
      mlir::MLIRContext* context, mlir::DialectAsmParser& parser, mlir::Type type)
  {
    mlir::StringAttr name;
    unsigned int order = 1;

    if (parser.parseLess() ||
        parser.parseAttribute(name) ||
        parser.parseComma() ||
        parser.parseInteger(order) ||
        parser.parseGreater()) {
      return mlir::Attribute();
    }

    return DerivativeAttr::get(context, name.getValue(), order);
  }

  void DerivativeAttr::print(mlir::DialectAsmPrinter& os) const
  {
    os << "derivative" << "<\"" << getName() << "\", " << getOrder() << ">";
  }
}
