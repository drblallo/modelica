#include "marco/Modeling/DimensionAccessDiv.h"
#include "llvm/Support/raw_ostream.h"

using namespace ::marco::modeling;

namespace marco::modeling
{
  DimensionAccessDiv::DimensionAccessDiv(
      mlir::MLIRContext* context,
      std::unique_ptr<DimensionAccess> first,
      std::unique_ptr<DimensionAccess> second)
      : DimensionAccess(DimensionAccess::Kind::Div, context),
        first(std::move(first)),
        second(std::move(second))
  {
  }

  DimensionAccessDiv::DimensionAccessDiv(const DimensionAccessDiv& other)
      : DimensionAccess(other),
        first(other.getFirst().clone()),
        second(other.getSecond().clone())
  {
  }

  DimensionAccessDiv::DimensionAccessDiv(DimensionAccessDiv&& other) = default;

  DimensionAccessDiv::~DimensionAccessDiv() = default;

  DimensionAccessDiv& DimensionAccessDiv::operator=(
      const DimensionAccessDiv& other)
  {
    DimensionAccessDiv result(other);
    swap(*this, result);
    return *this;
  }

  DimensionAccessDiv& DimensionAccessDiv::operator=(
      DimensionAccessDiv&& other) = default;

  void swap(DimensionAccessDiv& first, DimensionAccessDiv& second)
  {
    using std::swap;

    swap(static_cast<DimensionAccess&>(first),
         static_cast<DimensionAccess&>(second));

    swap(first.first, second.first);
    swap(first.second, second.second);
  }

  std::unique_ptr<DimensionAccess> DimensionAccessDiv::clone() const
  {
    return std::make_unique<DimensionAccessDiv>(*this);
  }

  bool DimensionAccessDiv::operator==(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessDiv>()) {
      return *this == *otherCasted;
    }

    return false;
  }

  bool DimensionAccessDiv::operator==(const DimensionAccessDiv& other) const
  {
    return getFirst() == other.getFirst() && getSecond() == other.getSecond();
  }

  bool DimensionAccessDiv::operator!=(const DimensionAccess& other) const
  {
    if (auto otherCasted = other.dyn_cast<DimensionAccessDiv>()) {
      return *this != *otherCasted;
    }

    return true;
  }

  bool DimensionAccessDiv::operator!=(const DimensionAccessDiv& other) const
  {
    return getFirst() != other.getFirst() || getSecond() != other.getSecond();
  }

  llvm::raw_ostream& DimensionAccessDiv::dump(llvm::raw_ostream& os) const
  {
    return os << "(" << getFirst() << " / " << getSecond() << ")";
  }

  bool DimensionAccessDiv::isAffine() const
  {
    return getFirst().isAffine() && getSecond().isAffine();
  }

  mlir::AffineExpr DimensionAccessDiv::getAffineExpr() const
  {
    assert(isAffine());
    return getFirst().getAffineExpr().floorDiv(getSecond().getAffineExpr());
  }

  mlir::AffineExpr DimensionAccessDiv::getAffineExpr(
      unsigned int numOfDimensions,
      DimensionAccess::FakeDimensionsMap& fakeDimensionsMap) const
  {
    mlir::AffineExpr firstExpr =
        getFirst().getAffineExpr(numOfDimensions, fakeDimensionsMap);

    mlir::AffineExpr secondExpr =
        getSecond().getAffineExpr(numOfDimensions, fakeDimensionsMap);

    return firstExpr.floorDiv(secondExpr);
  }

  IndexSet DimensionAccessDiv::map(const Point& point) const
  {
    const DimensionAccess& lhs = getFirst();
    const DimensionAccess& rhs = getSecond();

    IndexSet mappedLhs = lhs.map(point);
    IndexSet mappedRhs = rhs.map(point);

    IndexSet result;
    llvm::SmallVector<Point::data_type, 10> coordinates;

    for (Point mappedLhsPoint : mappedLhs) {
      for (Point mappedRhsPoint : mappedRhs) {
        assert(mappedLhsPoint.rank() == mappedRhsPoint.rank());
        coordinates.clear();

        for (size_t i = 0, e = mappedLhsPoint.rank(); i < e; ++i) {
          coordinates.push_back(mappedLhsPoint[i] / mappedRhsPoint[i]);
        }

        result += Point(coordinates);
      }
    }

    return result;
  }

  DimensionAccess& DimensionAccessDiv::getFirst()
  {
    assert(first && "First operand not set");
    return *first;
  }

  const DimensionAccess& DimensionAccessDiv::getFirst() const
  {
    assert(first && "First operand not set");
    return *first;
  }

  DimensionAccess& DimensionAccessDiv::getSecond()
  {
    assert(second && "Second operand not set");
    return *second;
  }

  const DimensionAccess& DimensionAccessDiv::getSecond() const
  {
    assert(second && "Second operand not set");
    return *second;
  }
}
