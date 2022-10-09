#include "marco/Codegen/Transforms/ModelSolving/CyclesCramerSolver.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;

#define DEBUG_TYPE "CyclesSolving"

SquareMatrix::SquareMatrix(
    std::vector<mlir::Value>& storage,
    size_t size) : size(size), storage(storage)
{
}

size_t SquareMatrix::getSize() const
{
  return size;
}

mlir::Value& SquareMatrix::operator()(size_t row, size_t col)
{
  assert(row*size + col < storage.size());
  return storage[row*size + col];
}

mlir::Value SquareMatrix::operator()(size_t row, size_t col) const
{
  assert(row*size + col < storage.size());
  return storage[row*size + col];
}

void SquareMatrix::print(llvm::raw_ostream &os)
{
  os << "MATRIX SIZE: " << size << "\n";
  for (size_t i = 0; i < size; i++) {
    os << "LINE #" << i << "\n";
    for (size_t j = 0; j < size; j++) {
      (*this)(i,j).print(os);
      os << "\n";
    }
    os << "\n";
  }
  os << "\n";
  os.flush();
}

void SquareMatrix::dump()
{
  print(llvm::dbgs());
}

SquareMatrix SquareMatrix::subMatrix(SquareMatrix out, size_t row, size_t col)
{
  for (size_t i = 0; i < size; i++)
  {
    if(i != row) {
      for (size_t j = 0; j < size; j++) {
        if(j != col) {
          out(i < row ? i : i - 1, j < col ? j : j - 1) =
              (*this)(i, j);
        }
      }
    }
  }
  return out;
}

SquareMatrix SquareMatrix::substituteColumn(
    SquareMatrix out,
    size_t colNumber,
    std::vector<mlir::Value>& col)
{
  assert(colNumber < size && out.getSize() == size);
  for (size_t i = 0; i < size; i++) {
    for (size_t j = 0; j < size; j++) {
      if(j != colNumber)
        out(i,j) = (*this)(i,j);
      else
        out(i,j) = col[i];
    }
  }
  return out;
}

mlir::Value SquareMatrix::det(mlir::OpBuilder& builder)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  SquareMatrix matrix = (*this);
  mlir::Value determinant;
  mlir::Location loc = builder.getUnknownLoc();


  if(size == 1)
    // The matrix is a scalar, the determinant is the scalar itself
    determinant = matrix(0, 0);

  else if(size == 2) {
    // The matrix is 2x2, the determinant is ad - bc where the matrix is:
    //    a b
    //    c d
    auto a = matrix(0,0);
    auto b = matrix(0,1);
    auto c = matrix(1,0);
    auto d = matrix(1,1);

    auto ad = builder.createOrFold<MulOp>(
        loc,
        getMostGenericType(a.getType(), d.getType()), a, d);
    auto bc = builder.createOrFold<MulOp>(
        loc,
        getMostGenericType(b.getType(), c.getType()), b, c);

    determinant = builder.createOrFold<SubOp>(
        loc,
        getMostGenericType(ad.getType(), bc.getType()), ad, bc);
  }
  else if (size > 2) {
    // The matrix is 3x3 or greater. Compute the determinant by taking the
    // first row of the matrix, and multiplying each scalar element of that
    // row with the determinant of the submatrix corresponding to that
    // element, that is the matrix constructed by removing from the
    // original one the row and the column of the scalar element.
    // Then add the ones in even position and subtract the ones in odd
    // (counting from zero).
    int sign = 1;
    size_t subMatrixSize = size - 1;

    // Create the matrix that will hold the n submatrices.
    auto subMatrixStorage = std::vector<mlir::Value>(
        subMatrixSize*subMatrixSize);
    SquareMatrix out = SquareMatrix(
        subMatrixStorage, subMatrixSize);

    // Initialize the determinant to zero.
    auto zeroAttr = RealAttr::get(
        builder.getContext(), 0);
    determinant = builder.create<ConstantOp>(
        loc, zeroAttr);

    // For each scalar element of the first row of the matrix:
    for (size_t i = 0; i < size; i++) {
      auto scalarDet = matrix(0,i);

      auto matrixDet = subMatrix(out, 0, i).det(builder);

      // Multiply the scalar element with the submatrix determinant.
      mlir::Value product = builder.createOrFold<MulOp>(
          loc,
          getMostGenericType(scalarDet.getType(),
                             matrixDet.getType()),
          scalarDet, matrixDet);

      // If in odd position, negate the value.
      if(sign == -1)
        product = builder.createOrFold<NegateOp>(
            loc, product.getType(), product);

      // Add the result to the rest of the determinant.
      determinant = builder.createOrFold<AddOp>(
          loc,
          getMostGenericType(determinant.getType(),
                             product.getType()),
          determinant, product);

      sign = -sign;
    }
  }

  return determinant;
}

//extract coefficients and constant terms from equations
//collect them into the system matrix and constant term vector

//for each eq in equations
//  set insertion point to beginning of eq
//  clone the system matrix inside of eq
//  clone the constant vector inside of eq
//  compute the solution with cramer
CramerSolver::CramerSolver(mlir::OpBuilder& builder, size_t systemSize) : builder(builder), systemSize(systemSize)
{
}

bool CramerSolver::solve(std::vector<MatchedEquation>& equations)
{
  mlir::OpBuilder::InsertionGuard guard(builder);
  auto loc = builder.getUnknownLoc();

  bool res = false;

  // Clone the system equations so that we can operate on them without
  // disrupting the rest of the compilation process.
  Equations<MatchedEquation> clones;

  for (const auto& equation : equations) {
    auto clone = Equation::build(equation.cloneIR(), equation.getVariables());

    auto matchedClone = std::make_unique<MatchedEquation>(
        std::move(clone), equation.getIterationRanges(), equation.getWrite().getPath());

    clones.add(std::move(matchedClone));
  }

  // Get the number of scalar equations in the system
  size_t subsystemSize = 0;
  for (const auto& equation : clones) {
    subsystemSize += equation->getIterationRanges().flatSize();
  }

  // Get the list of flat access indices
  std::map<size_t, std::pair<MatchedEquation*, Point>> flatMap;
  for (const auto& equation : clones) {
    for (const auto& point : equation->getIterationRanges()) {
      flatMap.emplace(equation->getFlatAccessIndex(point), std::pair(equation.get(), point));
    }
  }

  // Create the matrix to contain the coefficients of the system's
  // equations and the vector to contain the constant terms.
  auto storageSize = subsystemSize * subsystemSize;
  auto storage = std::vector<mlir::Value>(storageSize);
  auto matrix = SquareMatrix(storage, subsystemSize);
  auto constantVector = std::vector<mlir::Value>(subsystemSize);

  LLVM_DEBUG({
    llvm::dbgs() << "Populating the model matrix and constant vector.\n";
  });

  // Populate system coefficient matrix and constant term vector.
  res = getModelMatrixAndVector(
      matrix, constantVector, clones, subsystemSize, flatMap);

  LLVM_DEBUG({
    llvm::dbgs() << "COEFFICIENT MATRIX: \n";
    matrix.dump();
    llvm::dbgs() << "CONSTANT VECTOR: \n";
    for(auto el : constantVector)
      el.dump();
  });
//todo shuffling after getmatrix
  if(res) {
    size_t subsystemEquationIndex = 0;
    for (const auto& [index, pair] : flatMap) {
      auto& equation = pair.first;
      mlir::OpBuilder::InsertionGuard equationGuard(builder);
      builder.setInsertionPoint(equation->getOperation().bodyBlock()->getTerminator());
      // Allocate a new coefficient matrix and constant vector to contain the
      // cloned ones.
      auto cloneStorage = std::vector<mlir::Value>(storageSize);
      auto clonedMatrix = SquareMatrix(cloneStorage, subsystemSize);
      auto clonedVector = std::vector<mlir::Value>(subsystemSize);

      mlir::BlockAndValueMapping mapping;
      for (const auto& variable : equation->getVariables()) {
        auto variableValue = variable->getValue();
        mapping.map(variableValue, variableValue);
      }

      cloneCoefficientMatrix(builder, clonedMatrix, matrix, mapping);
      cloneConstantVector(builder, clonedVector, constantVector, mapping);

      // Create a temporary matrix to contain the matrices we are going to
      // derive from the original one, by substituting the constant term
      // vector to each of its columns.
      auto tempStorage = std::vector<mlir::Value>(subsystemSize * subsystemSize);
      auto temp = SquareMatrix(tempStorage, subsystemSize);

      // Compute the determinant of the system matrix.
      mlir::Value determinant = clonedMatrix.det(builder);

      auto access = equation->getWrite();
      auto& path = access.getPath();

      // Compute the determinant of each one of the matrices obtained by
      // substituting the constant term vector to each one of the matrix
      // columns, and divide them by the determinant of the system matrix.
      clonedMatrix.substituteColumn(
          temp, subsystemEquationIndex, clonedVector);

      auto tempDet = temp.det(builder);

      //TODO determine the type of a DivOp
      auto div = builder.createOrFold<DivOp>(
          loc,
          RealAttr::get(builder.getContext(), 0).getType(),
          tempDet, determinant);

      // Set the results computed as the right side of the cloned equations,
      // and the matched variables as the left side. Set the cloned equations
      // as the model equations.

      //todo take the point into account (pair.second)
      equation->setMatchSolution(builder, div);
      ++subsystemEquationIndex;
    }

    for (auto& equation : clones) {
      solution.add(std::move(equation));
    }
  } else {
    // Replace the newly found equations (if any) in the unsolved equations
    for (auto& readingEquation : clones) {
      auto accesses = readingEquation->getReads();

      // Map the variable position with the writing equation and indices
      std::multimap<unsigned int, std::pair<IndexSet, MatchedEquation*>> writesMap;
      for (const auto& equation : solution) {
        const auto& write = equation->getWrite();
        auto varPosition = write.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();
        auto writtenIndices = write.getAccessFunction().map(equation->getIterationRanges());
        writesMap.emplace(varPosition, std::make_pair(writtenIndices, equation.get()));
      }

      // Match the read accesses with a writing equation (if possible)
      std::vector<std::pair<const Access&, MatchedEquation*>> readWriteMatch;
      for (const auto& readingAccess : accesses) {
        auto readIndices = readingAccess.getAccessFunction().map(readingEquation->getIterationRanges());
        auto readPosition = readingAccess.getVariable()->getValue().cast<mlir::BlockArgument>().getArgNumber();

        for (const auto& [writePosition, pair] : llvm::make_range(writesMap.equal_range(readPosition))) {
          auto writingEquation = pair.second;
          auto writingAccess = writingEquation->getWrite();

          if (readingAccess.getVariable() != writingEquation->getWrite().getVariable() || !pair.first.overlaps(readIndices)) {
            continue;
          }

          readWriteMatch.emplace_back(readingAccess, writingEquation);
          break;
        }
      }

      // Replace the writing equation into the reading access
      bool replaced = false;
      auto clone = Equation::build(readingEquation->cloneIR(), readingEquation->getVariables());
      for (const auto& [readingAccess, writingEquation] : readWriteMatch) {
        auto iterationRanges = writingEquation->getIterationRanges();
        for(const auto& range : llvm::make_range(iterationRanges.rangesBegin(), iterationRanges.rangesEnd()))
        {
          auto res = writingEquation->replaceInto(
              builder, IndexSet(range), *clone,
              readingAccess.getAccessFunction(), readingAccess.getPath());

          if (mlir::failed(res)) {
            return false;
          }

          replaced = true;
        }
      }


      if (replaced) {
        auto matchedClone = std::make_unique<MatchedEquation>(
            Equation::build(clone->cloneIR(), clone->getVariables()),
            readingEquation->getIterationRanges(),readingEquation->getWrite().getPath());
        matchedClone->dumpIR();
        unsolved.add(std::move(matchedClone));
      } else {
        unsolved.add(std::move(readingEquation));
      }
    }
  }

  return res;
}

bool CramerSolver::getModelMatrixAndVector(
    SquareMatrix matrix,
    std::vector<mlir::Value>& constantVector,
    Equations<MatchedEquation> equations,
    size_t subsystemSize,
    const std::map<size_t, std::pair<MatchedEquation*, Point>>& flatMap)
{
  // Scan the accesses of each equation for missing matches
  for (const auto& [index, pair] : flatMap) {
    const auto& equation = pair.first;
    const auto& point = pair.second;

    for (const auto& readAccess : equation->getReads()) {
      if (flatMap.count(equation->getFlatAccessIndex(readAccess, point)) == 0) {
        return false;
      }
    }
  }

  // For each equation, get the coefficients of the variables and the
  // constant term, and save them respectively in the system matrix and
  // constant term vector.
  size_t subsystemEquationIndex = 0;
  for (const auto& [index, pair] : flatMap) {
    const auto& point = pair.second;
    const auto& equation = pair.first;

    // Collect the coefficients and the constant term
    auto coefficients = std::vector<mlir::Value>(systemSize);
    mlir::Value constantTerm;
    if(mlir::failed(equation->getCoefficients(builder, coefficients, constantTerm, point))) {
      return false;
    }

    // Copy the useful coefficients inside the matrix and constant vector
    size_t subsystemVariableIndex = 0;
    for (const auto& [variableIndex, _] : flatMap) {
      coefficients[variableIndex].dump();
      matrix(subsystemEquationIndex, subsystemVariableIndex) = coefficients[variableIndex];
      matrix.dump();
      ++subsystemVariableIndex;
    }
    assert(subsystemVariableIndex == subsystemSize);
    constantVector[subsystemEquationIndex] = constantTerm;
    ++subsystemEquationIndex;
  }

  return true;
}

mlir::Value CramerSolver::cloneValueAndDependencies(
    mlir::OpBuilder& builder,
    mlir::Value value,
    mlir::BlockAndValueMapping& mapping)
{
  mlir::OpBuilder::InsertionGuard guard(builder);

  std::stack<mlir::Operation*> cloneStack;
  std::vector<mlir::Operation*> toBeCloned;

  // Push the defining operation of the input value on the stack.
  if (auto op = value.getDefiningOp(); op != nullptr) {
    cloneStack.push(op);
  }

  // Until the stack is empty pop it, add the operand to the list, get its
  // operands (if any) and push them on the stack.
  while (!cloneStack.empty()) {
    auto op = cloneStack.top();
    cloneStack.pop();

    toBeCloned.push_back(op);

    for (const auto& operand : op->getOperands()) {
      if (auto operandOp = operand.getDefiningOp(); operandOp != nullptr) {
        cloneStack.push(operandOp);
      }
    }
  }

  // Clone the operations
  mlir::Operation* clonedOp = nullptr;
  for (auto it = toBeCloned.rbegin(); it != toBeCloned.rend(); ++it) {
    mlir::Operation* op = *it;
    clonedOp = builder.clone(*op, mapping);
  }

  assert(clonedOp != nullptr);
  auto op = clonedOp;
  auto results = op->getResults();
  assert(results.size() == 1);
  mlir::Value result = results[0];
  return result;
}

void CramerSolver::cloneCoefficientMatrix(
    mlir::OpBuilder& builder,
    SquareMatrix out,
    SquareMatrix in,
    mlir::BlockAndValueMapping& mapping)
{
  auto size = in.getSize();
  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      out(i,j) = cloneValueAndDependencies(builder, in(i,j), mapping);
    }
  }
}

void CramerSolver::cloneConstantVector(
    mlir::OpBuilder& builder,
    std::vector<mlir::Value>& out,
    std::vector<mlir::Value>& in,
    mlir::BlockAndValueMapping& mapping)
{
  for (size_t i = 0; i < in.size(); ++i) {
      out[i] = cloneValueAndDependencies(builder, in[i], mapping);
  }
}

bool CramerSolver::hasUnsolvedCycles() const
{
  return unsolved.size() != 0;
}

Equations<MatchedEquation> CramerSolver::getSolution() const
{
  return solution;
}

Equations<MatchedEquation> CramerSolver::getUnsolvedEquations() const
{
  return unsolved;
}
