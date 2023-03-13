#include "marco/Codegen/Transforms/ModelSolving/Utils.h"
#include "marco/Modeling/IndexSet.h"
#include "llvm/ADT/StringSwitch.h"

using namespace ::marco::codegen;
using namespace ::marco::modeling;
using namespace ::mlir::modelica;

static mlir::Attribute getRangeAttr(
    mlir::OpBuilder& builder,
    const MultidimensionalRange& multidimensionalRange)
{
  llvm::SmallVector<mlir::Attribute> rangesAttrs;

  for (unsigned int i = 0, rank = multidimensionalRange.rank(); i < rank; ++i) {
    const auto& range = multidimensionalRange[i];

    std::vector<mlir::Attribute> boundaries;
    boundaries.push_back(builder.getI64IntegerAttr(range.getBegin()));
    boundaries.push_back(builder.getI64IntegerAttr(range.getEnd() - 1));

    rangesAttrs.push_back(builder.getArrayAttr(boundaries));
  }

  return builder.getArrayAttr(rangesAttrs);
}

static llvm::Optional<MultidimensionalRange> getRangeFromAttr(mlir::Attribute attr)
{
  auto arrayAttr = attr.dyn_cast<mlir::ArrayAttr>();

  if (!arrayAttr) {
    return llvm::None;
  }

  llvm::SmallVector<Range> ranges;

  for (const auto& rangeAttr : arrayAttr) {
    auto rangeArrayAttr = rangeAttr.dyn_cast<mlir::ArrayAttr>();

    if (!rangeArrayAttr) {
      return llvm::None;
    }

    auto beginAttr = rangeArrayAttr[0].dyn_cast<mlir::IntegerAttr>();
    auto endAttr = rangeArrayAttr[1].dyn_cast<mlir::IntegerAttr>();

    if (!beginAttr || !endAttr) {
      return llvm::None;
    }

    ranges.emplace_back(beginAttr.getInt(), endAttr.getInt() + 1);
  }

  return MultidimensionalRange(ranges);
}

static void mergeRanges(mlir::OpBuilder& builder, llvm::SmallVectorImpl<mlir::Attribute>& ranges)
{
  using It = llvm::SmallVectorImpl<mlir::Attribute>::iterator;

  auto findCandidates = [&](It begin, It end) -> std::tuple<It, It, size_t> {
    for (It it1 = begin; it1 != end; ++it1) {
      auto it1Range = getRangeFromAttr(*it1);

      for (It it2 = std::next(it1); it2 != end; ++it2) {
        auto it2Range = getRangeFromAttr(*it2);

        if (auto mergePossibility = it1Range->canBeMerged(*it2Range); mergePossibility.first) {
          return std::make_tuple(it1, it2, mergePossibility.second);
        }
      }
    }

    return std::make_tuple(end, end, 0);
  };

  auto candidates = findCandidates(ranges.begin(), ranges.end());

  while (std::get<0>(candidates) != ranges.end() && std::get<1>(candidates) != ranges.end()) {
    auto& first = std::get<0>(candidates);
    auto& second = std::get<1>(candidates);
    size_t dimension = std::get<2>(candidates);

    auto firstRange = getRangeFromAttr(*first);
    auto secondRange = getRangeFromAttr(*second);

    MultidimensionalRange merged = firstRange->merge(*secondRange, dimension);
    *first = getRangeAttr(builder, merged);
    ranges.erase(second);
    candidates = findCandidates(ranges.begin(), ranges.end());
  }
}

static mlir::Attribute getIndexSetAttr(
    mlir::OpBuilder& builder,
    const IndexSet& indexSet,
    bool mergeAndSortRanges)
{
  llvm::SmallVector<mlir::Attribute> ranges;

  for (const auto& range : llvm::make_range(indexSet.rangesBegin(), indexSet.rangesEnd())) {
    ranges.push_back(getRangeAttr(builder, range));
  }

  // Merge the adjacent ranges for a 'deterministic' IR, allowing better tests.
  // This, however, is highly inefficient and should be used only for testing
  // purposes.
  if (mergeAndSortRanges) {
    mergeRanges(builder, ranges);

    llvm::sort(ranges, [](mlir::Attribute first, mlir::Attribute second) {
      return getRangeFromAttr(first) < getRangeFromAttr(second);
    });
  }

  return builder.getArrayAttr(ranges);
}

static llvm::Optional<IndexSet> getIndexSetFromAttr(mlir::Attribute attr)
{
  auto arrayAttr = attr.dyn_cast<mlir::ArrayAttr>();

  if (!arrayAttr) {
    return llvm::None;
  }

  IndexSet result;

  for (const auto& rangeAttr : arrayAttr) {
    llvm::Optional<MultidimensionalRange> range = getRangeFromAttr(rangeAttr);

    if (!range) {
      return llvm::None;
    }

    result += *range;
  }

  return result;
}

static mlir::Attribute getMatchedPathAttr(
    mlir::OpBuilder& builder,
    const EquationPath& path)
{
  std::vector<mlir::Attribute> pathAttrs;

  if (path.getEquationSide() == EquationPath::LEFT) {
    pathAttrs.push_back(builder.getStringAttr("L"));
  } else {
    pathAttrs.push_back(builder.getStringAttr("R"));
  }

  for (const auto& index : path) {
    pathAttrs.push_back(builder.getIndexAttr(index));
  }

  return builder.getArrayAttr(pathAttrs);
}

static llvm::Optional<EquationPath> getMatchedPathFromAttr(mlir::Attribute attr)
{
  auto pathAttr = attr.dyn_cast<mlir::ArrayAttr>();

  if (!pathAttr) {
    return llvm::None;
  }

  std::vector<size_t> pathIndices;

  for (size_t i = 1; i < pathAttr.size(); ++i) {
    auto indexAttr = pathAttr[i].dyn_cast<mlir::IntegerAttr>();

    if (!indexAttr) {
      return llvm::None;
    }

    pathIndices.push_back(indexAttr.getInt());
  }

  auto sideAttr = pathAttr[0].dyn_cast<mlir::StringAttr>();

  if (!sideAttr) {
    return llvm::None;
  }

  if (sideAttr.getValue() == "L") {
    return EquationPath(EquationPath::LEFT, pathIndices);
  }

  return EquationPath(EquationPath::RIGHT, pathIndices);
}

static mlir::Attribute getSchedulingDirectionAttr(
    mlir::OpBuilder& builder,
    scheduling::Direction direction)
{
  if (direction == scheduling::Direction::None) {
    return builder.getStringAttr("none");
  }

  if (direction == scheduling::Direction::Forward) {
    return builder.getStringAttr("forward");
  }

  if (direction == scheduling::Direction::Backward) {
    return builder.getStringAttr("backward");
  }

  if (direction == scheduling::Direction::Constant) {
    return builder.getStringAttr("constant");
  }

  if (direction == scheduling::Direction::Mixed) {
    return builder.getStringAttr("mixed");
  }

  return builder.getStringAttr("unknown");
}

static llvm::Optional<scheduling::Direction> getSchedulingDirectionFromAttr(mlir::Attribute attr)
{
  auto stringAttr = attr.dyn_cast<mlir::StringAttr>();

  if (!stringAttr) {
    return llvm::None;
  }

  return llvm::StringSwitch<scheduling::Direction>(stringAttr.getValue())
      .Case("none", scheduling::Direction::None)
      .Case("forward", scheduling::Direction::Forward)
      .Case("backward", scheduling::Direction::Backward)
      .Case("constant", scheduling::Direction::Constant)
      .Case("mixed", scheduling::Direction::Mixed)
      .Default(scheduling::Direction::Unknown);
}

namespace marco::codegen
{
  void writeDerivativesMap(
      mlir::OpBuilder& builder,
      ModelOp modelOp,
      const mlir::SymbolTable& symbolTable,
      const DerivativesMap& derivativesMap,
      const ModelSolvingIROptions& irOptions)
  {
    llvm::SmallVector<mlir::Attribute> derivativeAttrs;

    for (VariableOp variableOp : modelOp.getOps<VariableOp>()) {
      if (derivativesMap.hasDerivative(variableOp.getSymName())) {
        std::vector<mlir::NamedAttribute> namedAttrs;

        namedAttrs.emplace_back(
            builder.getStringAttr("variable"),
            mlir::SymbolRefAttr::get(builder.getContext(), variableOp.getSymName()));

        llvm::StringRef derName = derivativesMap.getDerivative(variableOp.getSymName());

        namedAttrs.emplace_back(
            builder.getStringAttr("derivative"),
            mlir::SymbolRefAttr::get(builder.getContext(), derName));

        auto memberType = variableOp.getMemberType();

        if (memberType.hasRank()) {
          // Add the indices only in case of array variable. Scalar variables
          // are implicitly considered as a single-element array.
          const auto& derivedIndices = derivativesMap.getDerivedIndices(variableOp.getSymName());

          namedAttrs.emplace_back(
              builder.getStringAttr("indices"),
              getIndexSetAttr(builder, derivedIndices, irOptions.mergeAndSortRanges));
        }

        derivativeAttrs.push_back(builder.getDictionaryAttr(namedAttrs));
      }
    }

    modelOp->setAttr("derivatives", builder.getArrayAttr(derivativeAttrs));
  }
}

static mlir::LogicalResult readDerivativesMapEntry(
    mlir::Attribute mapEntry,
    llvm::StringRef& variable,
    llvm::StringRef& derivative,
    IndexSet& derivedIndices)
{
  auto dict = mapEntry.dyn_cast<mlir::DictionaryAttr>();

  if (!dict) {
    return mlir::failure();
  }

  auto variableAttr = dict.getAs<mlir::SymbolRefAttr>("variable");
  auto derivativeAttr = dict.getAs<mlir::SymbolRefAttr>("derivative");

  if (!variableAttr || !derivativeAttr) {
    return mlir::failure();
  }

  if (mlir::Attribute indicesAttr = dict.get("indices")) {
    // Array variable.
    auto indicesArrayAttr = indicesAttr.dyn_cast<mlir::ArrayAttr>();

    if (!indicesArrayAttr) {
      return mlir::failure();
    }

    for (mlir::Attribute rangeAttr : indicesArrayAttr) {
      llvm::Optional<MultidimensionalRange> range = getRangeFromAttr(rangeAttr);

      if (!range) {
        return mlir::failure();
      }

      derivedIndices += *range;
    }
  } else {
    // Scalar variable.
    derivedIndices += Point(0);
  }

  variable = variableAttr.getLeafReference().getValue();
  derivative = derivativeAttr.getLeafReference().getValue();

  return mlir::success();
}

namespace marco::codegen
{
  mlir::LogicalResult readDerivativesMap(mlir::modelica::ModelOp modelOp, DerivativesMap& derivativesMap)
  {
    mlir::Attribute derivativesAttr = modelOp->getAttr("derivatives");

    if (!derivativesAttr) {
      // No derivatives specified.
      // Technically not a real error (even though it probably is).
      return mlir::success();
    }

    auto derivativesArrayAttr = derivativesAttr.dyn_cast<mlir::ArrayAttr>();

    if (!derivativesArrayAttr) {
      return mlir::failure();
    }

    std::atomic_bool success = true;
    std::mutex mutex;

    // Function to parse a chunk of entries.
    auto mapFn = [&](size_t from, size_t to) {
      for (size_t i = from; success && i < to; ++i) {
        mlir::Attribute mapEntry = derivativesArrayAttr[i];

        llvm::StringRef variable;
        llvm::StringRef derivative;
        IndexSet derivedIndices;

        if (mlir::failed(readDerivativesMapEntry(mapEntry, variable, derivative, derivedIndices))) {
          success = false;
          break;
        }

        std::lock_guard<std::mutex> lockGuard(mutex);
        derivativesMap.setDerivative(variable, derivative);
        derivativesMap.setDerivedIndices(variable, std::move(derivedIndices));
      }
    };

    // Shard the work among multiple threads.
    llvm::ThreadPool threadPool;

    size_t numOfEntries = derivativesArrayAttr.size();
    unsigned int numOfThreads = 1;
    size_t chunkSize = (numOfEntries + numOfThreads - 1) / numOfThreads;

    for (unsigned int i = 0; i < numOfThreads; ++i) {
      size_t from = std::min(numOfEntries, i * chunkSize);
      size_t to = std::min(numOfEntries, (i + 1) * chunkSize);

      if (from < to) {
        threadPool.async(mapFn, from, to);
      }
    }

    threadPool.wait();
    return mlir::LogicalResult::success(success);
  }

  void writeMatchingAttributes(
      mlir::OpBuilder& builder,
      Model<MatchedEquation>& model,
      const ModelSolvingIROptions& irOptions)
  {
    llvm::SmallVector<mlir::Attribute> matchingAttrs;

    for (const auto& equation : model.getEquations()) {
      equation->getOperation()->removeAttr("match");
    }

    for (const auto& equation : model.getEquations()) {
      std::vector<mlir::Attribute> matches;

      if (auto matchesAttr = equation->getOperation()->getAttrOfType<mlir::ArrayAttr>("match")) {
        for (const auto& match : matchesAttr) {
          matches.push_back(match);
        }
      }

      std::vector<mlir::NamedAttribute> namedAttrs;

      namedAttrs.emplace_back(
          builder.getStringAttr("path"),
          getMatchedPathAttr(builder, equation->getWrite().getPath()));

      namedAttrs.emplace_back(
          builder.getStringAttr("indices"),
          getIndexSetAttr(builder, equation->getIterationRanges(), irOptions.mergeAndSortRanges));

      matches.push_back(builder.getDictionaryAttr(namedAttrs));

      mlir::Attribute newMatchesAttr = builder.getArrayAttr(matches);
      equation->getOperation()->setAttr("match", newMatchesAttr);
    }

    if (irOptions.singleMatchAttr) {
      Equations<MatchedEquation> newEquations;
      llvm::DenseSet<Equation*> splitEquations;

      for (const auto& equation : model.getEquations()) {
        if (llvm::find_if(splitEquations, [&](Equation* eq) {
              return eq->getOperation() == equation->getOperation();
            }) != splitEquations.end()) {
          continue;
        }

        auto matchArrayAttr = equation->getOperation()->getAttrOfType<mlir::ArrayAttr>("match");

        if (!matchArrayAttr || matchArrayAttr.size() <= 1) {
          newEquations.add(std::make_unique<MatchedEquation>(*equation));
          continue;
        }

        for (const auto& matchAttr : matchArrayAttr) {
          EquationInterface clone = equation->cloneIR();
          clone->setAttr("match", builder.getArrayAttr(matchAttr));

          auto newEquation = Equation::build(clone, equation->getVariables());

          auto newMatchedEquation = std::make_unique<MatchedEquation>(
              std::move(newEquation),
              equation->getIterationRanges(),
              equation->getWrite().getPath());

          newEquations.add(std::move(newMatchedEquation));
        }

        splitEquations.insert(equation.get());
      }

      // Now that the equations have been cloned, we can erase the original ones.
      for (const auto& equation : splitEquations) {
        equation->eraseIR();
      }

      model.setEquations(newEquations);
    }
  }
}

static mlir::LogicalResult readMatchingAttribute(
    EquationInterface equationInt,
    mlir::Attribute matchAttr,
    Variables variables,
    std::vector<std::unique_ptr<MatchedEquation>>& equations)
{
  auto matchArrayAttr = matchAttr.dyn_cast<mlir::ArrayAttr>();

  if (!matchArrayAttr) {
    return mlir::failure();
  }

  for (mlir::Attribute match : matchArrayAttr) {
    auto dict = match.cast<mlir::DictionaryAttr>();
    llvm::Optional<IndexSet> indices = getIndexSetFromAttr(dict.get("indices"));
    llvm::Optional<EquationPath> path = getMatchedPathFromAttr(dict.get("path"));

    if (!indices || !path) {
      return mlir::failure();
    }

    auto equation = Equation::build(equationInt, variables);
    auto matchedEquation = std::make_unique<MatchedEquation>(std::move(equation), *indices, *path);
    equations.push_back(std::move(matchedEquation));
  }

  return mlir::success();
}

namespace marco::codegen
{
  mlir::LogicalResult readMatchingAttributes(
      Model<MatchedEquation>& result,
      std::function<bool(EquationInterface)> equationsFilter)
  {
    std::atomic_bool success = true;

    Equations<MatchedEquation> equations;
    std::mutex equationsMutex;

    // Collect the operations among which the matched equations should be
    // searched.
    llvm::SmallVector<EquationInterface> equationInterfaceOps;

    result.getOperation()->walk([&](EquationInterface op) {
      if (equationsFilter(op)) {
        equationInterfaceOps.push_back(op.getOperation());
      }
    });

    // Function to be applied to each chunk.
    auto mapFn = [&](size_t from, size_t to) {
      std::vector<std::unique_ptr<MatchedEquation>> matchedEquations;

      for (size_t i = from; i < to; ++i) {
        EquationInterface equationInt = equationInterfaceOps[i];
        mlir::Attribute matchAttr = equationInt->getAttr("match");

        if (!matchAttr) {
          break;
        }

        if (mlir::failed(readMatchingAttribute(equationInt, matchAttr, result.getVariables(), matchedEquations))) {
          success = false;
          break;
        }
      }

      if (success) {
        std::lock_guard<std::mutex> lockGuard(equationsMutex);

        for (auto& matchedEquation : matchedEquations) {
          equations.add(std::move(matchedEquation));
        }
      }
    };

    // Shard the work among multiple threads.
    llvm::ThreadPool threadPool;

    size_t numOfEquationInts = equationInterfaceOps.size();
    unsigned int numOfThreads = 1;
    size_t chunkSize = (numOfEquationInts + numOfThreads - 1) / numOfThreads;

    for (unsigned int i = 0; i < numOfThreads; ++i) {
      size_t from = std::min(numOfEquationInts, i * chunkSize);
      size_t to = std::min(numOfEquationInts, (i + 1) * chunkSize);

      if (from < to) {
        threadPool.async(mapFn, from, to);
      }
    }

    threadPool.wait();

    result.setEquations(equations);
    return mlir::LogicalResult::success(success);
  }

  void writeSchedulingAttributes(
      mlir::OpBuilder& builder,
      Model<ScheduledEquationsBlock>& model,
      const ModelSolvingIROptions& irOptions)
  {
    mlir::OpBuilder::InsertionGuard guard(builder);
    llvm::SmallVector<mlir::Attribute> schedulingAttrs;

    for (const auto& equationsBlock : llvm::enumerate(model.getScheduledBlocks())) {
      for (const auto& equation : *equationsBlock.value()) {
        equation->getOperation()->removeAttr("schedule");
      }
    }

    for (const auto& equationsBlock : llvm::enumerate(model.getScheduledBlocks())) {
      for (const auto& equation : *equationsBlock.value()) {
        std::vector<mlir::Attribute> schedules;

        if (auto schedulesAttr = equation->getOperation()->getAttrOfType<mlir::ArrayAttr>("schedule")) {
          for (const auto& schedule : schedulesAttr) {
            schedules.push_back(schedule);
          }
        }

        std::vector<mlir::NamedAttribute> namedAttrs;

        namedAttrs.emplace_back(
            builder.getStringAttr("block"),
            builder.getI64IntegerAttr(equationsBlock.index()));

        namedAttrs.emplace_back(
            builder.getStringAttr("cycle"),
            builder.getBoolAttr(equationsBlock.value()->hasCycle()));

        namedAttrs.emplace_back(
            builder.getStringAttr("path"),
            getMatchedPathAttr(builder, equation->getWrite().getPath()));

        namedAttrs.emplace_back(
            builder.getStringAttr("indices"),
            getIndexSetAttr(builder, equation->getIterationRanges(), irOptions.mergeAndSortRanges));

        namedAttrs.emplace_back(
            builder.getStringAttr("direction"),
            getSchedulingDirectionAttr(builder, equation->getSchedulingDirection()));

        schedules.push_back(builder.getDictionaryAttr(namedAttrs));

        mlir::Attribute newSchedulesAttr = builder.getArrayAttr(schedules);
        equation->getOperation()->setAttr("schedule", newSchedulesAttr);
      }
    }
  }

  mlir::LogicalResult readSchedulingAttributes(
      Model<ScheduledEquationsBlock>& result,
      std::function<bool(EquationInterface)> equationsFilter)
  {
    llvm::DenseSet<mlir::Operation*> equationInterfaceOps;

    result.getOperation()->walk([&](EquationInterface op) {
      if (equationsFilter(op)) {
        equationInterfaceOps.insert(op.getOperation());
      }
    });

    ScheduledEquationsBlocks scheduledEquationsBlocks;

    llvm::SmallVector<std::unique_ptr<ScheduledEquation>> scheduledEquations;
    llvm::DenseMap<int64_t, llvm::DenseSet<size_t>> blocks;
    llvm::DenseMap<int64_t, bool> cycles;

    for (const auto& equationIntOp : equationInterfaceOps) {
      auto equationInt = mlir::cast<EquationInterface>(equationIntOp);
      mlir::Attribute scheduleAttr = equationInt->getAttr("schedule");

      if (!scheduleAttr) {
        continue;
      }

      auto scheduleArrayAttr = scheduleAttr.dyn_cast<mlir::ArrayAttr>();

      if (!scheduleArrayAttr) {
        return mlir::failure();
      }

      for (auto schedule : scheduleArrayAttr) {
        auto dict = schedule.dyn_cast<mlir::DictionaryAttr>();

        if (!dict) {
          return mlir::failure();
        }

        // Block ID.
        auto blockAttr = dict.getAs<mlir::IntegerAttr>("block");

        if (!blockAttr) {
          return mlir::failure();
        }

        int64_t blockId = blockAttr.getInt();

        // Path
        llvm::Optional<EquationPath> path = getMatchedPathFromAttr(dict.get("path"));

        // Scheduled indices and direction.
        llvm::Optional<IndexSet> indices = getIndexSetFromAttr(dict.get("indices"));
        llvm::Optional<scheduling::Direction> direction = getSchedulingDirectionFromAttr(dict.get("direction"));

        if (!path || !indices || !direction) {
          return mlir::failure();
        }

        auto equation = Equation::build(equationInt, result.getVariables());
        auto matchedEquation = std::make_unique<MatchedEquation>(std::move(equation), *indices, *path);
        auto scheduledEquation = std::make_unique<ScheduledEquation>(std::move(matchedEquation), *indices, *direction);

        scheduledEquations.push_back(std::move(scheduledEquation));
        blocks[blockId].insert(scheduledEquations.size() - 1);

        // Cycle property for the block
        auto cycleAttr = dict.getAs<mlir::BoolAttr>("cycle");

        if (!cycleAttr) {
          return mlir::failure();
        }

        cycles[blockId] = cycleAttr.getValue();
      }
    }

    // Reorder the blocks by their ID.
    std::vector<int64_t> orderedBlocksIds;

    for (const auto& block : blocks) {
      orderedBlocksIds.push_back(block.getFirst());
    }

    llvm::sort(orderedBlocksIds);

    // Create the equations blocks.
    for (const auto& blockId : orderedBlocksIds) {
      Equations<ScheduledEquation> equations;

      for (const auto& equationIndex : blocks[blockId]) {
        equations.push_back(std::move(scheduledEquations[equationIndex]));
      }

      auto scheduledEquationsBlock = std::make_unique<ScheduledEquationsBlock>(equations, cycles[blockId]);
      scheduledEquationsBlocks.append(std::move(scheduledEquationsBlock));
    }

    result.setScheduledBlocks(scheduledEquationsBlocks);
    return mlir::success();
  }
}
