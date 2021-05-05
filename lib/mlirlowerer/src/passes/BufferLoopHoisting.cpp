//===- BufferOptimizations.cpp - pre-pass optimizations for bufferization -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements logic for three optimization passes. The first two
// passes try to move alloc nodes out of blocks to reduce the number of
// allocations and copies during buffer deallocation. The third pass tries to
// convert heap-based allocations to stack-based allocations, if possible.

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h.inc"

#include <modelica/mlirlowerer/passes/BufferLoopHoisting.h>

using namespace mlir;

/// Returns true if the given operation implements a known high-level region-
/// based control-flow interface.
static bool isKnownControlFlowInterface(Operation *op) {
	return isa<LoopLikeOpInterface, RegionBranchOpInterface>(op);
}

/// Check if the size of the allocation is less than the given size. The
/// transformation is only applied to small buffers since large buffers could
/// exceed the stack space.
static bool defaultIsSmallAlloc(Value alloc, unsigned maximumSizeInBytes,
																unsigned bitwidthOfIndexType,
																unsigned maxRankOfAllocatedMemRef) {
	auto type = alloc.getType().dyn_cast<ShapedType>();
	if (!type || !alloc.getDefiningOp<memref::AllocOp>())
		return false;
	if (!type.hasStaticShape()) {
		// Check if the dynamic shape dimension of the alloc is produced by RankOp.
		// If this is the case, it is likely to be small. Furthermore, the dimension
		// is limited to the maximum rank of the allocated memref to avoid large
		// values by multiplying several small values.
		if (type.getRank() <= maxRankOfAllocatedMemRef) {
			return llvm::all_of(
					alloc.getDefiningOp()->getOperands(),
					[&](Value operand) { return operand.getDefiningOp<RankOp>(); });
		}
		return false;
	}
	// For index types, use the provided size, as the type does not know.
	unsigned int bitwidth = type.getElementType().isIndex()
													? bitwidthOfIndexType
													: type.getElementTypeBitWidth();
	return type.getNumElements() * bitwidth <= maximumSizeInBytes * 8;
}

/// Checks whether the given aliases leave the allocation scope.
static bool
leavesAllocationScope(Region *parentRegion,
											const BufferAliasAnalysis::ValueSetT &aliases) {
	for (Value alias : aliases) {
		for (auto *use : alias.getUsers()) {
			// If there is at least one alias that leaves the parent region, we know
			// that this alias escapes the whole region and hence the associated
			// allocation leaves allocation scope.
			if (use->hasTrait<OpTrait::ReturnLike>() &&
					use->getParentRegion() == parentRegion)
				return true;
		}
	}
	return false;
}

/// Checks, if an automated allocation scope for a given alloc value exists.
static bool hasAllocationScope(Value alloc,
															 const BufferAliasAnalysis &aliasAnalysis) {
	Region *region = alloc.getParentRegion();
	do {
		if (Operation *parentOp = region->getParentOp()) {
			// Check if the operation is an automatic allocation scope and whether an
			// alias leaves the scope. This means, an allocation yields out of
			// this scope and can not be transformed in a stack-based allocation.
			if (parentOp->hasTrait<OpTrait::AutomaticAllocationScope>() &&
					!leavesAllocationScope(region, aliasAnalysis.resolve(alloc)))
				return true;
			// Check if the operation is a known control flow interface and break the
			// loop to avoid transformation in loops. Furthermore skip transformation
			// if the operation does not implement a RegionBeanchOpInterface.
			if (BufferPlacementTransformationBase::isLoop(parentOp) ||
					!isKnownControlFlowInterface(parentOp))
				break;
		}
	} while ((region = region->getParentRegion()));
	return false;
}

namespace {

	//===----------------------------------------------------------------------===//
	// BufferAllocationHoisting
	//===----------------------------------------------------------------------===//

	/// A base implementation compatible with the `BufferAllocationHoisting` class.
	struct BufferAllocationHoistingStateBase {
		/// A pointer to the current dominance info.
		DominanceInfo *dominators;

		/// The current allocation value.
		Value allocValue;

		/// The current placement block (if any).
		Block *placementBlock;

		/// Initializes the state base.
		BufferAllocationHoistingStateBase(DominanceInfo *dominators, Value allocValue,
																			Block *placementBlock)
				: dominators(dominators), allocValue(allocValue),
					placementBlock(placementBlock) {}
	};

	/// Implements the actual hoisting logic for allocation nodes.
	template <typename StateT>
	class BufferAllocationHoisting : public BufferPlacementTransformationBase {
		public:
		BufferAllocationHoisting(Operation *op)
				: BufferPlacementTransformationBase(op), dominators(op),
					postDominators(op)
		{
			op->walk([&](MemoryEffectOpInterface opInterface) {
				opInterface.dump();

				// Try to find a single allocation result.
				SmallVector<MemoryEffects::EffectInstance, 2> effects;
				opInterface.getEffects(effects);

				SmallVector<MemoryEffects::EffectInstance, 2> allocateResultEffects;
				llvm::copy_if(
						effects, std::back_inserter(allocateResultEffects),
						[=](MemoryEffects::EffectInstance &it) {
							Value value = it.getValue();

							bool x1 = isa<MemoryEffects::Allocate>(it.getEffect());
							bool x2 = value != nullptr;
							bool x3 = value.isa<OpResult>();
							bool x4 = it.getResource() !=
												SideEffects::AutomaticAllocationScopeResource::get();
							return x1 && x2 && x3;
						});
				// If there is one result only, we will be able to move the allocation and
				// (possibly existing) deallocation ops.
				if (allocateResultEffects.size() != 1)
					return;
				// Get allocation result.
				Value allocValue = allocateResultEffects[0].getValue();
				// Find the associated dealloc value and register the allocation entry.
				allocs.registerAlloc(std::make_tuple(allocValue, nullptr));
			});

		}

		/// Moves allocations upwards.
		void hoist() {
			for (BufferPlacementAllocs::AllocEntry &entry : allocs) {
				Value allocValue = std::get<0>(entry);
				Operation *definingOp = allocValue.getDefiningOp();
				assert(definingOp && "No defining op");
				auto operands = definingOp->getOperands();
				auto resultAliases = aliases.resolve(allocValue);
				// Determine the common dominator block of all aliases.
				Block *dominatorBlock =
						findCommonDominator(allocValue, resultAliases, dominators);
				// Init the initial hoisting state.
				StateT state(&dominators, allocValue, allocValue.getParentBlock());
				// Check for additional allocation dependencies to compute an upper bound
				// for hoisting.
				Block *dependencyBlock = nullptr;
				// If this node has dependencies, check all dependent nodes. This ensures
				// that all dependency values have been computed before allocating the
				// buffer.
				for (Value depValue : operands) {
					Block *depBlock = depValue.getParentBlock();
					if (!dependencyBlock || dominators.dominates(dependencyBlock, depBlock))
						dependencyBlock = depBlock;
				}

				// Find the actual placement block and determine the start operation using
				// an upper placement-block boundary. The idea is that placement block
				// cannot be moved any further upwards than the given upper bound.
				Block *placementBlock = findPlacementBlock(
						state, state.computeUpperBound(dominatorBlock, dependencyBlock));
				Operation *startOperation = BufferPlacementAllocs::getStartOperation(
						allocValue, placementBlock, liveness);

				// Move the alloc in front of the start operation.
				Operation *allocOperation = allocValue.getDefiningOp();
				allocOperation->moveBefore(startOperation);
			}
		}

		private:
		/// Finds a valid placement block by walking upwards in the CFG until we
		/// either cannot continue our walk due to constraints (given by the StateT
		/// implementation) or we have reached the upper-most dominator block.
		Block *findPlacementBlock(StateT &state, Block *upperBound) {
			Block *currentBlock = state.placementBlock;
			// Walk from the innermost regions/loops to the outermost regions/loops and
			// find an appropriate placement block that satisfies the constraint of the
			// current StateT implementation. Walk until we reach the upperBound block
			// (if any).

			// If we are not able to find a valid parent operation or an associated
			// parent block, break the walk loop.
			Operation *parentOp;
			Block *parentBlock;
			while ((parentOp = currentBlock->getParentOp()) &&
						 (parentBlock = parentOp->getBlock()) &&
						 (!upperBound ||
							dominators.properlyDominates(upperBound, currentBlock))) {
				// Try to find an immediate dominator and check whether the parent block
				// is above the immediate dominator (if any).
				DominanceInfoNode *idom = dominators.getNode(currentBlock)->getIDom();
				if (idom && dominators.properlyDominates(parentBlock, idom->getBlock())) {
					// If the current immediate dominator is below the placement block, move
					// to the immediate dominator block.
					currentBlock = idom->getBlock();
					state.recordMoveToDominator(currentBlock);
				} else {
					// We have to move to our parent block since an immediate dominator does
					// either not exist or is above our parent block. If we cannot move to
					// our parent operation due to constraints given by the StateT
					// implementation, break the walk loop. Furthermore, we should not move
					// allocations out of unknown region-based control-flow operations.
					if (!isKnownControlFlowInterface(parentOp) ||
							!state.isLegalPlacement(parentOp))
						break;
					// Move to our parent block by notifying the current StateT
					// implementation.
					currentBlock = parentBlock;
					state.recordMoveToParent(currentBlock);
				}
			}
			// Return the finally determined placement block.
			return state.placementBlock;
		}

		/// The dominator info to find the appropriate start operation to move the
		/// allocs.
		DominanceInfo dominators;

		/// The post dominator info to move the dependent allocs in the right
		/// position.
		PostDominanceInfo postDominators;

		/// The map storing the final placement blocks of a given alloc value.
		llvm::DenseMap<Value, Block *> placementBlocks;
	};

	/// A state implementation compatible with the `BufferAllocationHoisting` class
	/// that hoists allocations into dominator blocks while keeping them inside of
	/// loops.
	struct BufferAllocationHoistingState : BufferAllocationHoistingStateBase {
		using BufferAllocationHoistingStateBase::BufferAllocationHoistingStateBase;

		/// Computes the upper bound for the placement block search.
		Block *computeUpperBound(Block *dominatorBlock, Block *dependencyBlock) {
			// If we do not have a dependency block, the upper bound is given by the
			// dominator block.
			if (!dependencyBlock)
				return dominatorBlock;

			// Find the "lower" block of the dominator and the dependency block to
			// ensure that we do not move allocations above this block.
			return dominators->properlyDominates(dominatorBlock, dependencyBlock)
						 ? dependencyBlock
						 : dominatorBlock;
		}

		/// Returns true if the given operation does not represent a loop.
		bool isLegalPlacement(Operation *op) {
			return !BufferPlacementTransformationBase::isLoop(op);
		}

		/// Sets the current placement block to the given block.
		void recordMoveToDominator(Block *block) { placementBlock = block; }

		/// Sets the current placement block to the given block.
		void recordMoveToParent(Block *block) { recordMoveToDominator(block); }
	};

	/// A state implementation compatible with the `BufferAllocationHoisting` class
	/// that hoists allocations out of loops.
	struct BufferAllocationLoopHoistingState : BufferAllocationHoistingStateBase {
		using BufferAllocationHoistingStateBase::BufferAllocationHoistingStateBase;

		/// Remembers the dominator block of all aliases.
		Block *aliasDominatorBlock;

		/// Computes the upper bound for the placement block search.
		Block *computeUpperBound(Block *dominatorBlock, Block *dependencyBlock) {
			aliasDominatorBlock = dominatorBlock;
			// If there is a dependency block, we have to use this block as an upper
			// bound to satisfy all allocation value dependencies.
			return dependencyBlock ? dependencyBlock : nullptr;
		}

		/// Returns true if the given operation represents a loop and one of the
		/// aliases caused the `aliasDominatorBlock` to be "above" the block of the
		/// given loop operation. If this is the case, it indicates that the
		/// allocation is passed via a back edge.
		bool isLegalPlacement(Operation *op) {
			return BufferPlacementTransformationBase::isLoop(op) &&
						 !dominators->dominates(aliasDominatorBlock, op->getBlock());
		}

		/// Does not change the internal placement block, as we want to move
		/// operations out of loops only.
		void recordMoveToDominator(Block *block) {}

		/// Sets the current placement block to the given block.
		void recordMoveToParent(Block *block) { placementBlock = block; }
	};

	//===----------------------------------------------------------------------===//
	// BufferOptimizationPasses
	//===----------------------------------------------------------------------===//

	/// The buffer loop hoisting pass that hoists allocation nodes out of loops.
	struct BufferLoopHoistingPass : public ::mlir::FunctionPass {
		BufferLoopHoistingPass() : ::mlir::FunctionPass(::mlir::TypeID::get<BufferLoopHoistingPass>()) {}
		BufferLoopHoistingPass(const BufferLoopHoistingPass &) : ::mlir::FunctionPass(::mlir::TypeID::get<BufferLoopHoistingPass>()) {}

		void runOnFunction() override {
			// Hoist all allocations out of loops.
			BufferAllocationHoisting<BufferAllocationLoopHoistingState> optimizer(
					getFunction());
			optimizer.hoist();
		}

		static constexpr ::llvm::StringLiteral getArgumentName() {
			return ::llvm::StringLiteral("buffer-loop-hoisting");
		}

		/// Returns the derived pass name.
		static constexpr ::llvm::StringLiteral getPassName() {
			return ::llvm::StringLiteral("BufferLoopHoisting");
		}
		::llvm::StringRef getName() const override { return "BufferLoopHoisting"; }

		/// Support isa/dyn_cast functionality for the derived pass class.
		static bool classof(const ::mlir::Pass *pass) {
			return pass->getTypeID() == ::mlir::TypeID::get<BufferLoopHoistingPass>();
		}

		/// A clone method to create a copy of this pass.
		std::unique_ptr<::mlir::Pass> clonePass() const override {
			return std::make_unique<BufferLoopHoistingPass>(*static_cast<const BufferLoopHoistingPass *>(this));
		}

		/// Return the dialect that must be loaded in the context before this pass.
		void getDependentDialects(::mlir::DialectRegistry &registry) const override {

		}
	};

} // end anonymous namespace

std::unique_ptr<Pass> modelica::codegen::createBufferLoopHoistingPass() {
	return std::make_unique<BufferLoopHoistingPass>();
}
