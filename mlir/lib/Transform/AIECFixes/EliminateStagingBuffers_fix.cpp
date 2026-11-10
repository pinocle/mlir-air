//===- EliminateStagingBuffers_fix.cpp - herd operands without L2 staging -----===//
//
// See AIECEliminateStagingBuffers in mlir/include/air/AIECFixes/Passes_fix.td.
//
// Ported from hexagon-mlir IntermediateBufferEliminationPattern
// (qcom_hexagon_backend/lib/Transforms/CopyCanonicalizationPass.cpp, 9ec15e8,
// Qualcomm, BSD-3-Clause): a buffer that only passes data between a source and
// its readers is removed and the readers use the source. Here the readers (or
// writers) are herds whose tied argument is used only by DMA, and the buffer
// can also sit between herds and one copy that writes it out.
//
//===----------------------------------------------------------------------===//

#include "aiec/Transforms/Passes.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"

namespace aiec {
#define GEN_PASS_DEF_AIECELIMINATESTAGINGBUFFERS
#include "aiec/Transforms/Passes.h.inc"
} // namespace aiec

using namespace mlir;
namespace air = xilinx::air;

namespace {

enum class Role { Source, Destination };

// Every use of the herd argument tied to `operand` is a DMA that has it in
// `role`.
bool herdUsesOnlyAs(air::HerdOp herd, OpOperand &operand, Role role) {
  BlockArgument argument = herd.getTiedKernelArgument(operand.get());
  if (!argument)
    return false;
  for (Operation *user : argument.getUsers()) {
    auto dma = dyn_cast<air::DmaMemcpyNdOp>(user);
    if (!dma)
      return false;
    Value used = role == Role::Source ? dma.getSrcMemref() : dma.getDstMemref();
    Value other = role == Role::Source ? dma.getDstMemref() : dma.getSrcMemref();
    if (used != argument || other == argument)
      return false;
  }
  return true;
}

bool isAllocation(Value value) { return value && value.getDefiningOp<memref::AllocOp>(); }

// A herd writes only buffers allocated in the function: every DMA destination
// that is a herd argument is tied to an allocation.
bool herdWritesOnlyAllocations(air::HerdOp herd) {
  for (BlockArgument argument : herd.getKernelArguments())
    for (Operation *user : argument.getUsers())
      if (auto dma = dyn_cast<air::DmaMemcpyNdOp>(user))
        if (dma.getDstMemref() == argument && !isAllocation(herd.getTiedKernelOperand(argument)))
          return false;
  return true;
}

// `op` neither writes memory outside the function's allocations nor reads
// memory other than allocations through a herd or copy, so moving a read of
// a source earlier or a write to a destination later across it is safe.
bool inert(Operation *op) {
  if (isa<memref::AllocOp, memref::DeallocOp, memref::ReinterpretCastOp, memref::SubViewOp, memref::CastOp>(op))
    return true;
  if (auto copy = dyn_cast<memref::CopyOp>(op))
    return isAllocation(copy.getTarget()) && isAllocation(copy.getSource());
  if (auto herd = dyn_cast<air::HerdOp>(op)) {
    if (!herdWritesOnlyAllocations(herd))
      return false;
    for (Value operand : herd.getKernelOperands())
      if (isa<MemRefType, UnrankedMemRefType>(operand.getType()) && !isAllocation(operand))
        return false;
    return true;
  }
  return isMemoryEffectFree(op);
}

bool inertBetween(Operation *first, Operation *last) {
  for (Operation *op = first->getNextNode(); op && op != last; op = op->getNextNode())
    if (!inert(op))
      return false;
  return true;
}

// The herds that use `allocation`, each only as `role`, all in `block`.
// Returns false if any other user than `copy` and deallocations exists.
bool collectHerds(memref::AllocOp allocation, memref::CopyOp copy, Role role,
                  SmallVectorImpl<OpOperand *> &uses, SmallVectorImpl<memref::DeallocOp> &deallocs) {
  for (OpOperand &use : allocation->getUses()) {
    Operation *user = use.getOwner();
    if (user == copy)
      continue;
    if (auto dealloc = dyn_cast<memref::DeallocOp>(user)) {
      deallocs.push_back(dealloc);
      continue;
    }
    auto herd = dyn_cast<air::HerdOp>(user);
    if (!herd || herd->getBlock() != copy->getBlock() || !herdUsesOnlyAs(herd, use, role))
      return false;
    uses.push_back(&use);
  }
  return !uses.empty();
}

void retarget(ArrayRef<OpOperand *> uses, Value replacement) {
  for (OpOperand *use : uses) {
    auto herd = cast<air::HerdOp>(use->getOwner());
    herd.getTiedKernelArgument(use->get()).setType(replacement.getType());
    use->set(replacement);
  }
}

// memref.copy %source, %staging; herds read %staging by DMA.
bool eliminateInput(memref::CopyOp copy) {
  auto allocation = copy.getTarget().getDefiningOp<memref::AllocOp>();
  if (!allocation || isAllocation(copy.getSource()))
    return false;
  SmallVector<OpOperand *> uses;
  SmallVector<memref::DeallocOp> deallocs;
  if (!collectHerds(allocation, copy, Role::Source, uses, deallocs))
    return false;
  for (OpOperand *use : uses)
    if (!copy->isBeforeInBlock(use->getOwner()) || !inertBetween(copy, use->getOwner()))
      return false;
  retarget(uses, copy.getSource());
  copy->erase();
  for (memref::DeallocOp dealloc : deallocs)
    dealloc->erase();
  allocation->erase();
  return true;
}

// Moves the view ops defining `value` before `point` when their operands
// already dominate it. Returns false if `value` cannot be made available.
bool hoistBefore(Value value, Operation *point, DominanceInfo &dominance) {
  if (dominance.properlyDominates(value, point))
    return true;
  Operation *def = value.getDefiningOp();
  if (!def || def->getBlock() != point->getBlock() ||
      !isa<memref::ReinterpretCastOp, memref::SubViewOp, memref::CastOp>(def))
    return false;
  for (Value operand : def->getOperands())
    if (!dominance.properlyDominates(operand, point)) {
      Operation *producer = operand.getDefiningOp();
      if (!producer || producer->getBlock() != point->getBlock() || !isMemoryEffectFree(producer) ||
          producer->getNumRegions() || !llvm::all_of(producer->getOperands(), [&](Value v) {
            return dominance.properlyDominates(v, point);
          }))
        return false;
      producer->moveBefore(point);
    }
  def->moveBefore(point);
  return true;
}

// Herds write %staging by DMA; memref.copy %staging, %destination.
bool eliminateOutput(memref::CopyOp copy, DominanceInfo &dominance) {
  auto allocation = copy.getSource().getDefiningOp<memref::AllocOp>();
  if (!allocation || isAllocation(copy.getTarget()))
    return false;
  SmallVector<OpOperand *> uses;
  SmallVector<memref::DeallocOp> deallocs;
  if (!collectHerds(allocation, copy, Role::Destination, uses, deallocs))
    return false;
  Operation *first = uses.front()->getOwner();
  for (OpOperand *use : uses) {
    if (!use->getOwner()->isBeforeInBlock(copy))
      return false;
    if (use->getOwner()->isBeforeInBlock(first))
      first = use->getOwner();
  }
  // Nothing between the first herd and the copy may read the destination.
  if (!inertBetween(first, copy) || !hoistBefore(copy.getTarget(), first, dominance))
    return false;
  retarget(uses, copy.getTarget());
  copy->erase();
  for (memref::DeallocOp dealloc : deallocs)
    dealloc->erase();
  allocation->erase();
  return true;
}

struct AIECEliminateStagingBuffers : aiec::impl::AIECEliminateStagingBuffersBase<AIECEliminateStagingBuffers> {
  void runOnOperation() override {
    bool changed = true;
    while (changed) {
      changed = false;
      DominanceInfo dominance(getOperation());
      SmallVector<memref::CopyOp> copies;
      getOperation().walk([&](memref::CopyOp copy) {
        if (!copy->getParentOfType<air::HerdOp>())
          copies.push_back(copy);
      });
      for (memref::CopyOp copy : copies)
        if (eliminateInput(copy) || eliminateOutput(copy, dominance)) {
          changed = true;
          break; // operations moved or erased: recompute
        }
    }
  }
};

} // namespace
