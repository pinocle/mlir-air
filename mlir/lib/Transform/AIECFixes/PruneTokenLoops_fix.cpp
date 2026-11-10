//===- PruneTokenLoops_fix.cpp - remove token-only segment loops ---*- C++ -*-===//
//
// See AIECPruneTokenLoops in mlir/include/air/AIECFixes/Passes_fix.td.
//
//===----------------------------------------------------------------------===//

#include "aiec/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"

namespace aiec {
#define GEN_PASS_DEF_AIECPRUNETOKENLOOPS
#include "aiec/Transforms/Passes.h.inc"
} // namespace aiec

using namespace mlir;
namespace air = xilinx::air;

namespace {

bool isToken(Type type) { return isa<air::AsyncTokenType>(type); }

// A loop whose results are tokens and whose body holds only token joins,
// nested loops of the same kind, terminators and constants.
bool carriesOnlyTokens(Operation *loop) {
  if (loop->getNumResults() == 0 || !llvm::all_of(loop->getResultTypes(), isToken))
    return false;
  WalkResult inert = loop->walk([&](Operation *op) {
    if (op == loop)
      return WalkResult::advance();
    if (isa<air::WaitAllOp, scf::ForOp, scf::ParallelOp, scf::YieldOp, scf::ReduceOp, scf::ReduceReturnOp,
            arith::ConstantOp>(op))
      return WalkResult::advance();
    return WalkResult::interrupt();
  });
  return !inert.wasInterrupted();
}

ValueRange initialTokens(Operation *loop) {
  if (auto forOp = dyn_cast<scf::ForOp>(loop))
    return forOp.getInitArgs();
  return cast<scf::ParallelOp>(loop).getInitVals();
}

struct AIECPruneTokenLoops : aiec::impl::AIECPruneTokenLoopsBase<AIECPruneTokenLoops> {
  void runOnOperation() override {
    SmallVector<Operation *> loops;
    getOperation().walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (isa<air::HerdOp>(op))
        return WalkResult::skip();
      if (isa<scf::ForOp, scf::ParallelOp>(op) && op->getParentOfType<air::SegmentOp>()) {
        if (carriesOnlyTokens(op))
          loops.push_back(op);
        return WalkResult::skip(); // the outermost loop of a nest
      }
      return WalkResult::advance();
    });
    for (Operation *loop : loops) {
      OpBuilder builder(loop);
      auto done = air::WaitAllOp::create(builder, loop->getLoc(), air::AsyncTokenType::get(loop->getContext()),
                                         initialTokens(loop));
      for (Value result : loop->getResults())
        result.replaceAllUsesWith(done.getAsyncToken());
      loop->erase();
    }
  }
};

} // namespace
