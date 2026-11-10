//===- SynchronizeHost_fix.cpp - sequential host side of async AIR -*- C++ -*-===//
//
// See AIECSynchronizeHost in mlir/include/air/AIECFixes/Passes_fix.td.
//
//===----------------------------------------------------------------------===//

#include "aiec/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace aiec {
#define GEN_PASS_DEF_AIECSYNCHRONIZEHOST
#include "aiec/Transforms/Passes.h.inc"
} // namespace aiec

using namespace mlir;
namespace air = xilinx::air;

namespace {

bool isToken(Type type) { return isa<air::AsyncTokenType>(type); }

// Host operations: those of `function` outside device segments (the segment
// operations themselves included), in program order.
SmallVector<Operation *> hostOperations(func::FuncOp function) {
  SmallVector<Operation *> host;
  function.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == function)
      return WalkResult::advance();
    host.push_back(op);
    return isa<air::SegmentOp>(op) ? WalkResult::skip() : WalkResult::advance();
  });
  return host;
}

// An scf.parallel whose reductions only combined tokens, rebuilt without them.
void dropTokenReductions(scf::ParallelOp loop) {
  OpBuilder builder(loop);
  auto replacement = scf::ParallelOp::create(builder, loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(),
                                             loop.getStep(), ValueRange{});
  Block *from = loop.getBody(), *to = replacement.getBody();
  for (auto [old, now] : llvm::zip(from->getArguments(), to->getArguments()))
    old.replaceAllUsesWith(now);
  Operation *terminator = to->getTerminator();
  for (Operation &op : llvm::make_early_inc_range(from->without_terminator()))
    op.moveBefore(terminator);
  loop->erase();
}

// The operation rebuilt without its async dependencies and results.
void stripTokens(Operation *op) {
  auto sizes = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
  SmallVector<int32_t> segments(sizes.asArrayRef());
  const unsigned dependencies = segments[0];
  segments[0] = 0;
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(op->getOperands().drop_front(dependencies));
  for (NamedAttribute attribute : op->getAttrs())
    if (attribute.getName() != "operandSegmentSizes")
      state.addAttribute(attribute.getName(), attribute.getValue());
  state.addAttribute("operandSegmentSizes", DenseI32ArrayAttr::get(op->getContext(), segments));
  for (unsigned i = 0; i < op->getNumRegions(); ++i)
    state.addRegion();
  OpBuilder builder(op);
  Operation *replacement = builder.create(state);
  for (unsigned i = 0; i < op->getNumRegions(); ++i)
    replacement->getRegion(i).takeBody(op->getRegion(i));
  op->erase();
}

struct AIECSynchronizeHost : aiec::impl::AIECSynchronizeHostBase<AIECSynchronizeHost> {
  using Base::Base;

  LogicalResult synchronize(func::FuncOp function) {
    if (function.isExternal() || !function.getBody().hasOneBlock())
      return success();
    // One complete token per isolated scope (the function, each launch): a
    // launch region cannot use a value defined outside it.
    DenseMap<Operation *, Value> completes;
    auto complete = [&](Operation *user) -> Value {
      Operation *scope = user->getParentOp();
      while (scope != function && !scope->hasTrait<OpTrait::IsIsolatedFromAbove>())
        scope = scope->getParentOp();
      Value &done = completes[scope];
      if (!done) {
        OpBuilder builder = OpBuilder::atBlockBegin(&scope->getRegion(0).front());
        done = air::WaitAllOp::create(builder, scope->getLoc(), air::AsyncTokenType::get(&getContext()), ValueRange{})
                   .getAsyncToken();
      }
      return done;
    };
    SmallVector<Operation *> host = hostOperations(function);
    // Inline air.execute bodies; their values replace the execute's results.
    for (Operation *op : host)
      if (auto execute = dyn_cast<air::ExecuteOp>(op)) {
        Block &body = execute.getRegion().front();
        Operation *terminator = body.getTerminator();
        for (auto [result, value] : llvm::zip(execute->getResults().drop_front(), terminator->getOperands()))
          result.replaceAllUsesWith(value);
        for (Operation &child : llvm::make_early_inc_range(body.without_terminator()))
          child.moveBefore(execute);
      }
    for (Operation *op : host)
      for (Value result : op->getResults())
        if (isToken(result.getType()))
          for (OpOperand &use : llvm::make_early_inc_range(result.getUses()))
            use.set(complete(use.getOwner()));
    SmallVector<scf::ParallelOp> loops;
    for (Operation *op : host)
      if (auto loop = dyn_cast<scf::ParallelOp>(op); loop && loop->getNumResults())
        loops.push_back(loop);
    for (Operation *op : llvm::reverse(host))
      if (isa<air::ExecuteOp, air::WaitAllOp>(op))
        op->erase();
    // scf.for iter_args of tokens fold under canonicalization; scf.parallel
    // reductions do not.
    for (scf::ParallelOp loop : llvm::reverse(loops)) {
      if (llvm::any_of(loop->getResults(), [](Value r) { return !isToken(r.getType()) || !r.use_empty(); }))
        return loop.emitError("host scf.parallel reduces values other than unused tokens");
      dropTokenReductions(loop);
    }
    if (stripHierarchyTokens) {
      SmallVector<Operation *> hierarchy;
      function.walk<WalkOrder::PreOrder>([&](Operation *op) {
        if (isa<air::LaunchOp, air::SegmentOp, air::ChannelPutOp, air::ChannelGetOp>(op))
          hierarchy.push_back(op);
        return isa<air::SegmentOp>(op) ? WalkResult::skip() : WalkResult::advance();
      });
      for (Operation *op : hierarchy) {
        auto sizes = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
        if (!sizes || (sizes[0] == 0 && op->getNumResults() == 0))
          continue;
        if (llvm::any_of(op->getResults(), [](Value r) { return !isToken(r.getType()) || !r.use_empty(); }))
          return op->emitError("host hierarchy must only return completed tokens");
        stripTokens(op);
      }
    }
    for (auto &entry : completes)
      if (entry.second.use_empty())
        entry.second.getDefiningOp()->erase();
    return success();
  }

  void runOnOperation() override {
    for (auto function : llvm::make_early_inc_range(getOperation().getOps<func::FuncOp>()))
      if (failed(synchronize(function)))
        return signalPassFailure();
  }
};

} // namespace
