//===- AIRLocalizeAccumulators.cpp ------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// air-annotate-uniform-init and air-localize-accumulators: initialize a
// promoted reduction output tile in local memory from its proven uniform
// initial value instead of loading it. A core tile whose DMA has two input
// channels (AIE1) cannot otherwise run a reduction that also loads its
// accumulator next to two operands.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRLocalizeAccumulators.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/MapVector.h"

using namespace mlir;

namespace xilinx {
namespace air {
#define GEN_PASS_DEF_AIRANNOTATEUNIFORMINIT
#define GEN_PASS_DEF_AIRLOCALIZEACCUMULATORS
#include "air/Transform/Passes.h.inc"
} // namespace air
} // namespace xilinx

namespace air = xilinx::air;

namespace {

constexpr llvm::StringLiteral kInit = "air.uniform_output_init";

// The allocation a value views (through subview, cast, reshape).
Value root(Value value) {
  while (Operation *op = value.getDefiningOp()) {
    if (!isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp, memref::CollapseShapeOp,
             memref::ExpandShapeOp>(op))
      break;
    value = op->getOperand(0);
  }
  return value;
}

// Whether `value` is computed from `iv` (through defining operations).
bool dependsOn(Value value, Value iv, llvm::DenseSet<Value> &seen) {
  if (value == iv)
    return true;
  if (!seen.insert(value).second)
    return false;
  Operation *op = value.getDefiningOp();
  return op && llvm::any_of(op->getOperands(), [&](Value operand) { return dependsOn(operand, iv, seen); });
}

std::optional<int64_t> constantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantOp>())
    if (auto integer = dyn_cast<IntegerAttr>(constant.getValue()))
      return integer.getInt();
  return std::nullopt;
}

// Conservative reaching definitions of whole-buffer constant fills over a
// function's top-level operations. Unknown writes, calls and control flow
// kill every fact; a fill of a view never establishes one.
struct AIRAnnotateUniformInit
    : air::impl::AIRAnnotateUniformInitBase<AIRAnnotateUniformInit> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Metadata supplied by the input is never trusted.
    module.walk([](Operation *op) { op->removeAttr(kInit); });
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!function.getBody().hasOneBlock())
        continue;
      llvm::DenseMap<Value, Attribute> facts;
      for (Operation &op : function.getBody().front()) {
        if (auto fill = dyn_cast<linalg::FillOp>(op)) {
          Value destination = fill.getOutputs()[0];
          // Invalidate aliases even when the destination is a view.
          facts.erase(root(destination));
          if (root(destination) == destination)
            if (auto constant = fill.getInputs()[0].getDefiningOp<arith::ConstantOp>())
              facts[destination] = constant.getValue();
        } else if (auto copy = dyn_cast<memref::CopyOp>(op)) {
          Value source = copy.getSource(), destination = copy.getTarget();
          Attribute value = facts.lookup(root(source));
          if (isa<BlockArgument>(root(destination)))
            facts.clear(); // no noalias promise for function parameters
          else
            facts.erase(root(destination));
          if (root(destination) == destination && value)
            facts[destination] = value;
        } else if (op.getName().getStringRef().starts_with("linalg.")) {
          auto segments = op.getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes");
          if (!segments) {
            facts.clear();
            continue;
          }
          unsigned count = segments.asArrayRef().back();
          auto outputs = op.getOperands().take_back(count);
          SmallVector<Attribute> initializers;
          for (Value output : outputs) {
            Attribute known = facts.lookup(root(output));
            initializers.push_back(known ? known : UnitAttr::get(&getContext()));
          }
          if (op.getNumRegions() && !op.getRegion(0).empty()) {
            auto arguments = op.getRegion(0).front().getArguments().take_back(count);
            for (auto [init, argument] : llvm::zip(initializers, arguments))
              if (argument.use_empty())
                init = UnitAttr::get(&getContext());
          }
          if (llvm::any_of(initializers, [](Attribute a) { return isa<FloatAttr>(a); }))
            op.setAttr(kInit, ArrayAttr::get(&getContext(), initializers));
          // Bufferized parameters may alias: an unproven write kills every fact.
          facts.clear();
        } else if (isa<memref::SubViewOp, memref::CastOp, memref::ReinterpretCastOp, memref::CollapseShapeOp,
                       memref::ExpandShapeOp, memref::AllocOp, memref::AllocaOp, memref::DeallocOp,
                       memref::GetGlobalOp, func::ReturnOp>(op) ||
                   op.getName().getStringRef().starts_with("arith.")) {
          continue;
        } else {
          facts.clear();
        }
      }
    }
  }
};

struct AIRLocalizeAccumulators
    : air::impl::AIRLocalizeAccumulatorsBase<AIRLocalizeAccumulators> {

  // One output of `kernel` whose whole-buffer initial value is `initial`.
  LogicalResult localize(Operation *kernel, unsigned index, unsigned count, FloatAttr initial) {
    Value output = kernel->getOperand(kernel->getNumOperands() - count + index);
    auto alloc = output.getDefiningOp<memref::AllocOp>();
    if (!alloc)
      return kernel->emitError("Resident accumulator must be promoted by native Linalg codegen");
    auto space = dyn_cast_or_null<IntegerAttr>(cast<MemRefType>(output.getType()).getMemorySpace());
    if (!space || space.getInt() != 2)
      return kernel->emitError("Resident accumulator requires an AIE local allocation");
    SmallVector<memref::CopyOp> reads, writes;
    SmallVector<memref::DeallocOp> frees;
    for (Operation *user : output.getUsers()) {
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() == output)
          reads.push_back(copy);
        else if (copy.getSource() == output)
          writes.push_back(copy);
      } else if (auto free = dyn_cast<memref::DeallocOp>(user)) {
        frees.push_back(free);
      } else if (user != kernel) {
        return kernel->emitError("Resident accumulator needs one proven promotion copy pair");
      }
    }
    if (reads.size() != 1 || writes.size() != 1 || frees.size() != 1)
      return kernel->emitError("Resident accumulator needs one proven promotion copy pair");
    memref::CopyOp load = reads[0], store = writes[0];
    memref::DeallocOp free = frees[0];
    if (load.getSource() != store.getTarget())
      return kernel->emitError("Accumulator copy pair does not refer to the same output view");
    Value external = load.getSource();
    int64_t hoisted = 0;
    // The invariant copy-pair hoisting of AIR's HoistReduceBufferPattern,
    // restricted to synchronous memref copies and positive static loops.
    while (auto loop = dyn_cast<scf::ForOp>(alloc->getParentOp())) {
      Operation *definition = external.getDefiningOp();
      if (!definition)
        definition = external.getParentBlock()->getParentOp();
      if (loop->isAncestor(definition))
        break;
      for (Operation *op : {load.getOperation(), store.getOperation(), free.getOperation()})
        if (op->getParentOp() != loop)
          return kernel->emitError("Accumulator copy pair has incompatible control flow");
      auto lower = constantIndex(loop.getLowerBound()), upper = constantIndex(loop.getUpperBound()),
           step = constantIndex(loop.getStep());
      if (!lower || !upper || !step)
        return kernel->emitError("Accumulator hoisting needs a proven positive loop trip count");
      if (*step <= 0 || *upper <= *lower || loop->getNumResults())
        return kernel->emitError("Accumulator hoisting needs a positive loop without iter_args");
      Value external_root = root(external);
      WalkResult aliased = loop->walk([&](Operation *op) {
        if (op == load || op == store)
          return WalkResult::advance();
        for (Value operand : op->getOperands())
          if (isa<MemRefType>(operand.getType()) && root(operand) == external_root)
            return WalkResult::interrupt();
        return WalkResult::advance();
      });
      if (aliased.wasInterrupted())
        return kernel->emitError("Another access aliases the resident output inside the reduction loop");
      alloc->moveBefore(loop);
      load->moveBefore(loop);
      free->moveAfter(loop);
      store->moveAfter(loop);
      ++hoisted;
    }
    // Any enclosing serial loop must vary the output tile; otherwise a fill in
    // place of the load would reset a partial sum on every iteration.
    for (Operation *parent = load->getParentOp(); parent && !isa<func::FuncOp>(parent);
         parent = parent->getParentOp())
      if (auto loop = dyn_cast<scf::ForOp>(parent)) {
        llvm::DenseSet<Value> seen;
        if (!dependsOn(external, loop.getInductionVar(), seen))
          return kernel->emitError("Unhoisted invariant accumulator would reset a partial reduction");
      }
    OpBuilder builder(load);
    Value value = arith::ConstantOp::create(builder, load.getLoc(), initial.getType(), initial);
    linalg::FillOp::create(builder, load.getLoc(), value, output);
    load->erase();
    // The store of the localized tile writes into `external`; record its
    // buffer so a host fill that only initialized it can go.
    localizedStores[root(external)].push_back(store);
    return success();
  }

  // Stores of localized output tiles, by the buffer they write.
  llvm::MapVector<Value, SmallVector<Operation *>> localizedStores;

  // A whole-buffer fill of `buffer` at function scope is dead once every
  // output tile it initialized is filled in local memory: the next operation
  // touching the buffer is the loop nest holding the localized stores, and in
  // that nest the buffer is only written (by those stores, through views).
  void eraseDeadFills(func::FuncOp func) {
    for (auto &[buffer, stores] : localizedStores) {
      for (auto fill : llvm::make_early_inc_range(func.getBody().front().getOps<linalg::FillOp>())) {
        if (fill.getOutputs()[0] != buffer)
          continue;
        Operation *next = fill->getNextNode();
        auto touches = [&](Operation *op) {
          bool found = false;
          op->walk([&](Operation *nested) {
            for (Value operand : nested->getOperands())
              if (isa<BaseMemRefType>(operand.getType()) && root(operand) == buffer)
                found = true;
          });
          return found;
        };
        while (next && !touches(next))
          next = next->getNextNode();
        if (!next || !llvm::all_of(stores, [&](Operation *s) { return next->isAncestor(s); }))
          continue;
        bool onlyStores = true;
        next->walk([&](Operation *nested) {
          if (isa<memref::SubViewOp>(nested) && root(nested->getOperand(0)) == buffer)
            return;
          for (Value operand : nested->getOperands())
            if (isa<BaseMemRefType>(operand.getType()) && root(operand) == buffer &&
                !llvm::is_contained(stores, nested))
              onlyStores = false;
        });
        if (onlyStores)
          fill->erase();
      }
    }
  }

  void runOnOperation() override {
    SmallVector<Operation *> kernels;
    getOperation().walk([&](Operation *op) {
      if (op->hasAttr(kInit))
        kernels.push_back(op);
    });
    for (Operation *kernel : kernels) {
      auto values = kernel->getAttrOfType<ArrayAttr>(kInit);
      if (!values) {
        kernel->emitError("air.uniform_output_init must be an array");
        return signalPassFailure();
      }
      for (auto [index, value] : llvm::enumerate(values))
        if (auto initial = dyn_cast<FloatAttr>(value))
          if (failed(localize(kernel, index, values.size(), initial)))
            return signalPassFailure();
      kernel->removeAttr(kInit);
    }
    getOperation().walk([&](func::FuncOp func) {
      if (func.getBody().hasOneBlock())
        eraseDeadFills(func);
    });
  }
};

} // namespace

namespace xilinx {
namespace air {
std::unique_ptr<mlir::Pass> createAIRAnnotateUniformInitPass() {
  return std::make_unique<AIRAnnotateUniformInit>();
}
std::unique_ptr<mlir::Pass> createAIRLocalizeAccumulatorsPass() {
  return std::make_unique<AIRLocalizeAccumulators>();
}
} // namespace air
} // namespace xilinx
