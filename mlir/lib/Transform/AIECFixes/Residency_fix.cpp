//===- Residency_fix.cpp - accumulators and FIFO owners in local memory -*- C++ -*-===//
//
// See AIECLocalizeAccumulators and AIECSplitReadWriteFifo in
// mlir/include/air/AIECFixes/Passes_fix.td.
//
//===----------------------------------------------------------------------===//

#include "aiec/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

namespace aiec {
#define GEN_PASS_DEF_AIECANNOTATEUNIFORMINIT
#define GEN_PASS_DEF_AIECLOCALIZEACCUMULATORS
#define GEN_PASS_DEF_AIECSPLITREADWRITEFIFO
#include "aiec/Transforms/Passes.h.inc"
} // namespace aiec

using namespace mlir;
namespace air = xilinx::air;

namespace {

constexpr llvm::StringLiteral kInit = "aie1.uniform_output_init";

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

std::string text(Attribute attribute) {
  std::string out;
  llvm::raw_string_ostream stream(out);
  attribute.print(stream);
  return out;
}

LogicalResult writeReport(Operation *anchor, StringRef path, llvm::json::Object report) {
  if (path.empty())
    return success();
  std::error_code error;
  llvm::raw_fd_ostream stream(path, error, llvm::sys::fs::OF_Text);
  if (error)
    return anchor->emitError("cannot write report ") << path << ": " << error.message();
  stream << llvm::formatv("{0:2}", llvm::json::Value(std::move(report))) << "\n";
  return success();
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
struct AIECAnnotateUniformInit : aiec::impl::AIECAnnotateUniformInitBase<AIECAnnotateUniformInit> {
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

struct AIECLocalizeAccumulators : aiec::impl::AIECLocalizeAccumulatorsBase<AIECLocalizeAccumulators> {
  using Base::Base;

  // One output of `kernel` whose whole-buffer initial value is `initial`.
  LogicalResult localize(Operation *kernel, unsigned index, unsigned count, FloatAttr initial,
                         llvm::json::Array &records) {
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
    auto source = kernel->getAttrOfType<StringAttr>("aie1.source_op");
    records.push_back(llvm::json::Object{{"source_op", source ? source.getValue().str()
                                                              : kernel->getName().getStringRef().str()},
                                         {"output_index", index},
                                         {"initializer", text(initial)},
                                         {"hoisted_reduction_loops", hoisted},
                                         {"input_dma_eliminated", true},
                                         {"scalar_region_changed", false}});
    return success();
  }

  void runOnOperation() override {
    SmallVector<Operation *> kernels;
    getOperation().walk([&](Operation *op) {
      if (op->hasAttr(kInit))
        kernels.push_back(op);
    });
    llvm::json::Array records;
    for (Operation *kernel : kernels) {
      auto values = kernel->getAttrOfType<ArrayAttr>(kInit);
      if (!values) {
        kernel->emitError("aie1.uniform_output_init must be an array");
        return signalPassFailure();
      }
      for (auto [index, value] : llvm::enumerate(values))
        if (auto initial = dyn_cast<FloatAttr>(value))
          if (failed(localize(kernel, index, values.size(), initial, records)))
            return signalPassFailure();
      kernel->removeAttr(kInit);
    }
    llvm::json::Object summary{{"transform", "proven_uniform_initialization_and_invariant_copy_pair_hoisting"},
                               {"outputs", std::move(records)},
                               {"cpu_arithmetic_added", false}};
    if (failed(writeReport(getOperation(), report, std::move(summary))))
      signalPassFailure();
  }
};

struct AIECSplitReadWriteFifo : aiec::impl::AIECSplitReadWriteFifoBase<AIECSplitReadWriteFifo> {
  using Base::Base;

  void runOnOperation() override {
    int64_t count = 0;
    SmallVector<air::HerdOp> herds;
    getOperation().walk([&](air::HerdOp herd) { herds.push_back(herd); });
    for (air::HerdOp herd : herds) {
      // Channel endpoints of each local buffer, in program order.
      llvm::MapVector<Value, SmallVector<Operation *>> endpoints;
      herd.walk([&](Operation *op) {
        if (auto get = dyn_cast<air::ChannelGetOp>(op))
          endpoints[get.getDst()].push_back(op);
        else if (auto put = dyn_cast<air::ChannelPutOp>(op))
          endpoints[put.getSrc()].push_back(op);
      });
      for (auto &[value, ops] : endpoints) {
        bool gets = llvm::any_of(ops, [](Operation *op) { return isa<air::ChannelGetOp>(op); });
        bool puts = llvm::any_of(ops, [](Operation *op) { return isa<air::ChannelPutOp>(op); });
        if (!gets || !puts)
          continue;
        auto incoming = llvm::to_vector(llvm::make_filter_range(ops, [](Operation *op) {
          return isa<air::ChannelGetOp>(op);
        }));
        if (incoming.size() != 1 || ops.size() != 2) {
          herd.emitError("Read/write FIFO buffer needs one incoming and one outgoing owner");
          return signalPassFailure();
        }
        auto get = cast<air::ChannelGetOp>(incoming[0]);
        if (!isa<air::HerdOp, scf::ForOp>(get->getParentOp())) {
          get.emitError("Read/write FIFO split requires straight-line or static serial core control");
          return signalPassFailure();
        }
        if (!get.getAsyncDependencies().empty() || get->getNumResults() || !value.getDefiningOp<memref::AllocOp>()) {
          get.emitError("Read/write FIFO split needs a synchronous promoted local buffer");
          return signalPassFailure();
        }
        if (!get.getDynamicDstOffsets().empty() || !get.getDynamicDstSizes().empty() || !get.getDynamicDstStrides().empty() ||
            !get.getStaticDstOffsets().empty() || !get.getStaticDstSizes().empty() ||
            !get.getStaticDstStrides().empty()) {
          get.emitError("Read/write FIFO split requires a full-buffer input transfer");
          return signalPassFailure();
        }
        // The transfer lands in a temporary; a copy right after it gives the
        // consumed initial value and the produced result distinct owners.
        OpBuilder builder(get);
        auto temporary = memref::AllocOp::create(builder, get.getLoc(), cast<MemRefType>(value.getType()));
        get.getDstMutable().assign(temporary);
        builder.setInsertionPointAfter(get);
        auto copy = linalg::CopyOp::create(builder, get.getLoc(), temporary.getResult(), value);
        builder.setInsertionPointAfter(copy);
        memref::DeallocOp::create(builder, get.getLoc(), temporary);
        ++count;
      }
    }
    if (count) {
      // Keep copies structured until host/device operand identities are
      // bound: AIR eagerly scalarizes named L1 linalg.copy operations. As
      // linalg-generalize-named-ops, every named op becomes its generic form.
      SmallVector<linalg::LinalgOp> named;
      getOperation().walk([&](linalg::LinalgOp op) {
        if (!isa<linalg::GenericOp>(op))
          named.push_back(op);
      });
      IRRewriter rewriter(&getContext());
      for (linalg::LinalgOp op : named) {
        rewriter.setInsertionPoint(op);
        if (failed(linalg::generalizeNamedOp(rewriter, op))) {
          op->emitError("Cannot generalize a named Linalg operation");
          return signalPassFailure();
        }
      }
    }
    llvm::json::Object summary{{"readwrite_fifo_buffers_split", count}, {"source_values_preserved", true}};
    if (failed(writeReport(getOperation(), report, std::move(summary))))
      signalPassFailure();
  }
};

} // namespace
