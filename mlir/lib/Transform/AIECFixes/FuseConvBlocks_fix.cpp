//===- FuseConvBlocks_fix.cpp - conv prologue/epilogue fusion ------*- C++ -*-===//
//
// See AIECFuseConvBlocks in mlir/include/air/AIECFixes/Passes_fix.td.
//
//===----------------------------------------------------------------------===//

#include "aiec/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

namespace aiec {
#define GEN_PASS_DEF_AIECFUSECONVBLOCKS
#include "aiec/Transforms/Passes.h.inc"
} // namespace aiec

using namespace mlir;

namespace {

constexpr llvm::StringLiteral kMarker("__internal_linalg_transform__");

// --- IR queries ----------------------------------------------------------

bool isLocal(Value value) {
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!type)
    return false;
  Attribute space = type.getMemorySpace();
  if (!space)
    return false;
  auto integer = dyn_cast<IntegerAttr>(space);
  return !integer || integer.getInt() != 0;
}

int64_t numBytes(Type type) {
  auto memref = cast<MemRefType>(type);
  return memref.getNumElements() * memref.getElementTypeBitWidth() / 8;
}

// The allocation a chain of subviews views.
Value root(Value value) {
  while (auto subview = value.getDefiningOp<memref::SubViewOp>())
    value = subview.getSource();
  return value;
}

SmallVector<Operation *> usersOf(Value value) { return llvm::to_vector(value.getUsers()); }

bool isIdentity(AffineMap map, unsigned rank) { return map.getNumDims() == rank && map.isIdentity(); }

bool allParallel(linalg::GenericOp op) {
  return llvm::all_of(op.getIteratorTypesArray(), [](utils::IteratorType t) { return t == utils::IteratorType::parallel; });
}

// The ancestor of op in block, or null.
Operation *inBlock(Operation *op, Block *block) {
  while (op && op->getBlock() != block)
    op = op->getParentOp();
  return op;
}

bool after(Operation *op, Operation *anchor) { return op->getBlock() == anchor->getBlock() && anchor->isBeforeInBlock(op); }

// Defined where anchor runs: an argument of an enclosing block, an op before
// anchor in its block, or a static allocation after it (to be moved before).
bool available(Value value, Operation *anchor) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return inBlock(anchor, argument.getOwner()) != nullptr;
  Operation *owner = value.getDefiningOp();
  if (owner->getBlock() != anchor->getBlock())
    return false;
  auto alloc = dyn_cast<memref::AllocOp>(owner);
  return owner->isBeforeInBlock(anchor) || (alloc && alloc->getNumOperands() == 0);
}

// Whether an op other than end touches value's buffer after start and before end.
bool between(Value value, Operation *start, Operation *end) {
  SmallVector<Value> views = {value};
  for (Operation *user : value.getUsers())
    if (auto subview = dyn_cast<memref::SubViewOp>(user))
      views.push_back(subview.getResult());
  for (Value view : views)
    for (Operation *user : view.getUsers()) {
      Operation *op = inBlock(user, start->getBlock());
      if (!op || op == end || isa<memref::SubViewOp>(op))
        continue;
      if (start->isBeforeInBlock(op) && op->isBeforeInBlock(end))
        return true;
    }
  return false;
}

// Proves a nonnegative index is a multiple of divisor.
bool multipleOf(Value value, int64_t divisor) {
  if (divisor == 1)
    return true;
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    auto loop = dyn_cast<scf::ForOp>(argument.getOwner()->getParentOp());
    return loop && argument == loop.getInductionVar() && multipleOf(loop.getLowerBound(), divisor) &&
           multipleOf(loop.getStep(), divisor);
  }
  Operation *owner = value.getDefiningOp();
  if (auto constant = dyn_cast<arith::ConstantIndexOp>(owner))
    return constant.value() % divisor == 0;
  if (isa<arith::AddIOp, arith::SubIOp>(owner))
    return llvm::all_of(owner->getOperands(), [&](Value v) { return multipleOf(v, divisor); });
  if (isa<arith::MulIOp>(owner))
    return llvm::any_of(owner->getOperands(), [&](Value v) { return multipleOf(v, divisor); });
  if (auto apply = dyn_cast<affine::AffineApplyOp>(owner)) {
    AffineMap map = apply.getAffineMap();
    if (map.getNumResults() != 1)
      return false;
    std::function<bool(AffineExpr)> divides = [&](AffineExpr expr) -> bool {
      if (auto c = dyn_cast<AffineConstantExpr>(expr))
        return c.getValue() % divisor == 0;
      if (auto d = dyn_cast<AffineDimExpr>(expr))
        return multipleOf(apply.getOperand(d.getPosition()), divisor);
      if (auto binary = dyn_cast<AffineBinaryOpExpr>(expr)) {
        if (expr.getKind() == AffineExprKind::Add)
          return divides(binary.getLHS()) && divides(binary.getRHS());
        if (expr.getKind() == AffineExprKind::Mul)
          return divides(binary.getLHS()) || divides(binary.getRHS());
      }
      return false;
    };
    return divides(map.getResult(0));
  }
  return false;
}

// --- the pass ------------------------------------------------------------

struct AIECFuseConvBlocks : aiec::impl::AIECFuseConvBlocksBase<AIECFuseConvBlocks> {
  using Base::Base;

  SmallVector<std::string> fused, kept;
  int64_t prologues = 0, epilogues = 0, transposeEpilogues = 0;

  // Why a core that also holds `extraBytes` and `extraInputs` more inputs
  // would not fit its tile, or nothing.
  std::optional<std::string> fits(scf::ForallOp forall, int64_t extraBytes, int64_t extraInputs) {
    int64_t used = extraBytes, inputs = extraInputs, outputs = 0;
    forall.walk([&](Operation *op) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op); alloc && isLocal(alloc.getResult()))
        used += numBytes(alloc.getType());
      if (auto copy = dyn_cast<memref::CopyOp>(op)) {
        inputs += isLocal(copy.getTarget()) && !isLocal(copy.getSource());
        outputs += isLocal(copy.getSource()) && !isLocal(copy.getTarget());
      }
    });
    const int64_t room = moduleBytes - stackBytes - staticBytes;
    if (used > room)
      return "L1: " + std::to_string(used) + " bytes of buffers, " + std::to_string(room) +
             " after stack and static data";
    if (inputs > 2 * channelsPerDirection)
      return std::string("input transfers exceed the tile's and its neighbour's S2MM channels");
    SmallVector<int64_t> bounds(forall.getStaticUpperBound());
    if (llvm::none_of(bounds, ShapedType::isDynamic) && (streamTotal || streamInputs || streamOutputs)) {
      int64_t cores = 1;
      for (int64_t n : bounds)
        cores *= n;
      if ((streamTotal && cores * (inputs + outputs) > streamTotal) || (streamInputs && cores * inputs > streamInputs) ||
          (streamOutputs && cores * outputs > streamOutputs))
        return std::to_string(cores) + " cores x " + std::to_string(inputs + outputs) +
               " streams exceed the transport's budget";
    }
    return std::nullopt;
  }

  void record(Operation *op) {
    if (auto marker = op->getAttrOfType<StringAttr>(kMarker))
      fused.push_back(marker.str());
  }

  Operation *cloneStripped(OpBuilder &builder, Operation *op) {
    Operation *clone = builder.clone(*op);
    clone->removeAttr(kMarker);
    return clone;
  }

  // The view `subview` takes, of `base` instead of its source.
  Value retarget(OpBuilder &builder, memref::SubViewOp subview, Value base) {
    IRMapping mapping;
    mapping.map(subview.getSource(), base);
    return builder.clone(*subview, mapping)->getResult(0);
  }

  void eraseDeadAccumulator(Value cb) {
    for (Operation *user : usersOf(cb))
      if (auto subview = dyn_cast<memref::SubViewOp>(user); subview && subview->use_empty())
        subview->erase();
    if (cb.use_empty())
      cb.getDefiningOp()->erase();
  }

  // Prologue: the accumulator starts as a copy of a channel broadcast.
  void fusePrologue(scf::ForallOp forall, Value cb, Value out, memref::CopyOp copyIn) {
    linalg::GenericOp broadcast;
    memref::CopyOp producer;
    for (Operation *user : cb.getUsers())
      if (auto copy = dyn_cast<memref::CopyOp>(user); copy && copy.getTarget() == cb && !forall->isAncestor(user)) {
        if (producer) {
          producer = nullptr;
          break;
        }
        producer = copy;
      }
    if (producer && producer.getSource().getDefiningOp<memref::AllocOp>()) {
      Value bb = producer.getSource();
      SmallVector<Operation *> users = usersOf(bb);
      SmallVector<linalg::GenericOp> candidates;
      for (Operation *user : users)
        if (auto g = dyn_cast<linalg::GenericOp>(user); g && g->getNumOperands() == 2 && g->getOperand(1) == bb)
          candidates.push_back(g);
      if (users.size() == 2 && candidates.size() == 1) {
        linalg::GenericOp g = candidates.front();
        SmallVector<AffineMap> maps = g.getIndexingMapsArray();
        Block &body = g.getRegion().front();
        auto d1 = maps[0].getNumResults() == 1 ? dyn_cast<AffineDimExpr>(maps[0].getResult(0)) : AffineDimExpr(nullptr);
        if (d1 && d1.getPosition() == 1 && isIdentity(maps[1], 4) &&
            cast<MemRefType>(g->getOperand(0).getType()).getRank() == 1 && body.getOperations().size() == 1 &&
            body.getTerminator()->getOperand(0) == body.getArgument(0))
          broadcast = g;
      }
    }
    if (!broadcast) {
      kept.push_back("prologue: the accumulator does not start as a channel broadcast");
      return;
    }
    const int64_t channels = cast<MemRefType>(out.getType()).getShape()[1];
    if (auto reason = fits(forall, channels * 4, 0)) { // the bias tile replaces the accumulator's load
      kept.push_back("prologue: " + *reason);
      return;
    }
    auto subview = copyIn.getSource().getDefiningOp<memref::SubViewOp>();
    OpBuilder builder(copyIn);
    Location loc = copyIn.getLoc();
    Value bias = broadcast->getOperand(0);
    OpFoldResult channel = subview.getMixedOffsets()[1];
    Value tile = memref::SubViewOp::create(builder, loc, bias, ArrayRef<OpFoldResult>{channel},
                                           ArrayRef<OpFoldResult>{builder.getIndexAttr(channels)},
                                           ArrayRef<OpFoldResult>{builder.getIndexAttr(1)});
    auto l1Type = MemRefType::get({channels}, builder.getF32Type(), MemRefLayoutAttrInterface(),
                                  cast<MemRefType>(out.getType()).getMemorySpace());
    Value l1 = memref::AllocOp::create(builder, loc, l1Type);
    memref::CopyOp::create(builder, loc, tile, l1);
    Operation *init = cloneStripped(builder, broadcast);
    init->setOperand(0, l1);
    init->setOperand(1, out);
    copyIn->erase();
    record(broadcast);
    Value bb = producer.getSource();
    producer->erase();
    broadcast->erase();
    bb.getDefiningOp()->erase();
    ++prologues;
  }

  // Epilogue through an expand_shape and a transposing copy: the core writes
  // its tile directly into the permuted destination.
  bool fuseTranspose(scf::ForallOp forall, Value cb, Value out, memref::CopyOp copyOut) {
    SmallVector<Operation *> outside;
    for (Operation *user : cb.getUsers()) {
      auto copy = dyn_cast<memref::CopyOp>(user);
      if (!forall->isAncestor(user) && !(copy && copy.getTarget() == cb))
        outside.push_back(user);
    }
    auto expand = outside.size() == 1 ? dyn_cast<memref::ExpandShapeOp>(outside.front()) : nullptr;
    if (!expand)
      return false;
    Value expanded = expand.getResult();
    SmallVector<Operation *> consumers = usersOf(expanded);
    auto generic = consumers.size() == 1 ? dyn_cast<linalg::GenericOp>(consumers.front()) : nullptr;
    Block *block = forall->getBlock();
    if (!generic || expand->getBlock() != block || generic->getBlock() != block)
      return false;
    // Neither bypass writes nor escaped aliases of the accumulator: a direct
    // initializing copy before the loop is allowed; other users before the
    // reader must be inside the loop or the expand.
    for (Operation *user : cb.getUsers()) {
      Operation *op = inBlock(user, block);
      if (!op)
        return false;
      if (op == expand || op == generic || op == forall)
        continue;
      if (op->isBeforeInBlock(generic)) {
        auto copy = dyn_cast<memref::CopyOp>(op);
        if (copy && copy.getTarget() == cb && op->isBeforeInBlock(forall))
          continue;
        return false;
      }
    }
    if (generic.getNumDpsInputs() != 1 || generic.getNumDpsInits() != 1 || generic->getOperand(0) != expanded)
      return false;
    Value dest = generic->getOperand(1);
    if (!dest.getDefiningOp<memref::AllocOp>() || !available(dest, forall) || between(dest, forall, generic))
      return false;
    // A view made before the original write could be used while the
    // convolution runs: only later uses may observe the destination.
    for (Operation *user : dest.getUsers()) {
      Operation *op = inBlock(user, block);
      if (!op)
        return false;
      if (op == generic)
        continue;
      if (op == forall || op->isBeforeInBlock(generic))
        return false;
    }
    Block &body = generic.getRegion().front();
    if (body.getOperations().size() != 1 || body.getTerminator()->getOperand(0) != body.getArgument(0) ||
        !body.getArguments().back().use_empty())
      return false;
    auto expandedType = cast<MemRefType>(expanded.getType());
    auto destType = cast<MemRefType>(dest.getType());
    const int64_t rank = expandedType.getRank();
    if (!expandedType.hasStaticShape() || !destType.hasStaticShape() || destType.getRank() != rank)
      return false;
    SmallVector<AffineMap> maps = generic.getIndexingMapsArray();
    if (maps.size() != 2 || !isIdentity(maps[1], rank) || !allParallel(generic))
      return false;
    SmallVector<int64_t> perm;
    for (AffineExpr expr : maps[0].getResults()) {
      auto dim = dyn_cast<AffineDimExpr>(expr);
      if (!dim)
        return false;
      perm.push_back(dim.getPosition());
    }
    SmallVector<int64_t> sorted(perm);
    llvm::sort(sorted);
    for (int64_t i = 0; i < rank; ++i)
      if (sorted[i] != i || expandedType.getShape()[i] != destType.getShape()[perm[i]])
        return false;
    SmallVector<ReassociationIndices> groups = expand.getReassociationIndices();
    if (static_cast<int64_t>(groups.size()) != cast<MemRefType>(cb.getType()).getRank())
      return false;
    int64_t next = 0;
    for (const auto &group : groups) {
      if (group.size() != 1 && group.size() != 2)
        return false;
      for (int64_t dim : group)
        if (dim != next++)
          return false;
    }
    SmallVector<int64_t> destStrides;
    int64_t destOffset;
    if (failed(destType.getStridesAndOffset(destStrides, destOffset)) || destOffset != 0 ||
        llvm::any_of(destStrides, [](int64_t s) { return s < 0 || ShapedType::isDynamic(s); }))
      return false;
    auto sub = copyOut.getTarget().getDefiningOp<memref::SubViewOp>();
    if (!sub)
      return false;
    ArrayRef<int64_t> sizes = sub.getStaticSizes(), steps = sub.getStaticStrides(), staticOffsets = sub.getStaticOffsets();
    if (llvm::any_of(sizes, [](int64_t x) { return x <= 0; }) || llvm::any_of(steps, [](int64_t x) { return x != 1; }) ||
        llvm::any_of(staticOffsets, [](int64_t x) { return x < 0 && !ShapedType::isDynamic(x); }))
      return false;
    SmallVector<OpFoldResult> offsets = sub.getMixedOffsets();
    SmallVector<int64_t> strides;
    for (auto [axis, group] : llvm::enumerate(groups)) {
      if (group.size() == 1) {
        strides.push_back(destStrides[perm[group[0]]]);
        continue;
      }
      const int64_t inner = expandedType.getShape()[group[1]], tile = sizes[axis];
      std::optional<int64_t> known = getConstantIntValue(offsets[axis]);
      if (inner % tile || !(known ? *known % tile == 0 : multipleOf(cast<Value>(offsets[axis]), tile)))
        return false;
      strides.push_back(destStrides[perm[group[1]]]);
    }
    if (fits(forall, 0, 0))
      return false;
    if (after(dest.getDefiningOp(), forall))
      dest.getDefiningOp()->moveBefore(forall);
    OpBuilder builder(copyOut);
    Location loc = copyOut.getLoc();
    auto constant = [&](int64_t v) -> Value { return arith::ConstantIndexOp::create(builder, loc, v); };
    Value offset = constant(0);
    for (auto [axis, group] : llvm::enumerate(groups)) {
      Value value = getValueOrCreateConstantIndexOp(builder, loc, offsets[axis]);
      if (group.size() == 2) {
        const int64_t inner = expandedType.getShape()[group[1]];
        if (sizes[axis] != inner) {
          Value rem = arith::RemUIOp::create(builder, loc, value, constant(inner));
          Value term = arith::MulIOp::create(builder, loc, rem, constant(destStrides[perm[group[1]]]));
          offset = arith::AddIOp::create(builder, loc, offset, term);
        }
        value = arith::DivUIOp::create(builder, loc, value, constant(inner));
      }
      Value term = arith::MulIOp::create(builder, loc, value, constant(destStrides[perm[group[0]]]));
      offset = arith::AddIOp::create(builder, loc, offset, term);
    }
    auto layout = StridedLayoutAttr::get(builder.getContext(), ShapedType::kDynamic, strides);
    auto viewType = MemRefType::get(sizes, destType.getElementType(), layout, destType.getMemorySpace());
    Value view = memref::ReinterpretCastOp::create(builder, loc, viewType, dest, OpFoldResult(offset),
                                                   getAsIndexOpFoldResult(builder.getContext(), sizes),
                                                   getAsIndexOpFoldResult(builder.getContext(), strides));
    memref::CopyOp::create(builder, loc, out, view);
    copyOut->erase();
    record(generic);
    generic->erase();
    expand->erase();
    eraseDeadAccumulator(cb);
    return true;
  }

  // Epilogue: the accumulator's single elementwise reader runs in the core.
  void fuseEpilogue(scf::ForallOp forall, Value cb, Value out, memref::CopyOp copyOut) {
    SmallVector<Operation *> outside;
    for (Operation *user : cb.getUsers()) {
      auto copy = dyn_cast<memref::CopyOp>(user);
      if (!forall->isAncestor(user) && !(copy && copy.getTarget() == cb))
        outside.push_back(user);
    }
    auto reader = outside.size() == 1 ? dyn_cast<linalg::GenericOp>(outside.front()) : nullptr;
    if (!reader && fuseTranspose(forall, cb, out, copyOut)) {
      ++epilogues;
      ++transposeEpilogues;
      return;
    }
    std::optional<std::string> reason;
    if (!reader) {
      reason = outside.size() == 1 && isa<memref::ExpandShapeOp>(outside.front())
                   ? "reshape/transpose does not satisfy the current layout, lifetime or resource checks"
                   : "the accumulator has no single elementwise reader";
    } else {
      SmallVector<AffineMap> maps = reader.getIndexingMapsArray();
      Block &body = reader.getRegion().front();
      SmallVector<Value> operands(reader->getOperands());
      bool shaped = reader.getNumDpsInits() == 1 && reader.getNumLoops() == 4 && allParallel(reader) &&
                    llvm::all_of(maps, [](AffineMap m) { return isIdentity(m, 4); }) &&
                    body.getArguments().back().use_empty() &&
                    llvm::all_of(operands, [&](Value v) { return v.getType() == cb.getType(); });
      if (!shaped)
        reason = "the reader is not an identity-map elementwise op of buffers like the accumulator";
      else if (llvm::any_of(operands, [&](Value v) { return v != cb && !available(v, forall); }))
        reason = "the reader's other buffers are not defined where the convolution runs";
      else if (llvm::any_of(operands, [&](Value v) { return v != cb && between(v, forall, reader); }))
        reason = "the reader's other buffers are accessed between the convolution and the reader";
      else {
        int64_t others = llvm::count_if(reader.getDpsInputs(), [&](Value v) { return v != cb; });
        reason = fits(forall, others * numBytes(out.getType()), others);
      }
    }
    if (reason) {
      kept.push_back("epilogue: " + *reason);
      return;
    }
    // The reader's buffers exist before the convolution.
    for (Value v : reader->getOperands())
      if (v != cb && isa<OpResult>(v) && after(v.getDefiningOp(), forall))
        v.getDefiningOp()->moveBefore(forall);
    auto targetView = copyOut.getTarget().getDefiningOp<memref::SubViewOp>();
    OpBuilder builder(copyOut);
    Location loc = copyOut.getLoc();
    const unsigned inputs = reader.getNumDpsInputs();
    SmallVector<Value> tiles(inputs);
    for (unsigned i = 0; i < inputs; ++i) {
      Value v = reader->getOperand(i);
      if (v == cb)
        continue;
      tiles[i] = memref::AllocOp::create(builder, loc, cast<MemRefType>(out.getType()));
      memref::CopyOp::create(builder, loc, retarget(builder, targetView, v), tiles[i]);
    }
    Operation *epilogue = cloneStripped(builder, reader);
    for (unsigned i = 0; i < epilogue->getNumOperands(); ++i)
      epilogue->setOperand(i, i < inputs && tiles[i] ? tiles[i] : out);
    memref::CopyOp::create(builder, loc, out, retarget(builder, targetView, reader.getDpsInits()[0]));
    copyOut->erase();
    record(reader);
    reader->erase();
    eraseDeadAccumulator(cb);
    ++epilogues;
  }

  void runOnOperation() override {
    SmallVector<scf::ForallOp> foralls;
    getOperation().walk([&](scf::ForallOp op) { foralls.push_back(op); });
    for (scf::ForallOp forall : foralls) {
      SmallVector<memref::CopyOp> copies;
      SmallVector<linalg::GenericOp> convs;
      forall.walk([&](Operation *op) {
        if (auto copy = dyn_cast<memref::CopyOp>(op))
          copies.push_back(copy);
        if (auto g = dyn_cast<linalg::GenericOp>(op); g && g.getNumLoops() == 7)
          convs.push_back(g);
      });
      if (convs.size() != 1)
        continue;
      Value out = convs.front()->getOperand(2); // the accumulator tile in L1
      SmallVector<memref::CopyOp> ins, outs;
      for (memref::CopyOp copy : copies) {
        if (copy.getTarget() == out)
          ins.push_back(copy);
        if (copy.getSource() == out)
          outs.push_back(copy);
      }
      if (ins.size() != 1 || outs.size() != 1)
        continue;
      Value cb = root(ins.front().getSource());
      if (root(outs.front().getTarget()) != cb || !cb.getDefiningOp<memref::AllocOp>())
        continue;
      fusePrologue(forall, cb, out, ins.front());
      fuseEpilogue(forall, cb, out, outs.front());
    }
    if (failed(verify(getOperation()))) {
      signalPassFailure();
      return;
    }
    if (!report.empty()) {
      llvm::json::Array markers, reasons;
      for (const auto &m : fused)
        markers.push_back(m);
      for (const auto &r : kept)
        reasons.push_back(r);
      llvm::json::Object value{{"fused_markers", std::move(markers)}, {"prologues", prologues},
                               {"epilogues", epilogues}, {"transpose_epilogues", transposeEpilogues},
                               {"kept", std::move(reasons)}};
      std::error_code error;
      llvm::raw_fd_ostream stream(report, error);
      if (error) {
        getOperation().emitError("cannot write fusion report ") << report;
        signalPassFailure();
        return;
      }
      stream << llvm::formatv("{0:2}", llvm::json::Value(std::move(value))) << "\n";
    }
  }
};

} // namespace
