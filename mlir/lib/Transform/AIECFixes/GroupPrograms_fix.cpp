//===- GroupPrograms_fix.cpp - spatial AIR programs, device edges ---*- C++ -*-===//
//
// See AIECGroupPrograms in mlir/include/air/AIECFixes/Passes_fix.td.
//
// Consecutive unit launches of a function become one launch and segment with
// all their herds when every intermediate buffer between them is proven to be
// written once and read once in the same order (or retiled by address
// identity on the cores). Proven host channel pairs become device channels;
// anything unproved stays a program boundary with its reason in the report.
//
//===----------------------------------------------------------------------===//

#include "aiec/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IntegerSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"

#include <deque>
#include <map>
#include <memory>
#include <optional>

namespace aiec {
#define GEN_PASS_DEF_AIECGROUPPROGRAMS
#include "aiec/Transforms/Passes.h.inc"
} // namespace aiec

using namespace mlir;
namespace air = xilinx::air;

namespace {

using Ints = std::vector<int64_t>;

// --- static index expressions --------------------------------------------

// An index as a canonical expression of constants and enclosing loop
// induction variables (IV: the loop's position among the endpoint's loops).
struct Expr {
  enum Kind { Const, IV, Add, Mul, Div, Mod } kind;
  int64_t value = 0;
  std::vector<std::shared_ptr<const Expr>> args;
};
using E = std::shared_ptr<const Expr>;

int compare(const E &a, const E &b) {
  if (a->kind != b->kind)
    return a->kind < b->kind ? -1 : 1;
  if (a->value != b->value)
    return a->value < b->value ? -1 : 1;
  for (size_t i = 0; i < std::min(a->args.size(), b->args.size()); ++i)
    if (int c = compare(a->args[i], b->args[i]))
      return c;
  if (a->args.size() != b->args.size())
    return a->args.size() < b->args.size() ? -1 : 1;
  return 0;
}
bool same(const std::vector<E> &a, const std::vector<E> &b) {
  return a.size() == b.size() && llvm::all_of(llvm::seq<size_t>(0, a.size()),
                                               [&](size_t i) { return compare(a[i], b[i]) == 0; });
}

E constant(int64_t value) { return std::make_shared<const Expr>(Expr{Expr::Const, value, {}}); }
E node(Expr::Kind kind, std::vector<E> args) { return std::make_shared<const Expr>(Expr{kind, 0, std::move(args)}); }

int64_t floorDiv(int64_t a, int64_t b) {
  int64_t q = a / b;
  return (a % b != 0 && ((a < 0) != (b < 0))) ? q - 1 : q;
}
int64_t floorMod(int64_t a, int64_t b) { return a - b * floorDiv(a, b); }

// Same canonical form as the reference: flattened, constant-merged and sorted
// sums and products; quotient and remainder only fold for constants.
std::optional<E> fold(Expr::Kind kind, E a, E b, std::string &why) {
  if (a->kind == Expr::Const && b->kind == Expr::Const) {
    int64_t x = a->value, y = b->value;
    if ((kind == Expr::Div || kind == Expr::Mod) && y == 0) {
      why = "division by zero in a transfer index";
      return std::nullopt;
    }
    switch (kind) {
    case Expr::Add: return constant(x + y);
    case Expr::Mul: return constant(x * y);
    case Expr::Div: return constant(floorDiv(x, y));
    case Expr::Mod: return constant(floorMod(x, y));
    default: break;
    }
  }
  if (kind != Expr::Add && kind != Expr::Mul)
    return node(kind, {a, b});
  std::vector<E> items;
  for (const E &arg : {a, b}) {
    if (arg->kind == kind)
      items.insert(items.end(), arg->args.begin(), arg->args.end());
    else
      items.push_back(arg);
  }
  int64_t merged = kind == Expr::Add ? 0 : 1;
  std::vector<E> rest;
  for (const E &item : items) {
    if (item->kind == Expr::Const)
      merged = kind == Expr::Add ? merged + item->value : merged * item->value;
    else
      rest.push_back(item);
  }
  if (kind == Expr::Mul && merged == 0)
    return constant(0);
  if (merged != (kind == Expr::Add ? 0 : 1) || rest.empty())
    rest.push_back(constant(merged));
  if (rest.size() == 1)
    return rest[0];
  llvm::sort(rest, [](const E &x, const E &y) { return compare(x, y) < 0; });
  return node(kind, std::move(rest));
}

std::optional<E> subtract(E a, E b, std::string &why) {
  auto negative = fold(Expr::Mul, constant(-1), b, why);
  if (!negative)
    return std::nullopt;
  return fold(Expr::Add, a, *negative, why);
}

std::optional<int64_t> evaluate(const E &e, ArrayRef<int64_t> coordinates) {
  switch (e->kind) {
  case Expr::Const: return e->value;
  case Expr::IV: return coordinates[e->value];
  case Expr::Add: case Expr::Mul: {
    int64_t total = e->kind == Expr::Add ? 0 : 1;
    for (const E &arg : e->args) {
      auto v = evaluate(arg, coordinates);
      if (!v)
        return std::nullopt;
      total = e->kind == Expr::Add ? total + *v : total * *v;
    }
    return total;
  }
  case Expr::Div: case Expr::Mod: {
    auto x = evaluate(e->args[0], coordinates), y = evaluate(e->args[1], coordinates);
    if (!x || !y || *y == 0)
      return std::nullopt;
    return e->kind == Expr::Div ? floorDiv(*x, *y) : floorMod(*x, *y);
  }
  }
  return std::nullopt;
}

// --- endpoints -------------------------------------------------------------

struct Signature {
  int64_t base = 0;
  std::vector<std::array<int64_t, 3>> loops;
  Ints dims;
  std::vector<E> indices;
  std::array<std::vector<E>, 3> descriptor; // offsets, sizes, strides
  bool operator==(const Signature &o) const {
    return base == o.base && loops == o.loops && dims == o.dims && same(indices, o.indices) &&
           same(descriptor[0], o.descriptor[0]) && same(descriptor[1], o.descriptor[1]) &&
           same(descriptor[2], o.descriptor[2]);
  }
  bool operator!=(const Signature &o) const { return !(*this == o); }
};

struct Endpoint {
  Operation *op;
  std::string name;
  Value root;
  int source;
  int direction; // 1: the host receives (channel.get), 0: the host sends
  Signature signature;
  int64_t ports;
  bool broadcast;
};

// Ordered frames (element addresses) per channel slot, in slot first-seen order.
struct Frames {
  std::vector<Ints> slots;
  std::vector<std::vector<Ints>> frames;
  std::map<Ints, size_t> index;
};

std::optional<Frames> streamFrames(const Endpoint &e, std::string &why) {
  const Signature &s = e.signature;
  Frames out;
  int64_t slotCount = 1;
  for (int64_t d : s.dims)
    slotCount *= d;
  Ints coordinates(s.loops.size());
  std::function<bool(size_t)> iterate = [&](size_t depth) -> bool {
    if (depth < s.loops.size()) {
      auto [lb, ub, step] = s.loops[depth];
      for (int64_t v = lb; v < ub; v += step) {
        coordinates[depth] = v;
        if (!iterate(depth + 1))
          return false;
      }
      return true;
    }
    Ints slot;
    for (const E &index : s.indices) {
      auto v = evaluate(index, coordinates);
      if (!v) {
        why = "Unsupported static transfer expression";
        return false;
      }
      slot.push_back(*v);
    }
    if (slot.empty() && slotCount == 1)
      slot.assign(s.dims.size(), 0);
    if (slot.size() != s.dims.size() ||
        llvm::any_of(llvm::seq<size_t>(0, slot.size()), [&](size_t i) { return slot[i] < 0 || slot[i] >= s.dims[i]; })) {
      why = "Invalid channel slot traversal";
      return false;
    }
    std::array<Ints, 3> axes;
    for (int a = 0; a < 3; ++a)
      for (const E &x : s.descriptor[a]) {
        auto v = evaluate(x, coordinates);
        if (!v) {
          why = "Unsupported static transfer expression";
          return false;
        }
        axes[a].push_back(*v);
      }
    if (llvm::any_of(axes[1], [](int64_t n) { return n <= 0; })) {
      why = "Empty or dynamic transfer frame";
      return false;
    }
    int64_t origin = s.base;
    for (size_t i = 0; i < std::min(axes[0].size(), axes[2].size()); ++i)
      origin += axes[0][i] * axes[2][i];
    Ints frame;
    Ints position(axes[1].size(), 0);
    std::function<void(size_t, int64_t)> elements = [&](size_t axis, int64_t address) {
      if (axis == axes[1].size()) {
        frame.push_back(address);
        return;
      }
      int64_t stride = axis < axes[2].size() ? axes[2][axis] : 0;
      for (int64_t i = 0; i < axes[1][axis]; ++i)
        elements(axis + 1, address + i * stride);
    };
    elements(0, origin);
    auto [it, inserted] = out.index.try_emplace(slot, out.slots.size());
    if (inserted) {
      out.slots.push_back(slot);
      out.frames.emplace_back();
    }
    out.frames[it->second].push_back(std::move(frame));
    return true;
  };
  if (!iterate(0))
    return std::nullopt;
  if (static_cast<int64_t>(out.slots.size()) != slotCount) {
    why = "Incomplete channel slot traversal";
    return std::nullopt;
  }
  return out;
}

// Frame boundaries may differ; every slot must carry identical ordered data.
bool sameStreams(const Frames &a, const Frames &b) {
  if (a.index.size() != b.index.size())
    return false;
  for (auto &[slot, i] : a.index) {
    auto found = b.index.find(slot);
    if (found == b.index.end())
      return false;
    Ints x, y;
    for (const Ints &f : a.frames[i])
      x.insert(x.end(), f.begin(), f.end());
    for (const Ints &f : b.frames[found->second])
      y.insert(y.end(), f.begin(), f.end());
    if (x != y)
      return false;
  }
  return true;
}

int64_t elementCount(Type type) {
  int64_t n = 1;
  for (int64_t d : cast<MemRefType>(type).getShape())
    n *= d;
  return n;
}

struct Retile {
  Operation *source, *sink;
  Value sourceBuffer, sinkBuffer;
  std::map<std::pair<Ints, Ints>, std::pair<Ints, Ints>> routes;
  Endpoint *producer, *consumer;
};

struct Item {
  int id;
  SmallVector<int64_t> sourceIds;
  Operation *op;
  std::vector<Endpoint *> streams;
  std::vector<std::pair<Operation *, std::pair<int64_t, int64_t>>> herds;
  int64_t herdCount = 0;
  std::optional<std::string> reason;
};

template <class F> void walkWithin(Operation *root, F callback) {
  root->walk<WalkOrder::PreOrder>([&](Operation *op) { callback(op); });
}

bool within(Operation *op, Operation *root, StringRef ancestor) {
  for (Operation *p = op->getParentOp(); p; p = p->getParentOp()) {
    if (p->getName().getStringRef() == ancestor)
      return true;
    if (p == root)
      return false;
  }
  return false;
}

struct AIECGroupPrograms : aiec::impl::AIECGroupProgramsBase<AIECGroupPrograms> {
  using Base::Base;

  llvm::DenseMap<Value, Value> substitutions;
  llvm::DenseSet<Value> unitIds;
  llvm::StringMap<air::ChannelOp> declarations;
  std::deque<Endpoint> endpointStorage;

  Value resolve(Value v) {
    for (auto found = substitutions.find(v); found != substitutions.end(); found = substitutions.find(v))
      v = found->second;
    return v;
  }

  std::optional<E> affine(AffineExpr expr, ArrayRef<E> dims, ArrayRef<E> symbols, std::string &why) {
    if (auto c = dyn_cast<AffineConstantExpr>(expr))
      return constant(c.getValue());
    if (auto d = dyn_cast<AffineDimExpr>(expr))
      return dims[d.getPosition()];
    if (auto s = dyn_cast<AffineSymbolExpr>(expr))
      return symbols[s.getPosition()];
    auto binary = cast<AffineBinaryOpExpr>(expr);
    auto l = affine(binary.getLHS(), dims, symbols, why);
    if (!l)
      return std::nullopt;
    auto r = affine(binary.getRHS(), dims, symbols, why);
    if (!r)
      return std::nullopt;
    switch (expr.getKind()) {
    case AffineExprKind::Add: return fold(Expr::Add, *l, *r, why);
    case AffineExprKind::Mul: return fold(Expr::Mul, *l, *r, why);
    case AffineExprKind::FloorDiv: return fold(Expr::Div, *l, *r, why);
    case AffineExprKind::Mod: return fold(Expr::Mod, *l, *r, why);
    default: why = "unsupported affine expression"; return std::nullopt;
    }
  }

  std::optional<E> expression(Value v, const llvm::DenseMap<Value, E> &env, std::string &why) {
    v = resolve(v);
    if (unitIds.contains(v))
      return constant(0);
    if (auto found = env.find(v); found != env.end())
      return found->second;
    Operation *op = v.getDefiningOp();
    if (!op) {
      why = "dynamic index or unsupported block argument";
      return std::nullopt;
    }
    if (auto c = dyn_cast<arith::ConstantOp>(op)) {
      if (auto integer = dyn_cast<IntegerAttr>(c.getValue()))
        return constant(integer.getInt());
      why = "non-integer index constant";
      return std::nullopt;
    }
    if (isa<arith::IndexCastOp, arith::IndexCastUIOp>(op))
      return expression(op->getOperand(0), env, why);
    std::optional<Expr::Kind> kind;
    bool sub = false;
    if (isa<arith::AddIOp>(op))
      kind = Expr::Add;
    else if (isa<arith::MulIOp>(op))
      kind = Expr::Mul;
    else if (isa<arith::SubIOp>(op))
      kind = Expr::Add, sub = true;
    else if (isa<arith::DivUIOp, arith::DivSIOp>(op))
      kind = Expr::Div;
    else if (isa<arith::RemUIOp>(op))
      kind = Expr::Mod;
    if (kind) {
      auto a = expression(op->getOperand(0), env, why);
      if (!a)
        return std::nullopt;
      auto b = expression(op->getOperand(1), env, why);
      if (!b)
        return std::nullopt;
      return sub ? subtract(*a, *b, why) : fold(*kind, *a, *b, why);
    }
    if (auto apply = dyn_cast<affine::AffineApplyOp>(op)) {
      AffineMap map = apply.getAffineMap();
      std::vector<E> values;
      for (Value operand : apply.getOperands()) {
        auto e = expression(operand, env, why);
        if (!e)
          return std::nullopt;
        values.push_back(*e);
      }
      ArrayRef<E> all(values);
      return affine(map.getResult(0), all.take_front(map.getNumDims()), all.drop_front(map.getNumDims()), why);
    }
    why = "unsupported index operation: " + op->getName().getStringRef().str();
    return std::nullopt;
  }

  struct View {
    Value root;
    int64_t base;
    Ints strides;
  };

  std::optional<View> view(Value v, std::string &why) {
    v = resolve(v);
    auto type = dyn_cast<MemRefType>(v.getType());
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (!type || failed(type.getStridesAndOffset(strides, offset))) {
      why = "non-strided buffer layout";
      return std::nullopt;
    }
    Operation *op = v.getDefiningOp();
    if (op && isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp>(op)) {
      auto inner = view(op->getOperand(0), why);
      if (!inner)
        return std::nullopt;
      return View{inner->root, inner->base, Ints(strides.begin(), strides.end())};
    }
    if (auto subview = dyn_cast_or_null<memref::SubViewOp>(op)) {
      auto inner = view(subview.getSource(), why);
      if (!inner)
        return std::nullopt;
      ArrayRef<int64_t> offsets = subview.getStaticOffsets();
      if (llvm::any_of(offsets, [](int64_t n) { return n < 0; })) {
        why = "dynamic buffer view";
        return std::nullopt;
      }
      int64_t base = inner->base;
      for (size_t i = 0; i < std::min(offsets.size(), inner->strides.size()); ++i)
        base += offsets[i] * inner->strides[i];
      return View{inner->root, base, Ints(strides.begin(), strides.end())};
    }
    if (offset < 0 || llvm::any_of(strides, [](int64_t s) { return s < 0; })) {
      why = "dynamic buffer layout";
      return std::nullopt;
    }
    return View{v, offset, Ints(strides.begin(), strides.end())};
  }

  Endpoint *endpoint(Operation *op, Operation *launch, int sourceId, std::string &why) {
    auto symbol = op->getAttrOfType<FlatSymbolRefAttr>("chan_name");
    std::string name = symbol.getValue().str();
    auto declared = declarations.find(name);
    if (declared == declarations.end()) {
      why = "'" + name + "'";
      return nullptr;
    }
    air::ChannelOp declaration = declared->second;
    Ints dims;
    for (Attribute a : declaration.getSize())
      dims.push_back(cast<IntegerAttr>(a).getInt());
    auto fields = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
    if (fields[0] || op->getNumResults()) {
      why = "async transfer boundary";
      return nullptr;
    }
    Value payload = op->getOperand(fields[1]);
    auto seen = view(payload, why);
    if (!seen)
      return nullptr;
    SmallVector<Operation *> parents;
    for (Operation *p = op->getParentOp(); p != launch; p = p->getParentOp())
      parents.push_back(p);
    llvm::DenseMap<Value, E> env;
    std::vector<std::array<int64_t, 3>> loops;
    for (Operation *parent : llvm::reverse(parents)) {
      if (!isa<scf::ParallelOp, scf::ForOp>(parent)) {
        why = "conditional or unsupported host transfer loop";
        return nullptr;
      }
      Block &body = parent->getRegion(0).front();
      size_t n = body.getNumArguments();
      if (parent->getNumResults() || (isa<scf::ForOp>(parent) && n != 1)) {
        why = "loop-carried transfer state";
        return nullptr;
      }
      for (size_t i = 0; i < n; ++i) {
        std::array<int64_t, 3> bounds;
        for (size_t j = 0; j < 3; ++j) {
          auto e = expression(parent->getOperand(i + j * n), env, why);
          if (!e)
            return nullptr;
          if ((*e)->kind != Expr::Const) {
            why = "non-static transfer loop";
            return nullptr;
          }
          bounds[j] = (*e)->value;
        }
        if (bounds[2] <= 0) {
          why = "non-static transfer loop";
          return nullptr;
        }
        env[body.getArgument(i)] = std::make_shared<const Expr>(Expr{Expr::IV, static_cast<int64_t>(loops.size()), {}});
        loops.push_back(bounds);
      }
    }
    int direction = isa<air::ChannelGetOp>(op);
    StringRef prefix = direction ? "dst" : "src";
    unsigned cursor = fields[1] + 1;
    std::array<std::vector<E>, 3> descriptor;
    const char *names[] = {"offsets", "sizes", "strides"};
    for (int f = 0; f < 3; ++f) {
      auto statics = op->getAttrOfType<DenseI64ArrayAttr>(("static_" + prefix + "_" + names[f]).str());
      for (int64_t value : statics.asArrayRef()) {
        if (value == std::numeric_limits<int64_t>::min()) {
          auto e = expression(op->getOperand(cursor++), env, why);
          if (!e)
            return nullptr;
          descriptor[f].push_back(*e);
        } else {
          descriptor[f].push_back(constant(value));
        }
      }
    }
    if (descriptor[0].empty() && descriptor[1].empty() && descriptor[2].empty()) {
      auto shape = cast<MemRefType>(payload.getType()).getShape();
      for (int64_t d : shape) {
        descriptor[0].push_back(constant(0));
        descriptor[1].push_back(constant(d));
      }
      for (int64_t s : seen->strides)
        descriptor[2].push_back(constant(s));
    }
    if (descriptor[0].empty() || descriptor[1].empty() || descriptor[2].empty()) {
      why = "partial transfer descriptor";
      return nullptr;
    }
    std::vector<E> indices;
    for (unsigned i = 0; i < static_cast<unsigned>(fields[1]); ++i) {
      auto e = expression(op->getOperand(i), env, why);
      if (!e)
        return nullptr;
      indices.push_back(*e);
    }
    int64_t ports = 1;
    for (int64_t d : dims)
      ports *= d;
    endpointStorage.push_back(Endpoint{op, name, seen->root, sourceId, direction,
                                       Signature{seen->base, loops, dims, indices, descriptor}, ports,
                                       declaration->hasAttr("broadcast_shape")});
    return &endpointStorage.back();
  }

  // Single frame per slot (spatial retiling).
  std::optional<Frames> singleFrames(const Endpoint &e, std::string &why) {
    auto frames = streamFrames(e, why);
    if (!frames)
      return std::nullopt;
    if (llvm::any_of(frames->frames, [](auto &f) { return f.size() != 1; })) {
      why = "Temporal retiling needs a multi-frame schedule";
      return std::nullopt;
    }
    return frames;
  }

  std::optional<std::pair<Operation *, Value>> localEndpoint(const Endpoint &e, ArrayRef<Operation *> launches,
                                                             std::string &why) {
    SmallVector<Operation *> found;
    Operation *launch = launches[e.source];
    walkWithin(launch, [&](Operation *op) {
      if (isa<air::ChannelPutOp, air::ChannelGetOp>(op) && within(op, launch, "air.herd") &&
          op->getAttrOfType<FlatSymbolRefAttr>("chan_name").getValue() == e.name)
        found.push_back(op);
    });
    if (found.size() != 1) {
      why = "Retiling requires one local endpoint per channel";
      return std::nullopt;
    }
    Operation *op = found[0];
    if (!isa<air::HerdOp>(op->getParentOp()) || op->getNumResults()) {
      why = "Temporal or conditional local endpoint";
      return std::nullopt;
    }
    auto fields = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
    StringRef prefix = isa<air::ChannelPutOp>(op) ? "src" : "dst";
    bool partial = fields[0] || llvm::any_of(fields.drop_front(3), [](int32_t n) { return n != 0; });
    for (const char *f : {"offsets", "sizes", "strides"})
      partial |= !op->getAttrOfType<DenseI64ArrayAttr>(("static_" + prefix + "_" + f).str()).empty();
    if (partial) {
      why = "Retiling requires a complete local frame";
      return std::nullopt;
    }
    Value payload = op->getOperand(fields[1]);
    auto type = cast<MemRefType>(payload.getType());
    SmallVector<int64_t> strides;
    int64_t offset = 0;
    if (failed(type.getStridesAndOffset(strides, offset))) {
      why = "Non-contiguous local frame";
      return std::nullopt;
    }
    int64_t expected = 1;
    for (int64_t i = type.getRank() - 1; i >= 0; --i) {
      if (strides[i] != expected) {
        why = "Non-contiguous local frame";
        return std::nullopt;
      }
      expected *= type.getShape()[i];
    }
    if (offset != 0 || !type.getElementType().isF32()) {
      why = "Unsupported local frame layout";
      return std::nullopt;
    }
    if (elementCount(type) * 4 > localBytes - stackBytes) {
      why = "Retile frame exceeds tile memory";
      return std::nullopt;
    }
    return std::make_pair(op, payload);
  }

  static int64_t allocatedBytes(Operation *herd) {
    int64_t total = 0;
    walkWithin(herd, [&](Operation *op) {
      if (auto alloc = dyn_cast<memref::AllocOp>(op))
        total += elementCount(alloc.getType()) * 4;
    });
    return total;
  }
  static int64_t channelCount(Operation *herd) {
    int64_t total = 0;
    walkWithin(herd, [&](Operation *op) { total += isa<air::ChannelPutOp, air::ChannelGetOp>(op); });
    return total;
  }

  std::optional<Retile> planRetile(Endpoint &producer, Endpoint &consumer, ArrayRef<Operation *> launches,
                                   std::string &why) {
    auto source = localEndpoint(producer, launches, why);
    if (!source)
      return std::nullopt;
    auto sink = localEndpoint(consumer, launches, why);
    if (!sink)
      return std::nullopt;
    auto before = singleFrames(producer, why);
    if (!before)
      return std::nullopt;
    auto after = singleFrames(consumer, why);
    if (!after)
      return std::nullopt;
    int64_t sourceCount = elementCount(source->second.getType()), sinkCount = elementCount(sink->second.getType());
    if (llvm::any_of(before->frames, [&](auto &f) { return static_cast<int64_t>(f[0].size()) != sourceCount; }) ||
        llvm::any_of(after->frames, [&](auto &f) { return static_cast<int64_t>(f[0].size()) != sinkCount; })) {
      why = "Host and local transfer frame sizes disagree";
      return std::nullopt;
    }
    std::map<int64_t, std::pair<size_t, int64_t>> owner; // address -> (slot, element)
    for (size_t s = 0; s < before->slots.size(); ++s)
      for (auto [i, address] : llvm::enumerate(before->frames[s][0]))
        if (!owner.try_emplace(address, s, static_cast<int64_t>(i)).second) {
          why = "Overlapping producer writes";
          return std::nullopt;
        }
    Retile plan{source->first, sink->first, source->second, sink->second, {}, &producer, &consumer};
    for (size_t s = 0; s < after->slots.size(); ++s)
      for (auto [i, address] : llvm::enumerate(after->frames[s][0])) {
        auto found = owner.find(address);
        if (found == owner.end()) {
          why = "Consumer reads an unwritten intermediate element";
          return std::nullopt;
        }
        auto &pair = plan.routes[{before->slots[found->second.first], after->slots[s]}];
        pair.first.push_back(found->second.second);
        pair.second.push_back(static_cast<int64_t>(i));
      }
    for (int side = 0; side < 2; ++side) {
      Operation *op = side ? plan.sink : plan.source;
      std::map<Ints, int64_t> counts, bytes;
      for (auto &[slots, payloads] : plan.routes) {
        const Ints &slot = side ? slots.second : slots.first;
        counts[slot] += 1;
        bytes[slot] += 4 * static_cast<int64_t>((side ? payloads.second : payloads.first).size());
      }
      // Existing local allocations stay: they carry the original computation.
      // Each fragment adds a transfer buffer and a binary lock.
      Operation *herd = op->getParentOp();
      int64_t allocated = allocatedBytes(herd), channels = channelCount(herd) - 1;
      if (llvm::any_of(counts, [&](auto &c) { return channels + c.second + 2 > locksPerTile; })) {
        why = "Retile locks exceed AIE1 tile capacity";
        return std::nullopt;
      }
      if (llvm::any_of(bytes, [&](auto &b) {
            return allocated + b.second + stackBytes + retileReservedBytes > localBytes;
          })) {
        why = "Retile buffers exceed AIE1 tile memory";
        return std::nullopt;
      }
    }
    return plan;
  }

  void runOnOperation() override;
  LogicalResult groupFunction(func::FuncOp function, llvm::json::Array &groups, llvm::json::Array &boundaries,
                              llvm::json::Array &internal, int64_t &sourceLaunchCount);
  Operation *merge(std::vector<Item *> items);
  void emitRetile(Retile &plan);
};

// ---------------------------------------------------------------------------

void AIECGroupPrograms::runOnOperation() {
  ModuleOp module = getOperation();
  for (auto channel : module.getOps<air::ChannelOp>())
    declarations[channel.getSymName()] = channel;
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<air::LaunchOp, air::SegmentOp, air::HerdOp>(op))
      return;
    auto fields = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
    int32_t n = fields[1];
    Block &body = op->getRegion(0).front();
    auto values = op->getOperands().drop_front(fields[0]);
    for (auto [arg, value] : llvm::zip(body.getArguments().drop_front(n), values))
      substitutions[arg] = value;
    if (isa<air::LaunchOp>(op) && llvm::all_of(values.take_front(n), [](Value v) {
          auto c = v.getDefiningOp<arith::ConstantOp>();
          auto i = c ? dyn_cast<IntegerAttr>(c.getValue()) : IntegerAttr();
          return i && i.getInt() == 1;
        }))
      for (BlockArgument arg : body.getArguments().take_front(n))
        unitIds.insert(arg);
  });
  llvm::json::Array groups, boundaries, internal;
  int64_t sourceLaunchCount = 0;
  SmallVector<func::FuncOp> functions;
  module.walk<WalkOrder::PreOrder>([&](func::FuncOp f) {
    if (!f.getBody().empty())
      functions.push_back(f);
  });
  for (func::FuncOp function : functions)
    if (failed(groupFunction(function, groups, boundaries, internal, sourceLaunchCount)))
      return signalPassFailure();
  // Private allocations used only by dead initialization chains go. No
  // function argument, escaped allocation or original kernel is touched.
  for (bool changed = true; changed;) {
    changed = false;
    SmallVector<memref::AllocOp> allocations;
    module.walk<WalkOrder::PreOrder>([&](memref::AllocOp alloc) {
      if (!alloc->getParentOfType<air::HerdOp>())
        allocations.push_back(alloc);
    });
    for (memref::AllocOp alloc : allocations) {
      Value value = alloc.getResult();
      llvm::SetVector<Operation *> users;
      for (Operation *user : value.getUsers())
        users.insert(user);
      bool dead = llvm::all_of(users, [&](Operation *o) {
        if (isa<memref::DeallocOp>(o))
          return true;
        if (!isa<memref::CopyOp, linalg::FillOp>(o) || o->getOperands().back() != value)
          return false;
        return llvm::none_of(o->getOperands().drop_back(), [&](Value v) { return v == value; });
      });
      if (!dead)
        continue;
      for (Operation *user : users)
        user->erase();
      alloc->erase();
      changed = true;
    }
  }
  int64_t programs = 0;
  module.walk([&](air::LaunchOp) { ++programs; });
  if (report.empty())
    return;
  llvm::json::Object summary{{"groups", std::move(groups)},
                             {"boundaries", std::move(boundaries)},
                             {"internal_edges", std::move(internal)},
                             {"source_launch_count", sourceLaunchCount},
                             {"program_count", programs}};
  std::error_code error;
  llvm::raw_fd_ostream stream(report, error, llvm::sys::fs::OF_Text);
  if (error) {
    module.emitError("cannot write report ") << report << ": " << error.message();
    return signalPassFailure();
  }
  stream << llvm::formatv("{0:2}", llvm::json::Value(std::move(summary))) << "\n";
}

LogicalResult AIECGroupPrograms::groupFunction(func::FuncOp function, llvm::json::Array &groupsReport,
                                               llvm::json::Array &boundaries, llvm::json::Array &internal,
                                               int64_t &sourceLaunchCount) {
  if (!function.getBody().hasOneBlock())
    return success();
  Block &block = function.getBody().front();
  std::vector<Operation *> operations;
  for (Operation &op : block)
    operations.push_back(&op);
  auto position = [&](Operation *op) {
    return std::find(operations.begin(), operations.end(), op) - operations.begin();
  };
  std::vector<Operation *> launches;
  for (Operation *op : operations)
    if (isa<air::LaunchOp>(op))
      launches.push_back(op);
  int64_t sourceBase = sourceLaunchCount;
  std::deque<Item> descriptions;
  llvm::MapVector<Value, std::vector<Endpoint *>> uses;
  for (auto [sourceId, launch] : llvm::enumerate(launches)) {
    Item item;
    item.id = sourceId;
    item.op = launch;
    if (auto ids = launch->getAttrOfType<DenseI64ArrayAttr>("aie1.source_launches"))
      item.sourceIds.assign(ids.asArrayRef().begin(), ids.asArrayRef().end());
    else
      item.sourceIds.push_back(sourceBase + sourceId);
    walkWithin(launch, [&](Operation *op) { item.herdCount += isa<air::HerdOp>(op); });
    ++sourceLaunchCount;
    std::string why;
    auto fail = [&](std::string reason) {
      why = std::move(reason);
      return false;
    };
    auto check = [&]() -> bool {
      auto fields = launch->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
      bool unit = !launch->getNumResults() && !fields[0] && fields[1] == 2;
      for (unsigned i = 0; unit && i < 2; ++i) {
        auto e = expression(launch->getOperand(i), {}, why);
        if (!e)
          return false;
        unit = (*e)->kind == Expr::Const && (*e)->value == 1;
      }
      if (!unit)
        return fail("grouping requires synchronous unit launches");
      SmallVector<air::SegmentOp> segments;
      walkWithin(launch, [&](Operation *op) {
        if (auto s = dyn_cast<air::SegmentOp>(op))
          segments.push_back(s);
      });
      if (segments.size() != 1 || segments[0]->getNumResults() || segments[0]->getParentOp() != launch)
        return fail("unsupported segment control");
      auto segmentFields = segments[0]->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
      if (segmentFields[0] || segmentFields[1])
        return fail("sized or asynchronous segment control");
      for (Operation *hierarchy : {launch, segments[0].getOperation()})
        for (NamedAttribute a : hierarchy->getAttrs())
          if (!llvm::is_contained({"sym_name", "operandSegmentSizes", "id", "aie1.source_launches"},
                                  a.getName().getValue()))
            return fail("hierarchy attributes need an explicit grouping contract");
      bool ok = true;
      launch->walk<WalkOrder::PreOrder>([&](Operation *op) {
        bool inSegment = within(op, launch, "air.segment");
        StringRef name = op->getName().getStringRef();
        if (!inSegment && !name.starts_with("arith.") &&
            !llvm::is_contained({"air.launch", "air.launch_terminator", "air.segment", "air.channel.put",
                                 "air.channel.get", "scf.for", "scf.parallel", "scf.yield", "scf.reduce",
                                 "affine.apply"},
                                name)) {
          ok = fail("host effect inside launch: " + name.str());
          return WalkResult::interrupt();
        }
        if (auto herd = dyn_cast<air::HerdOp>(op)) {
          if (op->getParentOp() != segments[0]) {
            ok = fail("nested herd control");
            return WalkResult::interrupt();
          }
          auto f = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
          if (llvm::any_of(op->getResultTypes(), [](Type t) { return !isa<air::AsyncTokenType>(t); }) || f[1] != 2) {
            ok = fail("unsupported herd control");
            return WalkResult::interrupt();
          }
          int64_t size[2];
          for (int i = 0; i < 2; ++i) {
            auto e = expression(op->getOperand(f[0] + i), {}, why);
            if (!e) {
              ok = false;
              return WalkResult::interrupt();
            }
            size[i] = (*e)->kind == Expr::Const ? (*e)->value : -1;
          }
          item.herds.push_back({op, {size[0], size[1]}});
        }
        if (isa<air::ChannelPutOp, air::ChannelGetOp>(op) && !inSegment) {
          Endpoint *e = endpoint(op, launch, sourceId, why);
          if (!e) {
            ok = false;
            return WalkResult::interrupt();
          }
          item.streams.push_back(e);
          uses[e->root].push_back(e);
        }
        return WalkResult::advance();
      });
      if (!ok)
        return false;
      if (item.herds.empty())
        return fail("launch has no herd");
      return true;
    };
    if (!check())
      item.reason = why;
    descriptions.push_back(std::move(item));
  }

  // Escapes and host memory effects prevent removing an intermediate.
  auto safe = [](Operation *op) {
    return isa<memref::AllocOp, memref::DeallocOp, memref::CastOp, memref::SubViewOp, memref::CollapseShapeOp,
               memref::ExpandShapeOp, air::LaunchOp, air::SegmentOp, air::HerdOp, air::ChannelPutOp,
               air::ChannelGetOp>(op);
  };
  llvm::MapVector<Value, std::pair<Endpoint *, Endpoint *>> eligible;
  llvm::DenseMap<Value, std::string> rejected;
  llvm::DenseMap<Value, Retile> retiles;
  llvm::DenseMap<Value, std::vector<Operation *>> initializations;
  for (auto &[root, endpoints] : uses) {
    std::vector<Endpoint *> writers, readers;
    for (Endpoint *e : endpoints)
      (e->direction ? writers : readers).push_back(e);
    if (writers.empty() || readers.empty())
      continue;
    std::optional<std::string> why;
    if (!root.getDefiningOp<memref::AllocOp>())
      why = "intermediate storage escapes the model";
    else if (writers.size() != 1 || readers.size() != 1)
      why = "multiple writers or readers require a fanout/retile schedule";
    else if (llvm::any_of(endpoints, [](Endpoint *e) { return e->broadcast; }))
      why = "broadcast edge needs explicit fanout ownership";
    else if (writers[0]->source >= readers[0]->source)
      why = "in-place or reverse dependency";
    std::optional<Frames> frames;
    int64_t addressCount = 0, extent = elementCount(root.getType());
    if (!why) {
      std::string error;
      frames = streamFrames(*writers[0], error);
      if (frames) {
        std::set<int64_t> seen;
        bool bad = false;
        for (auto &stream : frames->frames)
          for (auto &frame : stream)
            for (int64_t a : frame) {
              ++addressCount;
              bad |= !seen.insert(a).second || a < 0 || a >= extent;
            }
        if (bad)
          error = "overlapping or out-of-bounds producer frames";
      }
      if (!error.empty())
        why = "producer frame ownership: " + error;
    }
    if (!why && writers[0]->signature != readers[0]->signature) {
      std::string error;
      auto consumed = streamFrames(*readers[0], error);
      if (consumed && !sameStreams(*frames, *consumed)) {
        auto plan = planRetile(*writers[0], *readers[0], launches, error);
        if (plan)
          retiles[root] = std::move(*plan);
      }
      if (!error.empty())
        why = "producer/consumer layout: " + error;
    }
    if (!why) {
      // A full producer overwrite makes earlier initialization dead, but
      // only for private storage with no other reads.
      std::vector<Operation *> dead;
      Operation *writerLaunch = launches[writers[0]->source];
      function->walk<WalkOrder::PreOrder>([&](Operation *op) {
        for (Value operand : op->getOperands()) {
          if (!isa<MemRefType>(operand.getType()))
            continue;
          std::string ignored;
          auto seen = view(operand, ignored);
          if (!seen || seen->root != root || safe(op))
            continue;
          if (op->getParentOp() == function && isa<memref::CopyOp, linalg::FillOp>(op) &&
              op->getOperands().back() == root && addressCount == extent &&
              position(op) < position(writerLaunch) &&
              llvm::none_of(op->getOperands().drop_back(), [&](Value v) { return v == root; })) {
            if (!llvm::is_contained(dead, op))
              dead.push_back(op);
          } else {
            why = "host use or memory effect on intermediate: " + op->getName().getStringRef().str();
          }
        }
      });
      initializations[root] = dead;
    }
    if (why)
      rejected[root] = *why;
    else
      eligible[root] = {writers[0], readers[0]};
  }

  using Edges = std::vector<std::pair<Endpoint *, Endpoint *>>;
  auto plan = [&](const std::vector<Item *> &items, Edges &edges) -> std::optional<std::string> {
    for (Item *i : items)
      if (i->reason)
        return llvm::formatv("launch {0}: {1}", sourceBase + i->id, *i->reason).str();
    std::set<int> ids;
    for (Item *i : items)
      ids.insert(i->id);
    edges.clear();
    for (auto &[key, pair] : eligible)
      if (ids.count(pair.first->source) && ids.count(pair.second->source))
        edges.push_back(pair);
    std::vector<Endpoint *> exposed;
    for (Item *i : items)
      for (Endpoint *e : i->streams)
        if (!e->root.getDefiningOp<memref::AllocOp>())
          exposed.push_back(e);
    for (Endpoint *w : exposed)
      for (Endpoint *r : exposed)
        if (w->direction && w->source < r->source)
          return std::string("externally owned buffers may alias across launches");
    for (auto &[key, endpoints] : uses) {
      std::set<int> sources;
      bool writes = false;
      for (Endpoint *e : endpoints)
        if (ids.count(e->source)) {
          sources.insert(e->source);
          writes |= e->direction;
        }
      if (sources.size() > 1 && writes && !eligible.count(key)) {
        auto found = rejected.find(key);
        return found != rejected.end() ? found->second : std::string("unproved shared storage");
      }
    }
    std::set<Endpoint *> removed;
    for (auto &[p, c] : edges)
      removed.insert(p), removed.insert(c);
    int64_t ports[2] = {0, 0};
    for (Item *i : items)
      for (Endpoint *e : i->streams)
        if (!removed.count(e))
          ports[e->direction] += e->ports;
    if (ports[0] > inputPorts || ports[1] > outputPorts)
      return std::string("GMIO endpoint capacity");
    bool packet = llvm::any_of(edges, [&](auto &edge) {
      return retiles.count(edge.first->root) || edge.first->signature != edge.second->signature;
    });
    for (Item *i : items)
      for (Endpoint *e : i->streams)
        if (auto type = declarations[e->name]->getAttrOfType<StringAttr>("channel_type"))
          packet |= type.getValue() == "npu_dma_packet";
    if (packet) {
      int64_t packets = ports[0] + ports[1];
      for (auto &[p, c] : edges) {
        auto retile = retiles.find(p->root);
        packets += retile != retiles.end() ? static_cast<int64_t>(retile->second.routes.size()) : p->ports;
      }
      if (packets > packetIds)
        return std::string("native AIR packet allocator capacity (per device program)");
    }
    // A middle herd can pack its outputs and unpack its inputs at once:
    // count all new fragments together, not one edge at a time.
    // Per herd, per slot in first-seen order: (bytes, fragments).
    llvm::MapVector<Operation *, std::pair<std::map<Ints, size_t>, std::vector<std::pair<int64_t, int64_t>>>> costs;
    for (auto &[producer, consumer] : edges) {
      auto retile = retiles.find(producer->root);
      if (retile == retiles.end())
        continue;
      for (int side = 0; side < 2; ++side) {
        Operation *herd = (side ? retile->second.sink : retile->second.source)->getParentOp();
        for (auto &[slots, payloads] : retile->second.routes) {
          auto &[order, values] = costs[herd];
          auto [at, added] = order.try_emplace(side ? slots.second : slots.first, values.size());
          if (added)
            values.push_back({0, 0});
          auto &extra = values[at->second];
          extra.first += 4 * static_cast<int64_t>((side ? payloads.second : payloads.first).size());
          extra.second += 1;
        }
      }
    }
    for (auto &[herd, slots] : costs) {
      int64_t allocated = allocatedBytes(herd), channels = channelCount(herd), replaced = 0;
      for (auto &[p, c] : edges) {
        auto retile = retiles.find(p->root);
        if (retile == retiles.end())
          continue;
        replaced += (retile->second.source->getParentOp() == herd) + (retile->second.sink->getParentOp() == herd);
      }
      for (auto &cost : slots.second) {
        if (allocated + cost.first + stackBytes + retileReservedBytes > localBytes)
          return std::string("combined retile buffers exceed AIE1 tile memory");
        if (channels - replaced + cost.second + 2 > locksPerTile)
          return std::string("combined retile locks exceed AIE1 tile capacity");
      }
    }
    std::set<std::pair<int64_t, int64_t>> occupied;
    for (Item *item : items)
      for (auto &[herd, size] : item->herds) {
        auto [width, height] = size;
        bool placed = false;
        for (int64_t x = 0; !placed && x < columns - width + 1; ++x)
          for (int64_t y = 0; !placed && y < rows - height + 1; ++y) {
            bool free = true;
            for (int64_t i = 0; free && i < width; ++i)
              for (int64_t j = 0; free && j < height; ++j)
                free = !occupied.count({x + i, y + j});
            if (!free)
              continue;
            for (int64_t i = 0; i < width; ++i)
              for (int64_t j = 0; j < height; ++j)
                occupied.insert({x + i, y + j});
            placed = true;
          }
        if (!placed)
          return std::string("simultaneous tile capacity");
      }
    return std::nullopt;
  };

  std::vector<std::vector<Item *>> groups;
  std::vector<Item *> current;
  auto pureGap = [](Operation *op) {
    return isa<arith::ConstantOp, memref::AllocOp, memref::GetGlobalOp, memref::CastOp, memref::SubViewOp,
               memref::CollapseShapeOp, memref::ExpandShapeOp>(op);
  };
  for (Item &item : descriptions) {
    std::optional<std::string> reason;
    if (!current.empty()) {
      auto a = position(current.back()->op), b = position(item.op);
      std::vector<Item *> trial = current;
      trial.push_back(&item);
      Edges planned;
      std::optional<std::string> failure = plan(trial, planned);
      std::set<Operation *> dead;
      if (!failure)
        for (auto &[p, c] : planned)
          if (auto found = initializations.find(p->root); found != initializations.end())
            dead.insert(found->second.begin(), found->second.end());
      for (auto i = a + 1; i < b; ++i)
        if (!pureGap(operations[i]) && !dead.count(operations[i])) {
          reason = "host effect between launches";
          break;
        }
      if (!reason)
        reason = failure;
    }
    if (!current.empty() && reason) {
      groups.push_back(current);
      current.clear();
      boundaries.push_back(llvm::json::Object{{"before", sourceBase + item.id}, {"reason", *reason}});
    }
    current.push_back(&item);
  }
  if (!current.empty())
    groups.push_back(current);

  for (auto &items : groups) {
    Edges edges;
    std::optional<std::string> reason = plan(items, edges);
    if (!reason)
      reason = items[0]->reason;
    llvm::json::Array sourceLaunches;
    int64_t herds = 0;
    for (Item *i : items) {
      for (int64_t n : i->sourceIds)
        sourceLaunches.push_back(n);
      herds += i->herdCount;
    }
    llvm::json::Object group{{"source_launches", std::move(sourceLaunches)}, {"herds", herds}};
    group["reason"] = reason ? llvm::json::Value(*reason) : llvm::json::Value(nullptr);
    if (items.size() < 2) {
      items[0]->op->setAttr("aie1.source_launches", DenseI64ArrayAttr::get(&getContext(), items[0]->sourceIds));
      groupsReport.push_back(std::move(group));
      continue;
    }
    bool retiled = llvm::any_of(edges, [](auto &e) { return e.first->signature != e.second->signature; });
    for (auto &[producer, consumer] : edges) {
      if (auto found = initializations.find(producer->root); found != initializations.end())
        for (Operation *op : found->second)
          op->erase();
      bool viaRetile = retiles.count(producer->root);
      int64_t fragments = 0;
      if (viaRetile) {
        Retile &r = retiles[producer->root];
        fragments = r.routes.size();
        emitRetile(r);
        retiled = true;
      } else {
        Operation *consumerLaunch = items[consumer->source - items[0]->id]->op;
        auto rename = FlatSymbolRefAttr::get(&getContext(), producer->name);
        walkWithin(consumerLaunch, [&](Operation *op) {
          if (isa<air::ChannelPutOp, air::ChannelGetOp>(op) &&
              op->getAttrOfType<FlatSymbolRefAttr>("chan_name").getValue() == consumer->name)
            op->setAttr("chan_name", rename);
        });
      }
      producer->op->erase();
      consumer->op->erase();
      int64_t payloadBytes = 0;
      for (Endpoint *e : {producer, consumer}) {
        std::string ignored;
        if (auto frames = streamFrames(*e, ignored))
          for (auto &stream : frames->frames)
            for (auto &frame : stream)
              payloadBytes += 4 * static_cast<int64_t>(frame.size());
      }
      internal.push_back(llvm::json::Object{
          {"producer", sourceBase + producer->source},
          {"consumer", sourceBase + consumer->source},
          {"channel", producer->name},
          {"eliminated_host_channels", llvm::json::Array{producer->name, consumer->name}},
          {"logical_ddr_bytes_removed", payloadBytes},
          {"frame_boundaries_changed", producer->signature != consumer->signature},
          {"transport", viaRetile ? "retiled_packet" : "direct"},
          {"fragments", fragments}});
    }
    Operation *merged = merge(items);
    if (!merged)
      return failure();
    if (retiled) {
      merged->setAttr("aie1.group_transport", StringAttr::get(&getContext(), "packet"));
      group["device_dataflow"] = "packet";
      llvm::StringSet<> names;
      merged->walk([&](Operation *op) {
        if (isa<air::ChannelPutOp, air::ChannelGetOp>(op))
          names.insert(op->getAttrOfType<FlatSymbolRefAttr>("chan_name").getValue());
      });
      // Native AIR uses this enum for packet channels on both AIE1 and AIE2,
      // tile-to-tile routes included.
      getOperation().walk([&](air::ChannelOp declaration) {
        if (names.contains(declaration.getSymName()))
          declaration->setAttr("channel_type", StringAttr::get(&getContext(), "npu_dma_packet"));
      });
    }
    groupsReport.push_back(std::move(group));
  }
  return success();
}

// Merge hierarchy bodies while preserving every original operation.
Operation *AIECGroupPrograms::merge(std::vector<Item *> items) {
  std::vector<Operation *> launches;
  for (Item *i : items)
    launches.push_back(i->op);
  llvm::SetVector<Value> values;
  for (Operation *launch : launches)
    for (Value v : launch->getOperands().drop_front(2))
      values.insert(v);
  OpBuilder builder(launches.back());
  auto merged = air::LaunchOp::create(builder, launches.back()->getLoc(), ValueRange{},
                                      launches[0]->getOperands().take_front(2), values.getArrayRef(), false);
  Block &body = merged.getBody().front();
  Operation *end = body.getTerminator();
  llvm::DenseMap<Value, Value> mapped;
  for (auto [v, a] : llvm::zip(values, body.getArguments().drop_front(4)))
    mapped[v] = a;
  std::vector<Operation *> segmentChildren;
  for (Operation *launch : launches) {
    Block &old = launch->getRegion(0).front();
    SmallVector<Value> replacements(body.getArguments().take_front(4));
    for (Value v : launch->getOperands().drop_front(2))
      replacements.push_back(mapped[v]);
    for (auto [arg, now] : llvm::zip(old.getArguments(), replacements))
      arg.replaceAllUsesWith(now);
    for (Operation &child : llvm::make_early_inc_range(old.without_terminator())) {
      auto segment = dyn_cast<air::SegmentOp>(child);
      if (!segment) {
        child.moveBefore(end);
        continue;
      }
      auto fields = segment->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
      if (fields[0] || fields[1]) {
        segment.emitError("sized segments need a grouped segment schedule");
        return nullptr;
      }
      Block &sb = segment.getBody().front();
      for (auto [arg, value] : llvm::zip(sb.getArguments(), segment->getOperands()))
        arg.replaceAllUsesWith(value);
      for (Operation &inner : sb.without_terminator())
        segmentChildren.push_back(&inner);
    }
  }
  llvm::StringSet<> herdNames;
  for (Operation *child : segmentChildren)
    child->walk<WalkOrder::PreOrder>([&](air::HerdOp herd) {
      std::string name = herd.getSymName()->str(), unique = name;
      for (int suffix = 1; herdNames.contains(unique); ++suffix)
        unique = name + "_group_" + std::to_string(suffix);
      if (unique != name)
        herd->setAttr("sym_name", StringAttr::get(&getContext(), unique));
      herdNames.insert(unique);
    });
  std::vector<Operation *> ordered;
  for (Operation *child : segmentChildren)
    child->walk<WalkOrder::PreOrder>([&](Operation *op) { ordered.push_back(op); });
  llvm::DenseSet<Operation *> inside(ordered.begin(), ordered.end());
  llvm::SetVector<Value> captures;
  for (Operation *op : ordered)
    for (Value value : op->getOperands()) {
      Operation *owner = value.getDefiningOp();
      if (!owner)
        owner = cast<BlockArgument>(value).getOwner()->getParentOp();
      if (!inside.contains(owner))
        captures.insert(value);
    }
  builder.setInsertionPoint(end);
  auto segment = air::SegmentOp::create(builder, merged.getLoc(), ValueRange{}, ValueRange{},
                                        captures.getArrayRef(), false);
  segment->setAttr("sym_name", StringAttr::get(&getContext(), "group_" + std::to_string(items[0]->id)));
  Block &sb = segment.getBody().front();
  for (auto [captured, arg] : llvm::zip(captures, sb.getArguments()))
    Value(captured).replaceUsesWithIf(arg, [&](OpOperand &use) { return inside.contains(use.getOwner()); });
  Operation *segmentEnd = sb.getTerminator();
  for (Operation *child : segmentChildren)
    child->moveBefore(segmentEnd);
  for (Operation *launch : launches)
    launch->erase();
  SmallVector<int64_t> sources;
  for (Item *i : items)
    sources.append(i->sourceIds.begin(), i->sourceIds.end());
  merged->setAttr("aie1.source_launches", DenseI64ArrayAttr::get(&getContext(), sources));
  return merged;
}

// Maximal affine runs keep packing code proportional to layout complexity.
std::vector<std::array<int64_t, 4>> runs(const Ints &indices) {
  std::vector<std::array<int64_t, 4>> out;
  for (size_t start = 0; start < indices.size();) {
    int64_t step = start + 1 < indices.size() ? indices[start + 1] - indices[start] : 1;
    size_t end = start + 1;
    while (end < indices.size() && indices[end] == indices[start] + static_cast<int64_t>(end - start) * step)
      ++end;
    out.push_back({static_cast<int64_t>(start), static_cast<int64_t>(end - start), indices[start], step});
    start = end;
  }
  return out;
}

void AIECGroupPrograms::emitRetile(Retile &plan) {
  MLIRContext *context = &getContext();
  Location loc = UnknownLoc::get(context);
  OpBuilder builder(context);
  auto index = [&](int64_t n) -> Value { return arith::ConstantIndexOp::create(builder, loc, n); };
  // Linear addressing is rebuilt in the original contiguous shape; the
  // original computation buffer keeps its type.
  auto copy = [&](Value payload, Value packed, const Ints &indices, bool packing) {
    auto shape = cast<MemRefType>(payload.getType()).getShape();
    for (auto [start, length, offset, stride] : runs(indices)) {
      Value lb = index(0), ub = index(length), step = index(1);
      auto loop = scf::ForOp::create(builder, loc, lb, ub, step);
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPoint(loop.getBody()->getTerminator());
      Value iv = loop.getInductionVar();
      Value first = index(offset);
      Value scaled = arith::MulIOp::create(builder, loc, iv, index(stride));
      Value linear = arith::AddIOp::create(builder, loc, first, scaled);
      SmallVector<Value> coordinates;
      for (size_t axis = 0; axis < shape.size(); ++axis) {
        int64_t divisor = 1;
        for (size_t k = axis + 1; k < shape.size(); ++k)
          divisor *= shape[k];
        Value coordinate = divisor != 1 ? arith::DivUIOp::create(builder, loc, linear, index(divisor)).getResult()
                                        : linear;
        if (axis)
          coordinate = arith::RemUIOp::create(builder, loc, coordinate, index(shape[axis]));
        coordinates.push_back(coordinate);
      }
      Value destination = arith::AddIOp::create(builder, loc, iv, index(start));
      if (packing) {
        Value v = memref::LoadOp::create(builder, loc, payload, coordinates);
        memref::StoreOp::create(builder, loc, v, packed, ValueRange{destination});
      } else {
        Value v = memref::LoadOp::create(builder, loc, packed, ValueRange{destination});
        memref::StoreOp::create(builder, loc, v, payload, coordinates);
      }
    }
  };
  ModuleOp module = getOperation();
  int64_t routeId = 0;
  for (auto &[slots, payloads] : plan.routes) {
    std::string name = "group_edge_" + plan.producer->name + "_" + plan.consumer->name + "_" + std::to_string(routeId++);
    builder.setInsertionPointToStart(module.getBody());
    air::ChannelOp::create(builder, loc, StringAttr::get(context, name),
                           builder.getI64ArrayAttr({1, 1}), StringAttr());
    for (int side = 0; side < 2; ++side) {
      Operation *op = side ? plan.sink : plan.source;
      Value payload = side ? plan.sinkBuffer : plan.sourceBuffer;
      const Ints &slot = side ? slots.second : slots.first;
      const Ints &positions = side ? payloads.second : payloads.first;
      auto fields = op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes").asArrayRef();
      SmallVector<Value> indices(op->getOperands().take_front(fields[1]));
      builder.setInsertionPoint(op);
      Block *body;
      if (!indices.empty()) {
        SmallVector<AffineExpr> constraints;
        for (auto [i, n] : llvm::enumerate(slot))
          constraints.push_back(getAffineDimExpr(i, context) - n);
        auto condition = IntegerSet::get(indices.size(), 0, constraints, SmallVector<bool>(constraints.size(), true));
        auto branch = affine::AffineIfOp::create(builder, loc, TypeRange{}, condition, indices, false);
        body = branch.getThenBlock();
        builder.setInsertionPoint(body->getTerminator());
      } else {
        // One slot, no herd indices in the original endpoint.
        auto branch = scf::ExecuteRegionOp::create(builder, loc, TypeRange{});
        body = &branch.getRegion().emplaceBlock();
        builder.setInsertionPointToStart(body);
        scf::YieldOp::create(builder, loc);
        builder.setInsertionPoint(body->getTerminator());
      }
      auto type = MemRefType::get({static_cast<int64_t>(positions.size())}, builder.getF32Type(), AffineMap(),
                                  builder.getI32IntegerAttr(2));
      Value packed = memref::AllocOp::create(builder, loc, type);
      SmallVector<Value> zeros{index(0), index(0)};
      if (side == 0) {
        copy(payload, packed, positions, true);
        air::ChannelPutOp::create(builder, loc, Type(), ValueRange{}, name, zeros, packed, ValueRange{}, ValueRange{},
                                  ValueRange{}, Value(), ArrayRef<int64_t>{}, ArrayRef<int64_t>{},
                                  ArrayRef<int64_t>{}, DenseI32ArrayAttr(), DenseI32ArrayAttr());
      } else {
        air::ChannelGetOp::create(builder, loc, Type(), ValueRange{}, name, zeros, packed, ValueRange{}, ValueRange{},
                                  ValueRange{}, ArrayRef<int64_t>{}, ArrayRef<int64_t>{},
                                  ArrayRef<int64_t>{}, DenseI32ArrayAttr(), DenseI32ArrayAttr());
        copy(payload, packed, positions, false);
      }
      memref::DeallocOp::create(builder, loc, packed);
    }
  }
  plan.source->erase();
  plan.sink->erase();
}

} // namespace
