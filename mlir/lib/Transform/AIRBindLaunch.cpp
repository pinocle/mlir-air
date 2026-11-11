//===- BindLaunch.cpp - bind a Triton launch into a kernel's signature ----===//
//
// See AIRBindLaunch in include/air/Transform/Passes.td. Not from a base
// repository: triton-shared, Triton-XDNA and hexagon-mlir keep the launch
// (pointer sizes, kernel scalars, grid) as runtime arguments.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRFrontendPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/BitVector.h"

using namespace mlir;

namespace xilinx {
namespace air {
#define GEN_PASS_DECL_AIRBINDLAUNCH
#define GEN_PASS_DEF_AIRBINDLAUNCH
#include "air/Transform/Passes.h.inc"
} // namespace air
} // namespace xilinx

using namespace mlir;

namespace {

constexpr unsigned kGridArguments = 6; // num_programs x, y, z; program_id x, y, z

FailureOr<TypedAttr> scalarValue(Type type, StringRef text, Location loc) {
  if (auto integer = dyn_cast<IntegerType>(type)) {
    APInt value;
    if (text.getAsInteger(10, value)) {
      emitError(loc) << "scalar '" << text << "' is not an integer";
      return failure();
    }
    return TypedAttr(IntegerAttr::get(type, value.sextOrTrunc(integer.getWidth())));
  }
  if (auto real = dyn_cast<FloatType>(type)) {
    APFloat value(real.getFloatSemantics());
    auto status = value.convertFromString(text, APFloat::rmNearestTiesToEven);
    if (!status) {
      llvm::consumeError(status.takeError());
      emitError(loc) << "scalar '" << text << "' is not a number";
      return failure();
    }
    return TypedAttr(FloatAttr::get(type, value));
  }
  emitError(loc) << "scalar argument of type " << type << " is not supported";
  return failure();
}

struct AIRBindLaunch : xilinx::air::impl::AIRBindLaunchBase<AIRBindLaunch> {
  using AIRBindLaunchBase::AIRBindLaunchBase;

  void runOnOperation() override {
    SmallVector<func::FuncOp> entries;
    getOperation().walk([&](func::FuncOp function) {
      if (!function.isExternal() && function.isPublic())
        entries.push_back(function);
    });
    if (entries.size() != 1) {
      getOperation().emitError("expected one defined public kernel");
      return signalPassFailure();
    }
    func::FuncOp function = entries.front();
    if (failed(bind(function)))
      signalPassFailure();
  }

  LogicalResult bind(func::FuncOp function) {
    Block &body = function.getBody().front();
    unsigned count = function.getNumArguments();
    if (count < kGridArguments)
      return function.emitError("expected the six grid arguments triton-shared appends");
    unsigned kernelArguments = count - kGridArguments;
    unsigned pointers = 0;
    for (unsigned i = 0; i < kernelArguments; ++i)
      pointers += isa<BaseMemRefType>(function.getArgument(i).getType());
    if (sizes.size() != pointers)
      return function.emitError() << "expected " << pointers << " sizes, got " << sizes.size();
    if (scalars.size() != kernelArguments - pointers)
      return function.emitError() << "expected " << kernelArguments - pointers << " scalars, got " << scalars.size();
    if (grid.size() != 3)
      return function.emitError("expected grid=x,y,z");
    for (unsigned i = count - 3; i < count; ++i)
      if (!function.getArgument(i).use_empty())
        return function.emitError("program ids are still used; wrap the body over the grid first "
                                  "(air-wrap-func-with-parallel)");

    OpBuilder builder = OpBuilder::atBlockBegin(&body);
    SmallVector<Type> inputs;
    llvm::BitVector erase(count);
    unsigned pointer = 0, scalar = 0;
    for (unsigned i = 0; i < kernelArguments; ++i) {
      BlockArgument argument = function.getArgument(i);
      if (auto memref = dyn_cast<BaseMemRefType>(argument.getType())) {
        auto ranked = MemRefType::get({sizes[pointer++]}, memref.getElementType());
        if (auto known = dyn_cast<MemRefType>(memref)) {
          if (known != ranked)
            return function.emitError() << "argument " << i << " is " << known << ", the launch gives " << ranked;
        } else {
          argument.setType(ranked);
          auto view = builder.create<memref::CastOp>(argument.getLoc(), memref, argument);
          argument.replaceAllUsesExcept(view, view);
        }
        inputs.push_back(ranked);
        continue;
      }
      FailureOr<TypedAttr> value = scalarValue(argument.getType(), scalars[scalar++], argument.getLoc());
      if (failed(value))
        return failure();
      argument.replaceAllUsesWith(builder.create<arith::ConstantOp>(argument.getLoc(), *value));
      erase.set(i);
    }
    for (unsigned i = 0; i < 3; ++i) {
      BlockArgument argument = function.getArgument(kernelArguments + i);
      auto type = dyn_cast<IntegerType>(argument.getType());
      if (!type)
        return function.emitError("grid size arguments must be integers");
      argument.replaceAllUsesWith(
          builder.create<arith::ConstantOp>(argument.getLoc(), IntegerAttr::get(type, grid[i])));
    }
    erase.set(kernelArguments, count);
    if (failed(function.eraseArguments(erase)))
      return failure();
    function.setFunctionType(FunctionType::get(function.getContext(), inputs, function.getResultTypes()));
    return success();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> xilinx::air::createAIRBindLaunchPass() {
  return std::make_unique<AIRBindLaunch>();
}
