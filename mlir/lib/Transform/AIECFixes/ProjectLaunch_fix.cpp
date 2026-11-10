//===- ProjectLaunch_fix.cpp - one launch of a mixed module ---------*- C++ -*-===//
//
// See AIECProjectLaunch in mlir/include/air/AIECFixes/Passes_fix.td.
//
//===----------------------------------------------------------------------===//

#include "aiec/Transforms/Passes.h"

#include "mlir/IR/Builders.h"
#include "llvm/ADT/StringSet.h"

namespace aiec {
#define GEN_PASS_DEF_AIECPROJECTLAUNCH
#include "aiec/Transforms/Passes.h.inc"
} // namespace aiec

using namespace mlir;
namespace air = xilinx::air;

namespace {

struct AIECProjectLaunch : aiec::impl::AIECProjectLaunchBase<AIECProjectLaunch> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<air::LaunchOp> launches;
    module.walk([&](air::LaunchOp launch) { launches.push_back(launch); });
    if (index < 0 || index >= static_cast<int64_t>(launches.size())) {
      module.emitError("aiec-project-launch index ") << static_cast<int64_t>(index) << " is not one of " << launches.size() << " launches";
      return signalPassFailure();
    }
    llvm::StringSet<> kept;
    launches[index].walk([&](Operation *op) {
      if (auto put = dyn_cast<air::ChannelPutOp>(op))
        kept.insert(put.getChanName());
      else if (auto get = dyn_cast<air::ChannelGetOp>(op))
        kept.insert(get.getChanName());
    });
    for (int64_t position = launches.size() - 1; position >= 0; --position) {
      if (position == index)
        continue;
      air::LaunchOp launch = launches[position];
      // Compiled separately: a dependency on another program is complete.
      if (Value token = launch.getAsyncToken()) {
        OpBuilder builder(launch);
        auto done = air::WaitAllOp::create(builder, launch.getLoc(), token.getType(), ValueRange{});
        token.replaceAllUsesWith(done.getAsyncToken());
      }
      launch->erase();
    }
    for (auto channel : llvm::make_early_inc_range(module.getOps<air::ChannelOp>()))
      if (!kept.contains(channel.getSymName()))
        channel->erase();
  }
};

} // namespace
