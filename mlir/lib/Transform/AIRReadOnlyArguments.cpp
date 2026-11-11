//===- ReadOnlyArguments.cpp - keep function tensor inputs unwritten ------===//
//
// See AIRReadOnlyArguments in include/air/Transform/Passes.td. Not from a
// base repository.
//
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRFrontendPasses.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

using namespace mlir;

namespace xilinx {
namespace air {
#define GEN_PASS_DEF_AIRREADONLYARGUMENTS
#include "air/Transform/Passes.h.inc"
} // namespace air
} // namespace xilinx

using namespace mlir;

namespace {

struct AIRReadOnlyArguments : xilinx::air::impl::AIRReadOnlyArgumentsBase<AIRReadOnlyArguments> {
  void runOnOperation() override {
    getOperation().walk([](func::FuncOp function) {
      if (function.isExternal())
        return;
      for (BlockArgument argument : function.getArguments())
        if (isa<TensorType>(argument.getType()))
          function.setArgAttr(argument.getArgNumber(), bufferization::BufferizationDialect::kWritableAttrName,
                              BoolAttr::get(function.getContext(), false));
    });
  }
};

} // namespace

std::unique_ptr<mlir::Pass> xilinx::air::createAIRReadOnlyArgumentsPass() {
  return std::make_unique<AIRReadOnlyArguments>();
}
