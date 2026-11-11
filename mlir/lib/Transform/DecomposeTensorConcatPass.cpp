//===- DecomposeTensorConcatPass.cpp - decompose tensor concat ------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
//   Wrapper pass for decompose tensor concat.
//
//
// Copied from hexagon-mlir 9ec15e8, qcom_hexagon_backend/lib/Transforms/DecomposeTensorConcatPass.cpp.
// Changes: xilinx::air namespace and include paths.
//===----------------------------------------------------------------------===//

#include "air/Transform/AIRFrontendPasses.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/TransformOps/TensorTransformOps.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hexagon-decompose-concat"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

using namespace mlir;

namespace xilinx {
namespace air {
#define GEN_PASS_DEF_DECOMPOSETENSORCONCAT
#include "air/Transform/Passes.h.inc"
} // namespace air
} // namespace xilinx

namespace {
struct DecomposeTensorConcatPass
    : public xilinx::air::impl::DecomposeTensorConcatBase<DecomposeTensorConcatPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(moduleOp.getContext());
    tensor::populateDecomposeTensorConcatPatterns(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
xilinx::air::createDecomposeTensorConcatPass() {
  return std::make_unique<DecomposeTensorConcatPass>();
}
