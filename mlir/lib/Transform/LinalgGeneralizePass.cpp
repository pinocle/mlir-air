//===- Generalize.cpp - linalg named op to generic ------------------------===//
//
// Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause.
// For more license information:
//   https://github.com/qualcomm/hexagon-mlir/LICENSE.txt
//
//===----------------------------------------------------------------------===//
//
// This file implements lowering most of the named to linalg generic.
//
//
// Copied from hexagon-mlir 9ec15e8, qcom_hexagon_backend/lib/Transforms/GeneralizePass.cpp.
// Changes: xilinx::air namespace and include paths, marked AIE1.
// AIE1: linalg.matmul stays named (air-linalg-codegen tiles named matmuls).
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/Support/Debug.h"
#include <algorithm>
#include <numeric>
#include <vector>

#include "air/Transform/AIRFrontendPasses.h"

#define DEBUG_TYPE "linalg-generalize"

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define DBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;

using namespace mlir;

namespace xilinx {
namespace air {
#define GEN_PASS_DEF_LINALGGENERALIZE
#include "air/Transform/Passes.h.inc"
} // namespace air
} // namespace xilinx

namespace {

struct LinalgGeneralizePass
    : public xilinx::air::impl::LinalgGeneralizeBase<LinalgGeneralizePass> {
  void runOnOperation() override;
  FailureOr<linalg::GenericOp> GeneralizeOp(IRRewriter &rewriter,
                                            linalg::LinalgOp linalgOp);
};

FailureOr<linalg::GenericOp>
LinalgGeneralizePass::GeneralizeOp(IRRewriter &rewriter,
                                   linalg::LinalgOp linalgOp) {
  rewriter.setInsertionPoint(linalgOp);
  FailureOr<linalg::GenericOp> generalizedOp =
      linalg::generalizeNamedOp(rewriter, linalgOp);
  return generalizedOp;
}

void LinalgGeneralizePass::runOnOperation() {
  auto funcOp = getOperation();
  IRRewriter rewriter(&getContext());
  // AIE1: hexagon-mlir generalizes every linalg.matmul here first.

  SmallVector<linalg::LinalgOp> namedOps;
  // Collect all linalg named ops
  funcOp.walk([&](linalg::LinalgOp linalgOp) {
    // AIE1: the MLIR this project pins (bdf04ebb) has no per-function
    // elementwise ops (AbsOp, AddOp, CeilOp, DivOp, DivUnsignedOp, ExpOp,
    // FloorOp, LogOp, MaxOp, MulOp, NegFOp, SubOp); linalg.elementwise
    // replaced them.
    if (isa_and_nonnull<
            linalg::BroadcastOp, linalg::CopyOp, linalg::ElementwiseOp,
            linalg::MapOp, linalg::ReduceOp, linalg::MatmulTransposeBOp,
            linalg::BatchMatmulTransposeBOp,
            // Fix for Flash-Attention : Don't generalize transpose
            linalg::TransposeOp>(linalgOp.getOperation())) {
      namedOps.push_back(linalgOp);
    }
  });

  for (auto linalgOp : namedOps) {
    FailureOr<linalg::GenericOp> generalizedOp =
        GeneralizeOp(rewriter, linalgOp);
    if (failed(generalizedOp)) {
      linalgOp->emitOpError("failed to generalize linalg named op");
      return signalPassFailure();
    }
  }
}
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
xilinx::air::createLinalgGeneralizePass() {
  return std::make_unique<LinalgGeneralizePass>();
}
