//===- AIRFrontendPasses.h --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//
//
// Front-end normalization passes before AIR tiling: hexagon-mlir 9ec15e8
// passes (BSD-3-Clause, Qualcomm) and the Triton launch binding.
//
//===----------------------------------------------------------------------===//

#ifndef AIR_FRONTEND_PASSES_H
#define AIR_FRONTEND_PASSES_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRBindLaunchPass();
std::unique_ptr<mlir::Pass> createAIRReadOnlyArgumentsPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createCopyCanonicalizationPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createDecomposeTensorConcatPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createEraseUnusedLinalgOperands();
std::unique_ptr<mlir::InterfacePass<mlir::FunctionOpInterface>> createLinalgGeneralizePass();
std::unique_ptr<mlir::Pass> createReduceContractionRankPass();

} // namespace air
} // namespace xilinx

#endif // AIR_FRONTEND_PASSES_H
