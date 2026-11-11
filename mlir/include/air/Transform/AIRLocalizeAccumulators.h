//===- AIRLocalizeAccumulators.h --------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_LOCALIZE_ACCUMULATORS_H
#define AIR_LOCALIZE_ACCUMULATORS_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIRAnnotateUniformInitPass();
std::unique_ptr<mlir::Pass> createAIRLocalizeAccumulatorsPass();

} // namespace air
} // namespace xilinx

#endif // AIR_LOCALIZE_ACCUMULATORS_H
