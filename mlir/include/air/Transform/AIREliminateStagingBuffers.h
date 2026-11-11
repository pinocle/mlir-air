//===- AIREliminateStagingBuffers.h -----------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_ELIMINATE_STAGING_BUFFERS_H
#define AIR_ELIMINATE_STAGING_BUFFERS_H

#include "air/Transform/PassDetail.h"

#include "mlir/Pass/Pass.h"
#include <memory>

namespace xilinx {
namespace air {

std::unique_ptr<mlir::Pass> createAIREliminateStagingBuffersPass();

} // namespace air
} // namespace xilinx

#endif // AIR_ELIMINATE_STAGING_BUFFERS_H
