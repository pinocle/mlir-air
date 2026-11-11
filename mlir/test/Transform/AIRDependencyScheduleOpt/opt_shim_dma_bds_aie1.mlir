//===- opt_shim_dma_bds_aie1.mlir ------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -pass-pipeline='builtin.module(func.func(air-opt-shim-dma-bds{device=xcvc1902}))' | FileCheck %s

// AIE1 shim DMAs are one-dimensional, so nothing is tiled; the launch body and
// the function body still end in the blocking air.launch_end wait that
// airrt-to-npu turns into the runtime sequence's awaits.
// CHECK-LABEL: func.func @f
// CHECK: %[[L:.*]] = air.launch async
// CHECK: %[[P:.*]] = air.channel.put async
// CHECK: %[[G:.*]] = air.channel.get async
// CHECK: air.wait_all [%[[P]], %[[G]]]  {air.launch_end}
// CHECK: air.wait_all [%[[L]]]  {air.launch_end}
// CHECK-NEXT: return
module {
  air.channel @in [1, 1]
  air.channel @out [1, 1]
  func.func @f(%a: memref<64xi32>, %b: memref<64xi32>) {
    %l = air.launch async () in () args(%x = %a, %y = %b) : memref<64xi32>, memref<64xi32> {
      %p = air.channel.put async @in[] (%x[] [] []) : (memref<64xi32>)
      %g = air.channel.get async @out[] (%y[] [] []) : (memref<64xi32>)
    }
    return
  }
}
