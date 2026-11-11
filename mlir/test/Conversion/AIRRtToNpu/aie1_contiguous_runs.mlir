//===- aie1_contiguous_runs.mlir -------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s --pass-pipeline='builtin.module(func.func(air-opt-shim-dma-bds{device=xcvc1902}),canonicalize,cse,air-to-std,symbol-dce,affine-expand-index-ops,canonicalize,cse,airrt-to-npu,canonicalize,cse)' | FileCheck %s

// An AIE1 (xcvc1902) program as aircc's air-to-aie leaves it: an 8x8 tile at
// columns 8..15 of a 16x16 matrix, plus one, into an 8x8 result. AIE1 shim
// BDs are one-dimensional, so airrt-to-npu splits the strided input into one
// contiguous run per row, in access order; air-opt-shim-dma-bds ends the
// launch with the air.launch_end wait whose drain becomes the output's await.

// CHECK-LABEL: aie.runtime_sequence
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 8 len = 8 sizes = [8] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 24 len = 8 sizes = [8] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 40 len = 8 sizes = [8] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 56 len = 8 sizes = [8] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 72 len = 8 sizes = [8] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 88 len = 8 sizes = [8] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 104 len = 8 sizes = [8] strides = [1])
// CHECK: aie.dma_bd(%arg0 : memref<16x16xi32> offset = 120 len = 8 sizes = [8] strides = [1])
// CHECK: %[[OUT:.*]] = aiex.dma_configure_task_for
// CHECK-NEXT: aie.dma_bd(%arg1 : memref<8x8xi32> offset = 0 len = 64 sizes = [64] strides = [1])
// CHECK: aiex.dma_start_task(%[[OUT]])
// CHECK: aiex.dma_await_task(%[[OUT]])

#loop_annotation = #llvm.loop_annotation<mustProgress = true>
module {
  aie.device(xcvc1902) @seg {
    %shim_noc_tile_6_0 = aie.tile(6, 0)
    %shim_noc_tile_7_0 = aie.tile(7, 0)
    %tile_7_2 = aie.tile(7, 2)
    aie.objectfifo @air_channel_0(%shim_noc_tile_7_0, {%tile_7_2}, 1 : i32) : !aie.objectfifo<memref<8x8xi32>>  
    aie.objectfifo @air_channel_1(%tile_7_2, {%shim_noc_tile_6_0}, 1 : i32) : !aie.objectfifo<memref<8x8xi32>>  
    %__air_external_buffer = aie.external_buffer {sym_name = "__air_external_buffer"} : memref<16x16xi32>
    %__air_external_buffer_1 = aie.external_buffer {sym_name = "__air_external_buffer_1"} : memref<8x8xi32>
    %core_7_2 = aie.core(%tile_7_2) {
      %c1_i32 = arith.constant 1 : i32
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      cf.br ^bb1
    ^bb1:  // 2 preds: ^bb0, ^bb2
      cf.br ^bb2
    ^bb2:  // pred: ^bb1
      %0 = aie.objectfifo.acquire @air_channel_0(Consume, 1) : memref<8x8xi32>
      %1 = builtin.unrealized_conversion_cast %0 : memref<8x8xi32> to memref<8x8xi32, 2>
      %2 = aie.objectfifo.acquire @air_channel_1(Produce, 1) : memref<8x8xi32>
      %3 = builtin.unrealized_conversion_cast %2 : memref<8x8xi32> to memref<8x8xi32, 2>
      scf.for %arg0 = %c0 to %c8 step %c1 {
        scf.for %arg1 = %c0 to %c8 step %c1 {
          %4 = memref.load %1[%arg0, %arg1] : memref<8x8xi32, 2>
          %5 = arith.addi %4, %c1_i32 : i32
          memref.store %5, %3[%arg0, %arg1] : memref<8x8xi32, 2>
        } {loop_annotation = #loop_annotation}
      } {loop_annotation = #loop_annotation}
      aie.objectfifo.release @air_channel_1(Produce, 1)
      aie.objectfifo.release @air_channel_0(Consume, 1)
      cf.br ^bb1
    } {air.herd_local_id = array<i64: 0, 0>, air.herd_name = "herd", air.herd_size = array<i64: 1, 1>, stack_size = 2048 : i32}
  } {dlti.dl_spec = #dlti.dl_spec<index = 32 : i64>}
  airrt.module_metadata{
    airrt.segment_metadata attributes {dma_allocations = [], sym_name = "seg"}{
      airrt.herd_metadata {dma_allocations = [], loc_x = 7 : i64, loc_y = 2 : i64, size_x = 1 : i64, size_y = 1 : i64, sym_name = "herd"}
    }
  }
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @tile_add1(%arg0: memref<16x16xi32>, %arg1: memref<8x8xi32>) {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<16x16xi32>, memref<8x8xi32> attributes {id = 1 : i32} {
      %c0 = arith.constant 0 : index
      %1 = air.channel.put async  @channel_0[%c0, %c0] (%arg6[0, 8] [8, 8] [16, 1]) {id = 1 : i32, metadata = @air_channel_0} : (memref<16x16xi32>)
      %2 = air.channel.get async  @channel_1[%c0, %c0] (%arg7[] [] []) {id = 2 : i32, metadata = @air_channel_1} : (memref<8x8xi32>)
      %3 = air.segment @seg async  attributes {id = 2 : i32, x_loc = 7 : i64, x_size = 10 : i64, y_loc = 2 : i64, y_size = 6 : i64} {
        %c1_0 = arith.constant 1 : index
        %4 = air.herd @herd async  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) attributes {id = 3 : i32, x_loc = 7 : i64, y_loc = 2 : i64} {
          %c0_1 = arith.constant 0 : index
          %c1_2 = arith.constant 1 : index
          %c8 = arith.constant 8 : index
          %c1_i32 = arith.constant 1 : i32
          %async_token, %results = air.execute -> (memref<8x8xi32, 2>) {
            %alloc = memref.alloc() : memref<8x8xi32, 2>
            air.execute_terminator %alloc : memref<8x8xi32, 2>
          }
          %async_token_3, %results_4 = air.execute -> (memref<8x8xi32, 2>) {
            %alloc = memref.alloc() : memref<8x8xi32, 2>
            air.execute_terminator %alloc : memref<8x8xi32, 2>
          }
          %5 = air.channel.get async [%async_token]  @channel_0[%arg8, %arg9] (%results[] [] []) {id = 3 : i32} : (memref<8x8xi32, 2>)
          %6 = air.wait_all async [%async_token_3, %5] 
          %7 = scf.for %arg12 = %c0_1 to %c8 step %c1_2 iter_args(%arg13 = %6) -> (!air.async.token) {
            %9 = scf.for %arg14 = %c0_1 to %c8 step %c1_2 iter_args(%arg15 = %arg13) -> (!air.async.token) {
              %async_token_7, %results_8 = air.execute [%arg15] -> (i32) {
                %12 = memref.load %results[%arg12, %arg14] : memref<8x8xi32, 2>
                air.execute_terminator %12 : i32
              }
              %10 = arith.addi %results_8, %c1_i32 : i32
              %async_token_9 = air.execute [%arg15] {
                memref.store %10, %results_4[%arg12, %arg14] : memref<8x8xi32, 2>
              }
              %11 = air.wait_all async [%async_token_7, %async_token_9] 
              scf.yield %11 : !air.async.token
            }
            scf.yield %9 : !air.async.token
          }
          %8 = air.channel.put async [%7]  @channel_1[%arg8, %arg9] (%results_4[] [] []) {id = 4 : i32} : (memref<8x8xi32, 2>)
          %async_token_5 = air.execute [%7] {
            memref.dealloc %results : memref<8x8xi32, 2>
          }
          %async_token_6 = air.execute [%8] {
            memref.dealloc %results_4 : memref<8x8xi32, 2>
          }
        }
      }
    }
    return
  }
}
