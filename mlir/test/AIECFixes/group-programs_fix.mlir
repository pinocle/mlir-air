// RUN: aiec-opt %s -aiec-group-programs -canonicalize -cse -symbol-dce | FileCheck %s
// RUN: aiec-opt %s -aiec-group-programs="columns=1 rows=1" | FileCheck %s --check-prefix=FULL

// Two unit launches joined by %tmp (written once by @c1's host get, read once
// by @c2's host put, same frames) become one launch with both herds; the
// consumer herd receives on the producer's channel and %tmp goes away.
// CHECK-NOT: air.channel @c2
// CHECK-LABEL: func.func @main
// CHECK-NOT: memref.alloc() : memref<32xf32>
// CHECK: air.launch {{.*}} attributes {aie1.source_launches = array<i64: 0, 1>}
// CHECK-NOT: air.launch
// CHECK: air.segment @group_0
// CHECK: air.herd @herd_0
// CHECK: air.channel.put @c1
// CHECK: air.herd @herd_1
// CHECK: air.channel.get @c1
// CHECK-NOT: air.channel.put @c2
// A one-tile array holds one herd at a time: two programs remain.
// FULL: air.launch {{.*}}aie1.source_launches = array<i64: 0>
// FULL: air.launch {{.*}}aie1.source_launches = array<i64: 1>
#map = affine_map<(d0) -> (d0)>
module {
  air.channel @c0 [1, 1]
  air.channel @c1 [1, 1]
  air.channel @c2 [1, 1]
  air.channel @c3 [1, 1]
  func.func @main(%arg0: memref<32xf32>, %arg1: memref<32xf32>) {
    %c1 = arith.constant 1 : index
    %alloc = memref.alloc() : memref<32xf32>
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%alloc) : memref<32xf32>, memref<32xf32> {
      %c0 = arith.constant 0 : index
      air.channel.put  @c0[%c0, %c0] (%arg6[] [] []) : (memref<32xf32>)
      air.channel.get  @c1[%c0, %c0] (%arg7[] [] []) : (memref<32xf32>)
      air.segment @stage_0  {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
          %cst = arith.constant 1.000000e+00 : f32
          %alloc_1 = memref.alloc() : memref<32xf32, 2 : i32>
          %alloc_2 = memref.alloc() : memref<32xf32, 2 : i32>
          air.channel.get  @c0[%arg8, %arg9] (%alloc_1[] [] []) : (memref<32xf32, 2 : i32>)
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_1 : memref<32xf32, 2 : i32>) outs(%alloc_2 : memref<32xf32, 2 : i32>) {
          ^bb0(%in: f32, %out: f32):
            %0 = arith.addf %in, %cst : f32
            linalg.yield %0 : f32
          }
          air.channel.put  @c1[%arg8, %arg9] (%alloc_2[] [] []) : (memref<32xf32, 2 : i32>)
          memref.dealloc %alloc_1 : memref<32xf32, 2 : i32>
          memref.dealloc %alloc_2 : memref<32xf32, 2 : i32>
        }
      }
    }
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%alloc, %arg7=%arg1) : memref<32xf32>, memref<32xf32> {
      %c0 = arith.constant 0 : index
      air.channel.put  @c2[%c0, %c0] (%arg6[] [] []) : (memref<32xf32>)
      air.channel.get  @c3[%c0, %c0] (%arg7[] [] []) : (memref<32xf32>)
      air.segment @stage_1  {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_1  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
          %cst = arith.constant 2.000000e+00 : f32
          %alloc_1 = memref.alloc() : memref<32xf32, 2 : i32>
          %alloc_2 = memref.alloc() : memref<32xf32, 2 : i32>
          air.channel.get  @c2[%arg8, %arg9] (%alloc_1[] [] []) : (memref<32xf32, 2 : i32>)
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc_1 : memref<32xf32, 2 : i32>) outs(%alloc_2 : memref<32xf32, 2 : i32>) {
          ^bb0(%in: f32, %out: f32):
            %0 = arith.addf %in, %cst : f32
            linalg.yield %0 : f32
          }
          air.channel.put  @c3[%arg8, %arg9] (%alloc_2[] [] []) : (memref<32xf32, 2 : i32>)
          memref.dealloc %alloc_1 : memref<32xf32, 2 : i32>
          memref.dealloc %alloc_2 : memref<32xf32, 2 : i32>
        }
      }
    }
    return
  }
}
