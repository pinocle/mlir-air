// RUN: aiec-opt %s -aiec-split-readwrite-fifo | FileCheck %s

// %alloc_1 is received on channel_1 and sent on channel_2: the get lands in a
// new buffer and a generic copy (generalized) fills the original after it.
// CHECK-LABEL: air.herd @herd_0
// CHECK: memref.alloc() : memref<16xf32, 2 : i32>
// CHECK-NEXT: %[[OUT:.*]] = memref.alloc() : memref<16xf32, 2 : i32>
// CHECK: air.channel.get @channel_0
// CHECK: %[[TMP:.*]] = memref.alloc() : memref<16xf32, 2 : i32>
// CHECK: air.channel.get @channel_1[{{.*}}] (%[[TMP]]{{\[}}
// CHECK: linalg.generic {{.*}} ins(%[[TMP]] : memref<16xf32, 2 : i32>) outs(%[[OUT]] : memref<16xf32, 2 : i32>)
// CHECK-NEXT: ^bb0(%[[IN:.*]]: f32, %{{.*}}: f32):
// CHECK-NEXT: linalg.yield %[[IN]] : f32
// CHECK: memref.dealloc %[[TMP]]
// CHECK: linalg.generic {{.*}} ins(%{{.*}} : memref<16xf32, 2 : i32>) outs(%[[OUT]] : memref<16xf32, 2 : i32>) attrs = {aie1.source_op
// CHECK: air.channel.put @channel_2[{{.*}}] (%[[OUT]]{{\[}}
#map = affine_map<(d0) -> (d0)>
module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  func.func @main(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
    %c1 = arith.constant 1 : index
    memref.copy %arg0, %arg1 : memref<16xf32> to memref<16xf32>
    air.launch (%arg2, %arg3) in (%arg4=%c1, %arg5=%c1) args(%arg6=%arg0, %arg7=%arg1) : memref<16xf32>, memref<16xf32> {
      %c0 = arith.constant 0 : index
      air.channel.put  @channel_0[%c0, %c0] (%arg6[] [] []) : (memref<16xf32>)
      air.channel.put  @channel_1[%c0, %c0] (%arg7[] [] []) : (memref<16xf32>)
      air.channel.get  @channel_2[%c0, %c0] (%arg7[] [] []) : (memref<16xf32>)
      air.segment @main_0  {
        %c1_0 = arith.constant 1 : index
        air.herd @herd_0  tile (%arg8, %arg9) in (%arg10=%c1_0, %arg11=%c1_0) {
          %alloc = memref.alloc() : memref<16xf32, 2 : i32>
          %alloc_1 = memref.alloc() : memref<16xf32, 2 : i32>
          air.channel.get  @channel_0[%arg8, %arg9] (%alloc[] [] []) : (memref<16xf32, 2 : i32>)
          air.channel.get  @channel_1[%arg8, %arg9] (%alloc_1[] [] []) : (memref<16xf32, 2 : i32>)
          linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%alloc : memref<16xf32, 2 : i32>) outs(%alloc_1 : memref<16xf32, 2 : i32>) attrs =  {aie1.source_op = "linalg.generic"} {
          ^bb0(%in: f32, %out: f32):
            %0 = arith.addf %in, %out : f32
            linalg.yield %0 : f32
          }
          air.channel.put  @channel_2[%arg8, %arg9] (%alloc_1[] [] []) : (memref<16xf32, 2 : i32>)
          memref.dealloc %alloc : memref<16xf32, 2 : i32>
          memref.dealloc %alloc_1 : memref<16xf32, 2 : i32>
        }
      }
    }
    return
  }
}
