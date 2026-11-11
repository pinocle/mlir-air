// RUN: air-opt %s -air-localize-accumulators -split-input-file -verify-diagnostics | FileCheck %s

// The L1 accumulator of the reduction is filled with its proven uniform
// initial value instead of loaded, and its copy pair leaves the reduction loop.
// CHECK-LABEL: func.func @main
// CHECK: scf.parallel
// CHECK:   %[[ACC:.*]] = memref.alloc() : memref<16x8xf32, 2 : i32>
// CHECK:   linalg.fill ins(%{{.*}} : f32) outs(%[[ACC]] : memref<16x8xf32, 2 : i32>)
// CHECK:   scf.for
// CHECK-NOT: memref.copy %{{.*}}, %[[ACC]]
// CHECK-NOT: air.uniform_output_init
// CHECK:   }
// CHECK:   memref.copy %[[ACC]], %{{.*}} : memref<16x8xf32, 2 : i32> to memref<16x8xf32, strided
// CHECK:   memref.dealloc %[[ACC]]
#map = affine_map<(d0, d1) -> ()>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map4 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @main(%arg0: memref<16x16xf32>, %arg1: memref<16x16xf32>, %arg2: memref<16x16xf32>) {
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %cst = arith.constant 1.250000e+00 : f32
    %alloc = memref.alloc() : memref<16x16xf32>
    linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst : f32) outs(%alloc : memref<16x16xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    }
    memref.copy %alloc, %arg2 : memref<16x16xf32> to memref<16x16xf32>
    scf.parallel (%arg3) = (%c0) to (%c16) step (%c8) {
      %subview = memref.subview %arg2[0, %arg3] [16, 8] [1, 1] : memref<16x16xf32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
      scf.for %arg4 = %c0 to %c16 step %c4 {
        %subview_0 = memref.subview %arg0[0, %arg4] [16, 4] [1, 1] : memref<16x16xf32> to memref<16x4xf32, strided<[16, 1], offset: ?>>
        %subview_1 = memref.subview %arg1[%arg4, %arg3] [4, 8] [1, 1] : memref<16x16xf32> to memref<4x8xf32, strided<[16, 1], offset: ?>>
        %alloc_2 = memref.alloc() : memref<16x4xf32, 2 : i32>
        %alloc_3 = memref.alloc() : memref<4x8xf32, 2 : i32>
        %alloc_4 = memref.alloc() : memref<16x8xf32, 2 : i32>
        memref.copy %subview_0, %alloc_2 : memref<16x4xf32, strided<[16, 1], offset: ?>> to memref<16x4xf32, 2 : i32>
        memref.copy %subview_1, %alloc_3 : memref<4x8xf32, strided<[16, 1], offset: ?>> to memref<4x8xf32, 2 : i32>
        memref.copy %subview, %alloc_4 : memref<16x8xf32, strided<[16, 1], offset: ?>> to memref<16x8xf32, 2 : i32>
        linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "reduction"]} ins(%alloc_2, %alloc_3 : memref<16x4xf32, 2 : i32>, memref<4x8xf32, 2 : i32>) outs(%alloc_4 : memref<16x8xf32, 2 : i32>) attrs =  {air.uniform_output_init = [1.250000e+00 : f32]} {
        ^bb0(%in: f32, %in_5: f32, %out: f32):
          %0 = arith.mulf %in, %in_5 : f32
          %1 = arith.addf %out, %0 : f32
          linalg.yield %1 : f32
        }
        memref.copy %alloc_4, %subview : memref<16x8xf32, 2 : i32> to memref<16x8xf32, strided<[16, 1], offset: ?>>
        memref.dealloc %alloc_2 : memref<16x4xf32, 2 : i32>
        memref.dealloc %alloc_3 : memref<4x8xf32, 2 : i32>
        memref.dealloc %alloc_4 : memref<16x8xf32, 2 : i32>
      }
      scf.reduce
    }
    return
  }
}


// -----

// An output that native promotion did not place in local memory is rejected.
func.func @unpromoted(%a: memref<4xf32>, %c: memref<4xf32>) {
  // expected-error @+1 {{Resident accumulator must be promoted by native Linalg codegen}}
  linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]}
      ins(%a : memref<4xf32>) outs(%c : memref<4xf32>) attrs = {air.uniform_output_init = [0.000000e+00 : f32]} {
  ^bb0(%in: f32, %out: f32):
    %s = arith.addf %in, %out : f32
    linalg.yield %s : f32
  }
  return
}
