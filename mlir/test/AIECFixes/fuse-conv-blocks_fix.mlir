// RUN: aiec-opt %s -aiec-fuse-conv-blocks -split-input-file | FileCheck %s
// RUN: aiec-opt %s -aiec-fuse-conv-blocks="module-bytes=2600 stack-bytes=2048 static-bytes=512" -split-input-file | FileCheck %s --check-prefix=SMALL

#bias = affine_map<(d0, d1, d2, d3) -> (d1)>
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#x = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
#w = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
#o = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>

// The bias broadcast becomes a per-core tile broadcast in L1 and the SiLU-like
// reader runs on the tile before it is stored to the reader's output.
// CHECK-LABEL: func.func @conv_bias_activation
// CHECK-NOT: linalg.generic {{.*}} ins(%{{.*}} : memref<8xf32>) outs(%{{.*}} : memref<1x8x4x4xf32>)
// CHECK: scf.forall
// CHECK:   memref.subview %{{.*}}[%{{.*}}] [4] [1] : memref<8xf32>
// CHECK:   memref.alloc() : memref<4xf32, 2 : i32>
// CHECK:   linalg.generic {{.*}} ins(%{{.*}} : memref<4xf32, 2 : i32>) outs(%{{.*}} : memref<1x4x4x4xf32, 2 : i32>)
// CHECK:   linalg.generic {{.*}}reduction
// CHECK:   arith.negf
// CHECK:   memref.copy %{{.*}}, %{{.*}} : memref<1x4x4x4xf32, 2 : i32> to memref<1x4x4x4xf32, strided
// CHECK: }
// CHECK-NOT: arith.negf
// 40 bytes after stack and static data cannot hold the 256-byte tile: nothing changes.
// SMALL-LABEL: func.func @conv_bias_activation
// SMALL: linalg.generic {{.*}} ins(%{{.*}} : memref<8xf32>) outs(%{{.*}} : memref<1x8x4x4xf32>)
// SMALL: scf.forall
// SMALL: }
// SMALL: arith.negf
func.func @conv_bias_activation(%input: memref<1x2x6x6xf32>, %weight: memref<8x2x3x3xf32>, %bias: memref<8xf32>,
                                %result: memref<1x8x4x4xf32>) {
  %broadcast = memref.alloc() : memref<1x8x4x4xf32>
  linalg.generic {indexing_maps = [#bias, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%bias : memref<8xf32>) outs(%broadcast : memref<1x8x4x4xf32>) attrs = {__internal_linalg_transform__ = "bias"} {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  }
  %acc = memref.alloc() : memref<1x8x4x4xf32>
  memref.copy %broadcast, %acc : memref<1x8x4x4xf32> to memref<1x8x4x4xf32>
  scf.forall (%t) in (2) {
    %c4 = arith.constant 4 : index
    %oc = arith.muli %t, %c4 : index
    %view = memref.subview %acc[0, %oc, 0, 0] [1, 4, 4, 4] [1, 1, 1, 1]
        : memref<1x8x4x4xf32> to memref<1x4x4x4xf32, strided<[128, 16, 4, 1], offset: ?>>
    %tile = memref.alloc() : memref<1x4x4x4xf32, 2 : i32>
    memref.copy %view, %tile : memref<1x4x4x4xf32, strided<[128, 16, 4, 1], offset: ?>> to memref<1x4x4x4xf32, 2 : i32>
    %wv = memref.subview %weight[%oc, 0, 0, 0] [4, 2, 3, 3] [1, 1, 1, 1]
        : memref<8x2x3x3xf32> to memref<4x2x3x3xf32, strided<[18, 9, 3, 1], offset: ?>>
    linalg.generic {indexing_maps = [#x, #w, #o],
                    iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]}
        ins(%input, %wv : memref<1x2x6x6xf32>, memref<4x2x3x3xf32, strided<[18, 9, 3, 1], offset: ?>>)
        outs(%tile : memref<1x4x4x4xf32, 2 : i32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %m = arith.mulf %a, %b : f32
      %s = arith.addf %c, %m : f32
      linalg.yield %s : f32
    }
    memref.copy %tile, %view : memref<1x4x4x4xf32, 2 : i32> to memref<1x4x4x4xf32, strided<[128, 16, 4, 1], offset: ?>>
    memref.dealloc %tile : memref<1x4x4x4xf32, 2 : i32>
  }
  linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%acc : memref<1x8x4x4xf32>) outs(%result : memref<1x8x4x4xf32>) attrs = {__internal_linalg_transform__ = "act"} {
  ^bb0(%in: f32, %out: f32):
    %n = arith.negf %in : f32
    linalg.yield %n : f32
  }
  return
}
