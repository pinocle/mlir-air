// RUN: air-opt %s -air-read-only-arguments -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" | FileCheck %s

// The generic's outs is the input %a: bufferization copies %a instead of
// writing it.
// CHECK-LABEL: func.func @accumulate_into_input
// CHECK-SAME: (%[[A:.*]]: memref<16xf32>)
// CHECK: %[[OUT:.*]] = memref.alloc() {{.*}}: memref<16xf32>
// CHECK: memref.copy %[[A]], %[[OUT]]
// CHECK: linalg.generic {{.*}} ins(%[[A]] : memref<16xf32>) outs(%[[OUT]] : memref<16xf32>)
// CHECK: return %[[OUT]]
func.func @accumulate_into_input(%a: tensor<16xf32>) -> tensor<16xf32> {
  %r = linalg.generic {indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>], iterator_types = ["parallel"]}
      ins(%a : tensor<16xf32>) outs(%a : tensor<16xf32>) {
  ^bb0(%x: f32, %old: f32):
    %y = arith.mulf %x, %old : f32
    linalg.yield %y : f32
  } -> tensor<16xf32>
  return %r : tensor<16xf32>
}

// Declarations keep their arguments as they are.
// CHECK-LABEL: func.func private @external
// CHECK-SAME: (memref<16xf32>)
func.func private @external(tensor<16xf32>)
