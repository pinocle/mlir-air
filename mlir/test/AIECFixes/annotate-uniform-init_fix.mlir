// RUN: aiec-opt %s -aiec-annotate-uniform-init -split-input-file | FileCheck %s

// A whole-buffer constant fill copied whole into the output reaches the
// matmul, which reads its output: the initializer is recorded.
// CHECK-LABEL: func.func @reaches
// CHECK: linalg.matmul {aie1.uniform_output_init = [1.250000e+00 : f32]}
func.func @reaches(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  %v = arith.constant 1.25 : f32
  %init = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%v : f32) outs(%init : memref<4x4xf32>)
  memref.copy %init, %c : memref<4x4xf32> to memref<4x4xf32>
  linalg.matmul ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>) outs(%c : memref<4x4xf32>)
  return
}

// -----

// An unknown write between them kills the fact, and a stale annotation from
// the input is removed.
// CHECK-LABEL: func.func @killed
// CHECK-NOT: aie1.uniform_output_init
func.func @killed(%a: memref<4x4xf32>, %b: memref<4x4xf32>, %c: memref<4x4xf32>) {
  %v = arith.constant 1.25 : f32
  %zero = arith.constant 0 : index
  %init = memref.alloc() : memref<4x4xf32>
  linalg.fill ins(%v : f32) outs(%init : memref<4x4xf32>)
  memref.copy %init, %c : memref<4x4xf32> to memref<4x4xf32>
  memref.store %v, %c[%zero, %zero] : memref<4x4xf32>
  linalg.matmul {aie1.uniform_output_init = [9.900000e+01 : f32]}
      ins(%a, %b : memref<4x4xf32>, memref<4x4xf32>) outs(%c : memref<4x4xf32>)
  return
}
