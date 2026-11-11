// RUN: air-opt %s -air-eliminate-staging-buffers -split-input-file | FileCheck %s

// The Triton-XDNA vec-add shape after its transform script: the block is
// staged in L2 on both sides of the herd. The herd takes the L3 views, and
// the output view moves before it.
// CHECK-LABEL: func.func @vecadd
// CHECK-NOT: memref.alloc() : memref<1024xf32, 1
// CHECK-NOT: memref.copy
// CHECK: %[[A:.*]] = memref.reinterpret_cast %arg0
// CHECK: %[[B:.*]] = memref.reinterpret_cast %arg1
// CHECK: %[[C:.*]] = memref.reinterpret_cast %arg2
// CHECK: air.herd @herd_0 {{.*}} args(%{{.*}}=%[[A]], %{{.*}}=%[[B]], %{{.*}}=%[[C]]) : memref<1024xf32, strided<[1], offset: ?>>, memref<1024xf32, strided<[1], offset: ?>>, memref<1024xf32, strided<[1], offset: ?>>
// CHECK: air.dma_memcpy_nd (%{{.*}}[] [] [], %{{.*}}[%{{.*}}] [%{{.*}}] [%{{.*}}]) : (memref<256xf32, 2>, memref<1024xf32, strided<[1], offset: ?>>)
// CHECK: return
#map = affine_map<(d0) -> (d0 * 256)>
func.func @vecadd(%arg0: memref<1024xf32>, %arg1: memref<1024xf32>, %arg2: memref<1024xf32>, %off: index) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %a = memref.reinterpret_cast %arg0 to offset: [%off], sizes: [1024], strides: [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  %sa = memref.alloc() : memref<1024xf32, 1 : i32>
  memref.copy %a, %sa : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32, 1 : i32>
  %b = memref.reinterpret_cast %arg1 to offset: [%off], sizes: [1024], strides: [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  %sb = memref.alloc() : memref<1024xf32, 1 : i32>
  memref.copy %b, %sb : memref<1024xf32, strided<[1], offset: ?>> to memref<1024xf32, 1 : i32>
  %sc = memref.alloc() : memref<1024xf32, 1>
  air.herd @herd_0 tile (%x, %y) in (%sx = %c4, %sy = %c1) args(%ha = %sa, %hb = %sb, %hc = %sc) : memref<1024xf32, 1 : i32>, memref<1024xf32, 1 : i32>, memref<1024xf32, 1> {
    %c1h = arith.constant 1 : index
    %c256 = arith.constant 256 : index
    %o = affine.apply #map(%x)
    %la = memref.alloc() : memref<256xf32, 2>
    air.dma_memcpy_nd (%la[] [] [], %ha[%o] [%c256] [%c1h]) : (memref<256xf32, 2>, memref<1024xf32, 1 : i32>)
    %lb = memref.alloc() : memref<256xf32, 2>
    air.dma_memcpy_nd (%lb[] [] [], %hb[%o] [%c256] [%c1h]) : (memref<256xf32, 2>, memref<1024xf32, 1 : i32>)
    %lc = memref.alloc() : memref<256xf32, 2>
    air.dma_memcpy_nd (%hc[%o] [%c256] [%c1h], %lc[] [] []) : (memref<1024xf32, 1>, memref<256xf32, 2>)
    memref.dealloc %la : memref<256xf32, 2>
    memref.dealloc %lb : memref<256xf32, 2>
    memref.dealloc %lc : memref<256xf32, 2>
  }
  %c = memref.reinterpret_cast %arg2 to offset: [%off], sizes: [1024], strides: [1] : memref<1024xf32> to memref<1024xf32, strided<[1], offset: ?>>
  memref.copy %sc, %c : memref<1024xf32, 1> to memref<1024xf32, strided<[1], offset: ?>>
  memref.dealloc %sc : memref<1024xf32, 1>
  return
}

// -----

// The staging buffer is a snapshot: a write to memory outside allocations
// between the copy and the herd keeps it.
// CHECK-LABEL: func.func @written_between
// CHECK: %[[S:.*]] = memref.alloc() : memref<16xf32, 1>
// CHECK: memref.copy %arg0, %[[S]]
// CHECK: memref.copy %arg1, %arg0
// CHECK: air.herd {{.*}} args(%{{.*}}=%[[S]])
func.func @written_between(%arg0: memref<16xf32>, %arg1: memref<16xf32>) {
  %c1 = arith.constant 1 : index
  %s = memref.alloc() : memref<16xf32, 1>
  memref.copy %arg0, %s : memref<16xf32> to memref<16xf32, 1>
  memref.copy %arg1, %arg0 : memref<16xf32> to memref<16xf32>
  air.herd tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%h = %s) : memref<16xf32, 1> {
    %l = memref.alloc() : memref<16xf32, 2>
    air.dma_memcpy_nd (%l[] [] [], %h[] [] []) : (memref<16xf32, 2>, memref<16xf32, 1>)
  }
  return
}

// -----

// A herd that also writes the staged input keeps the buffer: the copy is not
// its only writer.
// CHECK-LABEL: func.func @written_in_herd
// CHECK: memref.alloc() : memref<16xf32, 1>
// CHECK: memref.copy
func.func @written_in_herd(%arg0: memref<16xf32>) {
  %c1 = arith.constant 1 : index
  %s = memref.alloc() : memref<16xf32, 1>
  memref.copy %arg0, %s : memref<16xf32> to memref<16xf32, 1>
  air.herd tile (%x, %y) in (%sx = %c1, %sy = %c1) args(%h = %s) : memref<16xf32, 1> {
    %l = memref.alloc() : memref<16xf32, 2>
    air.dma_memcpy_nd (%l[] [] [], %h[] [] []) : (memref<16xf32, 2>, memref<16xf32, 1>)
    air.dma_memcpy_nd (%h[] [] [], %l[] [] []) : (memref<16xf32, 1>, memref<16xf32, 2>)
  }
  return
}
