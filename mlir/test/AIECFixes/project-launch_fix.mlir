// RUN: aiec-opt %s -aiec-project-launch=index=1 -canonicalize -cse -symbol-dce | FileCheck %s
// RUN: not aiec-opt %s -aiec-project-launch=index=2 2>&1 | FileCheck %s --check-prefix=RANGE

// Launch 1 stays with its channel; launch 0, its channel and the global only
// it read go. The dependency on launch 0 becomes a complete token.
// CHECK-NOT: @first
// CHECK: air.channel @second
// CHECK-NOT: memref.global
// CHECK-LABEL: func.func @host
// CHECK: memref.alloc() : memref<4xf32>
// CHECK: air.launch
// CHECK-NOT: air.launch
// CHECK: air.channel.put @second
// RANGE: aiec-project-launch index 2 is not one of 2 launches
air.channel @first [1, 1]
air.channel @second [1, 1]
memref.global "private" constant @weights : memref<4xf32> = dense<1.0>
func.func @host(%out: memref<4xf32>) {
  %c1 = arith.constant 1 : index
  %buffer = memref.alloc() : memref<4xf32>
  %w = memref.get_global @weights : memref<4xf32>
  %t0 = air.launch async (%x, %y) in (%sx = %c1, %sy = %c1) args(%a = %w) : memref<4xf32> {
    air.channel.put @first[] (%a[] [] []) : (memref<4xf32>)
  }
  %t1 = air.launch async [%t0] (%x, %y) in (%sx = %c1, %sy = %c1) args(%b = %buffer) : memref<4xf32> {
    air.channel.put @second[] (%b[] [] []) : (memref<4xf32>)
  }
  return
}
