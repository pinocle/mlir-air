// RUN: air-opt %s -split-input-file -verify-diagnostics -allow-unregistered-dialect -air-bind-launch="sizes=256,256 scalars=200,0.5 grid=4,1,1" | FileCheck %s

// A triton-shared kernel after air-wrap-func-with-parallel: pointers and
// scalars interleaved, num_programs still read, program ids unused.
// CHECK-LABEL: func.func @kernel(%arg0: memref<256xf32>, %arg1: memref<256xf32>) {
// CHECK-DAG: %[[N:.*]] = arith.constant 200 : i32
// CHECK-DAG: %[[A:.*]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG: %[[G:.*]] = arith.constant 4 : i32
// CHECK-DAG: %[[X:.*]] = memref.cast %arg0 : memref<256xf32> to memref<*xf32>
// CHECK-DAG: %[[Y:.*]] = memref.cast %arg1 : memref<256xf32> to memref<*xf32>
// CHECK: "test.use"(%[[X]], %[[N]], %[[Y]], %[[A]], %[[G]])
func.func @kernel(%x: memref<*xf32>, %n: i32, %y: memref<*xf32>, %alpha: f32,
                  %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32) {
  "test.use"(%x, %n, %y, %alpha, %gx) : (memref<*xf32>, i32, memref<*xf32>, f32, i32) -> ()
  return
}

// -----

// A program id still in use: the body was not wrapped over the grid.
// expected-error @+1 {{program ids are still used}}
func.func @unwrapped(%x: memref<*xf32>, %n: i32, %y: memref<*xf32>, %alpha: f32,
                     %gx: i32, %gy: i32, %gz: i32, %px: i32, %py: i32, %pz: i32) {
  "test.use"(%px) : (i32) -> ()
  return
}
