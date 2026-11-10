// RUN: aiec-opt %s -aiec-prune-token-loops -split-input-file | FileCheck %s

// A segment loop nest that only carries tokens becomes a wait on its initial
// tokens, which the herd still depends on.
// CHECK-LABEL: func.func @token_nest
// CHECK: air.segment
// CHECK-NOT: scf.parallel
// CHECK-NOT: scf.for
// CHECK: %[[DONE:.*]] = air.wait_all async [%{{.*}}]
// CHECK: air.herd @herd async [%[[DONE]]]
func.func @token_nest() {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%x, %y) in (%sx = %c1, %sy = %c1) {
    %s = air.segment @seg async {
      %c0 = arith.constant 0 : index
      %c1s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %w0 = air.wait_all async
      %p = scf.parallel (%i) = (%c0) to (%c2) step (%c1s) init (%w0) -> !air.async.token {
        %w1 = air.wait_all async
        %f = scf.for %j = %c0 to %c2 step %c1s iter_args(%a0 = %w1) -> (!air.async.token) {
          %g = scf.for %k = %c0 to %c2 step %c1s iter_args(%a1 = %a0) -> (!air.async.token) {
            %w3 = air.wait_all async
            scf.yield %w3 : !air.async.token
          }
          scf.yield %g : !air.async.token
        }
        %w4 = air.wait_all async
        scf.reduce(%w4 : !air.async.token) {
        ^bb0(%a: !air.async.token, %b: !air.async.token):
          %w2 = air.wait_all async [%a, %b]
          scf.reduce.return %w2 : !air.async.token
        }
      }
      %h = air.herd @herd async [%p] tile (%tx, %ty) in (%hx = %c1s, %hy = %c1s) {
      }
    }
  }
  return
}

// -----

// A loop that moves data is kept, as is any loop inside a herd.
// CHECK-LABEL: func.func @working_loops
// CHECK: air.segment
// CHECK: scf.for
// CHECK: air.channel.put
// CHECK: air.herd
// CHECK: scf.for
// CHECK: air.wait_all
air.channel @channel [1, 1]
func.func @working_loops(%arg0: memref<16xf32>) {
  %c1 = arith.constant 1 : index
  %t = air.launch async (%x, %y) in (%sx = %c1, %sy = %c1) args(%a = %arg0) : memref<16xf32> {
    %s = air.segment @seg async args(%b = %a) : memref<16xf32> {
      %c0 = arith.constant 0 : index
      %c1s = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %w0 = air.wait_all async
      %f = scf.for %j = %c0 to %c2 step %c1s iter_args(%a0 = %w0) -> (!air.async.token) {
        %put = air.channel.put async [%a0] @channel[] (%b[] [] []) : (memref<16xf32>)
        scf.yield %put : !air.async.token
      }
      %h = air.herd @herd async [%f] tile (%tx, %ty) in (%hx = %c1s, %hy = %c1s) {
        %z = arith.constant 0 : index
        %one = arith.constant 1 : index
        %two = arith.constant 2 : index
        %w = air.wait_all async
        %l = scf.for %k = %z to %two step %one iter_args(%a1 = %w) -> (!air.async.token) {
          %w3 = air.wait_all async
          scf.yield %w3 : !air.async.token
        }
      }
    }
  }
  return
}
