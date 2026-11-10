// RUN: aiec-opt %s -aiec-synchronize-host | FileCheck %s
// RUN: aiec-opt %s -aiec-synchronize-host=strip-hierarchy-tokens=true | FileCheck %s --check-prefix=STRIP

// Host air.execute bodies run inline, host token reductions disappear, and a
// launch region uses its own complete token. The device segment keeps its
// async graph.
// CHECK-LABEL: func.func @host
// CHECK-NOT: air.execute
// CHECK: memref.alloc() : memref<4xf32>
// CHECK: air.launch async
// CHECK:   %[[DONE:.*]] = air.wait_all async
// CHECK:   scf.parallel
// CHECK-NOT: scf.reduce(
// CHECK:     air.channel.put async [%[[DONE]]]
// CHECK:   air.segment @seg async
// CHECK:     air.execute
// STRIP-LABEL: func.func @host
// STRIP: air.launch (
// STRIP-NOT: air.launch async
// STRIP:   air.channel.put @channel
// STRIP:   air.segment @seg {
// STRIP:     air.execute
air.channel @channel [1, 1]
func.func @host() {
  %c1 = arith.constant 1 : index
  %t0, %buffer = air.execute -> (memref<4xf32>) {
    %a = memref.alloc() : memref<4xf32>
    air.execute_terminator %a : memref<4xf32>
  }
  %launch = air.launch async [%t0] (%x, %y) in (%sx = %c1, %sy = %c1) args(%b = %buffer) : memref<4xf32> {
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %one = arith.constant 1 : index
    %w = air.wait_all async
    %p = scf.parallel (%i) = (%c0) to (%c2) step (%one) init (%w) -> !air.async.token {
      %put = air.channel.put async [%w] @channel[] (%b[] [] []) : (memref<4xf32>)
      scf.reduce(%put : !air.async.token) {
      ^bb0(%l: !air.async.token, %r: !air.async.token):
        %j = air.wait_all async [%l, %r]
        scf.reduce.return %j : !air.async.token
      }
    }
    %s = air.segment @seg async [%p] {
      %c1s = arith.constant 1 : index
      %h = air.herd @herd async tile (%tx, %ty) in (%hx = %c1s, %hy = %c1s) {
        %e, %l1 = air.execute -> (memref<4xf32, 2 : i32>) {
          %a = memref.alloc() : memref<4xf32, 2 : i32>
          air.execute_terminator %a : memref<4xf32, 2 : i32>
        }
        %g = air.channel.get async [%e] @channel[%tx, %ty] (%l1[] [] []) : (memref<4xf32, 2 : i32>)
      }
    }
  }
  return
}
