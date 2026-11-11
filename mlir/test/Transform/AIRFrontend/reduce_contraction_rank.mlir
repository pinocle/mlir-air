// Copied from hexagon-mlir 9ec15e8, qcom_hexagon_backend/test/Conversion/LinalgToLLVM/reduce_contraction_rank.mlir
// (Qualcomm, BSD-3-Clause); linalg-hexagon-opt is air-opt.
// RUN: air-opt %s -pass-pipeline='builtin.module(func.func(reduce-contraction-rank))' | FileCheck %s

func.func @test_batch_matmul_to_matmul(%lhs: tensor<1x4x8xf32>, %rhs: tensor<1x8x16xf32>, %init: tensor<1x4x16xf32>) -> tensor<1x4x16xf32> {
  %result = linalg.batch_matmul
            ins(%lhs, %rhs : tensor<1x4x8xf32>, tensor<1x8x16xf32>) outs(%init : tensor<1x4x16xf32>) -> tensor<1x4x16xf32>
  return %result : tensor<1x4x16xf32>
}

// CHECK-LABEL: func.func @test_batch_matmul_to_matmul
// CHECK: %[[MATMUL:.+]] = linalg.matmul
// CHECK-SAME: ins(%[[LHS:.+]], %[[RHS:.+]] : tensor<4x8xf32>, tensor<8x16xf32>)
// CHECK-SAME: outs(%[[INIT:.+]] : tensor<4x16xf32>)
