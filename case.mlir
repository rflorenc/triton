#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {

tt.func @foo(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>) -> tensor<128x128xf32, #blocked1> {
  %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
  %cst1 = arith.constant dense<0.0> : tensor<128x128xf32, #blocked>
  %cst_0 = arith.constant dense<1> : tensor<128x128xi32, #blocked>
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32

  %0:4 = scf.for %arg6 = %arg0 to %arg1 step %arg2 iter_args(%k = %c0, %arg7 = %cst, %arg8 = %arg3, %arg9 = %arg4) -> (i32, tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>)  : i32 {
    %k_cond = arith.cmpi eq, %k, %c0 : i32
    %3 = scf.if %k_cond -> tensor<128x128xf32, #blocked> {
      %res = tt.load %arg8 : tensor<128x128x!tt.ptr<f32>, #blocked>
      scf.yield %res : tensor<128x128xf32, #blocked>
    } else {
      scf.yield %cst1 : tensor<128x128xf32, #blocked>
    }

    %4 = tt.load %arg9 : tensor<128x128x!tt.ptr<f32>, #blocked>
    %5 = ttg.convert_layout %3 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
    %6 = ttg.convert_layout %4 : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
    %7 = tt.dot %5, %6, %arg7 : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x128xf32, #blocked1>
    %8 = tt.addptr %arg8, %cst_0 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
    %9 = tt.addptr %arg9, %cst_0 : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>

    %next_k = arith.addi %k, %c1 : i32
    scf.yield %next_k, %7, %8, %9 : i32, tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>
  }
  tt.return %0#1 : tensor<128x128xf32, #blocked1>
}

}

