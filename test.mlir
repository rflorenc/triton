#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @foo(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>) -> tensor<128x128xf32, #blocked1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0:4 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %c0_i32, %arg7 = %cst, %arg8 = %arg3, %arg9 = %arg4) -> (i32, tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>)  : i32 {
      %1 = arith.cmpi eq, %arg6, %c0_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %2 = scf.if %1 -> (tensor<128x128xf32, #blocked>) {
        %10 = tt.load %arg8 : tensor<128x128x!tt.ptr<f32>, #blocked>
        scf.yield %10 : tensor<128x128xf32, #blocked>
      } else {
        scf.yield %cst_0 : tensor<128x128xf32, #blocked>
      } {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %3 = tt.load %arg9 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
      %4 = ttg.convert_layout %2 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      %5 = ttg.convert_layout %3 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
      %6 = tt.dot %4, %5, %arg7 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x128xf32, #blocked1>
      %7 = tt.addptr %arg8, %cst_1 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %8 = tt.addptr %arg9, %cst_1 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %9 = arith.addi %arg6, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : i32
      scf.yield %9, %6, %7, %8 : i32, tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    tt.return %0#1 : tensor<128x128xf32, #blocked1>
  }
}

