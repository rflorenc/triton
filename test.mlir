#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @foo(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>) -> tensor<128x128xf32, #blocked1> {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked1>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ttg.local_alloc  : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %1:4 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %c0_i32, %arg7 = %cst, %arg8 = %arg3, %arg9 = %arg4) -> (i32, tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>)  : i32 {
      %2 = arith.cmpi eq, %arg6, %c0_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %3 = arith.addi %arg6, %c1_i32 {loop.cluster = 5 : i32, loop.stage = 2 : i32} : i32
      scf.if %2 {
        %12 = ttg.async_copy_global_to_local %arg8, %0 : tensor<128x128x!tt.ptr<f32>, #blocked> -> <128x128xf32, #shared, #smem, mutable>
        %13 = ttg.async_commit_group %12
      } else {
      } {loop.cluster = 6 : i32, loop.stage = 2 : i32}
      %4 = scf.if %2 -> (tensor<128x128xf32, #blocked>) {
        %12 = ttg.async_wait  {num = 0 : i32}
        %13 = ttg.local_load %0 : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32, #blocked>
        scf.yield %13 : tensor<128x128xf32, #blocked>
      } else {
        scf.yield %cst_0 : tensor<128x128xf32, #blocked>
      } {loop.cluster = 0 : i32, loop.stage = 2 : i32}
      %5 = tt.load %arg9 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
      %6 = ttg.convert_layout %4 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>>
      %7 = ttg.convert_layout %5 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked> -> tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>>
      %8 = tt.dot %6, %7, %arg7, inputPrecision = tf32 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked1}>> * tensor<128x128xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked1}>> -> tensor<128x128xf32, #blocked1>
      %9 = tt.addptr %arg8, %cst_1 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %10 = tt.addptr %arg9, %cst_1 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %11 = arith.cmpi eq, %3, %arg1 {loop.cluster = 5 : i32, loop.stage = 2 : i32} : i32
      scf.if %11 {
        %12 = tt.load %arg9 : tensor<128x128x!tt.ptr<f32>, #blocked>
        %13 = ttg.convert_layout %8 : tensor<128x128xf32, #blocked1> -> tensor<128x128xf32, #blocked>
        %14 = arith.addf %12, %13 : tensor<128x128xf32, #blocked>
        tt.store %arg9, %14 : tensor<128x128x!tt.ptr<f32>, #blocked>
      } {loop.cluster = 5 : i32, loop.stage = 2 : i32}
      scf.yield %3, %8, %9, %10 : i32, tensor<128x128xf32, #blocked1>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    tt.return %1#1 : tensor<128x128xf32, #blocked1>
  }
}

