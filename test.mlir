#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 128, 8]}>
#shared = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 4, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @foo(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>) -> tensor<128x128xf64, #blocked1> {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x128xf64, #mma>
    %0:4 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %c0_i32, %arg7 = %cst_1, %arg8 = %arg3, %arg9 = %arg4) -> (i32, tensor<128x128xf64, #mma>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>)  : i32 {
      %2 = arith.cmpi eq, %arg6, %c0_i32 {loop.cluster = 0 : i32, loop.stage = 2 : i32} : i32
      %3 = scf.if %2 -> (tensor<128x128xf32, #blocked>) {
        scf.yield %cst_0 : tensor<128x128xf32, #blocked>
      } else {
        %11 = tt.load %arg8 : tensor<128x128x!tt.ptr<f32>, #blocked>
        scf.yield %11 : tensor<128x128xf32, #blocked>
      } {loop.cluster = 0 : i32, loop.stage = 2 : i32, ttg.conditional_load = true}
      %4 = ttg.local_alloc %3 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #shared, #smem>
      %5 = tt.load %arg9 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
      %6 = ttg.local_alloc %5 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : (tensor<128x128xf32, #blocked>) -> !ttg.memdesc<128x128xf32, #shared1, #smem>
      %7 = ttng.warp_group_dot %4, %6, %arg7 {inputPrecision = 0 : i32, loop.cluster = 1 : i32, loop.stage = 2 : i32} : !ttg.memdesc<128x128xf32, #shared, #smem> * !ttg.memdesc<128x128xf32, #shared1, #smem> -> tensor<128x128xf64, #mma>
      %8 = tt.addptr %arg8, %cst {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %9 = tt.addptr %arg9, %cst {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %10 = arith.addi %arg6, %c1_i32 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : i32
      scf.yield %10, %7, %8, %9 : i32, tensor<128x128xf64, #mma>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>
    }
    %1 = ttg.convert_layout %0#1 : tensor<128x128xf64, #mma> -> tensor<128x128xf64, #blocked1>
    tt.return %1 : tensor<128x128xf64, #blocked1>
  }
}

