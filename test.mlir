#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @foo(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_0 = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = ttg.local_alloc  : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %1 = ttg.local_alloc  : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %2:4 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %c0_i32, %arg7 = %arg4, %arg8 = %arg4, %arg9 = %arg4) -> (i32, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>)  : i32 {
      %3 = arith.cmpi eq, %arg6, %c0_i32 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
      %4 = arith.addi %arg6, %c1_i32 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : i32
      %5 = scf.if %3 -> (!ttg.async.token) {
        %14 = ttg.async_copy_global_to_local %arg8, %0 : tensor<128x128x!tt.ptr<f32>, #blocked> -> <128x128xf32, #shared, #smem, mutable>
        %15 = ttg.async_commit_group %14
        scf.yield %15 : !ttg.async.token
      } else {
        %14 = ub.poison : !ttg.async.token
        scf.yield %14 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt_latency = 2 : i32}
      %6 = scf.if %3 -> (tensor<128x128xf32, #blocked>) {
        %14 = ttg.async_wait %5 {num = 0 : i32}
        %15 = ttg.local_load %0 token %14 : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32, #blocked>
        scf.yield %15 : tensor<128x128xf32, #blocked>
      } else {
        scf.yield %cst : tensor<128x128xf32, #blocked>
      } {loop.cluster = 1 : i32, loop.stage = 2 : i32}
      %7 = tt.addptr %arg7, %cst_0 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %8 = tt.addptr %arg8, %cst_0 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %9 = tt.addptr %arg9, %cst_0 {loop.cluster = 3 : i32, loop.stage = 1 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %10 = tt.load %arg7 {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt_latency = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>
      %11 = arith.addf %10, %6 {loop.cluster = 1 : i32, loop.stage = 2 : i32} : tensor<128x128xf32, #blocked>
      %12 = arith.cmpi eq, %arg6, %arg1 {loop.cluster = 4 : i32, loop.stage = 0 : i32} : i32
      %13 = scf.if %12 -> (!ttg.async.token) {
        %14 = ttg.async_copy_global_to_local %arg9, %1 : tensor<128x128x!tt.ptr<f32>, #blocked> -> <128x128xf32, #shared, #smem, mutable>
        %15 = ttg.async_commit_group %14
        scf.yield %15 : !ttg.async.token
      } else {
        %14 = ub.poison : !ttg.async.token
        scf.yield %14 : !ttg.async.token
      } {loop.cluster = 4 : i32, loop.stage = 0 : i32, tt_latency = 2 : i32}
      scf.if %12 {
        %14 = ttg.async_wait %13 {num = 0 : i32}
        %15 = ttg.local_load %1 token %14 : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32, #blocked>
        %16 = arith.addf %15, %11 : tensor<128x128xf32, #blocked>
        tt.store %arg4, %16 : tensor<128x128x!tt.ptr<f32>, #blocked>
      } {loop.cluster = 5 : i32, loop.stage = 2 : i32}
      scf.yield %4, %7, %8, %9 : i32, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128x!tt.ptr<f32>, #blocked>
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

