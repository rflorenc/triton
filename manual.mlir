
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0, 1]}>
#smem = #ttg.shared_memory

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>

!tensor_ptr = tensor<128x128x!tt.ptr<f32>, #blocked>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @foo(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: tensor<128x128x!tt.ptr<f32>, #blocked>, %arg4: tensor<128x128x!tt.ptr<f32>, #blocked>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x128xf32, #blocked>
    %cst_1 = arith.constant dense<1> : tensor<128x128xi32, #blocked>
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32

    %buf = ttg.local_alloc  : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>
    %buf1 = ttg.local_alloc  : () -> !ttg.memdesc<128x128xf32, #shared, #smem, mutable>

    %0:4 = scf.for %arg5 = %arg0 to %arg1 step %arg2 iter_args(%arg6 = %c0_i32,
        %arg9 = %arg4,
        %arg10 = %arg4, %arg11 = %arg4) -> (i32, !tensor_ptr, !tensor_ptr, !tensor_ptr)  : i32 {
      %1 = arith.cmpi eq, %arg6, %c0_i32 : i32
      %2 = arith.addi %arg6, %c1_i32 : i32

      //"something"(%1) {tt_latency = 2 : i32} : (i1) -> ()
      %tok0 = scf.if %1 -> !ttg.async.token {
        %tok = ttg.async_copy_global_to_local %arg10, %buf : tensor<128x128x!tt.ptr<f32>, #blocked> -> <128x128xf32, #shared, #smem, mutable>
        %tok2 = ttg.async_commit_group %tok
        scf.yield %tok2 : !ttg.async.token
      } else {
        %undef = ub.poison : !ttg.async.token
        scf.yield %undef : !ttg.async.token
      } {tt_latency = 2 : i32}

      //%3 = "something"(%1) : (i1) -> tensor<128x128xf32, #blocked>
      %3 = scf.if %1 -> (tensor<128x128xf32, #blocked>) {
        %tt = ttg.async_wait %tok0 {num = 0 : i32}
        %11 = ttg.local_load %buf token %tt : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32, #blocked>
        scf.yield %11 : tensor<128x128xf32, #blocked>
      } else {
        scf.yield %cst_0 : tensor<128x128xf32, #blocked>
      }

      %9 = tt.addptr %arg9, %cst_1: tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %15 = tt.addptr %arg10, %cst_1: tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>
      %16 = tt.addptr %arg11, %cst_1: tensor<128x128x!tt.ptr<f32>, #blocked>, tensor<128x128xi32, #blocked>


      %4 = tt.load %arg9 {tt_latency = 2 : i32} : tensor<128x128x!tt.ptr<f32>, #blocked>

      %5 = arith.addf %4, %3 : tensor<128x128xf32, #blocked>


      %10 = arith.cmpi eq, %arg6, %arg1 : i32
      %tok1 = scf.if %10 -> !ttg.async.token {
        %tok = ttg.async_copy_global_to_local %arg11, %buf1 : tensor<128x128x!tt.ptr<f32>, #blocked> -> <128x128xf32, #shared, #smem, mutable>
        %tok2 = ttg.async_commit_group %tok
        scf.yield %tok2 : !ttg.async.token
      } else {
        %undef = ub.poison : !ttg.async.token
        scf.yield %undef : !ttg.async.token
      } {tt_latency = 2 : i32}

      scf.if %10 {
        %kk = ttg.async_wait %tok1 {num = 0 : i32}
        %11 = ttg.local_load %buf1 token %kk : !ttg.memdesc<128x128xf32, #shared, #smem, mutable> -> tensor<128x128xf32, #blocked>
        %x = arith.addf %11, %5 : tensor<128x128xf32, #blocked>
        tt.store %arg4, %x : tensor<128x128x!tt.ptr<f32>, #blocked>
      }
      scf.yield %2, %9, %15, %16 : i32, tensor<128x128x!tt.ptr<f32>, #blocked>, !tensor_ptr, !tensor_ptr
    } {tt.num_stages = 3 : i32}
    tt.return
  }
}

