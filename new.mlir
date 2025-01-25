#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked2 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [1, 32], warpsPerCTA = [8, 1], order = [1, 0]}>
#loc = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0)
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 256, 16]}>
#shared = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0], hasLeadingOffset = true}>
#shared1 = #ttg.shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [0, 1], hasLeadingOffset = true}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel_persistent(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg4: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg5: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg6: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg7: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0), %arg8: i32 {tt.divisibility = 16 : i32} loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":231:0)) attributes {noinline = false} {
    %c2_i64 = arith.constant 2 : i64 loc(#loc1)
    %c3_i32 = arith.constant 3 : i32 loc(#loc1)
    %c-1_i32 = arith.constant -1 : i32 loc(#loc1)
    %0 = ub.poison : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc1)
    %1 = ub.poison : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc1)
    %2 = ub.poison : tensor<128x256xf32, #mma> loc(#loc1)
    %3 = ub.poison : i32 loc(#loc1)
    %c1_i64 = arith.constant 1 : i64 loc(#loc1)
    %c0_i64 = arith.constant 0 : i64 loc(#loc1)
    %cst = arith.constant dense<0> : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc1)
    %cst_0 = arith.constant dense<0> : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc1)
    %c256_i32 = arith.constant 256 : i32 loc(#loc1)
    %c128_i32 = arith.constant 128 : i32 loc(#loc1)
    %c8_i32 = arith.constant 8 : i32 loc(#loc1)
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked1> loc(#loc1)
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<64x256xf16, #blocked> loc(#loc1)
    %c64_i32 = arith.constant 64 : i32 loc(#loc1)
    %c132_i32 = arith.constant 132 : i32 loc(#loc1)
    %c0_i32 = arith.constant 0 : i32 loc(#loc1)
    %c1_i32 = arith.constant 1 : i32 loc(#loc1)
    %c127_i32 = arith.constant 127 : i32 loc(#loc1)
    %c255_i32 = arith.constant 255 : i32 loc(#loc1)
    %c63_i32 = arith.constant 63 : i32 loc(#loc1)
    %4 = tt.get_program_id x : i32 loc(#loc2)
    %5 = arith.addi %arg3, %c127_i32 : i32 loc(#loc59)
    %6 = arith.divsi %5, %c128_i32 : i32 loc(#loc60)
    %7 = arith.addi %arg4, %c255_i32 : i32 loc(#loc61)
    %8 = arith.divsi %7, %c256_i32 : i32 loc(#loc62)
    %9 = arith.addi %arg5, %c63_i32 : i32 loc(#loc63)
    %10 = arith.divsi %9, %c64_i32 : i32 loc(#loc64)
    %11 = arith.muli %6, %8 : i32 loc(#loc8)
    %12 = arith.muli %8, %c8_i32 : i32 loc(#loc9)
    %13 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc10)
    %14 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc10)
    %15 = arith.subi %11, %4 : i32 loc(#loc11)
    %16 = arith.ceildivsi %15, %c132_i32 : i32 loc(#loc11)
    %17 = arith.extsi %10 : i32 to i64 loc(#loc11)
    %18 = arith.maxsi %17, %c1_i64 : i64 loc(#loc11)
    %19 = arith.extsi %16 : i32 to i64 loc(#loc11)
    %20 = arith.muli %19, %18 : i64 loc(#loc11)
    %21 = arith.subi %4, %c132_i32 : i32 loc(#loc11)
    %22 = ttg.local_alloc  : () -> !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> loc(#loc12)
    %23 = ttg.local_alloc  : () -> !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> loc(#loc13)
    %24 = arith.cmpi sgt, %20, %c0_i64 : i64 loc(#loc11)
    %25 = arith.remsi %c0_i64, %18 : i64 loc(#loc11)
    %26 = arith.cmpi eq, %25, %c0_i64 : i64 loc(#loc11)
    %27 = arith.select %26, %4, %21 : i32 loc(#loc11)
    %28 = arith.cmpi ne, %25, %c0_i64 : i64 loc(#loc11)
    %29:4 = scf.if %26 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>) {
      %110 = arith.divsi %4, %12 : i32 loc(#loc14)
      %111 = arith.muli %110, %c8_i32 : i32 loc(#loc15)
      %112 = arith.subi %6, %111 : i32 loc(#loc16)
      %113 = arith.minsi %112, %c8_i32 : i32 loc(#loc17)
      %114 = arith.remsi %4, %113 : i32 loc(#loc18)
      %115 = arith.addi %111, %114 : i32 loc(#loc19)
      %116 = arith.remsi %4, %12 : i32 loc(#loc20)
      %117 = arith.divsi %116, %113 : i32 loc(#loc21)
      %118 = arith.muli %115, %c128_i32 : i32 loc(#loc22)
      %119 = arith.muli %117, %c256_i32 : i32 loc(#loc23)
      %120 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc24)
      %121 = tt.splat %118 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc25)
      %122 = arith.addi %121, %120 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc25)
      %123 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc26)
      %124 = tt.splat %119 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc27)
      %125 = arith.addi %124, %123 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc27)
      %126 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc28)
      %127 = arith.cmpi slt, %122, %126 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc28)
      %128 = arith.select %127, %122, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc29)
      %129 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc30)
      %130 = arith.cmpi slt, %125, %129 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc30)
      %131 = arith.select %130, %125, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc31)
      scf.yield %118, %119, %128, %131 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc11)
    } else {
      scf.yield %3, %3, %1, %0 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc11)
    } loc(#loc11)
    %30 = tt.expand_dims %29#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc32)
    %31 = tt.splat %arg6 : i32 -> tensor<128x1xi32, #blocked1> loc(#loc33)
    %32 = arith.muli %30, %31 : tensor<128x1xi32, #blocked1> loc(#loc33)
    %33 = tt.expand_dims %13 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc34)
    %34 = tt.broadcast %32 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc35)
    %35 = tt.broadcast %33 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc35)
    %36 = arith.addi %34, %35 : tensor<128x64xi32, #blocked1> loc(#loc35)
    %37 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x64x!tt.ptr<f16>, #blocked1> loc(#loc36)
    %38 = tt.addptr %37, %36 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc36)
    %39 = tt.expand_dims %14 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc37)
    %40 = tt.expand_dims %29#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc38)
    %41 = tt.splat %arg7 : i32 -> tensor<1x256xi32, #blocked> loc(#loc39)
    %42 = arith.muli %40, %41 : tensor<1x256xi32, #blocked> loc(#loc39)
    %43 = tt.broadcast %39 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc40)
    %44 = tt.broadcast %42 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc40)
    %45 = arith.addi %43, %44 : tensor<64x256xi32, #blocked> loc(#loc40)
    %46 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<64x256x!tt.ptr<f16>, #blocked> loc(#loc41)
    %47 = tt.addptr %46, %45 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc41)
    %48 = tt.splat %arg5 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc42)
    %49 = arith.cmpi slt, %33, %48 : tensor<1x64xi32, #blocked1> loc(#loc42)
    %50 = tt.broadcast %49 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc12)
    %51 = ttg.memdesc_subview %22[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc12)
    %52 = tt.splat %24 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc11)
    %53 = arith.andi %52, %50 : tensor<128x64xi1, #blocked1> loc(#loc11)
    %54 = ttg.async_copy_global_to_local %38, %51 mask %53 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc12)
    %55 = ttg.async_commit_group %54 loc(#loc12)
    %56 = tt.splat %arg5 : i32 -> tensor<64x1xi32, #blocked> loc(#loc43)
    %57 = arith.cmpi slt, %39, %56 : tensor<64x1xi32, #blocked> loc(#loc43)
    %58 = tt.broadcast %57 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc13)
    %59 = ttg.memdesc_subview %23[%c0_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc13)
    %60 = tt.splat %24 : i1 -> tensor<64x256xi1, #blocked> loc(#loc11)
    %61 = arith.andi %60, %58 : tensor<64x256xi1, #blocked> loc(#loc11)
    %62 = ttg.async_copy_global_to_local %47, %59 mask %61 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc13)
    %63 = ttg.async_commit_group %62 loc(#loc13)
    %64 = arith.cmpi sgt, %20, %c1_i64 : i64 loc(#loc11)
    %65 = arith.addi %25, %c1_i64 : i64 loc(#loc11)
    %66 = arith.remsi %65, %18 : i64 loc(#loc11)
    %67 = arith.cmpi eq, %66, %c0_i64 : i64 loc(#loc11)
    %68 = arith.cmpi ne, %66, %c0_i64 : i64 loc(#loc11)
    %69 = arith.extui %68 : i1 to i32 loc(#loc11)
    %70:5 = scf.if %67 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
      %110 = arith.addi %27, %c132_i32 : i32 loc(#loc11)
      %111 = arith.divsi %110, %12 : i32 loc(#loc14)
      %112 = arith.muli %111, %c8_i32 : i32 loc(#loc15)
      %113 = arith.subi %6, %112 : i32 loc(#loc16)
      %114 = arith.minsi %113, %c8_i32 : i32 loc(#loc17)
      %115 = arith.remsi %110, %114 : i32 loc(#loc18)
      %116 = arith.addi %112, %115 : i32 loc(#loc19)
      %117 = arith.remsi %110, %12 : i32 loc(#loc20)
      %118 = arith.divsi %117, %114 : i32 loc(#loc21)
      %119 = arith.muli %116, %c128_i32 : i32 loc(#loc22)
      %120 = arith.muli %118, %c256_i32 : i32 loc(#loc23)
      %121 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc24)
      %122 = tt.splat %119 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc25)
      %123 = arith.addi %122, %121 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc25)
      %124 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc26)
      %125 = tt.splat %120 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc27)
      %126 = arith.addi %125, %124 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc27)
      %127 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc28)
      %128 = arith.cmpi slt, %123, %127 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc28)
      %129 = arith.select %128, %123, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc29)
      %130 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc30)
      %131 = arith.cmpi slt, %126, %130 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc30)
      %132 = arith.select %131, %126, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc31)
      scf.yield %119, %120, %129, %132, %110 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32 loc(#loc11)
    } else {
      scf.yield %29#0, %29#1, %29#2, %29#3, %27 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32 loc(#loc11)
    } loc(#loc11)
    %71 = arith.muli %69, %c64_i32 : i32 loc(#loc44)
    %72 = tt.splat %71 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc45)
    %73 = tt.splat %71 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc45)
    %74 = arith.addi %72, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc45)
    %75 = arith.addi %73, %14 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc45)
    %76 = tt.expand_dims %70#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc32)
    %77 = arith.muli %76, %31 : tensor<128x1xi32, #blocked1> loc(#loc33)
    %78 = tt.expand_dims %74 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc34)
    %79 = tt.broadcast %77 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc35)
    %80 = tt.broadcast %78 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc35)
    %81 = arith.addi %79, %80 : tensor<128x64xi32, #blocked1> loc(#loc35)
    %82 = tt.addptr %37, %81 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc36)
    %83 = tt.expand_dims %75 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc37)
    %84 = tt.expand_dims %70#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc38)
    %85 = arith.muli %84, %41 : tensor<1x256xi32, #blocked> loc(#loc39)
    %86 = tt.broadcast %83 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc40)
    %87 = tt.broadcast %85 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc40)
    %88 = arith.addi %86, %87 : tensor<64x256xi32, #blocked> loc(#loc40)
    %89 = tt.addptr %46, %88 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc41)
    %90 = arith.subi %arg5, %71 : i32 loc(#loc46)
    %91 = tt.splat %90 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc42)
    %92 = arith.cmpi slt, %33, %91 : tensor<1x64xi32, #blocked1> loc(#loc42)
    %93 = tt.broadcast %92 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc12)
    %94 = ttg.memdesc_subview %22[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc12)
    %95 = tt.splat %64 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc11)
    %96 = arith.andi %95, %93 : tensor<128x64xi1, #blocked1> loc(#loc11)
    %97 = ttg.async_copy_global_to_local %82, %94 mask %96 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc12)
    %98 = ttg.async_commit_group %97 loc(#loc12)
    %99 = tt.splat %90 : i32 -> tensor<64x1xi32, #blocked> loc(#loc43)
    %100 = arith.cmpi slt, %39, %99 : tensor<64x1xi32, #blocked> loc(#loc43)
    %101 = tt.broadcast %100 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc13)
    %102 = ttg.memdesc_subview %23[%c1_i32, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc13)
    %103 = tt.splat %64 : i1 -> tensor<64x256xi1, #blocked> loc(#loc11)
    %104 = arith.andi %103, %101 : tensor<64x256xi1, #blocked> loc(#loc11)
    %105 = ttg.async_copy_global_to_local %89, %102 mask %104 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc13)
    %106 = ttg.async_commit_group %105 loc(#loc13)
    %107:20 = scf.for %arg9 = %c0_i64 to %20 step %c1_i64 iter_args(%arg10 = %66, %arg11 = %70#4, %arg12 = %2, %arg13 = %70#0, %arg14 = %70#1, %arg15 = %70#2, %arg16 = %70#3, %arg17 = %c1_i32, %arg18 = %c-1_i32, %arg19 = %69, %arg20 = %63, %arg21 = %106, %arg22 = %28, %arg23 = %68, %arg24 = %25, %arg25 = %66, %arg26 = %29#0, %arg27 = %70#0, %arg28 = %29#1, %arg29 = %70#1) -> (
      i64, i32,
      tensor<128x256xf32, #mma>,
      i32, i32,
      tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>,
      tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>,
      i32, i32, i32, !ttg.async.token, !ttg.async.token, i1, i1, i64, i64, i32, i32, i32, i32)  : i64 {
      %110 = arith.subi %20, %c2_i64 : i64 loc(#loc11)
      %111 = arith.cmpi slt, %arg9, %110 : i64 loc(#loc11)
      %112 = arith.addi %arg19, %c1_i32 : i32 loc(#loc11)
      %113 = arith.addi %arg10, %c1_i64 : i64 loc(#loc11)
      %114 = arith.remsi %113, %18 : i64 loc(#loc11)
      %115 = arith.cmpi eq, %114, %c0_i64 : i64 loc(#loc11)
      %116 = arith.select %115, %c0_i32, %112 : i32 loc(#loc11)
      %117 = arith.cmpi ne, %114, %c0_i64 : i64 loc(#loc11)
      %118:5 = scf.if %115 -> (i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32) {
        %168 = arith.addi %arg11, %c132_i32 : i32 loc(#loc11)
        %169 = arith.divsi %168, %12 : i32 loc(#loc14)
        %170 = arith.muli %169, %c8_i32 : i32 loc(#loc15)
        %171 = arith.subi %6, %170 : i32 loc(#loc16)
        %172 = arith.minsi %171, %c8_i32 : i32 loc(#loc17)
        %173 = arith.remsi %168, %172 : i32 loc(#loc18)
        %174 = arith.addi %170, %173 : i32 loc(#loc19)
        %175 = arith.remsi %168, %12 : i32 loc(#loc20)
        %176 = arith.divsi %175, %172 : i32 loc(#loc21)
        %177 = arith.muli %174, %c128_i32 : i32 loc(#loc22)
        %178 = arith.muli %176, %c256_i32 : i32 loc(#loc23)
        %179 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc24)
        %180 = tt.splat %177 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc25)
        %181 = arith.addi %180, %179 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc25)
        %182 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc26)
        %183 = tt.splat %178 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc27)
        %184 = arith.addi %183, %182 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc27)
        %185 = tt.splat %arg3 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc28)
        %186 = arith.cmpi slt, %181, %185 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc28)
        %187 = arith.select %186, %181, %cst_0 {tt.contiguity = dense<128> : tensor<1xi32>, tt.divisibility = dense<128> : tensor<1xi32>} : tensor<128xi1, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> loc(#loc29)
        %188 = tt.splat %arg4 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc30)
        %189 = arith.cmpi slt, %184, %188 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc30)
        %190 = arith.select %189, %184, %cst {tt.contiguity = dense<256> : tensor<1xi32>, tt.divisibility = dense<256> : tensor<1xi32>} : tensor<256xi1, #ttg.slice<{dim = 0, parent = #blocked}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> loc(#loc31)
        scf.yield %177, %178, %187, %190, %168 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32 loc(#loc11)
      } else {
        scf.yield %arg13, %arg14, %arg15, %arg16, %arg11 : i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32 loc(#loc11)
      } loc(#loc11)
      %119 = arith.addi %arg18, %c1_i32 : i32 loc(#loc11)
      %120 = arith.cmpi slt, %119, %c3_i32 : i32 loc(#loc11)
      %121 = arith.select %120, %119, %c0_i32 : i32 loc(#loc11)
      %122 = ttg.memdesc_subview %22[%121, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc12)
      %123 = ttg.async_wait %arg20 {num = 2 : i32} loc(#loc12)
      %124 = ttg.memdesc_subview %23[%121, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc13)
      %125 = ttng.warp_group_dot %122, %124, %arg12, %arg22 {inputPrecision = 0 : i32, isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> * !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> -> tensor<128x256xf32, #mma> loc(#loc47)
      %126:3 = ttng.warp_group_dot_wait %125, %122, %124 {pendings = 1 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc47)
      %127 = arith.addi %arg17, %c1_i32 : i32 loc(#loc11)
      %128 = arith.cmpi slt, %127, %c3_i32 : i32 loc(#loc11)
      %129 = arith.select %128, %127, %c0_i32 : i32 loc(#loc11)
      %130 = arith.muli %116, %c64_i32 : i32 loc(#loc44)
      %131 = tt.splat %130 : i32 -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc45)
      %132 = tt.splat %130 : i32 -> tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc45)
      %133 = arith.addi %131, %13 : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> loc(#loc45)
      %134 = arith.addi %132, %14 : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> loc(#loc45)
      %135 = tt.expand_dims %118#2 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<128x1xi32, #blocked1> loc(#loc32)
      %136 = arith.muli %135, %31 : tensor<128x1xi32, #blocked1> loc(#loc33)
      %137 = tt.expand_dims %133 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x64xi32, #blocked1> loc(#loc34)
      %138 = tt.broadcast %136 : tensor<128x1xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc35)
      %139 = tt.broadcast %137 : tensor<1x64xi32, #blocked1> -> tensor<128x64xi32, #blocked1> loc(#loc35)
      %140 = arith.addi %138, %139 : tensor<128x64xi32, #blocked1> loc(#loc35)
      %141 = tt.addptr %37, %140 : tensor<128x64x!tt.ptr<f16>, #blocked1>, tensor<128x64xi32, #blocked1> loc(#loc36)
      %142 = tt.expand_dims %134 {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked> loc(#loc37)
      %143 = tt.expand_dims %118#3 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x256xi32, #blocked> loc(#loc38)
      %144 = arith.muli %143, %41 : tensor<1x256xi32, #blocked> loc(#loc39)
      %145 = tt.broadcast %142 : tensor<64x1xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc40)
      %146 = tt.broadcast %144 : tensor<1x256xi32, #blocked> -> tensor<64x256xi32, #blocked> loc(#loc40)
      %147 = arith.addi %145, %146 : tensor<64x256xi32, #blocked> loc(#loc40)
      %148 = tt.addptr %46, %147 : tensor<64x256x!tt.ptr<f16>, #blocked>, tensor<64x256xi32, #blocked> loc(#loc41)
      %149 = arith.subi %arg5, %130 : i32 loc(#loc46)
      %150 = tt.splat %149 : i32 -> tensor<1x64xi32, #blocked1> loc(#loc42)
      %151 = arith.cmpi slt, %33, %150 : tensor<1x64xi32, #blocked1> loc(#loc42)
      %152 = tt.broadcast %151 : tensor<1x64xi1, #blocked1> -> tensor<128x64xi1, #blocked1> loc(#loc12)
      %153 = ttg.memdesc_subview %22[%129, %c0_i32, %c0_i32] : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc12)
      %154 = tt.splat %111 : i1 -> tensor<128x64xi1, #blocked1> loc(#loc11)
      %155 = arith.andi %154, %152 : tensor<128x64xi1, #blocked1> loc(#loc11)
      %156 = ttg.async_copy_global_to_local %141, %153 mask %155 other %cst_1 : tensor<128x64x!tt.ptr<f16>, #blocked1> -> <128x64xf16, #shared, #smem, mutable, 3x128x64> loc(#loc12)
      %157 = ttg.async_commit_group %156 loc(#loc12)
      %158 = tt.splat %149 : i32 -> tensor<64x1xi32, #blocked> loc(#loc43)
      %159 = arith.cmpi slt, %39, %158 : tensor<64x1xi32, #blocked> loc(#loc43)
      %160 = tt.broadcast %159 : tensor<64x1xi1, #blocked> -> tensor<64x256xi1, #blocked> loc(#loc13)
      %161 = ttg.memdesc_subview %23[%129, %c0_i32, %c0_i32] : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> -> !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc13)
      %162 = tt.splat %111 : i1 -> tensor<64x256xi1, #blocked> loc(#loc11)
      %163 = arith.andi %162, %160 : tensor<64x256xi1, #blocked> loc(#loc11)
      %164 = ttg.async_copy_global_to_local %148, %161 mask %163 other %cst_2 : tensor<64x256x!tt.ptr<f16>, #blocked> -> <64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc13)
      %165 = ttg.async_commit_group %164 loc(#loc13)
      %166 = arith.subi %18, %c1_i64 : i64 loc(#loc11)
      %167 = arith.cmpi eq, %arg24, %166 : i64 loc(#loc11)
      scf.if %167 {
        %168:3 = ttng.warp_group_dot_wait %126#0, %122, %124 {pendings = 0 : i32} : tensor<128x256xf32, #mma>, !ttg.memdesc<128x64xf16, #shared, #smem, mutable, 3x128x64>, !ttg.memdesc<64x256xf16, #shared1, #smem, mutable, 3x64x256> loc(#loc47)
        %169 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc24)
        %170 = tt.splat %arg26 : i32 -> tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc25)
        %171 = arith.addi %170, %169 : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> loc(#loc25)
        %172 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc26)
        %173 = tt.splat %arg28 : i32 -> tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc27)
        %174 = arith.addi %173, %172 : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> loc(#loc27)
        %175 = tt.expand_dims %171 {axis = 1 : i32} : tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked2}>> -> tensor<128x1xi32, #blocked2> loc(#loc48)
        %176 = tt.splat %arg8 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc49)
        %177 = arith.muli %176, %175 : tensor<128x1xi32, #blocked2> loc(#loc49)
        %178 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>, #blocked2> loc(#loc50)
        %179 = tt.addptr %178, %177 : tensor<128x1x!tt.ptr<f16>, #blocked2>, tensor<128x1xi32, #blocked2> loc(#loc50)
        %180 = tt.expand_dims %174 {axis = 0 : i32} : tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked2}>> -> tensor<1x256xi32, #blocked2> loc(#loc51)
        %181 = tt.broadcast %179 : tensor<128x1x!tt.ptr<f16>, #blocked2> -> tensor<128x256x!tt.ptr<f16>, #blocked2> loc(#loc52)
        %182 = tt.broadcast %180 : tensor<1x256xi32, #blocked2> -> tensor<128x256xi32, #blocked2> loc(#loc52)
        %183 = tt.addptr %181, %182 : tensor<128x256x!tt.ptr<f16>, #blocked2>, tensor<128x256xi32, #blocked2> loc(#loc52)
        %184 = tt.splat %arg3 : i32 -> tensor<128x1xi32, #blocked2> loc(#loc53)
        %185 = arith.cmpi slt, %175, %184 : tensor<128x1xi32, #blocked2> loc(#loc53)
        %186 = tt.splat %arg4 : i32 -> tensor<1x256xi32, #blocked2> loc(#loc54)
        %187 = arith.cmpi slt, %180, %186 : tensor<1x256xi32, #blocked2> loc(#loc54)
        %188 = tt.broadcast %185 : tensor<128x1xi1, #blocked2> -> tensor<128x256xi1, #blocked2> loc(#loc55)
        %189 = tt.broadcast %187 : tensor<1x256xi1, #blocked2> -> tensor<128x256xi1, #blocked2> loc(#loc55)
        %190 = arith.andi %188, %189 : tensor<128x256xi1, #blocked2> loc(#loc55)
        %191 = arith.truncf %168#0 : tensor<128x256xf32, #mma> to tensor<128x256xf16, #mma> loc(#loc56)
        %192 = ttg.convert_layout %191 : tensor<128x256xf16, #mma> -> tensor<128x256xf16, #blocked2> loc(#loc57)
        tt.store %183, %192, %190 : tensor<128x256x!tt.ptr<f16>, #blocked2> loc(#loc57)
      } loc(#loc11)
      scf.yield %114, %118#4, %126#0, %118#0, %118#1, %118#2, %118#3, %129, %121, %116, %arg21, %165, %arg23, %117, %arg25, %114, %arg27, %118#0, %arg29, %118#1 : i64, i32, tensor<128x256xf32, #mma>, i32, i32, tensor<128xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>, tensor<256xi32, #ttg.slice<{dim = 0, parent = #blocked}>>, i32, i32, i32, !ttg.async.token, !ttg.async.token, i1, i1, i64, i64, i32, i32, i32, i32 loc(#loc11)
    } loc(#loc11)
    %108 = ttng.warp_group_dot_wait %107#2 {pendings = 0 : i32} : tensor<128x256xf32, #mma> loc(#loc11)
    %109 = ttg.async_wait  {num = 0 : i32} loc(#loc11)
    ttg.local_dealloc %22 : !ttg.memdesc<3x128x64xf16, #shared, #smem, mutable> loc(#loc11)
    ttg.local_dealloc %23 : !ttg.memdesc<3x64x256xf16, #shared1, #smem, mutable> loc(#loc11)
    tt.return loc(#loc58)
  } loc(#loc)
} loc(#loc)
#loc1 = loc(unknown)
#loc2 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":242:30)
#loc3 = loc("/root/.pyenv/versions/3.11.8/lib/python3.11/site-packages/triton/language/standard.py":40:22)
#loc4 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":243:27)
#loc5 = loc("/root/.pyenv/versions/3.11.8/lib/python3.11/site-packages/triton/language/standard.py":40:28)
#loc6 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":244:27)
#loc7 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":245:25)
#loc8 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":246:28)
#loc9 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":247:38)
#loc10 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":249:35)
#loc11 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":251:47)
#loc12 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":273:24)
#loc13 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":274:24)
#loc14 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":252:30)
#loc15 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":253:33)
#loc16 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":254:39)
#loc17 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":254:52)
#loc18 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":255:41)
#loc19 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":255:31)
#loc20 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":256:27)
#loc21 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":256:48)
#loc22 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":258:26)
#loc23 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":259:26)
#loc24 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":260:41)
#loc25 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":260:28)
#loc26 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":261:41)
#loc27 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":261:28)
#loc28 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":262:37)
#loc29 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":262:49)
#loc30 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":263:37)
#loc31 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":263:49)
#loc32 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":270:38)
#loc33 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":270:49)
#loc34 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":270:68)
#loc35 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":270:61)
#loc36 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":270:30)
#loc37 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":271:37)
#loc38 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":271:68)
#loc39 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":271:79)
#loc40 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":271:60)
#loc41 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":271:30)
#loc42 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":273:64)
#loc43 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":274:64)
#loc44 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":269:26)
#loc45 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":269:41)
#loc46 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":273:68)
#loc47 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":275:39)
#loc48 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":279:45)
#loc49 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":279:37)
#loc50 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":279:25)
#loc51 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":279:76)
#loc52 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":279:56)
#loc53 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":280:37)
#loc54 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":280:62)
#loc55 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":280:43)
#loc56 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":284:31)
#loc57 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":285:25)
#loc58 = loc("/root/code/triton/python/tutorials/09-persistent-matmul.py":251:4)
#loc59 = loc(callsite(#loc3 at #loc4))
#loc60 = loc(callsite(#loc5 at #loc4))
#loc61 = loc(callsite(#loc3 at #loc6))
#loc62 = loc(callsite(#loc5 at #loc6))
#loc63 = loc(callsite(#loc3 at #loc7))
#loc64 = loc(callsite(#loc5 at #loc7))

