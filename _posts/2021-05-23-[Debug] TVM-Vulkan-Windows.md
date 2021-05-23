---
title:  "[Debug] TVM，Vulkan，Windows，我，到底谁出了毛病"
---

最近想用 TVM 生成 Vulkan 模型，恰巧还必须用 Windows，恰巧 Windows 上还出了问题。

* TOC
{:toc}

## 计算错误

以前没怎么用 Win 开发过，小心翼翼——
- 装好依赖（Vulkan SDK）
- 编译 LLVM（Windows 上必须编译安装 LLVM），不过这个可以不用
- 编译 TVM

  由于现在问题已经得到解决，复现问题需要用到旧版的 commit，这里用了 [2f29679e](https://github.com/apache/tvm/tree/2f29679e8f2c24c7b8faab824654a1ee1290f7bb)
- 跑一个测试试试 `python .\tests\python\integration\test_gemm.py`

{::options parse_block_html="true" /}

<details><summary markdown="span">于是有了这个计算错误</summary>

```powershell
PS I:\tvm> python .\tests\python\integration\test_gemm.py
[10:50:57] I:\tvm\src\runtime\vulkan\vulkan.cc:760: Initialize Vulkan with 1 devices..
[10:50:57] I:\tvm\src\runtime\vulkan\vulkan.cc:762: vulkan(0)='NVIDIA GeForce RTX 2080' phy_dev_id=0000018AE044DDC0 use_immediate=1
vulkan(0): exec=0.0017107 sec/op
Traceback (most recent call last):
  File ".\tests\python\integration\test_gemm.py", line 113, in <module>
    test_gemm()
  File ".\tests\python\integration\test_gemm.py", line 104, in test_gemm
    check_device("vulkan")
  File ".\tests\python\integration\test_gemm.py", line 102, in check_device
    tvm.testing.assert_allclose(c.numpy(), np.dot(a_np, b_np.T), rtol=1e-5)
  File "i:\tvm\python\tvm\testing.py", line 82, in assert_allclose
    np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol, verbose=True)
  File "C:\Python38\lib\site-packages\numpy\testing\_private\utils.py", line 1528, in assert_allclose
    assert_array_compare(compare, actual, desired, err_msg=str(err_msg),
  File "C:\Python38\lib\site-packages\numpy\testing\_private\utils.py", line 842, in assert_array_compare
    raise AssertionError(msg)
AssertionError:
Not equal to tolerance rtol=1e-05, atol=1e-07

Mismatched elements: 1048315 / 1048576 (100%)
Max absolute difference: 119.47925
Max relative difference: 0.48160496
x: array([[338.6976 , 245.89548, 259.1391 , ..., 261.33603, 259.62222,
        258.07376],
      [245.89548, 331.28613, 251.41951, ..., 248.87415, 245.80385,...
y: array([[249.09152, 241.75012, 256.6666 , ..., 249.89967, 246.78192,
        248.47548],
      [256.62817, 251.93724, 263.58716, ..., 260.56418, 263.1849 ,...
```

</details><br>

{::options parse_block_html="false" /}

## 基本诊断

使用相同版本、配置编译的 TVM，在 Ubuntu 下使用 nvidia 显卡跑上述测试，是能通过的。我们可以确定，两方唯一的区别仅在系统版本（或者说驱动实现）。按照直觉，按照可能性从高到低列举两点猜测

- windows 下的 Vulkan 驱动（由nv提供）是有问题的

  如果是这样，那解决办法主要是

  - 等 Vulkan 驱动的修复
  - 修改现在 SPIRV 代码以绕过可能存在的 bug

- TVM 生成的 SPIRV 代码本身有问题

  如果是这样，这意味着目前能够正确通过测试的 Vulkan 平台的驱动实现有不规范的地方，这反而导致了其运行结果**错误地***正确*了。

  那么可能的解决办法是
  - 调试 Windows 下的 codegen 直到正确，然后在 Linux 上回归测试

而即便是等待驱动修复，也需要开发者尽可能地缩小问题范围，这一步恰好也是和自主 debug 的工作一致的。所以还需要更进一步定位到问题。至于需要定位到哪一步，如果不上手的话是很难提前预测的。

根据之前的报错信息，计算完全错误，但结果值的范围看起来都还正常，没有出现inf、nan或0，错误值和正确值也都在一个数量级上，这说明计算过程本身应该是通顺的，input 能够对 output 产生作用。

### 构造输入，判断错误性质

gemm 计算有两个输入矩阵，A 和 B，在测试中两个矩阵都是随机输入，现在把随机性去掉，并且一次只让一个输入影响输出，即让另一个输入为全0。

  - A[:] = 0; B[:] = 1

    ```powershell
    Mismatched elements: 1048576 / 1048576 (100%)
    Max absolute difference: 1024.
    Max relative difference: inf
    x: array([[1024., 1024., 1024., ..., 1024., 1024., 1024.],
          [1024., 1024., 1024., ..., 1024., 1024., 1024.],
          [1024., 1024., 1024., ..., 1024., 1024., 1024.],...
    y: array([[0., 0., 0., ..., 0., 0., 0.],
          [0., 0., 0., ..., 0., 0., 0.],
          [0., 0., 0., ..., 0., 0., 0.],...
    ```


  - A[:] = 1; B[:] = 0

    passed

现象很有意思，先忽略掉这里的 max diff 居然出现了 inf，只看可视范围内，看起来符合一个假设：kernel 内部计算只用到了矩阵 B 而把 A 忽略了，乘法与加法本身的计算还是生效的。不过 1 本身是个特殊值，把它换成 0.5 的话，可以结果里的 1024 变成了 256， 这就也更验证了猜测。但是不能忘了那个 inf，所以这个假设虽然在可视范围内成立，实际上是不成立的。

这应该是个很常用的初步定位错误位置的方法，针对计算错误，构造特殊输入值（比如0、1），固定随机值，然后观察输出结果得出一些结论。通过上面的现象，矩阵 A 很可能根本没有参与运算，矩阵 B 一定程度上取代了矩阵 A。

更加严谨的做法，当我们假设 A 矩阵没有参与计算时，还需要把其他一下特殊值给到 A，比如 1， inf， nan，random 等等，才能够更有把握。不过初步诊断到此基本不再能得到其他有用结论了。目前先仅仅记住这个猜测吧。

## 进一步诊断

接下来依然要用到之前构造的各种不同的输入组合，但是现在还要将这个 gemm 计算的 schedule 纳入考虑范围。看这个 gemm 的 schedule 片段：

```py
  # ...
  CC = s.cache_write(C, "local")
  AA = s.cache_read(A, "shared", [CC])
  BB = s.cache_read(B, "shared", [CC])
  # ...
  s[AA].compute_at(s[CC], k)
  s[BB].compute_at(s[CC], k)
  s[AA].double_buffer()
  s[BB].double_buffer()
  ty, xi = s[AA].split(s[AA].op.axis[0], nparts=num_thread)
  tx, xi = s[AA].split(xi, nparts=num_thread)
  s[AA].bind(ty, thread_y)
  s[AA].bind(tx, thread_x)

  ty, xi = s[BB].split(s[BB].op.axis[0], nparts=num_thread)
  tx, xi = s[BB].split(xi, nparts=num_thread)
  s[BB].bind(ty, thread_y)
  s[BB].bind(tx, thread_x)
  # ...
```

很简单的 kernel，根本没法找到哪里错了。之前正怀疑矩阵 A 没有参与运算，而 schedule 中计算部分没有直接用 A，而是用了 AA，也就是 A 在 shared memory 上的 cache。那么就把 A 和 B 的 cache，AA 和 BB，都关掉试试。还好这个操作很方便，注释掉 schedule 中包含 AA 和 BB 的代码行就可以了。

经过各种组合实验，发现：

1. 输入随机，关掉 AA 和 BB，计算结果正确
2. 输入随机，只关掉 AA，计算结果正确
3. 输入随机，只关掉 BB，计算结果正确

更新之前的推论，并不是矩阵 A 没有参与计算，而是两个 shared memory 的 buffer，最多只有一个正常工作了。

### 导出 spv 文件

我们更愿意相信，一份代码如果在某个地方能跑对，那这份代码本身错的可能性很小，这也是[基本诊断](#基本诊断)中排行的来源。但是现在不能够纠结于这一点，最好能找到一段能在 Windows 上跑对的 vulkan 程序，它最好是一段 gemm 程序，用到了 shared memory。

但我连 cuda 都不想写（这也是我喜欢 TVM 的原因），何况 HLSL，GLSL？而且就算手写好了，也不能保证可以方便调用起来，毕竟 Vulkan 接口是出了名的啰嗦。天无绝人之路，TVM 生成的是 SPIR-V IR（二进制文件spv），而 Vulkan SDK 里就有 `spirv-cross` 工具可以把 spv 转成 GLSL，可以先转来看看。

先导出 spv 文件。熟悉 TVM 的小伙伴都知道 TVM 总是为各种中间代码（比如 cuda kernel、spv）准备了后处理回调接口，通过这个接口我们可以很方便的导出 spv，或者对这个 spv 进行修改然后传给 TVM 的后续处理。把下面这段代码贴到 test_gemm.py 一个合适的地方就好了。

```py
@tvm.register_func("tvm_callback_vulkan_postproc")
def ff(spv):
    open('a.spv', 'wb').write(spv)
    return spv
```

可以用 `spirv-dis a.spv -o a.txt` 工具把 spv 转成文本格式备用。

{::options parse_block_html="true" /}

<details><summary markdown="span">a.txt</summary>

```
; SPIR-V
; Version: 1.0
; Generator: Khronos; 0
; Bound: 229
; Schema: 0
               OpCapability Shader
               OpExtension "SPV_KHR_storage_buffer_storage_class"
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %15 "default_function_kernel0" %gl_WorkGroupID %gl_LocalInvocationID
               OpExecutionMode %15 LocalSize 8 8 1
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_struct_10 0 Offset 0
               OpDecorate %_struct_10 Block
               OpDecorate %12 DescriptorSet 0
               OpDecorate %12 Binding 0
               OpDecorate %13 DescriptorSet 0
               OpDecorate %13 Binding 1
               OpDecorate %14 DescriptorSet 0
               OpDecorate %14 Binding 2
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %_arr_float_uint_64 ArrayStride 4
               OpMemberDecorate %_struct_26 0 Offset 0
               OpDecorate %_struct_26 Block
               OpDecorate %_arr_float_uint_128 ArrayStride 4
               OpMemberDecorate %_struct_31 0 Offset 0
               OpDecorate %_struct_31 Block
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
        %int = OpTypeInt 32 1
       %uint = OpTypeInt 32 0
       %bool = OpTypeBool
      %float = OpTypeFloat 32
      %int_0 = OpConstant %int 0
       %void = OpTypeVoid
          %8 = OpTypeFunction %void
%_runtimearr_float = OpTypeRuntimeArray %float
 %_struct_10 = OpTypeStruct %_runtimearr_float
%_ptr_StorageBuffer__struct_10 = OpTypePointer StorageBuffer %_struct_10
         %12 = OpVariable %_ptr_StorageBuffer__struct_10 StorageBuffer
         %13 = OpVariable %_ptr_StorageBuffer__struct_10 StorageBuffer
         %14 = OpVariable %_ptr_StorageBuffer__struct_10 StorageBuffer
      %v3int = OpTypeVector %int 3
%_ptr_Input_v3int = OpTypePointer Input %v3int
%gl_WorkGroupID = OpVariable %_ptr_Input_v3int Input
%_ptr_Input_int = OpTypePointer Input %int
      %int_1 = OpConstant %int 1
    %uint_64 = OpConstant %uint 64
%_arr_float_uint_64 = OpTypeArray %float %uint_64
 %_struct_26 = OpTypeStruct %_arr_float_uint_64
%_ptr_Function__struct_26 = OpTypePointer Function %_struct_26
   %uint_128 = OpConstant %uint 128
%_arr_float_uint_128 = OpTypeArray %float %uint_128
 %_struct_31 = OpTypeStruct %_arr_float_uint_128
%_ptr_Workgroup__struct_31 = OpTypePointer Workgroup %_struct_31
         %33 = OpVariable %_ptr_Workgroup__struct_31 Workgroup
         %34 = OpVariable %_ptr_Workgroup__struct_31 Workgroup
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3int Input
      %int_8 = OpConstant %int 8
    %float_0 = OpConstant %float 0
%_ptr_Function_float = OpTypePointer Function %float
%_ptr_StorageBuffer_float = OpTypePointer StorageBuffer %float
   %int_1024 = OpConstant %int 1024
   %int_8192 = OpConstant %int 8192
  %int_65536 = OpConstant %int 65536
%_ptr_Workgroup_float = OpTypePointer Workgroup %float
   %int_1023 = OpConstant %int 1023
    %int_272 = OpConstant %int 272
      %int_2 = OpConstant %int 2
     %int_64 = OpConstant %int 64
         %15 = OpFunction %void None %8
         %16 = OpLabel
         %28 = OpVariable %_ptr_Function__struct_26 Function
         %22 = OpAccessChain %_ptr_Input_int %gl_WorkGroupID %int_1
         %23 = OpLoad %int %22
         %35 = OpAccessChain %_ptr_Input_int %gl_WorkGroupID %int_0
         %36 = OpLoad %int %35
         %38 = OpAccessChain %_ptr_Input_int %gl_LocalInvocationID %int_1
         %39 = OpLoad %int %38
         %40 = OpAccessChain %_ptr_Input_int %gl_LocalInvocationID %int_0
         %41 = OpLoad %int %40
               OpBranch %43
         %43 = OpLabel
         %47 = OpPhi %int %int_0 %16 %61 %45
         %48 = OpSLessThan %bool %47 %int_8
               OpLoopMerge %46 %45 None
               OpBranchConditional %48 %44 %46 128 1
         %44 = OpLabel
               OpBranch %49
         %49 = OpLabel
         %53 = OpPhi %int %int_0 %44 %60 %51
         %54 = OpSLessThan %bool %53 %int_8
               OpLoopMerge %52 %51 None
               OpBranchConditional %54 %50 %52 128 1
         %50 = OpLabel
         %57 = OpIMul %int %47 %int_8
         %58 = OpIAdd %int %57 %53
         %59 = OpInBoundsAccessChain %_ptr_Function_float %28 %int_0 %58
               OpStore %59 %float_0 None
               OpBranch %51
         %51 = OpLabel
         %60 = OpIAdd %int %53 %int_1
               OpBranch %49
         %52 = OpLabel
               OpBranch %45
         %45 = OpLabel
         %61 = OpIAdd %int %47 %int_1
               OpBranch %43
         %46 = OpLabel
         %64 = OpIMul %int %41 %int_1024
         %66 = OpIMul %int %39 %int_8192
         %68 = OpIMul %int %23 %int_65536
         %69 = OpIAdd %int %68 %66
         %70 = OpIAdd %int %69 %64
         %71 = OpInBoundsAccessChain %_ptr_StorageBuffer_float %12 %int_0 %70
         %72 = OpLoad %float %71 None
         %74 = OpIMul %int %39 %int_8
         %75 = OpIAdd %int %74 %41
         %76 = OpInBoundsAccessChain %_ptr_Workgroup_float %33 %int_0 %75
               OpStore %76 %72 None
         %77 = OpIMul %int %41 %int_1024
         %78 = OpIMul %int %39 %int_8192
         %79 = OpIMul %int %36 %int_65536
         %80 = OpIAdd %int %79 %78
         %81 = OpIAdd %int %80 %77
         %82 = OpInBoundsAccessChain %_ptr_StorageBuffer_float %13 %int_0 %81
         %83 = OpLoad %float %82 None
         %84 = OpIMul %int %39 %int_8
         %85 = OpIAdd %int %84 %41
         %86 = OpInBoundsAccessChain %_ptr_Workgroup_float %34 %int_0 %85
               OpStore %86 %83 None
               OpBranch %88
         %88 = OpLabel
         %92 = OpPhi %int %int_0 %46 %166 %90
         %93 = OpSLessThan %bool %92 %int_1023
               OpLoopMerge %91 %90 None
               OpBranchConditional %93 %89 %91 128 1
         %89 = OpLabel
               OpControlBarrier %int_2 %int_2 %int_272
         %96 = OpIMul %int %41 %int_1024
         %97 = OpIMul %int %39 %int_8192
         %98 = OpIMul %int %23 %int_65536
         %99 = OpIAdd %int %98 %97
        %100 = OpIAdd %int %99 %96
        %101 = OpIAdd %int %100 %92
        %102 = OpIAdd %int %101 %int_1
        %103 = OpInBoundsAccessChain %_ptr_StorageBuffer_float %12 %int_0 %102
        %104 = OpLoad %float %103 None
        %105 = OpIMul %int %39 %int_8
        %107 = OpIAdd %int %92 %int_1
        %108 = OpBitwiseAnd %int %107 %int_1
        %109 = OpIMul %int %108 %int_64
        %110 = OpIAdd %int %109 %105
        %111 = OpIAdd %int %110 %41
        %112 = OpInBoundsAccessChain %_ptr_Workgroup_float %33 %int_0 %111
               OpStore %112 %104 None
        %113 = OpIMul %int %41 %int_1024
        %114 = OpIMul %int %39 %int_8192
        %115 = OpIMul %int %36 %int_65536
        %116 = OpIAdd %int %115 %114
        %117 = OpIAdd %int %116 %113
        %118 = OpIAdd %int %117 %92
        %119 = OpIAdd %int %118 %int_1
        %120 = OpInBoundsAccessChain %_ptr_StorageBuffer_float %13 %int_0 %119
        %121 = OpLoad %float %120 None
        %122 = OpIMul %int %39 %int_8
        %123 = OpIAdd %int %92 %int_1
        %124 = OpBitwiseAnd %int %123 %int_1
        %125 = OpIMul %int %124 %int_64
        %126 = OpIAdd %int %125 %122
        %127 = OpIAdd %int %126 %41
        %128 = OpInBoundsAccessChain %_ptr_Workgroup_float %34 %int_0 %127
               OpStore %128 %121 None
               OpBranch %129
        %129 = OpLabel
        %133 = OpPhi %int %int_0 %89 %165 %131
        %134 = OpSLessThan %bool %133 %int_8
               OpLoopMerge %132 %131 None
               OpBranchConditional %134 %130 %132 128 1
        %130 = OpLabel
               OpBranch %135
        %135 = OpLabel
        %139 = OpPhi %int %int_0 %130 %164 %137
        %140 = OpSLessThan %bool %139 %int_8
               OpLoopMerge %138 %137 None
               OpBranchConditional %140 %136 %138 128 1
        %136 = OpLabel
        %141 = OpIMul %int %41 %int_8
        %142 = OpBitwiseAnd %int %92 %int_1
        %143 = OpIMul %int %142 %int_64
        %144 = OpIAdd %int %143 %141
        %145 = OpIAdd %int %144 %139
        %146 = OpInBoundsAccessChain %_ptr_Workgroup_float %34 %int_0 %145
        %147 = OpLoad %float %146 None
        %148 = OpIMul %int %39 %int_8
        %149 = OpBitwiseAnd %int %92 %int_1
        %150 = OpIMul %int %149 %int_64
        %151 = OpIAdd %int %150 %148
        %152 = OpIAdd %int %151 %133
        %153 = OpInBoundsAccessChain %_ptr_Workgroup_float %33 %int_0 %152
        %154 = OpLoad %float %153 None
        %155 = OpFMul %float %154 %147
        %156 = OpIMul %int %133 %int_8
        %157 = OpIAdd %int %156 %139
        %158 = OpInBoundsAccessChain %_ptr_Function_float %28 %int_0 %157
        %159 = OpLoad %float %158 None
        %160 = OpFAdd %float %159 %155
        %161 = OpIMul %int %133 %int_8
        %162 = OpIAdd %int %161 %139
        %163 = OpInBoundsAccessChain %_ptr_Function_float %28 %int_0 %162
               OpStore %163 %160 None
               OpBranch %137
        %137 = OpLabel
        %164 = OpIAdd %int %139 %int_1
               OpBranch %135
        %138 = OpLabel
               OpBranch %131
        %131 = OpLabel
        %165 = OpIAdd %int %133 %int_1
               OpBranch %129
        %132 = OpLabel
               OpBranch %90
         %90 = OpLabel
        %166 = OpIAdd %int %92 %int_1
               OpBranch %88
         %91 = OpLabel
               OpControlBarrier %int_2 %int_2 %int_272
               OpBranch %167
        %167 = OpLabel
        %171 = OpPhi %int %int_0 %91 %199 %169
        %172 = OpSLessThan %bool %171 %int_8
               OpLoopMerge %170 %169 None
               OpBranchConditional %172 %168 %170 128 1
        %168 = OpLabel
               OpBranch %173
        %173 = OpLabel
        %177 = OpPhi %int %int_0 %168 %198 %175
        %178 = OpSLessThan %bool %177 %int_8
               OpLoopMerge %176 %175 None
               OpBranchConditional %178 %174 %176 128 1
        %174 = OpLabel
        %179 = OpIMul %int %41 %int_8
        %180 = OpIAdd %int %179 %177
        %181 = OpIAdd %int %180 %int_64
        %182 = OpInBoundsAccessChain %_ptr_Workgroup_float %34 %int_0 %181
        %183 = OpLoad %float %182 None
        %184 = OpIMul %int %39 %int_8
        %185 = OpIAdd %int %184 %171
        %186 = OpIAdd %int %185 %int_64
        %187 = OpInBoundsAccessChain %_ptr_Workgroup_float %33 %int_0 %186
        %188 = OpLoad %float %187 None
        %189 = OpFMul %float %188 %183
        %190 = OpIMul %int %171 %int_8
        %191 = OpIAdd %int %190 %177
        %192 = OpInBoundsAccessChain %_ptr_Function_float %28 %int_0 %191
        %193 = OpLoad %float %192 None
        %194 = OpFAdd %float %193 %189
        %195 = OpIMul %int %171 %int_8
        %196 = OpIAdd %int %195 %177
        %197 = OpInBoundsAccessChain %_ptr_Function_float %28 %int_0 %196
               OpStore %197 %194 None
               OpBranch %175
        %175 = OpLabel
        %198 = OpIAdd %int %177 %int_1
               OpBranch %173
        %176 = OpLabel
               OpBranch %169
        %169 = OpLabel
        %199 = OpIAdd %int %171 %int_1
               OpBranch %167
        %170 = OpLabel
               OpBranch %200
        %200 = OpLabel
        %204 = OpPhi %int %int_0 %170 %228 %202
        %205 = OpSLessThan %bool %204 %int_8
               OpLoopMerge %203 %202 None
               OpBranchConditional %205 %201 %203 128 1
        %201 = OpLabel
               OpBranch %206
        %206 = OpLabel
        %210 = OpPhi %int %int_0 %201 %227 %208
        %211 = OpSLessThan %bool %210 %int_8
               OpLoopMerge %209 %208 None
               OpBranchConditional %211 %207 %209 128 1
        %207 = OpLabel
        %212 = OpIMul %int %204 %int_8
        %213 = OpIAdd %int %212 %210
        %214 = OpInBoundsAccessChain %_ptr_Function_float %28 %int_0 %213
        %215 = OpLoad %float %214 None
        %216 = OpIMul %int %41 %int_8
        %217 = OpIMul %int %36 %int_64
        %218 = OpIMul %int %204 %int_1024
        %219 = OpIMul %int %39 %int_8192
        %220 = OpIMul %int %23 %int_65536
        %221 = OpIAdd %int %220 %219
        %222 = OpIAdd %int %221 %218
        %223 = OpIAdd %int %222 %217
        %224 = OpIAdd %int %223 %216
        %225 = OpIAdd %int %224 %210
        %226 = OpInBoundsAccessChain %_ptr_StorageBuffer_float %14 %int_0 %225
               OpStore %226 %215 None
               OpBranch %208
        %208 = OpLabel
        %227 = OpIAdd %int %210 %int_1
               OpBranch %206
        %209 = OpLabel
               OpBranch %202
        %202 = OpLabel
        %228 = OpIAdd %int %204 %int_1
               OpBranch %200
        %203 = OpLabel
               OpReturn
               OpFunctionEnd
```

</details><br>

{::options parse_block_html="false" /}

熟悉 TVM 的小伙伴可能要问了，在 build 好的 module 上 `get_source` 也是能拿到文本格式的 IR 的，为啥要用 callback？

一方面，我觉得用 callback 这个方法比较有营养，知道这个接口的人肯定比知道 `get_source` 的人要少吧。另一方面，spv 二进制文件是必须的，如果用 `get_source` 方法得到了文本 IR，在用 `spirv-as` 进行汇编时要记得传入 `--target-env spv1.0` 来指定 1.0 的 spirv 版本，因为目前 TVM 写死了生成的 spirv 代码版本为 1.0。

不过最重要的理由之后会说。

### 构造正确 spv 文件

用 `spirv-cross a.spv --output a.glsl` 得到 GLSL 代码。

{::options parse_block_html="true" /}

<details><summary markdown="span">a.glsl</summary>

```glsl
#version 450
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(binding = 0, std430) buffer _10_12
{
    float _m0[];
} _12;

layout(binding = 1, std430) buffer _10_13
{
    float _m0[];
} _13;

layout(binding = 2, std430) buffer _10_14
{
    float _m0[];
} _14;

shared _31 _33;
shared _31 _34;

void main()
{
    _26 _28;
    for (int _47 = 0; _47 < 8; _47++)
    {
        for (int _53 = 0; _53 < 8; _53++)
        {
            _28._m0[(_47 * 8) + _53] = 0.0;
        }
    }
    _33._m0[(int(gl_LocalInvocationID.y) * 8) + int(gl_LocalInvocationID.x)] = _12._m0[((int(gl_WorkGroupID.y) * 65536) + (int(gl_LocalInvocationID.y) * 8192)) + (int(gl_LocalInvocationID.x) * 1024)];
    _34._m0[(int(gl_LocalInvocationID.y) * 8) + int(gl_LocalInvocationID.x)] = _13._m0[((int(gl_WorkGroupID.x) * 65536) + (int(gl_LocalInvocationID.y) * 8192)) + (int(gl_LocalInvocationID.x) * 1024)];
    for (int _92 = 0; _92 < 1023; _92++)
    {
        barrier();
        _33._m0[((((_92 + 1) & 1) * 64) + (int(gl_LocalInvocationID.y) * 8)) + int(gl_LocalInvocationID.x)] = _12._m0[((((int(gl_WorkGroupID.y) * 65536) + (int(gl_LocalInvocationID.y) * 8192)) + (int(gl_LocalInvocationID.x) * 1024)) + _92) + 1];
        _34._m0[((((_92 + 1) & 1) * 64) + (int(gl_LocalInvocationID.y) * 8)) + int(gl_LocalInvocationID.x)] = _13._m0[((((int(gl_WorkGroupID.x) * 65536) + (int(gl_LocalInvocationID.y) * 8192)) + (int(gl_LocalInvocationID.x) * 1024)) + _92) + 1];
        for (int _133 = 0; _133 < 8; _133++)
        {
            for (int _139 = 0; _139 < 8; _139++)
            {
                _28._m0[(_133 * 8) + _139] += (_33._m0[(((_92 & 1) * 64) + (int(gl_LocalInvocationID.y) * 8)) + _133] * _34._m0[(((_92 & 1) * 64) + (int(gl_LocalInvocationID.x) * 8)) + _139]);
            }
        }
    }
    barrier();
    for (int _171 = 0; _171 < 8; _171++)
    {
        for (int _177 = 0; _177 < 8; _177++)
        {
            _28._m0[(_171 * 8) + _177] += (_33._m0[((int(gl_LocalInvocationID.y) * 8) + _171) + 64] * _34._m0[((int(gl_LocalInvocationID.x) * 8) + _177) + 64]);
        }
    }
    for (int _204 = 0; _204 < 8; _204++)
    {
        for (int _210 = 0; _210 < 8; _210++)
        {
            _14._m0[(((((int(gl_WorkGroupID.y) * 65536) + (int(gl_LocalInvocationID.y) * 8192)) + (_204 * 1024)) + (int(gl_WorkGroupID.x) * 64)) + (int(gl_LocalInvocationID.x) * 8)) + _210] = _28._m0[(_204 * 8) + _210];
        }
    }
}
```
</details><br>

{::options parse_block_html="false" /}


Vulkan SDK 里还有一个 `glslc` 工具可以用来把 GLSL 代码编译到 spv。先把这个 `a.glsl` 复制一份 `b.glsl`。

`glslc --target-spv=spv1.0 -x glsl -fshader-stage=comp b.glsl -o b.spv` 尝试把 `b.glsl` 编译成 `b.spv`。

不过很明显，这时 `b.glsl` 因为这几行的问题而无法编译。

```glsl
// ...
shared _31 _33; // _31 not declared
shared _31 _34; // _31 not declared

void main()
{
    _26 _28; // _26 not declared
// ...
```

巧了，这三个变量刚好对应两个 read cache(shared) 和一个 write cache(local)。我们有很多办法知道这三个数组的实际大小，比如 

{::options parse_block_html="true" /}

<details><summary markdown="span">`print(tvm.lower(s, [A, B, C], simple_mode=True))`</summary>

```
primfn(A_1: handle, B_1: handle, CC_1: handle) -> ()
  attr = {"global_symbol": "main", "tir.noalias": True}
  buffers = {B: Buffer(B_2: Pointer(float32), float32, [1024, 1024], []),
             A: Buffer(A_2: Pointer(float32), float32, [1024, 1024], []),
             CC: Buffer(CC_2: Pointer(float32), float32, [1024, 1024], [])}
  buffer_map = {A_1: A, B_1: B, CC_1: CC} {
  attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 16;
  attr [CC.local: Pointer(float32)] "storage_scope" = "local";
  allocate(CC.local, float32, [64]);
  attr [A.shared: Pointer(float32)] "storage_scope" = "shared";
  allocate(A.shared, float32, [128]);
  attr [B.shared: Pointer(float32)] "storage_scope" = "shared";
  allocate(B.shared, float32, [128]);
  attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 16;
  attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 8;
  attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 8 {
    for (ii.c.init: int32, 0, 8) {
      for (jj.c.init: int32, 0, 8) {
        CC.local[((ii.c.init*8) + jj.c.init)] = 0f32
      }
    }
    A.shared[((threadIdx.y*8) + threadIdx.x)] = (float32*)A_2[(((blockIdx.y*65536) + (threadIdx.y*8192)) + (threadIdx.x*1024))]
    B.shared[((threadIdx.y*8) + threadIdx.x)] = (float32*)B_2[(((blockIdx.x*65536) + (threadIdx.y*8192)) + (threadIdx.x*1024))]
    for (k.outer: int32, 0, 1023) {
      attr [A.shared] "double_buffer_write" = 1;
      A.shared[(((floormod((k.outer + 1), 2)*64) + (threadIdx.y*8)) + threadIdx.x)] = (float32*)A_2[(((((blockIdx.y*65536) + (threadIdx.y*8192)) + (threadIdx.x*1024)) + k.outer) + 1)]
      attr [B.shared] "double_buffer_write" = 1;
      B.shared[(((floormod((k.outer + 1), 2)*64) + (threadIdx.y*8)) + threadIdx.x)] = (float32*)B_2[(((((blockIdx.x*65536) + (threadIdx.y*8192)) + (threadIdx.x*1024)) + k.outer) + 1)]
      for (ii.c: int32, 0, 8) {
        for (jj.c: int32, 0, 8) {
          CC.local[((ii.c*8) + jj.c)] = ((float32*)CC.local[((ii.c*8) + jj.c)] + ((float32*)A.shared[(((floormod(k.outer, 2)*64) + (threadIdx.y*8)) + ii.c)]*(float32*)B.shared[(((floormod(k.outer, 2)*64) + (threadIdx.x*8)) + jj.c)]))
        }
      }
    }
    for (ii.c_1: int32, 0, 8) {
      for (jj.c_1: int32, 0, 8) {
        CC.local[((ii.c_1*8) + jj.c_1)] = ((float32*)CC.local[((ii.c_1*8) + jj.c_1)] + ((float32*)A.shared[(((threadIdx.y*8) + ii.c_1) + 64)]*(float32*)B.shared[(((threadIdx.x*8) + jj.c_1) + 64)]))
      }
    }
    for (ii.inner.inner: int32, 0, 8) {
      for (jj.inner.inner: int32, 0, 8) {
        CC_2[((((((blockIdx.y*65536) + (threadIdx.y*8192)) + (ii.inner.inner*1024)) + (blockIdx.x*64)) + (threadIdx.x*8)) + jj.inner.inner)] = (float32*)CC.local[((ii.inner.inner*8) + jj.inner.inner)]
      }
    }
  }
}
```

</details><br>

{::options parse_block_html="false" /}

可以看出来，两个 shared memory 的大小为 128xfloat32，一个 local memory 的大小为 64xfloat32。

可以给 `b.glsl` 加上

```glsl
// ...

struct _31 {
    float _m0[128];
};

struct _26 {
    float _m0[64];
};

shared _31 _33; // _31 not declared
shared _31 _34; // _31 not declared

void main()
{
    _26 _28; // _26 not declared
// ...
```

这就能编译过得到 `b.spv` 了，反编译得到 `b.txt`

{::options parse_block_html="true" /}

<details><summary markdown="span">b.txt</summary>

```
; SPIR-V
; Version: 1.0
; Generator: Google Shaderc over Glslang; 10
; Bound: 359
; Schema: 0
               OpCapability Shader
          %1 = OpExtInstImport "GLSL.std.450"
               OpMemoryModel Logical GLSL450
               OpEntryPoint GLCompute %main "main" %gl_LocalInvocationID %gl_WorkGroupID
               OpExecutionMode %main LocalSize 8 8 1
               OpSource GLSL 450
               OpSourceExtension "GL_GOOGLE_cpp_style_line_directive"
               OpSourceExtension "GL_GOOGLE_include_directive"
               OpName %main "main"
               OpName %_47 "_47"
               OpName %_53 "_53"
               OpName %_26 "_26"
               OpMemberName %_26 0 "_m0"
               OpName %_28 "_28"
               OpName %_31 "_31"
               OpMemberName %_31 0 "_m0"
               OpName %_33 "_33"
               OpName %gl_LocalInvocationID "gl_LocalInvocationID"
               OpName %_10_12 "_10_12"
               OpMemberName %_10_12 0 "_m0"
               OpName %_12 "_12"
               OpName %gl_WorkGroupID "gl_WorkGroupID"
               OpName %_34 "_34"
               OpName %_10_13 "_10_13"
               OpMemberName %_10_13 0 "_m0"
               OpName %_13 "_13"
               OpName %_92 "_92"
               OpName %_133 "_133"
               OpName %_139 "_139"
               OpName %_171 "_171"
               OpName %_177 "_177"
               OpName %_204 "_204"
               OpName %_210 "_210"
               OpName %_10_14 "_10_14"
               OpMemberName %_10_14 0 "_m0"
               OpName %_14 "_14"
               OpDecorate %gl_LocalInvocationID BuiltIn LocalInvocationId
               OpDecorate %_runtimearr_float ArrayStride 4
               OpMemberDecorate %_10_12 0 Offset 0
               OpDecorate %_10_12 BufferBlock
               OpDecorate %_12 DescriptorSet 0
               OpDecorate %_12 Binding 0
               OpDecorate %gl_WorkGroupID BuiltIn WorkgroupId
               OpDecorate %_runtimearr_float_0 ArrayStride 4
               OpMemberDecorate %_10_13 0 Offset 0
               OpDecorate %_10_13 BufferBlock
               OpDecorate %_13 DescriptorSet 0
               OpDecorate %_13 Binding 1
               OpDecorate %_runtimearr_float_1 ArrayStride 4
               OpMemberDecorate %_10_14 0 Offset 0
               OpDecorate %_10_14 BufferBlock
               OpDecorate %_14 DescriptorSet 0
               OpDecorate %_14 Binding 2
               OpDecorate %gl_WorkGroupSize BuiltIn WorkgroupSize
       %void = OpTypeVoid
          %3 = OpTypeFunction %void
        %int = OpTypeInt 32 1
%_ptr_Function_int = OpTypePointer Function %int
      %int_0 = OpConstant %int 0
      %int_8 = OpConstant %int 8
       %bool = OpTypeBool
      %float = OpTypeFloat 32
       %uint = OpTypeInt 32 0
    %uint_64 = OpConstant %uint 64
%_arr_float_uint_64 = OpTypeArray %float %uint_64
        %_26 = OpTypeStruct %_arr_float_uint_64
%_ptr_Function__26 = OpTypePointer Function %_26
    %float_0 = OpConstant %float 0
%_ptr_Function_float = OpTypePointer Function %float
      %int_1 = OpConstant %int 1
   %uint_128 = OpConstant %uint 128
%_arr_float_uint_128 = OpTypeArray %float %uint_128
        %_31 = OpTypeStruct %_arr_float_uint_128
%_ptr_Workgroup__31 = OpTypePointer Workgroup %_31
        %_33 = OpVariable %_ptr_Workgroup__31 Workgroup
     %v3uint = OpTypeVector %uint 3
%_ptr_Input_v3uint = OpTypePointer Input %v3uint
%gl_LocalInvocationID = OpVariable %_ptr_Input_v3uint Input
     %uint_1 = OpConstant %uint 1
%_ptr_Input_uint = OpTypePointer Input %uint
     %uint_0 = OpConstant %uint 0
%_runtimearr_float = OpTypeRuntimeArray %float
     %_10_12 = OpTypeStruct %_runtimearr_float
%_ptr_Uniform__10_12 = OpTypePointer Uniform %_10_12
        %_12 = OpVariable %_ptr_Uniform__10_12 Uniform
%gl_WorkGroupID = OpVariable %_ptr_Input_v3uint Input
  %int_65536 = OpConstant %int 65536
   %int_8192 = OpConstant %int 8192
   %int_1024 = OpConstant %int 1024
%_ptr_Uniform_float = OpTypePointer Uniform %float
%_ptr_Workgroup_float = OpTypePointer Workgroup %float
        %_34 = OpVariable %_ptr_Workgroup__31 Workgroup
%_runtimearr_float_0 = OpTypeRuntimeArray %float
     %_10_13 = OpTypeStruct %_runtimearr_float_0
%_ptr_Uniform__10_13 = OpTypePointer Uniform %_10_13
        %_13 = OpVariable %_ptr_Uniform__10_13 Uniform
   %int_1023 = OpConstant %int 1023
     %uint_2 = OpConstant %uint 2
   %uint_264 = OpConstant %uint 264
     %int_64 = OpConstant %int 64
%_runtimearr_float_1 = OpTypeRuntimeArray %float
     %_10_14 = OpTypeStruct %_runtimearr_float_1
%_ptr_Uniform__10_14 = OpTypePointer Uniform %_10_14
        %_14 = OpVariable %_ptr_Uniform__10_14 Uniform
     %uint_8 = OpConstant %uint 8
%gl_WorkGroupSize = OpConstantComposite %v3uint %uint_8 %uint_8 %uint_1
       %main = OpFunction %void None %3
          %5 = OpLabel
        %_47 = OpVariable %_ptr_Function_int Function
        %_53 = OpVariable %_ptr_Function_int Function
        %_28 = OpVariable %_ptr_Function__26 Function
        %_92 = OpVariable %_ptr_Function_int Function
       %_133 = OpVariable %_ptr_Function_int Function
       %_139 = OpVariable %_ptr_Function_int Function
       %_171 = OpVariable %_ptr_Function_int Function
       %_177 = OpVariable %_ptr_Function_int Function
       %_204 = OpVariable %_ptr_Function_int Function
       %_210 = OpVariable %_ptr_Function_int Function
               OpStore %_47 %int_0
               OpBranch %10
         %10 = OpLabel
               OpLoopMerge %12 %13 None
               OpBranch %14
         %14 = OpLabel
         %15 = OpLoad %int %_47
         %18 = OpSLessThan %bool %15 %int_8
               OpBranchConditional %18 %11 %12
         %11 = OpLabel
               OpStore %_53 %int_0
               OpBranch %20
         %20 = OpLabel
               OpLoopMerge %22 %23 None
               OpBranch %24
         %24 = OpLabel
         %25 = OpLoad %int %_53
         %26 = OpSLessThan %bool %25 %int_8
               OpBranchConditional %26 %21 %22
         %21 = OpLabel
         %34 = OpLoad %int %_47
         %35 = OpIMul %int %34 %int_8
         %36 = OpLoad %int %_53
         %37 = OpIAdd %int %35 %36
         %40 = OpAccessChain %_ptr_Function_float %_28 %int_0 %37
               OpStore %40 %float_0
               OpBranch %23
         %23 = OpLabel
         %41 = OpLoad %int %_53
         %43 = OpIAdd %int %41 %int_1
               OpStore %_53 %43
               OpBranch %20
         %22 = OpLabel
               OpBranch %13
         %13 = OpLabel
         %44 = OpLoad %int %_47
         %45 = OpIAdd %int %44 %int_1
               OpStore %_47 %45
               OpBranch %10
         %12 = OpLabel
         %56 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
         %57 = OpLoad %uint %56
         %58 = OpBitcast %int %57
         %59 = OpIMul %int %58 %int_8
         %61 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
         %62 = OpLoad %uint %61
         %63 = OpBitcast %int %62
         %64 = OpIAdd %int %59 %63
         %70 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_1
         %71 = OpLoad %uint %70
         %72 = OpBitcast %int %71
         %74 = OpIMul %int %72 %int_65536
         %75 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
         %76 = OpLoad %uint %75
         %77 = OpBitcast %int %76
         %79 = OpIMul %int %77 %int_8192
         %80 = OpIAdd %int %74 %79
         %81 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
         %82 = OpLoad %uint %81
         %83 = OpBitcast %int %82
         %85 = OpIMul %int %83 %int_1024
         %86 = OpIAdd %int %80 %85
         %88 = OpAccessChain %_ptr_Uniform_float %_12 %int_0 %86
         %89 = OpLoad %float %88
         %91 = OpAccessChain %_ptr_Workgroup_float %_33 %int_0 %64
               OpStore %91 %89
         %93 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
         %94 = OpLoad %uint %93
         %95 = OpBitcast %int %94
         %96 = OpIMul %int %95 %int_8
         %97 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
         %98 = OpLoad %uint %97
         %99 = OpBitcast %int %98
        %100 = OpIAdd %int %96 %99
        %105 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_0
        %106 = OpLoad %uint %105
        %107 = OpBitcast %int %106
        %108 = OpIMul %int %107 %int_65536
        %109 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %110 = OpLoad %uint %109
        %111 = OpBitcast %int %110
        %112 = OpIMul %int %111 %int_8192
        %113 = OpIAdd %int %108 %112
        %114 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %115 = OpLoad %uint %114
        %116 = OpBitcast %int %115
        %117 = OpIMul %int %116 %int_1024
        %118 = OpIAdd %int %113 %117
        %119 = OpAccessChain %_ptr_Uniform_float %_13 %int_0 %118
        %120 = OpLoad %float %119
        %121 = OpAccessChain %_ptr_Workgroup_float %_34 %int_0 %100
               OpStore %121 %120
               OpStore %_92 %int_0
               OpBranch %123
        %123 = OpLabel
               OpLoopMerge %125 %126 None
               OpBranch %127
        %127 = OpLabel
        %128 = OpLoad %int %_92
        %130 = OpSLessThan %bool %128 %int_1023
               OpBranchConditional %130 %124 %125
        %124 = OpLabel
               OpControlBarrier %uint_2 %uint_2 %uint_264
        %133 = OpLoad %int %_92
        %134 = OpIAdd %int %133 %int_1
        %135 = OpBitwiseAnd %int %134 %int_1
        %137 = OpIMul %int %135 %int_64
        %138 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %139 = OpLoad %uint %138
        %140 = OpBitcast %int %139
        %141 = OpIMul %int %140 %int_8
        %142 = OpIAdd %int %137 %141
        %143 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %144 = OpLoad %uint %143
        %145 = OpBitcast %int %144
        %146 = OpIAdd %int %142 %145
        %147 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_1
        %148 = OpLoad %uint %147
        %149 = OpBitcast %int %148
        %150 = OpIMul %int %149 %int_65536
        %151 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %152 = OpLoad %uint %151
        %153 = OpBitcast %int %152
        %154 = OpIMul %int %153 %int_8192
        %155 = OpIAdd %int %150 %154
        %156 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %157 = OpLoad %uint %156
        %158 = OpBitcast %int %157
        %159 = OpIMul %int %158 %int_1024
        %160 = OpIAdd %int %155 %159
        %161 = OpLoad %int %_92
        %162 = OpIAdd %int %160 %161
        %163 = OpIAdd %int %162 %int_1
        %164 = OpAccessChain %_ptr_Uniform_float %_12 %int_0 %163
        %165 = OpLoad %float %164
        %166 = OpAccessChain %_ptr_Workgroup_float %_33 %int_0 %146
               OpStore %166 %165
        %167 = OpLoad %int %_92
        %168 = OpIAdd %int %167 %int_1
        %169 = OpBitwiseAnd %int %168 %int_1
        %170 = OpIMul %int %169 %int_64
        %171 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %172 = OpLoad %uint %171
        %173 = OpBitcast %int %172
        %174 = OpIMul %int %173 %int_8
        %175 = OpIAdd %int %170 %174
        %176 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %177 = OpLoad %uint %176
        %178 = OpBitcast %int %177
        %179 = OpIAdd %int %175 %178
        %180 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_0
        %181 = OpLoad %uint %180
        %182 = OpBitcast %int %181
        %183 = OpIMul %int %182 %int_65536
        %184 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %185 = OpLoad %uint %184
        %186 = OpBitcast %int %185
        %187 = OpIMul %int %186 %int_8192
        %188 = OpIAdd %int %183 %187
        %189 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %190 = OpLoad %uint %189
        %191 = OpBitcast %int %190
        %192 = OpIMul %int %191 %int_1024
        %193 = OpIAdd %int %188 %192
        %194 = OpLoad %int %_92
        %195 = OpIAdd %int %193 %194
        %196 = OpIAdd %int %195 %int_1
        %197 = OpAccessChain %_ptr_Uniform_float %_13 %int_0 %196
        %198 = OpLoad %float %197
        %199 = OpAccessChain %_ptr_Workgroup_float %_34 %int_0 %179
               OpStore %199 %198
               OpStore %_133 %int_0
               OpBranch %201
        %201 = OpLabel
               OpLoopMerge %203 %204 None
               OpBranch %205
        %205 = OpLabel
        %206 = OpLoad %int %_133
        %207 = OpSLessThan %bool %206 %int_8
               OpBranchConditional %207 %202 %203
        %202 = OpLabel
               OpStore %_139 %int_0
               OpBranch %209
        %209 = OpLabel
               OpLoopMerge %211 %212 None
               OpBranch %213
        %213 = OpLabel
        %214 = OpLoad %int %_139
        %215 = OpSLessThan %bool %214 %int_8
               OpBranchConditional %215 %210 %211
        %210 = OpLabel
        %216 = OpLoad %int %_133
        %217 = OpIMul %int %216 %int_8
        %218 = OpLoad %int %_139
        %219 = OpIAdd %int %217 %218
        %220 = OpLoad %int %_92
        %221 = OpBitwiseAnd %int %220 %int_1
        %222 = OpIMul %int %221 %int_64
        %223 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %224 = OpLoad %uint %223
        %225 = OpBitcast %int %224
        %226 = OpIMul %int %225 %int_8
        %227 = OpIAdd %int %222 %226
        %228 = OpLoad %int %_133
        %229 = OpIAdd %int %227 %228
        %230 = OpAccessChain %_ptr_Workgroup_float %_33 %int_0 %229
        %231 = OpLoad %float %230
        %232 = OpLoad %int %_92
        %233 = OpBitwiseAnd %int %232 %int_1
        %234 = OpIMul %int %233 %int_64
        %235 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %236 = OpLoad %uint %235
        %237 = OpBitcast %int %236
        %238 = OpIMul %int %237 %int_8
        %239 = OpIAdd %int %234 %238
        %240 = OpLoad %int %_139
        %241 = OpIAdd %int %239 %240
        %242 = OpAccessChain %_ptr_Workgroup_float %_34 %int_0 %241
        %243 = OpLoad %float %242
        %244 = OpFMul %float %231 %243
        %245 = OpAccessChain %_ptr_Function_float %_28 %int_0 %219
        %246 = OpLoad %float %245
        %247 = OpFAdd %float %246 %244
        %248 = OpAccessChain %_ptr_Function_float %_28 %int_0 %219
               OpStore %248 %247
               OpBranch %212
        %212 = OpLabel
        %249 = OpLoad %int %_139
        %250 = OpIAdd %int %249 %int_1
               OpStore %_139 %250
               OpBranch %209
        %211 = OpLabel
               OpBranch %204
        %204 = OpLabel
        %251 = OpLoad %int %_133
        %252 = OpIAdd %int %251 %int_1
               OpStore %_133 %252
               OpBranch %201
        %203 = OpLabel
               OpBranch %126
        %126 = OpLabel
        %253 = OpLoad %int %_92
        %254 = OpIAdd %int %253 %int_1
               OpStore %_92 %254
               OpBranch %123
        %125 = OpLabel
               OpControlBarrier %uint_2 %uint_2 %uint_264
               OpStore %_171 %int_0
               OpBranch %256
        %256 = OpLabel
               OpLoopMerge %258 %259 None
               OpBranch %260
        %260 = OpLabel
        %261 = OpLoad %int %_171
        %262 = OpSLessThan %bool %261 %int_8
               OpBranchConditional %262 %257 %258
        %257 = OpLabel
               OpStore %_177 %int_0
               OpBranch %264
        %264 = OpLabel
               OpLoopMerge %266 %267 None
               OpBranch %268
        %268 = OpLabel
        %269 = OpLoad %int %_177
        %270 = OpSLessThan %bool %269 %int_8
               OpBranchConditional %270 %265 %266
        %265 = OpLabel
        %271 = OpLoad %int %_171
        %272 = OpIMul %int %271 %int_8
        %273 = OpLoad %int %_177
        %274 = OpIAdd %int %272 %273
        %275 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %276 = OpLoad %uint %275
        %277 = OpBitcast %int %276
        %278 = OpIMul %int %277 %int_8
        %279 = OpLoad %int %_171
        %280 = OpIAdd %int %278 %279
        %281 = OpIAdd %int %280 %int_64
        %282 = OpAccessChain %_ptr_Workgroup_float %_33 %int_0 %281
        %283 = OpLoad %float %282
        %284 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %285 = OpLoad %uint %284
        %286 = OpBitcast %int %285
        %287 = OpIMul %int %286 %int_8
        %288 = OpLoad %int %_177
        %289 = OpIAdd %int %287 %288
        %290 = OpIAdd %int %289 %int_64
        %291 = OpAccessChain %_ptr_Workgroup_float %_34 %int_0 %290
        %292 = OpLoad %float %291
        %293 = OpFMul %float %283 %292
        %294 = OpAccessChain %_ptr_Function_float %_28 %int_0 %274
        %295 = OpLoad %float %294
        %296 = OpFAdd %float %295 %293
        %297 = OpAccessChain %_ptr_Function_float %_28 %int_0 %274
               OpStore %297 %296
               OpBranch %267
        %267 = OpLabel
        %298 = OpLoad %int %_177
        %299 = OpIAdd %int %298 %int_1
               OpStore %_177 %299
               OpBranch %264
        %266 = OpLabel
               OpBranch %259
        %259 = OpLabel
        %300 = OpLoad %int %_171
        %301 = OpIAdd %int %300 %int_1
               OpStore %_171 %301
               OpBranch %256
        %258 = OpLabel
               OpStore %_204 %int_0
               OpBranch %303
        %303 = OpLabel
               OpLoopMerge %305 %306 None
               OpBranch %307
        %307 = OpLabel
        %308 = OpLoad %int %_204
        %309 = OpSLessThan %bool %308 %int_8
               OpBranchConditional %309 %304 %305
        %304 = OpLabel
               OpStore %_210 %int_0
               OpBranch %311
        %311 = OpLabel
               OpLoopMerge %313 %314 None
               OpBranch %315
        %315 = OpLabel
        %316 = OpLoad %int %_210
        %317 = OpSLessThan %bool %316 %int_8
               OpBranchConditional %317 %312 %313
        %312 = OpLabel
        %322 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_1
        %323 = OpLoad %uint %322
        %324 = OpBitcast %int %323
        %325 = OpIMul %int %324 %int_65536
        %326 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_1
        %327 = OpLoad %uint %326
        %328 = OpBitcast %int %327
        %329 = OpIMul %int %328 %int_8192
        %330 = OpIAdd %int %325 %329
        %331 = OpLoad %int %_204
        %332 = OpIMul %int %331 %int_1024
        %333 = OpIAdd %int %330 %332
        %334 = OpAccessChain %_ptr_Input_uint %gl_WorkGroupID %uint_0
        %335 = OpLoad %uint %334
        %336 = OpBitcast %int %335
        %337 = OpIMul %int %336 %int_64
        %338 = OpIAdd %int %333 %337
        %339 = OpAccessChain %_ptr_Input_uint %gl_LocalInvocationID %uint_0
        %340 = OpLoad %uint %339
        %341 = OpBitcast %int %340
        %342 = OpIMul %int %341 %int_8
        %343 = OpIAdd %int %338 %342
        %344 = OpLoad %int %_210
        %345 = OpIAdd %int %343 %344
        %346 = OpLoad %int %_204
        %347 = OpIMul %int %346 %int_8
        %348 = OpLoad %int %_210
        %349 = OpIAdd %int %347 %348
        %350 = OpAccessChain %_ptr_Function_float %_28 %int_0 %349
        %351 = OpLoad %float %350
        %352 = OpAccessChain %_ptr_Uniform_float %_14 %int_0 %345
               OpStore %352 %351
               OpBranch %314
        %314 = OpLabel
        %353 = OpLoad %int %_210
        %354 = OpIAdd %int %353 %int_1
               OpStore %_210 %354
               OpBranch %311
        %313 = OpLabel
               OpBranch %306
        %306 = OpLabel
        %355 = OpLoad %int %_204
        %356 = OpIAdd %int %355 %int_1
               OpStore %_204 %356
               OpBranch %303
        %305 = OpLabel
               OpReturn
               OpFunctionEnd
```

</details><br>

{::options parse_block_html="false" /}

### 将生成的 spv 文件放回测试 

**这一步很重要**

`b.txt` 中的

```
OpEntryPoint GLCompute %main "main" %gl_LocalInvocationID %gl_WorkGroupID
```

需要改成

```
OpEntryPoint GLCompute %main "default_function_kernel0" %gl_WorkGroupID %gl_LocalInvocationID
```

也是跟 `a.txt` 里这个接口函数名保持一致。然后再 `spirv-as --target-env spv1.0 .\b.txt -o bb.spv` 把它汇编为 `bb.spv`。


现在 `a.spv` 和 `bb.spv` 的程序逻辑、接口名称应该是完全一致的，所以理论上可以把 `bb.spv` 的内容替换到这个 `test_gemm.py` 中去跑一跑试试。改一改前面导出 `a.spv` 用到的代码就可以方便的做到了。

```py
@tvm.register_func("tvm_callback_vulkan_postproc")
def ff(spv):
    return open('bb.spv', 'rb').read()
    # open('a.spv', 'wb').write(spv)
    # return spv
```

现在测试完全可以通过了！


## 定位错误点，明确修复目标

到这里我们甚至可以跑对了，总结流程

1. 假设 `a.spv` 本身有问题，得到不合法 GLSL 代码 `a.glsl`
2. 通过分析将 `a.glsl`，进行改正，得到 `b.glsl`，然后对其汇编得到 `b.spv` 以及反汇编文本 `b.txt`
3. 修改文本 `b.txt` 里的接口函数名，并汇编得到 `bb.spv`
4. 将 `bb.spv` 替换进原测试流程，测试通过

现在只需要对比 `a.txt` 和 `b.txt` 两个文件 shared memory 相关部分，很可能就得出结论了，我把关键的部分摘取出来

- `a.txt`

  ```
        ; ...
        OpDecorate %_arr_float_uint_128 ArrayStride 4
        OpMemberDecorate %_struct_31 0 Offset 0
        OpDecorate %_struct_31 Block
        ; ...
  %_arr_float_uint_128 = OpTypeArray %float %uint_128
  %_struct_31 = OpTypeStruct %_arr_float_uint_128
  %_ptr_Workgroup__struct_31 = OpTypePointer Workgroup %_struct_31
        %33 = OpVariable %_ptr_Workgroup__struct_31 Workgroup
        %34 = OpVariable %_ptr_Workgroup__struct_31 Workgroup
        ; ...
  ```

- `b.txt`

  ```
    ; ...
    OpName %_31 "_31"
    OpMemberName %_31 0 "_m0"
    ; ...
    %uint_128 = OpConstant %uint 128
%_arr_float_uint_128 = OpTypeArray %float %uint_128
    %_31 = OpTypeStruct %_arr_float_uint_128
%_ptr_Workgroup__31 = OpTypePointer Workgroup %_31
    %_33 = OpVariable %_ptr_Workgroup__31 Workgroup
    ; ...
    %_34 = OpVariable %_ptr_Workgroup__31 Workgroup
    ; ...
  ```

`OpName`，`OpMemberName`之类的多半只是一些 metadata，不影响程序逻辑和结构。我个人第一眼主要觉得 `a.txt` 中的 `OpDecorate %_struct_31 Block`可能有问题，因为只有这一行是最看不懂的。结合[SPIRV文档](https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html#Decoration)

> **Block**
>
> Apply only to a structure type to establish it is a non-SSBO-like shader-interface block.

和[GLSL文档](https://www.khronos.org/opengl/wiki/Interface_Block_(GLSL))

> An **Interface Block** is a group of GLSL input, output, uniform, or storage buffer variables. 

这个 Block Decoration 很可能是多余的。根据这个假设，修改 `a.txt`，删除两行

```
OpDecorate %_struct_26 Block
OpDecorate %_struct_31 Block
```

汇编，替换，运行，测试通过！

那么接下来的修复目标就是修改 TVM 的 codegen，使得产生的 spirv 代码中，非 `interface block` 类型的变量不要有 Block Decoration。

## 修复和总结

根据上面的结论，[修复的 PR](https://github.com/apache/tvm/pull/8102) 已经提交并 merge。

最开始遇到这个问题之后，完全没有能够定位到问题的半点思路，即使假设生成的 spirv 有问题也并不知道如何调试。spirv 相比 cuda，代码不是给人看的，Vulkan 的调用接口也很复杂。所以在挣扎两天之后还是上论坛求助了，然而并不能得到帮助。

其实最大的难点在于，人类很难通过阅读 spirv 文档就学会 spirv。

后来翻文档的时候突然发现，spirv 用到的很多术语和 glsl 很相似，和 hlsl 有些区别，也许可以通过 glsl 可以帮助我理解 spirv，反过来 vulkan 里的错误也许在 glsl 里也能得到揭示。

抱着试一试的心态，我尝试看看 spv 转出的 GLSL 代码，还真就在 shared memory 相关的地方有问题，然后手动改正后放入测试流程里，瞬间就搞定了。

其实整个过程还有别的困难，比如一开始没注意到 GLSL 的函数入口名默认为`main`，但 TVM 需要的入口名为`default_function_kernel0`，还想了老半天才反应过来可以再把 spv 翻译回文本格式，改完再汇编回去。而更多的困难来自于对于 Vulkan、GLSL 这些东西的未知，每进行一步都需要翻看文档，如何从海量文档中切割出最需要的部分也是难点。

另一个头疼的问题是，一开始我是用的 `get_source`，而使用 `spirv-as` 工具时没有指定 spv1.0 版本，默认汇编成了 1.5 版本，而基于这个版本的 spv 还有其他的问题，妨碍了我的调试进程。

吐槽一下吧

- `spirv-as` 是真的不看开头的几行注释（写了spv1.0版本，但是忽略掉了）
- `spirv-val` 不能检查出本文所说的错误，即给非 interface block 变量进行了 Block 装饰
- Vulkan 驱动也不认为这是个问题输入，有的驱动甚至还跑对了，这不是误导人吗

不论是这个问题的产生还是解决，都可以用机缘巧合四个字来概括。

---

调试的过程也能学到很多知识，比如 cuda 代码中的 raw pointer 传参、shared memory、local memory 和 vulkan 中 descriptor set、workgroup variable、function variable 的、和 glsl 中 interface block、shared variable、local variable 之间的对应关系，Vulkan SDK 里有一大堆好用的工具。

很久（一年多）都没有写文章了，主要是平时学习工作都比较零散（其实以前写的时候也很零散）。不过这次 debug 还是很有意思的，感觉可以记录一下。

怎么说呢，debug 很刺激。心境可能从一开始的失望，然后心信满满开始 debug，然后逐渐绝望，在求助无果后陷入抓狂，最后柳暗花明，到现在终于可以说：“舒服了”。