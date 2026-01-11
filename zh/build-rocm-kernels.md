---
title: "使用 Hugging Face 轻松构建并共享 ROCm 内核"
thumbnail: /blog/assets/build-rocm-kernels/thumbnail.png
authors:
- user: badaoui
- user: daniehua
- user: ColorsWind
- user: ftyghome
translators:
- user: chenglu
---

# 使用 Hugging Face 轻松构建并共享 ROCm 内核

![Easily Build and Share ROCm Kernels with Hugging Face](/blog/assets/build-rocm-kernels/thumbnail.png)

## 简介

自定义内核是高性能深度学习的基础，它让 GPU 操作能完全贴合你的工作负载需求——无论是图像处理、张量变换，还是其他计算密集型任务。然而，要为正确的架构编译这些内核、配置各种编译标志并干净地整合到 PyTorch 扩展中，往往会变成一团乱麻（CMake/Nix、编译错误、ABI 问题等）。
Hugging Face 提供的 [**kernel-builder**](https://github.com/huggingface/kernel-builder) 和 [**kernels**](https://github.com/huggingface/kernels) 库，让你能轻松地在 [**kernels-community**](https://huggingface.co/kernels-community) 上分享这些自定义内核，支持多种 GPU 和加速器后端，包括 CUDA、ROCm、Metal 和 XPU。这保证了你的内核既高效、又可移植，还能无缝集成到 PyTorch 中。

本文专注于构建 **ROCm 兼容内核**，展示如何使用 [kernel-builder](https://github.com/huggingface/kernel-builder/tree/main) 进行构建、测试与分享。你将学习如何在 AMD GPU 上高效运行自定义内核，以及可复现性、打包与部署的最佳实践。

本文是针对 ROCm 的简化版教程。如果你想了解 CUDA 相关内容，可参阅原文：[A Guide to Building and Scaling Production-Ready CUDA Kernels](https://huggingface.co/blog/kernel-builder)。

## 构建步骤

我们以 [RadeonFlow_Kernels](https://github.com/RadeonFlow/RadeonFlow_Kernels) 中的 GEMM 内核为例。若想直接查看教程，可[点击此处](#step-1-project-structure)。

### 关于这个内核

> [!NOTE]
> 本节由 **RadeonFlow GEMM** 内核作者撰写。
> 作者：[ColorsWind](https://huggingface.co/ColorsWind)、[Zesen Liu](https://huggingface.co/ftyghome)、[Andy](https://huggingface.co/jpy794)

**RadeonFlow GEMM** 内核是一个针对 AMD Instinct MI300X GPU 优化的高性能 FP8 分块矩阵乘法实现。
GEMM（通用矩阵乘法）是大多数深度学习计算的核心：给定矩阵 A 与 B，计算它们的乘积 C = A × B。
该实现使用 **FP8**（低精度浮点格式），以少量精度换取更高的吞吐量和更低的显存带宽需求。此内核为 [AMD Developer Challenge 2025](https://www.datamonsters.com/amd-developer-challenge-2025) 开发，并在 2025 年 6 月获得 🏆 **特等奖**，以表彰其在 AMD 硬件上的性能与创新表现。

该内核使用 `e4m3fnuz` 浮点格式进行量化计算，并通过分块缩放保持低精度计算的准确性。`e4m3fnuz` 是一种 FP8 变体，拥有 4 位指数和 3 位尾数，专为神经网络任务设计。由于 FP8 的动态范围较小，我们为每个块应用缩放因子（a_scale 和 b_scale），以在计算前后将数值调整到合理范围，从而尽可能保留精度。

函数接口如下：

```
(a, b, a_scale, b_scale, c)
```

参数含义：

* `a`: 输入矩阵 A，大小为 K × M，类型 e4m3fnuz
* `b`: 输入矩阵 B，大小为 K × N，类型 e4m3fnuz
* `a_scale`: 大小 (K // 128) × M，类型 fp32
* `b_scale`: 大小 (K // 128) × (N // 128)，类型 fp32
* `c`: 输出矩阵，大小 M × N，类型 bf16

该内核针对特定矩阵形状进行了预编译，并假设内存为转置布局（比赛要求）。若需支持更多形状或布局，需要修改启动代码。

现在我们已有一个高性能 ROCm 内核，接下来要做的是：**如何将它整合到 PyTorch 中并分享？**
接下来我们将利用 `kernel-builder` 与 `kernels` 进行项目结构化、构建与发布。

> [!NOTE]
> 本教程技术性较强，但你可以照着一步步操作，无需理解所有细节，也能顺利运行。想深入学习时可随时回头阅读。

### 步骤 1：项目结构

Hugging Face 的 Kernel Builder 期望项目文件按以下方式组织：

```
gemm/
├── build.toml
├── gemm
│   └── gemm_kernel.h
├── flake.nix
└── torch-ext
    ├── torch_binding.cpp
    ├── torch_binding.h
    └── gemm
        └── __init__.py
```

* **build.toml**：项目构建配置文件，定义整个编译过程。
* **gemm/**：GPU 源码目录。
* **flake.nix**：可复现构建环境配置。
* **torch-ext/**：PyTorch 扩展的 Python 封装。

实际项目可能还包含测试、脚本等附加文件，可自由添加。
在本文示例中，结构如下：

```
gemm/
├── build.toml
├── gemm
│   ├── gemm_kernel.h
│   ├── gemm_kernel_legacy.h
│   ├── transpose_kernel.h
│   └── gemm_launcher.hip
├── include
│   ├── clangd_workaround.h
│   ├── gpu_libs.h
│   ├── gpu_types.h
│   └── timer.h
├── src/utils
│   ├── arithmetic.h
│   └── timer.hip
├── tests/checker
│   ├── checker.cpp
│   ├── metrics.h
│   └── checker.h
├── flake.nix
└── torch-ext
    ├── torch_binding.cpp
    ├── torch_binding.h
    └── gemm
        └── __init__.py
```

如果你查看 RadeonFlow Kernels 中 GEMM 内核的原始文件，会发现它们是以 `.cpp` 为后缀的 HIP 源文件。
在开始之前，第一步需要根据文件内容和用途，将这些扩展名修改为 `.h` 或 `.hip`：

* 使用 `.h`：适用于包含内核声明、内联函数或模板代码的头文件，这些文件通常会被其他文件引用。
* 使用 `.hip`：适用于包含需要单独编译的 HIP/GPU 实现代码的文件（例如内核启动器、复杂的设备函数等）。

例如：`gemm_kernel.h`、`gemm_kernel_legacy.h`、`transpose_kernel.h` 是头文件，`gemm_launcher.hip` 是实现文件。
这种命名方式有助于 kernel-builder 正确识别和编译。

### 步骤 2：配置文件设置

#### `build.toml` 构建清单

这个文件负责统筹整个构建过程，告诉 kernel-builder 要编译哪些内容以及它们之间如何关联。

```toml
[general]
name = "gemm"
universal = false

[torch]
src = [
  "torch-ext/torch_binding.cpp",
  "torch-ext/torch_binding.h",
]

[kernel.gemm]
backend = "rocm"
rocm-archs = [
    "gfx942",
]

depends = ["torch"]

src = [
  "include/clangd_workaround.h",
  "include/gpu_libs.h",
  "include/gpu_types.h",
  "include/timer.h",
  "gemm/gemm_kernel.h",
  "gemm/gemm_kernel_legacy.h",
  "gemm/gemm_launcher.hip",
  "gemm/transpose_kernel.h",
  "src/utils/arithmetic.h",
  "src/utils/timer.hip",
  "tests/checker/metrics.h",
]

include = ["include"]
```

**general**

这一部分定义项目的基本配置。

* **name**（必填）：项目名称，应与内核名一致，也会作为 Python 包名使用。
* **universal**（可选）：设为 `true` 时表示该内核是通用内核（纯 Python 实现，无需编译）。通用内核不会使用下方的其他配置部分。典型示例是 Triton 内核。默认值：`false`。

**torch**

这一部分描述 PyTorch 扩展的配置，用于定义将内核暴露给 PyTorch 的 Python 绑定接口。

* **src**（必填）：列出用于构建 PyTorch 扩展的源文件与头文件。在这里，我们包含创建 Python 接口的 C++ 绑定文件。

**kernel.gemm**

定义名为 “gemm” 的内核。若项目中包含多个内核，可在同一个 `build.toml` 文件中添加多个 `[kernel.xxx]` 部分。

* **backend**（必填）：计算后端类型，这里使用 “rocm” 表示 AMD GPU。
* **rocm-archs**（ROCm 必填）：指定编译目标的 ROCm 架构列表，例如 “gfx942” 对应 MI300 系列 GPU。
* **depends**（必填）：依赖项列表。此处依赖 “torch”，以便使用 PyTorch 张量操作。
* **include**（可选）：相对项目根目录的头文件搜索路径，方便编译器查找依赖。

#### `flake.nix` 可复现性配置文件

为了让任何人都能在任意机器上构建你的内核，我们使用 `flake.nix` 文件来锁定 kernel-builder 及其依赖的确切版本。（可以直接复制下面的示例并修改描述即可）

```nix
{
  description = "Flake for GEMM kernel";

  inputs = {
    kernel-builder.url = "github:huggingface/kernel-builder";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:

    kernel-builder.lib.genFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
```

#### 编写内核

接下来是 GPU 代码。在 `gemm/gemm_launcher.hip` 文件中，我们定义 GEMM 内核的启动逻辑。
根据配置，程序会选择调用新的优化版 `gemm/gemm_kernel`，或在必要时回退到旧版 `gemm/gemm_kernel_legacy`。

```C
// ... previous includes and definitions
extern "C" void run(
    void *a, void *b, void *as, void *bs, void *c,
    int m, int n, int k,
    PerfMetrics *metrics, hipStream_t job_stream0
) {
    const __FP8_TYPE *a_ptr = static_cast<const __FP8_TYPE *>(a);
    const __FP8_TYPE *b_ptr = static_cast<const __FP8_TYPE *>(b);
    __BF16_TYPE *c_ptr = static_cast<__BF16_TYPE *>(c);
    const float *as_ptr = static_cast<const float *>(as);
    const float *bs_ptr = static_cast<const float *>(bs);

    KernelTimerScoped timer(timers, 2LL * m * n * k,
        metrics ? &metrics->entries[0].time : nullptr,
        metrics ? &metrics->entries[0].gflops : nullptr, job_stream0);

    // Dispatch GEMM to the fastest available implementation
    switch (pack_shape(m, n, k)) {
        DISPATCH_GEMM(1024, 1536, 7168, 256, 128, 128, 4, 2, 512, 4, 16);
        DISPATCH_GEMM(6144, 7168, 2304, 256, 128, 128, 4, 2, 512, 1, 16);
        default: {
            printf("Error: Unsupported shape M=%d, K=%d, N=%d\n", m, k, n);
            abort();
        }
    }
}
// ...
```

#### 注册原生 PyTorch 运算符

这一步非常关键。我们不仅是让函数能在 Python 中被调用，而是要把它注册为 **原生 PyTorch 运算符**，让它成为 `torch.ops` 下的一等成员。

`torch-ext/torch_binding.cpp` 文件负责这个注册过程。

```C
#include <torch/all.h>
#include <torch/library.h>
#include <hip/hip_runtime.h>

#include "registration.h"
#include "torch_binding.h"

// Forward declaration of the C function from gemm_launcher.hip
extern "C" {
    struct PerfMetrics;
    void run(void *a, void *b, void *as, void *bs, void *c, int m, int n, int k, PerfMetrics *metrics, hipStream_t job_stream0);
}

void gemm(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b, 
          torch::Tensor const &as, torch::Tensor const &bs) {
    
    // Validate tensor properties
    TORCH_CHECK(a.device().is_cuda(), "Input tensor a must be on GPU device");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor b must be on GPU device");
    TORCH_CHECK(as.device().is_cuda(), "Scale tensor as must be on GPU device");
    TORCH_CHECK(bs.device().is_cuda(), "Scale tensor bs must be on GPU device");
    TORCH_CHECK(out.device().is_cuda(), "Output tensor out must be on GPU device");
    
    TORCH_CHECK(a.is_contiguous(), "Input tensor a must be contiguous");
    TORCH_CHECK(b.is_contiguous(), "Input tensor b must be contiguous");
    TORCH_CHECK(as.is_contiguous(), "Scale tensor as must be contiguous");
    TORCH_CHECK(bs.is_contiguous(), "Scale tensor bs must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "Output tensor out must be contiguous");
    
    // Get matrix dimensions from tensor shapes
    // Assuming a is [M, K], b is [K, N], out is [M, N]
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(1);
    
    TORCH_CHECK(b.size(0) == K, "Matrix dimensions mismatch: a.size(1) != b.size(0)");
    TORCH_CHECK(out.size(0) == M, "Output tensor dimension mismatch: out.size(0) != M");
    TORCH_CHECK(out.size(1) == N, "Output tensor dimension mismatch: out.size(1) != N");
    
    // Use default HIP stream (stream 0)
    const hipStream_t stream = 0;
    
    // Call the C function
    run(a.data_ptr(), b.data_ptr(), as.data_ptr(), bs.data_ptr(), out.data_ptr(),
        M, N, K, nullptr, stream);
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("gemm(Tensor! out, Tensor a, Tensor b, Tensor a_scale, Tensor b_scale) -> ()");
  ops.impl("gemm", torch::kCUDA, &gemm);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
```
`torch_binding.h` 文件用于声明函数。例如，对于 `gemm` 内核，其在 `torch_binding.h` 中的函数声明如下：

```h
#pragma once

#include <torch/torch.h>

void gemm(torch::Tensor &out, torch::Tensor const &a, torch::Tensor const &b, 
          torch::Tensor const &as, torch::Tensor const &bs);
```

#### 创建 `__init__.py` 封装

在 `torch-ext/gemm/` 目录下，需要一个 `__init__.py` 文件，使其成为 Python 包，并以更易用的方式暴露自定义运算符。

```python
from typing import Optional
import torch
from ._ops import ops

def gemm(a: torch.Tensor, b: torch.Tensor, as_: torch.Tensor, bs: torch.Tensor, 
         out: Optional[torch.Tensor] = None) -> torch.Tensor:
         
    if out is None:
        # Create output tensor with appropriate shape and dtype
        M, K = a.shape
        K_b, N = b.shape
        assert K == K_b, f"Matrix dimension mismatch: A has {K} cols, B has {K_b} rows"
        
        # Output should be BF16 type on the same device as inputs
        out = torch.empty((M, N), dtype=torch.bfloat16, device=a.device)
    
    ops.gemm(out, a, b, as_, bs)
    return out
```

### 步骤 3：构建内核

`kernel-builder` 使用 **Nix** 来构建内核。只要你的系统中安装了 Nix，就可以直接编译或运行这些内核。推荐的安装方式如下：

* **Linux**：使用 [官方 Nix 安装器](https://nixos.org/download/)。
* **macOS**：使用 [Determinate Nix 安装器](https://docs.determinate.systems/determinate-nix/)。另外，目前构建内核还需要 Xcode 16.x。

#### 开始使用 Nix

首先，运行以下命令：

```bash
nix flake update
```

该命令会生成一个 `flake.lock` 文件，用于锁定 kernel-builder 及其所有依赖的确切版本。
请将 `flake.nix` 和 `flake.lock` 一并提交到仓库中，以确保构建结果在任何环境下都可复现。

由于 kernel-builder 依赖许多软件包（例如不同版本的 PyTorch），建议启用 Hugging Face 的缓存以避免重复构建，节省大量时间：

```bash
# Install cachix and configure the cache
cachix use huggingface
```

或者如果你不想永久安装 cachix，可以仅临时启用一次：

```bash
# Use cachix without installing it
nix run nixpkgs#cachix -- use huggingface

#### 使用 Nix 构建内核

如果项目中包含 `flake.nix` 文件，可以直接通过以下命令构建：

```bash
cd Build_RadeonFlow_Kernels/gemm
nix build . -L
```

编译完成后，生成的内核文件会保存在本地的 `build/` 目录中。


#### 本地开发环境（Development Shell）

`kernel-builder` 提供了适用于开发的 Shell 环境。在这种环境中，所有依赖项都会被自动配置好，同时提供 `build2cmake` 工具用于生成 CMake 项目文件：

```bash
$ nix develop
$ build2cmake generate-torch build.toml
$ cmake -B build-ext
$ cmake --build build-ext
```

如果你想将内核作为 Python 包进行测试，也可以直接在这个环境中完成。`nix develop` 会自动在当前目录下创建并激活一个虚拟环境 `.venv`：

```bash
$ nix develop
$ build2cmake generate-torch build.toml
$ pip install --no-build-isolation -e .
```

开发环境可针对不同的构建配置进行切换。
例如，若你想在 **Torch 2.7 + ROCm 6.3** 环境中进行开发，可使用以下命令进入对应的 Shell：

```bash
$ rm -rf .venv  # Remove existing venv if any
$ nix develop .#devShells.torch27-cxx11-rocm63-x86_64-linux
```

这样，你就能在完整的可复现环境中构建、调试并测试你的 ROCm 内核。

### 步骤 4：上传内核到 Hugging Face Hub

现在我们已经完成了内核的构建，接下来可以进行测试并将其上传到 Hugging Face Hub。

#### 为所有 PyTorch 和 ROCm 版本构建内核

在分享之前，先清理掉构建过程中生成的临时文件和开发产物，以避免上传不必要的文件：

```bash
build2cmake clean build.toml 
```

为了同时构建所有支持的 PyTorch 和 ROCm 版本，可以使用 kernel-builder 工具自动完成这一过程：

```bash
# Outside of the dev shell, run the following command
# if you are inside of the sandbox you can leave with `exit`
nix build . -L
``` 

> **注意：**
> 这个过程可能会花费较长时间，因为它会为所有支持的 PyTorch 与 ROCm 版本进行编译。
> 构建结果会输出到 `result` 目录中。

最后，将构建结果移动到项目中预期的 `build/` 目录中（`kernels` 库会从这里读取编译好的文件）：

```bash
mkdir -p build
rsync -av --delete --chmod=Du+w,Fu+w result/ build/
```

#### 推送到 Hugging Face Hub

将构建产物推送到 Hugging Face Hub 后，其他开发者就能直接下载并使用你的内核。

首先，创建一个新的仓库：

```bash
hf repo create gemm
```

> 请确保已使用 `huggingface-cli login` 登录 Hugging Face 账户。

然后在项目目录中，将本地项目与刚创建的仓库连接并推送代码：

```bash
# Initialize git and connect to the Hugging Face Hub
git init
git remote add origin https://huggingface.co/<your-username>/gemm

# Pull the changes (just the default .gitattributes file)
git pull origin main
git xet install
git checkout -b main

# Update to use Xet for the binary files
git xet track "*.so"

# Add and commit your changes (being careful to only include the necessary files
# since our build2cmake command generated a lot of dev-specific files)
git add \
  build/ gemm/ include/ src/utils tests/checker \
  torch-ext/torch_binding.cpp torch-ext/torch_binding.h torch-ext/gemm \
  flake.nix flake.lock build.toml

git commit -m "feat: Created a compliant gemm kernel"
git push -u origin main
```ush -u origin main
```

🎉 **完成！** 你的内核现已成功上传到 Hugging Face Hub。其他开发者可以直接使用它，并且它已完全符合 `kernels` 库的规范。

### 步骤 5：让我们来使用它吧 :)

在 **kernels** 库中，你并不需要像传统方式那样“安装”内核，而是可以直接从 Hugging Face Hub 加载它。加载后，内核会自动注册为新的 PyTorch 运算符。

```python
import torch
from kernels import get_kernel

# Load the kernel from the Hub
gemm = get_kernel("kernels-community/gemm")

# Matrix dimensions (must be supported - see gemm_launcher.cpp)
M, N, K = 1024, 1536, 7168
QUANT_SIZE = 128

# Setup device
device = torch.device("cuda")

# Create inputs - kernel expects A:(K,M), B:(K,N)
A_fp32 = torch.randn(M, K, device=device)
B_fp32 = torch.randn(K, N, device=device)

# Convert to FP8
A_fp8 = A_fp32.to(torch.float8_e4m3fnuz)
B_fp8 = B_fp32.to(torch.float8_e4m3fnuz)

# Create scale factors (uniform scaling)
A_scale = torch.ones(K // QUANT_SIZE, M, device=device, dtype=torch.float32)
B_scale = torch.ones(K // QUANT_SIZE, N // QUANT_SIZE, device=device, dtype=torch.float32)

C = torch.zeros(M, N, device=device, dtype=torch.bfloat16)

# Use the kernel
result = gemm.gemm(A_fp8, B_fp8, A_scale, B_scale, C)
```

就是这样！🎉
你的 ROCm 内核现在已经可以直接从 Hugging Face Hub 使用啦。

## 总结

借助 Hugging Face 提供的工具，构建和分享 ROCm 内核变得前所未有的简单。
通过 Nix 实现的可复现构建流程与 PyTorch 的无缝集成，开发者可以把更多精力投入到性能优化上，而不再被环境配置困扰。

一旦构建完成，只需上传到 Hugging Face Hub，社区中的其他人就能直接使用你的自定义内核——几行代码即可完成集成，轻松共享高性能成果。🚀

## 相关库与资源

* [kernel-builder](https://github.com/huggingface/kernel-builder) —— 用于构建与编译自定义内核的工具
* [kernels](https://github.com/huggingface/kernels) —— 用于从 Hub 管理与加载内核的库
* [Kernels Community Hub](https://huggingface.co/kernels-community) —— 发现与分享社区自定义内核的平台

