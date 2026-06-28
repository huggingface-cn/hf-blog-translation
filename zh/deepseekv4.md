---
title: "DeepSeek-V4：智能体真正可用的百万 token 上下文"
thumbnail: /blog/assets/deepseekv4/thumbnail.png
authors:
- user: burtenshaw
translators:
- user: HCS9527
---

# DeepSeek-V4：智能体真正可用的百万 token 上下文

DeepSeek 今天发布了 V4。Hub 上已有两个 MoE 检查点：DeepSeek-V4-Pro 总参数量为 1.6T、激活参数量为 49B；DeepSeek-V4-Flash 总参数量为 284B、激活参数量为 13B。二者都拥有 1M token 上下文窗口。它们的基准测试分数有竞争力，但不是 SOTA。这并不重要。真正的创新在于 DeepSeek v4 如何为高效支持大上下文长度而设计，也因此成为智能体任务的最佳候选模型之一。

更重要的是，V4 面向长时间运行的智能体工作负载。今天，把一个前沿开放模型作为智能体运行时，常会以可预见的方式出问题：模型停住，需要重新发 prompt；轨迹超出了上下文预算，KV cache 填满了 GPU，或工具调用的往返交互在长任务执行到一半时开始劣化。**V4 正是为修复这些已知故障而构建的**，也为社区后续工作指出了方向。

本文会介绍三件事：这个架构为了降低长上下文推理成本做了哪些不同设计；叠加在架构之上的、面向智能体的后训练决策；以及论文中一些有助于理解这些变化的要点。

## 智能体的 KV cache 问题

1M 上下文窗口只是容量，不等于性能。能否真正使用它，取决于在这种深度下每次前向传播的成本。对于运行长工具使用轨迹的智能体（例如 SWE-bench 任务、多步骤浏览会话，或包含数百条命令的终端会话），每个工具结果都会被追加到上下文中，而之后生成的每个 token 都要对此前全部内容计算完整注意力。

有两个数字很关键：单 token 推理 FLOPs 和 KV cache 大小。二者都会随序列长度增长。在 1M token 下，与 DeepSeek-V3.2 相比，DeepSeek-V4-Pro 只需要 27% 的单 token 推理 FLOPs，因此能在相同硬件上运行得更快。它也只使用 10% 的 KV cache 内存。V4-Flash 将这些数字进一步降低：FLOPs 为 10%，KV cache 为 7%。

如果拿 KV cache 内存与一种成熟架构比较，例如使用 8 个头、并以常见 bfloat16 格式存储的 grouped query attention（分组查询注意力），那么 DeepSeek v4 大约只需要 2% 的 cache 大小。这让它更容易部署到需要处理超大上下文的场景中。

![DeepSeek-V4 技术报告图 1，左侧为基准测试，右侧为推理 FLOPs 与 KV cache 缩放](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig1_efficiency.png)
*图 1：基准测试对比（左），每 token FLOPs 以及随序列长度累积的 KV cache（右）。*

## 混合注意力：CSA 和 HCA

效率提升来自把注意力拆成两种机制，并在不同层之间交错使用。

**Compressed Sparse Attention（CSA，压缩稀疏注意力）** 使用带可学习位置偏置的 softmax 门控池化，在序列维度上将 KV 条目压缩 4 倍。lightning indexer（FP4、ReLU 打分的多头点积）会为每个 query（查询向量）选择 top-k 压缩块。它继承了 DeepSeek Sparse Attention 在 V3.2 中的稀疏选择思路，但运行在已经比原始序列短 4 倍的块上。因此 indexer 的搜索空间也随之缩小。

![图 3：Compressed Sparse Attention，展示 compressor、在压缩块上运行的 lightning indexer，以及 sliding-window 分支](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig3_csa.png)
*图 3：CSA。compressor 将每 4 个 token 折叠为一个压缩 KV 条目。lightning indexer 为每个 query 选择 top-k 压缩块。sliding-window 分支处理最近的未压缩 token 。*

**Heavily Compressed Attention（HCA，重压缩注意力）** 将 KV 条目压缩 128 倍，并去掉稀疏选择。每个查询向量都会密集关注每个压缩块。压缩后的序列足够短，因此密集注意力的成本也很低。

![图 4：Heavily Compressed Attention，以 128 倍压缩和密集 MQA 处理压缩块](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig4_hca.png)
*图 4：HCA。更强的 compressor（128x，相比 CSA 的 4x）之后，是对压缩流的密集注意力，并使用同样的 sliding-window 分支保留近期信息。*

各层会在 CSA 与 HCA 之间交替。不同层承载不同注意力模式，如果强行让所有层使用同一种机制，就会浪费容量。在 V4-Pro 的 61 层堆栈中，第 0–1 层是 HCA，第 2–60 层交替使用 CSA 和 HCA，最后的 MTP block 只使用 sliding-window。

两条路径都会对大多数 KV 条目使用 FP8 存储，仅对 RoPE 维度使用 BF16。CSA 内部的 lightning indexer 使用 FP4。这些存储选择与压缩比例叠加，最终得到 2% KV cache 这个数字。

![图 2：整体架构，展示 embedding、混合 CSA/HCA 注意力、DeepSeekMoE、manifold-constrained hyper-connections](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig2_architecture.png)
*图 2：整体架构。注意力层在 CSA 和 HCA 之间交替。前馈层使用 DeepSeekMoE。残差连接被 manifold-constrained hyper-connections（mHC）取代。*

## 智能体会发生什么变化

高效长上下文注意力是智能体工作流的必要条件，但还不够。论文描述了三个直接面向智能体用例的后训练和基础设施选择。

### 跨工具调用的交错思考

V3.2 会在工具结果轮次之间保留推理轨迹，但一旦出现新的用户消息就会丢弃它们。对于处理单个用户轮次的智能体来说，这没问题。但对于多轮智能体工作流来说，如果用户在智能体已经串联多次工具调用后发来追问，模型就会丢失已积累的推理，并不得不重新构建状态。

当对话包含工具调用时，V4 会跨用户消息边界保留推理内容。模型会在所有轮次中保留完整推理历史，包括跨用户轮次的历史。这让长时程智能体任务可以拥有连贯、累积的思维链。对于不使用工具的普通对话，旧行为会保留：每个轮次都会清空推理，以保持上下文简洁。

![图 7：thinking 管理，有工具时（上）会跨轮次保留推理；无工具时（下）会在每条新用户消息处丢弃推理](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig7_thinking.png)
*图 7：带工具的 thinking（上）会跨所有轮次保留推理。不带工具的 thinking（下）会在每条新用户消息处丢弃推理。*

### 使用专用 token 的工具调用 schema

V4 引入了 `|DSML|` 特殊 token 和一种基于 XML 的工具调用格式。与把 JSON 放进字符串的工具调用相比，XML 格式减少了转义失败。这类失败是模型输出嵌套引号内容时常见的问题。

该 schema 将字符串参数（以 `string="true"` 原样传递）与结构化参数（以 `string="false"` 作为 JSON 传递）分开。这消除了一类 JSON 工具调用格式经常遇到的数字和布尔值解析错误。

### DSec：为 RL rollout 构建的沙盒

智能体行为是通过在真实工具环境中进行 RL 训练得到的。论文描述了为此构建的沙盒基础设施。DeepSeek Elastic Compute（DSec）是一个 Rust 平台，通过一个 Python SDK 暴露四类执行后端：函数调用、容器、microVM（Firecracker）和完整 VM（QEMU）。单个集群可以运行数十万个并发沙盒。

有三个 DSec 特性对智能体训练很重要：通过分层 3FS 存储快速加载镜像（这样 RL rollout 不必等待容器启动）、可安全处理中断的轨迹重放（这样被中断的训练步骤可以恢复，而无需重新运行工具调用）、跨执行后端的统一 API（这样训练 harness 可以面向函数调用或完整 VM，而不必重写）。这些基础设施决策支撑了智能体基准测试分数。

## 智能体基准测试结果

知识和推理分数有竞争力，但不是领先水平。智能体分数才是 V4-Pro-Max 与其他模型拉开差距的地方。

![DeepSeek-V4-Pro-Max 与前沿模型的基准测试对比](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/resolve/main/assets/dsv4_performance.png)

表 6 智能体部分的具体数字如下：

- Terminal Bench 2.0：V4-Pro-Max 得分 67.9，领先 GLM-5.1（63.5）和 K2.6（66.7），落后于 GPT-5.4-xHigh（75.1）和 Gemini-3.1-Pro（68.5）。
- SWE Verified：解决率 80.6，距离 Opus-4.6-Max（80.8）不到 1 个百分点，与 Gemini-3.1-Pro（80.6）持平。
- MCPAtlas Public：73.6，仅次于 Opus-4.6-Max（73.8）。
- Toolathlon：51.8，领先 K2.6（50.0）、GLM-5.1（40.7）和 Gemini-3.1-Pro（48.8）。

在论文内部的 R&D 编程基准测试中，30 个精心挑选的任务覆盖 PyTorch、CUDA、Rust 和 C++，V4-Pro-Max 达到 67% 通过率，而 Sonnet 4.5 为 47%，Opus 4.5 为 70%。在一项针对 85 位 DeepSeek 开发者的调查中，他们把 V4-Pro 作为日常主力模型使用，其中 52% 表示它已经可以替代当前的主要编程模型，另有 39% 倾向于认为可以。

长上下文检索数字见图 9。MRCR 8-needle 准确率在 256K token 内保持在 0.82 以上，并在 1M 处保持 0.59。

![图 9：MRCR 8-needle 检索性能，覆盖最高 1M token 的上下文长度](https://huggingface.co/buckets/burtenshaw/deepseek-v4-figures/resolve/v4_fig9_mrcr.png)
*图 9：MRCR 8-needle 检索。V4-Pro-Max 在 256K 内保持 0.82 以上，并在 1M 处保持 0.59。*

## 使用这些模型

Hub 上有四个检查点。instruct 模型对 MoE 专家权重使用 FP4，对其他所有部分使用 FP8。base 模型全程使用 FP8。

- [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)（1.6T / 49B 激活，instruct）
- [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)（284B / 13B 激活，instruct）
- [deepseek-ai/DeepSeek-V4-Pro-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-Base)（1.6T / 49B 激活，base）
- [deepseek-ai/DeepSeek-V4-Flash-Base](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Base)（284B / 13B 激活，base）

两个 instruct 模型都支持三种推理模式：Non-think（快速，不生成思维链）、Think High（在 `<think>` 块中进行显式推理）和 Think Max（使用专用 system prompt，以最大推理强度运行）。Think Max 需要至少 384K token 的上下文窗口。所有模式推荐的采样参数都是 `temperature=1.0, top_p=1.0`。

V4-Pro 在 SWE Verified、MCPAtlas 和内部 R&D 基准上的成绩，让它在智能体任务上达到与前沿闭源模型相当的水平。开放问题在于，社区的工具 harness 会如何适配 `|DSML|` schema，以及交错思考带来的收益能否迁移到其他领域的智能体框架。

本文图片来自技术报告 [DeepSeek\_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)。
