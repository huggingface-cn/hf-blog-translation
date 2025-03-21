---
title: "Open R1 项目进展第二期"
thumbnail: /blog/assets/open-r1/thumbnails_update2.png
authors:
- user: lewtun
- user: loubnabnl
- user: anton-l
- user: eliebak
- user: guipenedo
- user: hynky
- user: gabrielmbmb
translators:
- user: yaoqih
---
# Open R1 项目进展第二期

我们启动 [Open R1 项目](https://github.com/huggingface/open-r1) 已经两周了，这个项目是为了把 DeepSeek R1 缺失的部分补齐，特别是训练流程和合成数据。

这篇文章里，我们很高兴跟大家分享一个大成果：[OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/openr1-220k-math)，这是我们打造的第一个大规模数学推理数据集！

除此之外，我们还聊聊社区里一些让人兴奋的进展，比如怎么整理出小而精的高质量数据集来微调模型，以及如何在训练和推理时控制推理模型的“思考步数”。

一起来看看吧！

## OpenR1-Math-220k 数据集

DeepSeek R1 的厉害之处在于，它能把高级推理能力“传授”给小模型。DeepSeek 团队生成了 60 万条推理记录，用来微调 Qwen 和 Llama 系列模型，结果证明，不用强化学习，直接从 R1 “蒸馏”出来的效果也很棒。比如，DeepSeek-R1-Distill-Qwen-7B 在 AIME 2024 上拿下了 55.5% 的成绩，比更大的 QwQ-32B-Preview 还强。

不过，这些推理记录没公开，这就促使社区自食其力，重新创建了几个类似的数据集。比如 [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)、[Bespoke-Stratos-17k](https://huggingface.co/datasets/HuggingFaceH4/Bespoke-Stratos-17k)、[Dolphin-R1](https://huggingface.co/datasets/cognitivecomputations/dolphin-r1/viewer/reasoning-deepseek) 和 [LIMO](https://huggingface.co/datasets/GAIR/LIMO)。

🐳 **隆重介绍 OpenR1-Math-220k**！这是一个用 512 台 H100 机器本地跑出来的大规模数学推理数据集，每个问题还配了好几个答案。我们跟 [Numina](https://projectnumina.ai) 合作，基于他们超受欢迎的 [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 数据集，推出了全新升级版。

这个 OpenR1 数据集跟其他的有啥不一样：

- **80 万条推理记录**：我们用 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) 为 40 万道题各生成了两个答案，筛完后剩下 **22 万道题**，每道题都有靠谱的推理过程。
- **512 台 H100 本地跑**：没用 API，我们靠 [vLLM](https://github.com/vllm-project/vllm/) 和 [SGLang](https://github.com/sgl-project/sglang?) 在自家科学计算集群上搞定，每天能生成 **18 万条推理记录**。
- **基于 [NuminaMath 1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5)**：我们主攻数学推理，针对 NuminaMath 1.5（[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) 的升级版）里的题目生成答案。
- **自动筛选**：用 [Math Verify](https://github.com/huggingface/Math-Verify) 只留下至少一个正确答案的题目，还请 [Llama3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) 当“裁判”，捞回更多靠谱答案（比如有些答案格式乱了，规则解析器认不出来）。
- **性能追平 [DeepSeek-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)**：我们在数据集上微调 [Qwen-7B-Math-Instruct](https://huggingface.co/Qwen/Qwen2.5-Math-7B-Instruct)，效果不输原版[ DeepSeek-Distill-Qwen-7B ](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)。

我们希望这个可扩展、高质量的推理数据生成方法，不仅能用在数学上，还能拓展到代码生成等领域。

### 数据怎么来的

为了搞出 OpenR1-220k，我们让 [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) 根据模型卡建议的参数去解 NuminaMath 1.5 里的 40 万道题。还在每道题的提示前加了句：

“请一步步推理，最后把答案写在 \boxed{} 里。”

每道题最多给 16k token ，因为我们发现 75% 的题 8k token 就能搞定，剩下的基本得用满 16k。一开始用 vLLM 跑推理，每台 H100 一小时能生成 15 个答案，脚本也在之前的更新和 OpenR1 [仓库](https://github.com/huggingface/open-r1) 里分享了。后来我们试了 **[SGLang](https://github.com/sgl-project/sglang)**，速度翻倍，每张 H100 一小时能搞 25 个答案！靠着 512 张 H100，我们一天能生成 30 万个答案，几天就攒了 80 万条推理记录。

每道题我们生成了两份答案，有些甚至四份，这样筛选和训练时更灵活。这种做法跟 DeepSeek R1 的拒绝采样差不多，还能支持 DPO 这种偏好优化方法。

生成脚本在这儿：[https://github.com/huggingface/open-r1/tree/main/slurm](https://github.com/huggingface/open-r1/tree/main/slurm)

没筛过的数据集在这儿：[https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw](https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw)

### 数据怎么筛的

为了确保只留下高质量、正确的推理过程，我们用 [Math Verify](https://github.com/huggingface/Math-Verify) 来把关。这是个专门评测 LLM 答案的数学表达式评估系统，我们把模型给的最终答案跟数据集里的标准答案对比。

结果发现，55% 的题至少有一个正确答案。但 NuminaMath 1.5 里有些标准答案是空的，或者格式没法自动校验，挺麻烦的。虽然我们升级了 Math-Verify，让它能更好地处理这些怪格式（后面会讲改进），但还是找了个备用方案：用 Llama-3.3-70B-Instruct 当“裁判”，从被拒的答案里救回一些靠谱的。先把不完整或标准答案空的样本筛掉，只看格式OK、答案框得清楚的，最后救回了 2.8 万道题。

我们给 **Llama3.3-70B-Instruct** 的指令是：

```
你是数学答案的检查员。给你一道题，你得对比标准答案和模型的最终答案，看看是不是一个意思，哪怕格式不一样。

题目：

{problem}

标准答案：

{answer}

模型答案：

{generation}

只看模型给的最终数学答案，别管这些差别：

- 格式（比如 \boxed{} 和普通文本）
- 多选题形式（比如 “A” 和完整答案）
- 坐标对或答案的顺序
- 等价的数学表达或符号差异
- 如果模型答案乱七八糟，就说“结果：不确定”

先简单说两三句你的对比思路，然后给出结论，用这几种格式：

- “结果：一样”
- “结果：不一样”
- “结果：不确定”
```

结合规则校验（Math Verify）和 LLM 判断，我们既保证了数据质量，又没牺牲规模。最终数据集有 22 万道题，推理过程都经过验证，是个训练推理模型的好资源。每道题多份答案也方便社区筛选更好的结果，或者根据 NuminaMath 的数据来源和题型再做调整。

![图片/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/8wY4H7-J_nBmfXDBMHwnJ.png)

数据集分两块：

- `default`（9.4 万道题）：SFT 微调后效果最好。
- `extended`（13.1 万道题）：加了 NuminaMath 1.5 的其他来源，比如 `cn_k12`，推理记录更多。但在这部分微调后效果不如 `default`，可能是 `cn_k12` 的题太简单了。

对于多正确答案的题，我们还试了用奖励模型（RM）挑最好的。每道题如果 R1 给了好几个正确答案，我们去掉思考过程（`<think>…</think>`），把问题和答案丢给 [Qwen/Qwen2.5-Math-RM-72B](https://huggingface.co/Qwen/Qwen2.5-Math-RM-72B) 打分，用 vLLM 跑。按分数排了个序，挑了第一名的答案放进训练集。可惜实验发现，这么挑跟随便选一个正确的没啥差别。以后可以试试评分时带上推理过程，别只看最终答案。

### 跟 DeepSeek-Distill-Qwen-7B 比比性能

我们用 5e-5 的学习率，在 `default` 数据集上微调了 Qwen2.5-Math-Instruct 三轮。为了把上下文长度从 4k 拉到 32k，我们把 RoPE 频率调到 300k。训练用的是线性学习率，前面 10% 是预热。下面是用 [lighteval](https://github.com/huggingface/open-r1?tab=readme-ov-file#evaluating-models) 比较 [OpenR1-Qwen-7B](https://huggingface.co/open-r1/OpenR1-Qwen-7B)、[DeepSeek-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) 和 [OpenThinker-7B](https://huggingface.co/open-thoughts/OpenThinker-7B) 的表现：

| 模型                  | MATH-500 | AIME24 | AIME25 |
|-----------------------|----------|--------|--------|
| DeepSeek-Distill-Qwen-7B | 91.6     | 43.3   | 40     |
| OpenR1-Qwen-7B        | 90.6     | 36.7   | 40     |
| OpenThinker-7B        | 89.6     | 30.0   | 33.3   |

这版数据集只是个起点，社区还能再优化，比如用 DeepSeek R1 的拒绝采样法提高质量。

### Math-Verify 升级了啥

我们在检查 Math-Verify 的结果时发现了一些问题，就做了大修。强烈建议大家升到最新版（0.5.2），体验这些改进：

```python
pip install math-verify==0.5.2
```

主要升级有：

- 改进了纯文本答案的解析和验证（比如 $\text{E}$ 和 $E$ 算一样）。
- 改进了答案列表的解析（比如 $1$ 和 $2$ 和 $3$ 跟 $1,2,3$ 等价）。
- 修了个 bug，单个 LaTeX 里多个框的答案也能认了（比如 $\boxed{1},\boxed{2}$ 等于 {1,2}）。
- 加了有序元组。因为判断列表是元组还是集合非常困难，我们靠标准答案来定：
  - (1,2,3) ≠ {3,2,1}; 1,2,3 == {3,2,1}; {3,2,1} == {1,2,3}。
- 支持标准答案的关系表达（比如小于）和预测的区间（比如 $1 < x < 2$ 等价于 $(1,2)$）。

## 社区热点

这周社区从各种角度玩转了 GRPO，还有研究表明，只要 1000 个优质样本，就能让现有开源模型引发推理。

### GRPO 的一些实践

- nrehiew [证明](https://x.com/nrehiew_/status/1887874867225063543) 把 GRPO 用在 Qwen2.5-0.5B 基础模型上，在 GSM8k 测试中拿下 51% 的准确率，比 Qwen2.5-0.5B-Instruct 高了 10 个点。这成绩太亮眼，引发了大家对预训练中指令数据作用的 [热议](https://x.com/abacaj/status/1888644577604563240)。不过，把 GRPO 用到其他基础模型（比如 Llama 3）上还没啥大突破。[Sea AI Lab (SAIL) 的研究](https://www.notion.so/Open-R1-Update-2-1961384ebcac80efb364e947cec44c91?pvs=21) 发现，基础模型稍微提示一下就能自我反思，DeepSeek-R1 论文里的“开悟”可能更多是模型本身牛，而不是 RL 优化的功劳。
- Unsloth 施展 [优化魔法](https://unsloth.ai/blog/r1-reasoning)，让 15B 参数的模型只用 15GB 显存就能跑 GRPO 🤯。这下 Google Colab 免费也能玩了！
- [Axolotl](https://github.com/axolotl-ai-cloud/axolotl) 的 Wing Lian 发现 [DoRA 比 LoRA 和全微调收敛快](https://x.com/winglian/status/1888951180606202028)。
- Alexander Doria 搞出了 [给诗歌设计的奖励函数](https://x.com/Dorialexander/status/1886176543593894387)，这很酷，因为 GRPO 第一次公开跳出“可验证”领域。

### 测试表现

这周 [2025 AIME I](https://artofproblemsolving.com/wiki/index.php/2025_AIME_I?srsltid=AfmBOoqknvf_6DwLAOY55UF1k21ilYYaSwo7QWzl9impFvE_XXMpfY7r) 第一部分放出来了，有 15 道难题，是给高中生备战国际数学奥赛用的。过去一年，AIME 2024 是测 LLM 数学能力的主力，大家很期待LLM在新题上的表现：

- [ETH Zurich](https://x.com/mbalunovic/status/1887962694659060204) 的研究人员测了一堆模型，发现 [性能波动](https://x.com/9hills/status/1888742869625905536) 远小于预期，只有 10-20 个百分点。
- 但 [Dimitris Papailiopoulos](https://x.com/DimitrisPapail/status/1888325914603516214) 发现 AIME 2025 有几道题网上早就有了！这可能不小心泄了题，凸显了 [给 LLM 出新题有多难](https://x.com/hyhieu226/status/1888653916663132319)。

### LLM 必须用自然语言推理吗？

![图片/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/I2Z27QYRjqmF7pbSdFJQc.png)
一篇新 [论文](https://arxiv.org/abs/2502.05171) 挺有意思，用循环语言模型在潜在空间隐式推理，能扩展测试时的计算。这跟 [Meta 的 Coconut 项目](https://arxiv.org/abs/2412.06769) 在潜在空间训语言模型有点像，但现在用在了推理上。好处是效率高，不用生成一堆“思考”token 也能出好成绩。

### 小而精的推理数据成趋势？

DeepSeek R1 用 60 万条推理记录搞蒸馏，但最近研究发现，不用海量训练，少量精心挑的样本也能让模型学会复杂推理。

比如 [s1K](https://huggingface.co/datasets/simplescaling/s1K) 数据集，只有 1000 道数学题，推理过程从 [Gemini Flash](https://deepmind.google/technologies/gemini/flash-thinking/) 蒸馏而来，挑题时看重难度、多样性和质量。作者用它微调 Qwen2.5-32B-Instruct，在竞赛数学测试中比 OpenAI 的 o1-preview 高了 27%。

另一个 [LIMO](https://huggingface.co/GAIR/LIMO) 数据集更狠，只用 817 个样本就在 AIME 和 MATH 上表现抢眼。作者猜，如果模型预训练时已经学了很多领域知识，可能几百个好例子就够让它推理开窍。

### 控制思维链长度：预算强制和奖励设计

[s1K](https://huggingface.co/datasets/simplescaling/s1K) 微调的 Qwen2.5-32B-Instruct 这么牛，一个关键是 **预算强制**。这招能在测试时调整推理时间，要么加个“Wait”让它多想，要么加个结束标记让它停。作者发现，模型有测试时扩展性：多给点思考时间，数学测试的准确率就涨。

![图片/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/C_BAQUuhHUEoEYqQzRBKl.png)

类似地，[《揭秘 LLM 长链推理》](https://arxiv.org/abs/2502.03373)（Yeo 等人）也研究了思维链（CoT）长度对效果的影响。他们搞了个 **余弦奖励(Cosine Reward)**，正确答案鼓励短 CoT，错的推长 CoT，稳住了 RL 训练，尤其在上下文长度有限、回答容易爆炸时。还有个 **重复惩罚**，模型要是为刷奖励在难题上重复废话，就罚它，逼它好好解题。

## 下一步干啥？

![图片/png](https://cdn-uploads.huggingface.co/production/uploads/61c141342aac764ce1654e43/ed4LVSonlqAxYOfahvtVi.png)

GRPO 在 TRL 里跑得挺顺，我们正在大干一场实验，看看哪些超参数和奖励函数最管用。想知道进展，可以去 [社区页](https://huggingface.co/spaces/open-r1/README/discussions/15) 瞧瞧，下次更新会写详细报告！

想加入我们？去 [GitHub 的 open-r1 仓库](https://github.com/huggingface/open-r1) 或关注 [Hugging Face 的 open-r1 组织](https://huggingface.co/open-r1) 吧。

