---
title: "使用 RapidFire AI，TRL 微调提速 20 倍"
thumbnail: /blog/assets/rapidfireai/thumbnail.png
authors:
- user: kbigdelysh
  guest: true
  org: rapidfire-ai-inc
- user: arunkk09
  guest: true
  org: rapidfire-ai-inc
- user: qgallouedec
translators:
- user: chenglu
---

# 使用 RapidFire AI，TRL 微调提速 20 倍

Hugging Face 的 TRL（Transformer Reinforcement Learning）现在已正式集成 RapidFire AI，大大加快了微调和训练后实验的效率。对于 TRL 用户来说，RapidFire AI 提供了一种更快的方式，帮助他们在不修改大量代码、也不增加 GPU 负担的前提下，轻松安装并运行多个微调或后训练配置，从而快速对比结果、定制 LLM 模型。

## 为什么这很重要？

在微调或训练大语言模型（LLM）时，团队往往因为时间紧或预算有限，无法同时测试多个配置，尽管这样做通常可以显著提升模型评估指标。RapidFire AI 的出现，正好解决了这个问题。它支持你 **同时运行多个 TRL 配置**，即便只用一张 GPU，也能通过一种新型的“自适应分块调度执行机制”实现近乎实时的对比分析。根据 TRL 官方页面引用的内部基准测试数据，RapidFire AI 的实验吞吐量相比传统串行测试提高了约 **16 到 24 倍**，让你更快获得更优的模型表现。

![RapidFire AI 架构图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/rf-usage.png)
*RapidFire AI 实现了 IDE、指标面板与多 GPU 执行后端之间的实时三向通信*

## 开箱即用的功能

* **即插即用的 TRL 包装器** — 你可以使用 `RFSFTConfig`、`RFDPOConfig` 和 `RFGRPOConfig` 来替代 TRL 中的 SFT、DPO 和 GRPO 配置，几乎无需修改原有代码。

* **自适应分块并发训练** — RapidFire AI 会将数据集拆分为若干数据块，并在块与块之间切换不同配置，既能更早实现公平对比，也能最大化 GPU 利用率。

* **交互式控制操作（IC Ops）** — 你可以直接在仪表盘中对正在运行的任务进行停止、恢复、删除或克隆修改操作，还支持热启动（Warm-Start）。这样可以及时停止效果差的配置、集中资源优化表现更好的配置，无需重启任务，也不需要手动管理 GPU 或集群，避免资源浪费。

![交互式控制操作示意图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/icop-clone.png)
*在实时仪表盘中克隆表现优秀的配置，可修改超参数，并可选择从父模型的权重热启动*

* **多 GPU 协同调度** — RapidFire AI 的调度器会通过高效的共享内存机制，自动将不同配置分配到可用的 GPU 上，并在数据块之间协调运行。你只需专注于模型训练和评估指标，无需操心底层资源调度。

* **基于 MLflow 的仪表盘** — 一旦开始实验，即可在同一个界面中实时查看训练指标、日志以及执行 IC Ops 操作。未来还将支持 Trackio、W&B、TensorBoard 等更多可视化工具。

## 工作原理

RapidFire AI 会将你的数据集随机划分为多个“数据块”，并在每个数据块的边界处轮流调度不同的 LLM 配置在 GPU 上运行。这样可以更快地获取所有配置在评估指标上的初步信号，实现快速对比。
同时，系统通过高效的共享内存机制实现模型的自动保存与加载（checkpointing），保证训练过程的平稳、稳定与一致性。
你还可以使用 IC Ops 功能，在训练中途灵活调整：提前停止表现不佳的配置，克隆并优化表现优异的配置，必要时还能从原始模型的权重热启动，进一步提升实验效率。

![GPU 调度方式对比](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/rapidfireai_intro/gantt-2gpu.png)
*顺序执行 vs. 任务并行 vs. RapidFire AI：自适应调度器能够在多个配置和多张 GPU 之间最大化资源利用率。最下方展示了 IC Ops 的实际效果——在训练过程中实时停止、克隆和修改运行任务。*

## 快速上手

一分钟内安装并运行 RapidFire AI：

```bash
pip install rapidfireai

# Authenticate with Hugging Face
huggingface-cli login --token YOUR_TOKEN

# Workaround for current issue
pip uninstall -y hf-xet

# Initialize and start RapidFire AI
rapidfireai init
rapidfireai start
```

仪表盘地址：`http://localhost:3000`，可实时查看与管理所有实验。

## 支持的 TRL 训练器

* 支持使用 `RFSFTConfig` 的 SFT（监督微调）
* 支持使用 `RFDPOConfig` 的 DPO（直接偏好优化）
* 支持使用 `RFGRPOConfig` 的 GRPO（强化学习优化）

这些配置是专门设计为“即插即用”的替代方案，让你在保留 TRL 原有使用习惯的同时，获得更高的并发能力和对微调/训练后任务的更强控制力。

## 最简 TRL SFT 示例

下面是一个示例，展示了如何在单张 GPU 上 **并发训练多个配置**：

```python
from rapidfireai import Experiment
from rapidfireai.automl import List, RFGridSearch, RFModelConfig, RFLoraConfig, RFSFTConfig
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup: load your dataset and define formatting
dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
train_dataset = dataset["train"].select(range(128)).shuffle(seed=42)

def formatting_function(row):
    return {
        "prompt": [
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": row["instruction"]},
        ],
        "completion": [{"role": "assistant", "content": row["response"]}]
    }

dataset = dataset.map(formatting_function)

# Define multiple configs to compare
config_set = List([
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=RFLoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]),
        training_args=RFSFTConfig(learning_rate=1e-3, max_steps=128, fp16=True),
    ),
    RFModelConfig(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        peft_config=RFLoraConfig(r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"]),
        training_args=RFSFTConfig(learning_rate=1e-4, max_steps=128, fp16=True),
        formatting_func=formatting_function,
    )
])

# Run all configs concurrently with chunk-based scheduling
experiment = Experiment(experiment_name="sft-comparison")
config_group = RFGridSearch(configs=config_set, trainer_type="SFT")

def create_model(model_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_config["model_name"], 
        device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"])
    return (model, tokenizer)

experiment.run_fit(config_group, create_model, train_dataset, num_chunks=4, seed=42)
experiment.end()
```

**运行时会发生什么？**

假设你在一台拥有 2 张 GPU 的机器上运行上述代码。与传统的顺序训练方式（配置 1 → 等待 → 配置 2 → 再等待）不同，这两个配置将同时并发训练：

| 方式                | 得出比较结论所需时间 | GPU 利用率 |
| ----------------- | ---------- | ------- |
| 传统顺序训练            | 约 15 分钟    | 60%     |
| RapidFire AI 并发训练 | 约 5 分钟     | 超过 95%  |

你可以在两个配置处理完首个数据块后，提前 **3 倍速度** 做出判断，而无需等两次完整训练流程结束。打开 `http://localhost:3000`，可实时查看指标并使用 IC Ops 停止、克隆、调整任务。

## 性能实测：真实提速效果

以下是一些团队使用 RapidFire AI 并行实验替代传统顺序比较后，在达到相似训练损失时的耗时对比：

| 场景           | 传统方式   | 使用 RapidFire AI | 加速比     |
| ------------ | ------ | --------------- | ------- |
| 4 个配置，1 GPU  | 120 分钟 | 7.5 分钟          | **16×** |
| 8 个配置，1 GPU  | 240 分钟 | 12 分钟           | **20×** |
| 4 个配置，2 GPUs | 60 分钟  | 4 分钟            | **15×** |

*测试平台：NVIDIA A100 40GB，模型为 TinyLlama-1.1B 和 Llama-3.2-1B*

## 立即开始使用

**🚀 在线试用**：[Colab 交互笔记本](http://tinyurl.com/rapidfireai-colab) — 浏览器一键运行
**📚 完整文档**：[oss-docs.rapidfire.ai](https://oss-docs.rapidfire.ai) — 全套教程、示例和 API
**💻 GitHub**：[RapidFireAI/rapidfireai](https://github.com/RapidFireAI/rapidfireai) — 开源，生产可用
**📦 PyPI 安装**：[pypi.org/project/rapidfireai](https://pypi.org/project/rapidfireai) — `pip install rapidfireai`
**💬 加入社区**：[Discord](https://discord.gg/6vSTtncKNN) — 获取帮助、反馈建议、交流成果

---

RapidFire AI 的初衷是打破“一次只能测试一个配置”的低效常态，它浪费了宝贵的时间和 GPU 资源。通过与 TRL 的官方集成，用户现在可以更更高效地进行微调和训练后优化，加快迭代速度，打造更优质的模型。

**欢迎试用并告诉我们你的反馈**：你的实验速度提升了多少？你希望我们下一个开发什么功能？我们才刚起步，你的反馈将指引我们前进的方向。
