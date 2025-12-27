---
title: "Open ASR 排行榜：多语种与长音频识别的趋势与洞察"
thumbnail: /blog/assets/open-asr-leaderboard/thumbnail.png
authors:
- user: bezzam
- user: Steveeeeeeen
- user: eustlb
- user: reach-vb
translators:
- user: chenglu
---


# Open ASR 排行榜：多语种与长音频识别的趋势与洞察

如今几乎人人都在开发新的语音识别（ASR）模型，连“奶奶 👵”都不例外。感觉比挑选下一部 Netflix 剧集还要困难。截至 2025 年 11 月 21 日，[Hugging Face 模型库](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)上已经有 **150 个 [音频转文本模型](https://huggingface.co/models?pipeline_tag=audio-text-to-text&sort=trending)（Audio-Text-to-Text）** 和 **2.7 万个 [语音识别](https://huggingface.co/models?pipeline_tag=automatic-speech-recognition&sort=trending)（ASR）模型** 🤯

然而，目前大多数评测只关注于 **英文短音频转录（<30秒）**，却忽视了一些同样重要的任务，比如 (1) 多语言识别能力 和 (2) 模型在长音频处理中的效率 —— 这对于会议、播客等应用场景至关重要。

在过去两年中，[**Open ASR 排行榜**](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) 已成为对比开源和闭源语音识别模型在 **准确性** 与 **效率** 方面的行业标准。最近，该排行榜新增加了 **多语种** 和 **长音频识别** 两个赛道 🎉

### TL;DR - [Open ASR 排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)

* 📝 **新论文预印本**：关于排行榜中的ASR趋势：[https://hf.co/papers/2510.06961](https://hf.co/papers/2510.06961)
* 🧠 **准确率最高**：Conformer 编码器 + LLM 解码器（开源大胜 🥳）
* ⚡ **最快**：CTC / TDT 解码器
* 🌍 **多语种识别**：会牺牲部分单语性能
* ⌛ **长音频识别**：目前闭源系统仍占优势（暂时😉）
* 🧑‍💻 **微调指南**：支持 [Parakeet](https://github.com/Deep-unlearning/Finetune-Parakeet)、[Voxtral](https://github.com/Deep-unlearning/Finetune-Voxtral-ASR)、[Whisper](https://huggingface.co/learn/audio-course/chapter5/fine-tuning)，助你提升性能

# 从 60+ 个模型中总结出的要点

截至 2025 年 11 月 21 日，*Open ASR 排行榜* 已对来自 **18 个组织** 的 **60 多个开源与闭源模型** 进行了对比，覆盖 **11 个不同数据集**。

我们在最近的[论文预印本](https://hf.co/papers/2510.06961)中，详细介绍了评测方法与当前ASR技术的主要趋势。以下是几个关键观察👇

## 1. Conformer 编码器 🤝 LLM 解码器 成绩最佳 📈

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/leaderboard_WER.png" width="1024px" alt="thumbnail" />
</div>

结合 [**Conformer 编码器**](https://huggingface.co/papers/2005.08100) 与 **大型语言模型（LLM）解码器** 的模型，在英文转录任务中表现最优。例如：

* **NVIDIA 的 [Canary-Qwen-2.5B](https://huggingface.co/nvidia/canary-qwen-2.5b)**
* **IBM 的 [Granite-Speech-3.3-8B](https://huggingface.co/ibm-granite/granite-speech-3.3-8b)**
* **Microsoft 的 [Phi-4-Multimodal-Instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct)**

这些模型的词错误率 ([WER](https://huggingface.co/learn/audio-course/en/chapter5/evaluation#word-error-rate)) 非常低，证明融合语言模型推理能力可以显著提升识别准确率。

💡 *小提示：NVIDIA 推出的 [Fast Conformer](https://huggingface.co/papers/2305.05084) 是 Conformer 的高效版本，速度提升约 2 倍，广泛用于 Canary 与 Parakeet 系列模型中。*

## 2. 准确率与速度的权衡 ⚖️

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/leaderboard_RTX.png" width="1024px" alt="thumbnail" />
</div>

虽然这些 LLM 解码器在准确率上表现出色，但相较于一些更简单的方案，它们的速度通常 **较慢**。在 *Open ASR 排行榜* 中，效率使用 *实时因子的倒数（RTFx）* 来衡量，数值越高表示模型越高效。

如果追求更快的推理速度，[**CTC**](https://huggingface.co/learn/audio-course/en/chapter3/ctc#ctc-architectures) 与 [**TDT**](https://huggingface.co/papers/2304.06795) 解码器则提供 **10 到 100 倍更高的吞吐率**，尽管词错误率会略有上升。这种类型非常适合用于 **实时识别**、**离线处理** 或 **批量转录**（如会议、讲座、播客等）场景。

## 3. 多语言识别 🌍

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/multilingual.png" width="1024px" alt="thumbnail" />
</div>

OpenAI 推出的 [**Whisper Large v3**](https://huggingface.co/openai/whisper-large-v3) 依然是一个强大的多语种语音识别基线模型，支持多达 **99 种语言**。不过，一些经过 **微调或蒸馏** 的版本，如 [**Distil-Whisper**](https://huggingface.co/distil-whisper/distil-large-v3.5) 和 [**CrisperWhisper**](https://huggingface.co/nyrahealth/CrisperWhisper)，在 **仅限英语** 的任务中常常表现优于原始模型，说明有针对性的微调可以有效提升模型的专业化能力。

*想学习如何进行微调？可以参考这些教程：
[Whisper 微调指南](https://huggingface.co/learn/audio-course/chapter5/fine-tuning)、[Parakeet 微调指南](https://github.com/Deep-unlearning/Finetune-Parakeet)、[Voxtral 微调指南](https://github.com/Deep-unlearning/Finetune-Voxtral-ASR)*

不过，专注于英语优化通常会 **削弱多语种的覆盖能力** 👉 这是“专业化 vs 泛化”的经典权衡。同样，虽然像 Meta 的 [**Massively Multilingual Speech (MMS)**](https://huggingface.co/facebook/mms-1b-all) 和 [**Omnilingual ASR**](https://github.com/facebookresearch/omnilingual-asr) 这样的 **自监督学习系统** 可以支持超过 1000 种语言，但在准确率上，仍不及针对单一语言优化的模型。

⭐ *目前排行榜只覆盖了 5 种语言，但我们计划扩展到更多语言，欢迎通过 GitHub [pull request](https://github.com/huggingface/open_asr_leaderboard) 贡献新的数据集和模型，一起推动多语种语音识别的发展。*

🎯 除了多语种评测之外，还有一些由 **社区驱动的排行榜** 专注于单一语言的语音识别任务。例如：

* [**Open Universal Arabic ASR Leaderboard**](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard)：评估模型在 **现代标准阿拉伯语及其地区方言** 上的表现，展示了语音变体和双语现象带来的挑战
* [**Russian ASR Leaderboard**](https://huggingface.co/spaces/Vikhrmodels/Russian_ASR_Leaderboard)：专注于评测编码器-解码器和 CTC 模型在 **俄语语音特点和形态结构** 方面的识别能力

这些本地化排行榜与多语种主榜目标一致，都是为了推动 **数据集共享、模型微调成果的开放发布，以及透明、公平的模型评估**，尤其是在资源匮乏语言场景下的实际应用。

## 4. 长音频识别是另一场战斗 ⏳

<div align="center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open_asr_leaderboard/long_form.png" width="1024px" alt="thumbnail" />
</div>

对于 **长音频内容**（如播客、讲座、会议等），目前 **闭源系统** 依然略胜一筹。这可能得益于更深入的领域优化、自定义的音频切分策略，或是更成熟的生产级部署能力。

在开源模型中，**OpenAI 的 Whisper Large v3** 表现最为出色。但若从 **吞吐量（RTFx）** 来看，**基于 CTC 的 Conformer 模型** 更具优势 👉 举例来说，**NVIDIA 的 [Parakeet CTC 1.1B](https://huggingface.co/nvidia/parakeet-ctc-1.1b)** 的 RTFx 达到 **2793.75**，而 Whisper Large v3 为 **68.56**，两者在词错误率（WER）上差距并不大（分别为 **6.68** 和 **6.43**）。

不过这也带来了一个权衡：Parakeet 是 **仅支持英语** 的，再次提醒我们，在多语种覆盖与任务专精之间始终存在取舍 🫠。

⭐ *虽然闭源系统目前仍占上风，但开源在这一领域的潜力巨大。长音频语音识别仍是社区亟待攻克的下一片热土！*

# 🎤 演出继续

ASR 技术正快速演进，我们也很期待新的架构如何推动准确率与效率进一步提升。同时，*Open ASR 排行榜* 也将继续作为一个 **透明、社区驱动的基准平台**，为语音识别领域提供参考，也为其他排行榜（如[俄语](https://huggingface.co/spaces/Vikhrmodels/Russian_ASR_Leaderboard)、[阿拉伯语](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard)、[语音深度伪造检测](https://huggingface.co/spaces/Speech-Arena-2025/Speech-DF-Arena)）提供借鉴。

我们会持续扩展 *Open ASR 排行榜*，纳入 **更多模型、语言与数据集**，敬请关注 👀

👉 **想参与贡献？** 欢迎访问 [GitHub 仓库](https://github.com/huggingface/open_asr_leaderboard) 发起  *Pull Request* 🚀
