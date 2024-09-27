---
title: "欢迎使用Llama 3.2，现在Llama可以在您的设备上查看和运行了" 
thumbnail: /blog/assets/llama32/thumbnail.jpg
authors:
- user: merve
- user: philschmid
- user: osanseviero
- user: reach-vb
- user: lewtun
- user: ariG23498
- user: pcuenq
translators:
- user: roseking
---

# 欢迎使用Llama 3.2，现在Llama可以在您的设备上查看和运行了

Llama 3.2 来了！今天，我们欢迎 Llama 系列的下一个版本加入 Hugging Face。这次，我们很高兴与 Meta 合作发布多模态和小型模型。在 Hub 上提供了十个开源模型（5 个多模态模型和 5 个仅文本模型）。

Llama 3.2 Vision 有两种尺寸：11B 适用于在消费级 GPU 上的高效部署和开发，90B 适用于大规模应用。两种版本都有基础版和指令微调版。除了这四个多模态模型外，Meta 还发布了支持视觉的新版 Llama Guard。Llama Guard 3 是一个安全模型，可以分类模型输入和生成内容，包括检测有害的多模态提示或助手响应。

Llama 3.2 还包括可以在设备上运行的小型仅文本语言模型。它们有两种新尺寸（1B 和 3B），有基础版和指令版，并且在它们的大小范围内具有强大的能力。还有一个小型 1B 版本的 Llama Guard，可以在生产用例中与这些或更大的文本模型一起部署。

在发布的功能和集成中，我们有：
- [Hub 上的模型检查点](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- Hugging Face Transformers 和 TGI 集成用于 Vision 模型
- 与 Inference Endpoints、Google Cloud、Amazon SageMaker 和 DELL Enterprise Hub 的推理与部署集成
- 使用 [transformers🤗](https://github.com/huggingface/huggingface-llama-recipes/tree/main/Llama-Vision FT.ipynb) 和 [TRL](https://github.com/huggingface/trl/tree/main/examples/scripts/sft_vlm.py) 在单个 GPU 上微调 Llama 3.2 11B Vision

## 目录

- [什么是 Llama 3.2 Vision？](#what-is-llama-32-vision)
- [Llama 3.2 许可证变更。对不起，欧盟 :(](#llama-32-license-changes-sorry-eu-)
- [Llama 3.2 1B 和 3B 有什么特别之处？](#what-is-special-about-llama-32-1b-and-3b)
- [演示](#demo)
- [使用 Hugging Face Transformers](#using-hugging-face-transformers)
- [Llama 3.2 1B & 3B 语言模型](#llama-32-1b--3b-language-models)
- [Llama 3.2 Vision](#llama-32-vision)
- [设备上运行](#on-device)
- [Llama.cpp & Llama-cpp-python](#llamacpp--llama-cpp-python)
- [Transformers.js](#transformersjs)
- [微调 Llama 3.2](#fine-tuning-llama-32)
- [Hugging Face 合作伙伴集成](#hugging-face-partner-integrations)
- [其他资源](#additional-resources)
- [致谢](#acknowledgements)

## 什么是 Llama 3.2 Vision？

Llama 3.2 Vision 是 Meta 发布的最强大的开源多模态模型。它具有出色的视觉理解和推理能力，可以用于完成各种任务，包括视觉推理和定位、文档问答和图文检索。思维链（Chain of Thought, CoT）答案通常非常好，这使得视觉推理特别强大。

Llama 3.2 Vision 可以处理文本和图像，也可以仅处理文本。通过图文提示，模型可以接受英文输入，而对于仅文本提示，模型可以处理多种语言。仅文本模式支持的语言包括英语、德语、法语、意大利语、葡萄牙语、印地语、西班牙语和泰语。

这些模型的架构基于 Llama 3.1 LLM 与视觉塔和图像适配器的结合。使用的文本模型是 Llama 3.1 8B 用于 Llama 3.2 11B Vision 模型，Llama 3.1 70B 用于 3.2 90B Vision 模型。据我们所知，在视觉模型的训练过程中，文本模型被冻结以保留仅文本性能。

下面你可以找到一些来自 11B 指令微调模型的推理示例，展示了真实世界知识、文档推理和信息图理解能力。

| 图像 | 描述 |
|------|------|
| ![Wat Arun](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/wat_arun.jpg) | **这张图片位于哪里？附近有哪些其他景点？**<br>这张图片描绘了泰国曼谷的大皇宫。大皇宫是一个建筑群，曾作为暹罗国王的官方住所和政府行政中心长达 150 年。它是曼谷最受欢迎的旅游景点之一，是任何访问该城市的游客必看的地方。<br><br>大皇宫附近的一些景点包括：<br>* 玉佛寺：这座寺庙位于大皇宫建筑群内，是著名的翡翠佛像的所在地。<br>* 卧佛寺：这座寺庙位于大皇宫附近，是巨大的卧佛雕像的所在地。<br>* 隆披尼公园：这个公园位于曼谷市中心，为城市提供了一个宁静的避风港。<br>* 吉姆·汤普森之家：这个博物馆展示了吉姆·汤普森的生活和工作，他是美国商人，在 1950 年代和 1960 年代帮助振兴了泰国的丝绸产业。 |
| ![Weather Events](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/weather_events.png) | **根据这张图，干旱发生在哪里？**<br>**2016 年干旱地区**<br>根据这张图，2016 年干旱发生在两个不同的地区：<br>* **东非和南部非洲**：这个地区经历了严重的干旱。<br>* **智利**：智利在 1 月份经历了 50 多年来最干燥的天气。这些地区在那个时期面临严重的水资源短缺问题。 |
| ![Invoice](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/invoice.png) | **从发票日期到截止日期需要多长时间？简短明了。**<br>**发票日期：** 2019 年 2 月 11 日<br>**截止日期：** 2019 年 2 月 26 日<br>**时间差计算：**<br>2019 年 2 月 26 日 - 2019 年 2 月 11 日 = 15 天<br><br>发票日期和截止日期之间的时间差是 **15 天。** |

Vision 模型具有 128k 个 token 的上下文长度，这允许包含图像的多轮对话。然而，模型在处理单张图像时效果最佳，因此 `transformers` 实现仅关注输入中提供的最后一张图像。这既保证了质量又节省了内存。

11B 基础模型支持 448 的 tile 大小，而指令版本和 90B 模型都使用 560 的 tile 大小。这些模型在包含 60 亿对图像-文本对的庞大数据集上进行了训练，数据混合多样。这使得它们非常适合在下游任务上进行微调。作为参考，你可以看到下面 11B、90B 及其指令微调版本在一些基准测试中的比较，这些基准测试由 Meta 报告。请参阅模型卡片以获取更多基准测试和详细信息。

| 模型 | 11B | 11B (指令微调) | 90B | 90B (指令微调) | 指标 |
|------|-----|----------------|-----|----------------|------|
| MMMU (val) | 41.7 | 50.7 (CoT) | 49.3 (零样本) | 60.3 (CoT) | 微平均准确率 |
| VQAv2 | 66.8 (val) | 75.2 (test) | 73.6 (val) | 78.1 (test) | 准确率 |
| DocVQA | 62.3 (val) | 88.4 (test) | 70.7 (val) | 90.1 (test) | ANLS |
| AI2D | 62.4 | 91.1 | 75.3 | 92.3 | 准确率 |

我们预计这些模型的文本能力将与 8B 和 70B Llama 3.1 模型相当，因为我们的理解是，在 Vision 模型的训练过程中，文本模型被冻结。因此，文本基准测试应与 8B 和 70B 一致。

## Llama 3.2 许可证变更。对不起，欧盟 :(

![License Change](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/license_change.png)

关于许可条款，Llama 3.2 的许可证与 Llama 3.1 非常相似，但有一个关键区别在于可接受的使用政策：任何居住在欧盟的个人或公司在欧盟主要营业地的公司不被授予使用 Llama 3.2 中包含的多模态模型的许可权利。此限制不适用于包含此类多模态模型的产品或服务的最终用户，因此人们仍然可以使用视觉变体构建全球产品。

有关完整详情，请务必阅读[官方许可证](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/LICENSE.txt)和[可接受的使用政策](https://huggingface.co/meta-llama/Llama-3.2-1B/blob/main/USE_POLICY.md)。

## Llama 3.2 1B 和 3B 有什么特别之处？

Llama 3.2 系列包括 1B 和 3B 文本模型。这些模型设计用于设备上的用例，如提示重写、多语言知识检索、摘要任务、工具使用和本地运行的助手。它们在这些尺寸上优于许多可用的开源模型，并与大得多的模型竞争。在后面的部分中，我们将展示如何在离线状态下运行这些模型。

这些模型遵循与 Llama 3.1 相同的架构。它们在高达 9 万亿个 token 的数据上进行了训练，并且仍然支持 128k 个 token 的长上下文长度。这些模型是多语言的，支持英语、德语、法语、意大利语、葡萄牙语、印地语、西班牙语和泰语。

还有一个新的 Llama Guard 小版本，Llama Guard 3 1B，可以与这些模型一起部署，以评估多轮对话中的最后一个用户或助手响应。它使用一组预定义的类别（新版本中可以自定义或排除）来适应开发者的用例。有关 Llama Guard 使用的更多详情，请参阅模型卡片。

额外提示：Llama 3.2 接触到的语言比上面提到的 8 种支持语言更广泛。开发者鼓励对 Llama 3.2 模型进行特定语言用例的微调。

我们通过 Open LLM Leaderboard 评估套件运行了基础模型，而指令模型在三个流行的基准测试中进行了评估，这些基准测试衡量指令遵循能力，并与 LMSYS Chatbot Arena 高度相关：[IFEval](https://arxiv.org/abs/2311.07911)、[AlpacaEval](https://arxiv.org/abs/2404.04475) 和 [MixEval-Hard](https://arxiv.org/abs/2406.06565)。以下是基础模型的结果，包括 Llama-3.1-8B 作为参考：

| 模型 | BBH | MATH Lvl 5 | GPQA | MUSR | MMLU-PRO | 平均 |
|------|-----|------------|------|------|----------|------|
| Meta-Llama-3.2-1B | 4.37 | 0.23 | 0.00 | 2.56 | 2.26 | 1.88 |
| Meta-Llama-3.2-3B | 14.73 | 1.28 | 4.03 | 3.39 | 16.57 | 8.00 |
| Meta-Llama-3.1-8B | 25.29 | 4.61 | 6.15 | 8.98 | 24.95 | 14.00 |

以下是指令模型的结果，包括 Llama-3.1-8B-Instruct 作为参考：

| 模型 | AlpacaEval (LC) | IFEval | MixEval-Hard | 平均 |
|------|-----------------|--------|--------------|------|
| Meta-Llama-3.2-1B-Instruct | 7.17 | 58.92 | 26.10 | 30.73 |
| Meta-Llama-3.2-3B-Instruct | 20.88 | 77.01 | 31.80 | 43.23 |
| Meta-Llama-3.1-8B-Instruct | 25.74 | 76.49 | 44.10 | 48.78 |

值得注意的是，3B 模型在 IFEval 上的表现与 8B 模型相当！这使得该模型非常适合代理应用，在这些应用中，遵循指令对于提高可靠性至关重要。这个高 IFEval 分数对于这个尺寸的模型来说非常令人印象深刻。

工具使用在 1B 和 3B 指令微调模型中都得到了支持。工具由用户在零样本设置中指定（模型之前没有关于开发者将使用的工具的信息）。因此，Llama 3.1 模型中内置的工具（`brave_search` 和 `wolfram_alpha`）不再可用。

由于其尺寸，这些小型模型可以用作更大模型的助手，并执行[辅助生成](https://huggingface.co/blog/assisted-generation)（也称为推测解码）。[这里](https://github.com/huggingface/huggingface-llama-recipes/tree/main)是一个使用 Llama 3.2 1B 模型作为 Llama 3.1 8B 模型助手的示例。对于离线用例，请查看后面的设备上运行部分。

## 演示

您可以在以下演示中体验三个指令模型：

- [Gradio Space with Llama 3.2 11B Vision Instruct](https://huggingface.co/spaces/huggingface-projects/llama-3.2-vision-11B)
- [Gradio-powered Space with Llama 3.2 3B](https://huggingface.co/spaces/huggingface-projects/llama-3.2-3B-Instruct)
- Llama 3.2 3B 在 WebGPU 上运行

![Demo GIF](https://huggingface.co/datasets/huggingface/release-assets/resolve/main/demo_gif.gif)

## 使用 Hugging Face Transformers

仅文本检查点具有与之前版本相同的架构，因此无需更新您的环境。然而，由于新的架构，Llama 3.2 Vision 需要更新 Transformers。请确保将您的安装升级到 4.45.0 或更高版本。

```bash
pip install "transformers>=4.45.0" --upgrade
```

升级后，您可以使用新的 Llama 3.2 模型并利用 Hugging Face 生态系统的所有工具。

## Llama 3.2 1B & 3B 语言模型

您只需几行代码就可以使用 Transformers 运行 1B 和 3B 文本模型检查点。模型检查点以 `bfloat16` 精度上传，但您也可以使用 float16 或量化权重。内存需求取决于模型大小和权重的精度。以下是使用不同配置进行推理所需的近似内存：

| 模型大小 | BF16/FP16 | FP8 | INT4 |
|----------|-----------|-----|------|
| 3B | 6.5 GB | 3.2 GB | 1.75 GB |
| 1B | 2.5 GB | 1.25 GB | 0.75 GB |

```python
from transformers import pipeline
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [
    {"role": "user", "content": "你是谁？请用海盗的语言回答。"},
]
outputs = pipe(
    messages,
    max_new_tokens=256,
)
response = outputs[0]["generated_text"][-1]["content"]
print(response)
# 啊哈，伙计！你在找关于我的信息，是吗？好吧，伙计！我是一个语言生成的海盗，一个数字海盗，擅长将文字变成知识的金块！我的名字是……（戏剧性停顿）……助手！是的，那是我的名字，我在这里帮助你穿越问题的七海，找到隐藏的答案宝藏！所以升起帆，启航去冒险吧，伙计！你的第一个问题是什么？
```

一些细节：

- 我们以 `bfloat16` 加载模型。如上所述，这是 Meta 发布的原始检查点使用的类型，因此建议以确保最佳精度或进行评估。根据您的硬件，float16 可能会更快。

- 默认情况下，transformers 使用与原始 Meta 代码库相同的采样参数（temperature=0.6 和 top_p=0.9）。我们还没有进行广泛的测试，欢迎探索！

## Llama 3.2 Vision

Vision 模型更大，因此比小型文本模型需要更多的内存来运行。作为参考，11B Vision 模型在 4 位模式下进行推理大约需要 10 GB 的 GPU RAM。

使用指令微调的 Llama Vision 模型进行推理的最简单方法是使用内置的聊天模板。输入有 `user` 和 `assistant` 角色来指示对话轮次。与文本模型的一个区别是，不支持系统角色。用户轮次可以包括图文输入或仅文本输入。要指示输入包含图像，请在输入内容部分添加 `{"type": "image"}`，然后将图像数据传递给 `processor`：

```python
import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor

model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device="cuda",
)
processor = AutoProcessor.from_pretrained(model_id)

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
image = Image.open(requests.get(url, stream=True).raw)

messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "你能用一句话描述这张图片吗？"}
    ]}
]

input_text = processor.apply_chat_template(
    messages, add_generation_prompt=True,
)
inputs = processor(
    image, input_text, return_tensors="pt"
).to(model.device)

output = model.generate(**inputs, max_new_tokens=70)

print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))

## 这张图片描绘了一只穿着蓝色外套和棕色背心的兔子，站在石屋前的泥土路上。
```

你可以继续关于这张图片的对话。然而，请记住，如果你在新的用户轮次中提供了一张新图片，模型将从那时起参考新图片。你不能同时查询两张不同的图片。以下是继续之前对话的示例，我们在对话中添加了助手轮次并询问了一些更多细节：

```python
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "你能用一句话描述这张图片吗？"}
    ]},
    {"role": "assistant", "content": "这张图片描绘了一只穿着蓝色外套和棕色背心的兔子，站在石屋前的泥土路上。"},
    {"role": "user", "content": "背景里有什么？"}
]

input_text = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
)
inputs = processor(image, input_text, return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=70)
print(processor.decode(output[0][inputs["input_ids"].shape[-1]:]))
```

这是我们得到的响应：

```
背景里有一座石屋，屋顶是茅草的，泥土路，一片花田，还有连绵起伏的丘陵。
```

你还可以自动量化模型，以 8 位甚至 4 位模式加载它，使用 `bitsandbytes` 库。以下是如何以 4 位模式加载生成管道的示例：

```diff
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
+from transformers import BitsAndBytesConfig

+bnb_config = BitsAndBytesConfig(
+    load_in_4bit=True,
+    bnb_4bit_quant_type="nf4",
+    bnb_4bit_compute_dtype=torch.bfloat16
)
 
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
-   torch_dtype=torch.bfloat16,
-   device="cuda",
+   quantization_config=bnb_config,
)
```

然后，你可以应用聊天模板，使用处理器，并像之前一样调用模型。

## 设备上运行

你可以在设备上的 CPU/GPU/浏览器中使用以下几个开源库直接运行 Llama 3.2 1B 和 3B。

### Llama.cpp & Llama-cpp-python

[Llama.cpp](https://github.com/ggerganov/llama.cpp) 是所有跨平台设备上 ML 推理的首选框架。我们为 1B 和 3B 模型提供了 4 位和 8 位的量化权重。我们预计社区会接受这些模型并创建额外的量化和微调。你可以在[这里](https://huggingface.co/models?search=hugging-quants/Llama-3.2-)找到所有量化的 Llama 3.2 模型。

以下是如何直接使用 llama.cpp 运行这些检查点的方法。

通过 brew 安装 llama.cpp（适用于 Mac 和 Linux）。

```bash
brew install llama.cpp
```

你可以使用 CLI 运行单次生成或调用 llama.cpp 服务器，该服务器与 Open AI 消息规范兼容。

你可以使用如下命令运行 CLI：

```bash
llama-cli --hf-repo hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF --hf-file llama-3.2-3b-instruct-q8_0.gguf -p "生命的意义和宇宙的意义是"
```

你可以像这样启动服务器：

```bash
llama-server --hf-repo hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF --hf-file llama-3.2-3b-instruct-q8_0.gguf -c 2048
```

你还可以使用 [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) 在 Python 中以编程方式访问这些模型。

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id="hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF",
    filename="*q8_0.gguf",
)
llm.create_chat_completion(
      messages = [
          {
              "role": "user",
              "content": "法国的首都是哪里？"
          }
      ]
)
```

### Transformers.js

你甚至可以在浏览器（或任何 JavaScript 运行时如 Node.js、Deno 或 Bun）中使用 [Transformers.js](https://huggingface.co/docs/transformers.js) 运行 Llama 3.2。你可以在 Hub 上找到 [ONNX 模型](https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct)。如果你还没有安装该库，你可以从 [NPM](https://www.npmjs.com/package/@huggingface/transformers) 使用以下命令安装：

```bash
npm i @huggingface/transformers
```

然后，你可以像这样运行模型：

```js
import { pipeline } from "@huggingface/transformers";

// 创建一个文本生成管道
const generator = await pipeline("text-generation", "onnx-community/Llama-3.2-1B-Instruct");

// 定义消息列表
const messages = [
  { role: "system", content: "你是一个有帮助的助手。" },
  { role: "user", content: "讲个笑话。" },
];

// 生成响应
const output = await generator(messages, { max_new_tokens: 128 });
console.log(output[0].generated_text.at(-1).content);
```

<details>

<summary>示例输出</summary>

```
给你讲个笑话：

你叫什么假面条？

一个 impasta！

希望这让你笑了！你想再听一个吗？
```

</details>

## 微调 Llama 3.2

TRL 支持与 Llama 3.2 文本模型进行聊天和微调：

```bash
# 聊天
trl chat --model_name_or_path meta-llama/Llama-3.2-3B

# 微调
trl sft  --model_name_or_path meta-llama/Llama-3.2-3B \
         --dataset_name HuggingFaceH4/no_robots \
         --output_dir Llama-3.2-3B-Instruct-sft \
         --gradient_checkpointing
```

TRL 还支持使用[这个脚本](https://github.com/huggingface/trl/tree/main/examples/scripts/sft_vlm.py)微调 Llama 3.2 Vision。

```bash
# 在 8x H100 GPU 上测试
accelerate launch --config_file=examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/sft_vlm.py \
    --dataset_name HuggingFaceH4/llava-instruct-mix-vsft \
    --model_name_or_path meta-llama/Llama-3.2-11B-Vision-Instruct \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --output_dir Llama-3.2-11B-Vision-Instruct-sft \
    --bf16 \
    --torch_dtype bfloat16 \
    --gradient_checkpointing
```

你还可以查看[这个笔记本](https://github.com/huggingface/huggingface-llama-recipes/blob/main/Llama-Vision%20FT.ipynb)，了解使用 transformers 和 PEFT 进行 LoRA 微调。

## Hugging Face 合作伙伴集成

我们目前正在与 AWS、Google Cloud、Microsoft Azure 和 DELL 的合作伙伴合作，将 Llama 3.2 11B、90B 添加到 Amazon SageMaker、Google Kubernetes Engine、Vertex AI Model Catalog、Azure AI Studio、DELL Enterprise Hub。我们将在容器可用时更新此部分，您可以订阅 [Hugging Squad](https://mailchi.mp/huggingface/squad) 以获取电子邮件更新。

## 其他资源

- [Hub 上的模型](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf)
- [Hugging Face Llama Recipes](https://github.com/huggingface/huggingface-llama-recipes)
- [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Meta Blog](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)
- [评估数据集](https://huggingface.co/collections/meta-llama/llama-32-evals-66f44b3d2df1c7b136d821f0)

## 致谢

如果没有成千上万的社区成员对 transformers、text-generation-inference、vllm、pytorch、LM Eval Harness 和许多其他项目的贡献，发布这些模型并支持生态系统中的评估是不可能的。向 VLLM 团队致敬，感谢他们在测试和报告问题方面的帮助。如果没有 Clémentine、Alina、Elie 和 Loubna 对 LLM 评估的支持，Nicolas Patry、Olivier Dehaene 和 Daniël de Kok 对 Text Generation Inference 的支持；Lysandre、Arthur、Pavel、Edward Beeching、Amy、Benjamin、Joao、Pablo、Raushan Turganbay、Matthew Carrigan 和 Joshua Lochner 对 transformers、transformers.js、TRL 和 PEFT 的支持；Nathan Sarrazin 和 Victor 使 Llama 3.2 在 Hugging Chat 中可用；Brigitte Tousignant 和 Florent Daudens 对沟通的支持；Julien、Simon、Pierric、Eliott、Lucain、Alvaro、Caleb 和 Mishig 来自 Hub 团队对 Hub 开发和发布功能的贡献，这一切都不可能实现。

还要特别感谢 Meta 团队发布 Llama 3.2 并使其对开源 AI 社区可用！
