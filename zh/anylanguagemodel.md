---
title: "隆重推出 AnyLanguageModel：在 Apple 平台统一本地与远程大语言模型的 API "
thumbnail: /blog/assets/anylanguagemodel/banner.png
authors:
- user: mattt
  guest: true
translators:
- user: chenglu
---

# 隆重推出 AnyLanguageModel：在 Apple 平台统一本地与远程大语言模型的 API

大语言模型（LLM）已成为构建现代软件不可或缺的工具。
但对于 Apple 平台的开发者来说，集成这些模型仍然不够友好。

在开发 AI 驱动的应用时，开发者通常采用混合方案，比如：

* 使用 Core ML 或 MLX 运行本地模型，提升隐私性并支持离线运行
* 通过 OpenAI、Anthropic 等云服务获取先进模型能力
* 使用 Apple 的 Foundation Models 作为系统级的默认模型

但每种方案都有各自不同的 API、配置要求和集成方式，开发门槛高、整合成本大。
我在采访开发者时，模型集成的困难几乎是所有人都提到的问题。
一位开发者直言不讳地说：

> 我本来想随便跑个 demo 测试一下，快速做个雏形，
> 没想到浪费了这么多时间，搞得我头都大了。

高昂的尝试成本阻碍了开发者发现：其实本地开源模型就能很好地满足很多场景需求。

现在我们发布了 [AnyLanguageModel](https://github.com/mattt/AnyLanguageModel)，
这是一个 Swift 包，可作为 Apple Foundation Models 框架的直接替代，
同时支持多个模型服务商。我们的目标是：

* 降低在 Apple 平台上使用 LLM 的难度
* 鼓励开发者采用可本地运行的开源模型

## 解决方案：只需替换 `import`，API 保持不变

核心理念很简单：
**只需替换导入语句，原有代码几乎不用改。**

```diff
- import FoundationModels
+ import AnyLanguageModel
```

实战示例如下：
先来看 Apple 内置模型的用法：

```swift
let model = SystemLanguageModel.default
let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "Explain quantum computing in one sentence")
print(response.content)
```

然后用 MLX 本地运行的开源模型（如 Qwen）：

```swift
let model = MLXLanguageModel(modelId: "mlx-community/Qwen3-4B-4bit")
let session = LanguageModelSession(model: model)

let response = try await session.respond(to: "Explain quantum computing in one sentence")
print(response.content)
```

**AnyLanguageModel 支持多种模型提供商，包括：**

* **Apple Foundation Models**：原生集成 Apple 的系统模型（支持 macOS 26+ / iOS 26+）
* **Core ML**：通过神经引擎加速运行已转换的本地模型
* **MLX**：在 Apple Silicon 上高效运行量化模型
* **llama.cpp**：通过 llama.cpp 后端加载 GGUF 格式模型
* **Ollama**：通过 Ollama 的 HTTP API 连接本地模型服务
* **OpenAI、Anthropic、Google Gemini**：云端模型服务商，便于对比和兜底
* **Hugging Face Inference Providers**：支持数百个云端模型，由业界领先的 [推理服务商](https://huggingface.co/docs/inference-providers/en/index) 提供计算支持

我们特别关注本地模型，建议通过 [Hugging Face Hub](https://huggingface.co/docs/hub/) 下载使用。
当然，也保留云服务选项，降低上手门槛，并为迁移提供路径：**“先跑起来，再优化。”**

## 为什么以 Foundation Models 作为基础 API？

我们在设计 AnyLanguageModel 时，面临一个选择：

* 是重新设计一套抽象？
* 还是基于已有的 API 构建？

我们选择后者，以 Apple 推出的 [Foundation Models 框架](https://developer.apple.com/documentation/FoundationModels) 为模板。

这看起来可能有点反直觉——为什么要绑定 Apple 的设计？
但我们有几个理由：

1. Foundation Models 的 API 设计真的很棒。
   它充分利用了 Swift 的特性，比如宏，开发体验优秀；
   它的抽象（如会话、工具、生成等）也高度契合 LLM 的使用方式。

2. 它的功能是有意做了限制的。
   Foundation Models 可以看作是大语言模型能力的“最小公分母”。
   我们并不认为这是缺点，反而把它当作一个稳定的基础（嘿嘿）。
   几乎所有面向 Apple 平台的 Swift 开发者都需要接触这套 API，
   直接在它之上构建，可以大大减少理解和上手成本。

3. 它让我们保持专注。
   每多加一层抽象，就会让你离实际问题更远。
   抽象固然有用，但叠加太多，反而会带来新的复杂性。

最终的好处是：在不同模型服务商之间切换时，代码几乎不用改；
而且核心抽象干净、统一、可预测。

## 包的可选特性：按需加载，避免依赖臃肿

多后端的库常常会出现依赖太多的问题。
比如你只想跑 MLX 模型，却不得不引入 llama.cpp 及其庞大依赖。

**AnyLanguageModel 使用 Swift 6.1 的包特性（Package Traits）解决这一问题。**
你可以按需引入需要的后端支持：

```swift
dependencies: [
    .package(
        url: "https://github.com/mattt/AnyLanguageModel.git",
        from: "0.4.0",
        traits: ["MLX"]  // Pull in MLX dependencies only
    )
]
```

支持的 traits 包括：`CoreML`、`MLX`、`Llama`（支持 llama.cpp 和 [llama.swift](https://github.com/mattt/llama.swift)）
默认情况下，只包含基础 API 和云服务支持，不引入重依赖，后者仅依赖标准的 `URLSession` 网络请求。

对于 Xcode 项目（目前还不支持直接声明 trait），
你可以创建一个内部的 Swift 包，指定所需 trait 依赖 AnyLanguageModel，
然后将这个包作为本地依赖添加到你的项目中。
具体操作方法可以参考 [README](https://github.com/mattt/AnyLanguageModel#using-traits-in-xcode-projects) 中的详细说明。

## 图像支持（以及 API 设计的权衡）

[视觉语言模型](https://huggingface.co/blog/vlms-2025) 如今非常强大且广泛应用。
它们可以对图片进行描述、从截图中提取文字、分析图表，甚至可以回答与图片内容相关的问题。
但遗憾的是，Apple 的 Foundation Models 框架目前还不支持在 prompt 中发送图片。

基于已有 API 进行开发，就意味着要接受它的限制。
Apple 未来大概率会在后续版本（也许 iOS 27？）加入图像支持，
但视觉语言模型实在太实用了，等不了那么久。
因此，我们在现有 Foundation Models 的基础上进行了扩展，提供了更多功能。

例如，下面是发送图片给 Claude 的代码：

```swift
let model = AnthropicLanguageModel(
    apiKey: ProcessInfo.processInfo.environment["ANTHROPIC_API_KEY"]!,
    model: "claude-sonnet-4-5-20250929"
)

let session = LanguageModelSession(model: model)
let response = try await session.respond(
    to: "What's in this image?",
    image: .init(url: URL(fileURLWithPath: "/path/to/image.png"))
)
```

我们在这里做了有意识的取舍，
也许我们现在的设计将来会和 Apple 官方的实现产生冲突，
但这正是废弃警告（deprecation warnings）存在的意义。
有时候，你必须为“还不存在的框架”提前设计好 API。

## 快速体验：chat-ui-swift 应用示例

![chat-ui-swift 应用截图](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/any-swift-llm/chat-ui-swift.png)

想要直观体验 AnyLanguageModel 的实际效果，
可以试用 [chat-ui-swift](https://github.com/mattt/chat-ui-swift) 这个 SwiftUI 聊天应用，
它完整展示了该库的各种功能。

这个应用包含：

* 通过 Foundation Models 集成 Apple Intelligence（支持 macOS 26+）
* 支持 Hugging Face OAuth 登录，访问受限模型
* 支持流式响应
* 支持聊天记录持久化

这个项目是一个起点：
你可以 fork 它、扩展它、替换不同的模型，
了解各个模块如何协作，并根据自己的需求进行定制。

## 下一步计划

AnyLanguageModel 当前版本仍为 pre-1.0。
核心 API 已稳定，
但我们仍在努力将 Foundation Models 的全部特性引入到所有适配器中，包括：

* **工具调用（Tool calling）**：适配所有模型服务商
* **MCP 集成（MCP integration）**，用于工具调用与引导
* **结构化输出的引导生成（Guided generation）**
* 本地推理的性能优化

这个库只是迈向更大目标的第一步。
统一的推理 API 能为 Apple 平台上的智能代理类应用打下基础——
让模型能够调用工具、访问系统资源、完成复杂任务。
更多内容，敬请期待。🤫

## 一起参与进来

我们非常欢迎你来一起完善这个项目：

* **试用它** —— 实际开发点什么，体验下功能
* **分享你的反馈** —— 哪些地方顺手？哪些有痛点？我们很想听听你在集成 AI 到应用时遇到的挑战
* **提交 issue** —— 不管是功能建议、Bug 报告还是技术问题都欢迎
* **参与贡献** —— 欢迎提交 PR

## 链接汇总

* [AnyLanguageModel on GitHub](https://github.com/mattt/AnyLanguageModel)
* [chat-ui-swift on GitHub](https://github.com/mattt/chat-ui-swift)

我们期待看到你能用它打造出什么精彩作品 🦾
