---
title: "Open Responses：需要了解的关键内容"
thumbnail: /blog/assets/openresponses/thumbnail.png
authors:
- user: evalstate
- user: burtenshaw
- user: merve
- user: pcuenq
translators:
- user: HCS9527
---

Open Responses 是一个新的开放推理标准。它由 OpenAI 发起，由开源 AI 社区构建，并得到 Hugging Face 生态系统支持。Open Responses 基于 Responses API，面向智能体的未来而设计。本文将介绍 Open Responses 的工作方式，以及开源社区为什么应该使用 Open Responses。

聊天机器人的时代早已过去，智能体正在主导推理工作负载。开发者正在转向能够在较长时间跨度内推理、规划并行动的自主系统。尽管发生了这种转变，生态系统中的许多项目仍在使用 **Chat Completion** 格式。它最初是为回合制对话设计的，难以满足智能体化用例。**Responses 格式** 旨在解决这些限制，但它是封闭的，也尚未被广泛采用。因此，即使存在替代方案，**Chat Completion** 格式仍然是事实标准。

智能体化工作流的需求与既有接口之间的错配，正是开放推理标准的必要性所在。未来几个月，我们将与社区和推理提供方合作，把 Open Responses 实现并适配为一种共享格式，使其在实践中能够替代 Chat Completion 接口。

Open Responses 沿着 OpenAI 在 2025 年 3 月发布 [*Responses API*](https://platform.openai.com/docs/api-reference/responses) 时设定的方向继续推进。Responses API 取代了既有的 Completion 和 Assistants API，用一种一致的方式支持：

- 生成文本、图像和 JSON 结构化输出
- 通过单独的基于任务的端点创建视频内容
- 在提供方侧运行智能体循环，自主执行工具调用并返回最终结果。

## 什么是 Open Responses？

Open Responses 扩展并开源了 Responses API，让构建者和路由提供方更容易互操作，并围绕共同需求协作。

其中一些关键点包括：

- 默认无状态，并支持有需要的提供方使用加密推理。
- 标准化的模型配置参数。
- 流式输出被建模为一系列语义事件，而不是原始文本或对象增量。
- 可通过特定模型提供方的可配置参数进行扩展。

## 使用 Open Responses 构建应用需要了解什么？

下面会简要介绍会影响多数社区成员的核心变化。如果想深入了解规范本身，可以查看 [Open Responses 文档](https://www.openresponses.org/)。

### 向 Open Responses 发起客户端请求

向 Open Responses 发起客户端请求的方式与现有 Responses API 类似。下面用 curl 演示一次对 Open Responses API 的请求。这里调用的是一个代理端点，它会使用 Open Responses API schema 路由到 Inference Providers。

```diff
 curl https://evalstate-openresponses.hf.space/v1/responses \
   -H "Content-Type: application/json" \
   -H "Authorization: Bearer $HF_TOKEN" \
+  -H "OpenResponses-Version: latest" \
   -N \
   -d '{
         "model": "moonshotai/Kimi-K2-Thinking:nebius",
         "input": "explain the theory of life"
       }'
```

### 推理客户端和提供方的变化

已经支持 Responses API 的客户端，可以用相对较少的工作量迁移到 Open Responses。主要变化涉及推理内容的暴露方式：

- 扩展推理可见性：Open Responses 为推理项正式定义了三个可选字段：`content`（原始推理轨迹）、`encrypted_content`（提供方特定的受保护内容）和 `summary`（从原始轨迹清洗而来的摘要）。

OpenAI 模型过去只暴露 `summary` 和 `encrypted_content`。有了 Open Responses，提供方可以通过 API 暴露原始推理内容。从过去只返回摘要和加密内容的提供方迁移过来的客户端，现在有机会在所选提供方支持时接收并处理原始推理流。

- 实现更丰富的状态变化和载荷，包括更详细的可观测性。例如，托管的 Code Interpreter 可以发送一个具体的 `interpreting` 状态，以便在长时间运行的操作中提升智能体和用户可见性。

对于模型提供方来说，如果它们已经遵循 Responses API 规范，实现 Open Responses 所需的变更应该比较直接。对于路由服务来说，现在则有机会围绕一致的端点实现标准化，并在需要时支持用于定制的配置选项。

随着时间推移，当提供方持续创新时，某些功能会逐步进入基础规范并标准化。

总结来说，迁移到 Open Responses 会让推理体验更加一致，也会提升质量，因为旧版 Completions API 中那些未文档化的扩展、解释方式和变通方案，会在 Open Responses 中得到规范化。

下面可以看到如何流式输出推理片段。

```json
 {
  "model": "moonshotai/Kimi-K2-Thinking:together",
  "input": [
    {
      "type": "message",
      "role": "user",
      "content": "explain photosynthesis."
    }
  ],
  "stream": true
}

```

下面展示的是获取 Open Response 与使用 OpenAI Responses 获取推理增量之间的差异：

```json
// Open weight models stream raw reasoning
event: response.reasoning.delta
data: { "delta": "User asked: 'Where should I eat...' Step 1: Parse location...", ... }

// Models with encrypted reasoning send summaries, or sent as a convenience by Open Weight models
event: response.reasoning_summary_text.delta
data: { "delta": "Determined user wants restaurant recommendations", ... }

```

### 面向路由的 Open Responses

Open Responses 区分「模型提供方」（Model Providers）和「路由服务」（Routers）：前者提供推理，后者是在多个提供方之间进行编排的中间层。

客户端现在可以在发起请求时指定提供方，并附带提供方特定的 API 选项，从而让中间路由服务在上游提供方之间编排请求。

### 工具

Open Responses 原生支持两类工具：内部工具和外部工具。外部托管工具在模型提供方系统之外实现，例如需要在客户端侧执行的函数，或 MCP 服务器。内部托管工具则位于模型提供方系统之内，例如 OpenAI 的文件搜索或 Google Drive 集成。模型完全在提供方基础设施内调用、执行并检索结果，不需要开发者介入。

### 子智能体循环

Open Responses 正式定义了智能体化循环。这个循环通常由推理、工具调用和响应生成构成，并不断重复，让模型能够自主完成多步骤任务。

![流程图](https://huggingface.co/huggingface/documentation-images/resolve/main/openresponses/image1.png)

[图片来源：openresponses.org](https://www.openresponses.org/specification#the-agentic-loop)

这个循环的运行方式如下：

1. API 接收用户请求，并从模型采样
2. 如果模型发出工具调用，API 会执行该调用（内部或外部）
3. 工具结果被反馈给模型，用于继续推理
4. 循环不断重复，直到模型发出完成信号

对于内部托管工具，提供方会管理整个循环：执行工具、把结果返回给模型，并流式输出结果。这意味着「搜索文档、总结发现，然后起草一封邮件」这样的多步骤工作流可以通过单次请求完成。

客户端可以通过 `max_tool_calls` 控制循环行为、限制迭代次数，并通过 `tool_choice` 约束哪些工具可以被调用：

```json
{
  "model": "zai-org/GLM-4.7",
  "input": "Find Q3 sales data and email a summary to the team",
  "tools": [...],
  "max_tool_calls": 5,
  "tool_choice": "auto"
}
```

响应会包含所有中间项：工具调用、结果和推理内容。

## 下一步

Open Responses 扩展并改进了 Responses API，提供更丰富、更详细的内容定义、兼容性和部署选项。它还提供了一种标准方式，用于在主要推理调用期间执行子智能体循环，从而为 AI 应用打开更强大的能力。我们期待与 Open Responses 团队**以及更广泛的社区**共同推进该规范的未来发展。

![验收测试](https://huggingface.co/huggingface/documentation-images/resolve/main/openresponses/image2.png)

现在可以通过 [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers/index) 试用 Open Responses。我们已经提供了一个可在 [Hugging Face Spaces](https://huggingface.co/spaces/evalstate/openresponses) 上使用的早期访问版本。欢迎今天就用你的客户端和 Open Responses Compliance Tool 试试看！
