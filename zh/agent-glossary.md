---
title: "Harness、Scaffold，以及值得厘清的 AI 智能体术语"
thumbnail: /blog/assets/agent-glossary/thumbnail.png
authors:
- user: sergiopaniego
- user: ariG23498
translators:
- user: HCS9527
---

# Harness、Scaffold，以及值得厘清的 AI 智能体术语

当一个领域快速发展时，它的词汇往往比共识变化得更快。术语会开始变得含混，被用于不同语境，或者变成某些想法的简写，而这些想法本身从未被充分解释。我们现在也在 AI 智能体领域看到这种情况：一些概念被混在一起，一些概念换了名字，还有一些词流行几个月后又悄然消失。

这会让刚入门的人感到不知所措，即使是试图跟上最新进展的从业者也一样。ICLR 2026 之后，我们中的一位作者（[@ariG23498](https://x.com/ariG23498/status/2049668725511737663)）提出了一个问题，很好地概括了这种困惑：

> *「在智能体语境中，你们所说的「harness」和「scaffold」到底是什么意思？我在 ICLR 听过很多解释，但不明白为什么这些解释始终没能收敛成同一种说法。」*

这份术语表试图厘清那些反复出现、却缺少清晰一致解释的术语。它并不是这个领域的完整词典。我们关注的是那些经常被混用、在不同场景下反复出现，或被默认「大家都懂」但其实并不显然的概念。

无论是在构建智能体、部署智能体，还是使用 Claude Code、Codex、Hermes Agent 这类工具，这些术语大多都会出现。最后一节会介绍一些模型训练特有的概念，更适合关注训练侧工作的读者。

> [!NOTE]
> 许多术语目前还没有公认定义，不同框架也会用同一个词指代不同事物。这里的目标不是强行规定一套「正确词汇」，而是提供一个实用的心智模型，让相关讨论更容易理解和跟进。

下面开始。

## 目录

- [Model（模型）](#model)
- [Scaffolding（脚手架）](#scaffolding)
- [Harness](#harness)
- [Agent（智能体）](#agent)
- [Context Engineering（上下文工程）](#context-engineering)
- [Policy（策略）](#policy)
- [Tool Use（工具使用）](#tool-use)
- [Skills（技能）](#skills)
- [Sub-agents（子智能体）](#sub-agents)
- [Training（训练）](#training)
  - [RL Environment（RL 环境）](#rl-environment)
  - [Trainer（训练器）](#trainer)
  - [Rollout（回滚）](#rollout)
  - [Reward（奖励）](#reward)
- [延伸阅读](#learn-more)

## Model

模型就是 LLM：接收文本并输出文本（例如 Claude、Qwen、GPT、Kimi、DeepSeek……）。模型本身在两次调用之间没有记忆，也不具备循环执行能力。它可以表达调用工具的意图，但需要 harness 来真正执行这个调用。它回答一个 prompt，然后停止。配上 scaffolding 和 harness 后，它才会成为智能体。

## Scaffolding

Scaffolding 是围绕模型、定义其行为的一层：system prompt、工具描述、模型响应的解析方式，以及跨步骤保留哪些信息（上下文管理）。无论在训练还是推理阶段，它都会塑造模型如何理解外部世界，以及如何在其中行动。

Claude Code、Codex、Antigravity CLI 等产品会把整套系统都称为 harness。Claude Code 的[官方文档](https://code.claude.com/docs/en/how-claude-code-works)说得很直接：「Claude Code 是围绕 Claude 构建的智能体 harness。」这是广义用法：harness 指模型之外的一切。只有在需要分别推理 scaffold 和 harness 时，这一区分才最重要，例如在训练流水线里。也有人会把「scaffold」用得更宽泛，用来指 harness 依赖的任何基础设施：hooks、运行时配置，甚至目录结构。

有些产品（例如 Claude Code 和 Codex）与其提供方的模型紧密耦合。另一些产品（例如 Antigravity CLI 和 Hermes Agent）则允许接入任意模型。

## Harness

Harness 是智能体内部的执行层：它调用模型、处理模型发出的工具调用，并决定何时停止。harness 让智能体真正运行起来。上面定义的 scaffolding 则是模型工作的依据：它的指令、工具和格式。

**Harness engineering** 是把这一层设计好的工程实践：决定智能体应该何时停止、如何处理错误，以及哪些护栏可以让它不偏离目标。它同时适用于训练和推理。[Addy Osmani 的文章](https://www.oreilly.com/radar/agent-harness-engineering/)和 [OpenAI 关于使用 Codex 构建的经验](https://openai.com/index/harness-engineering/)都从推理侧讨论了这个主题。

在评估时，同一种模式会表现为 **eval harness**：它不收集训练数据，而是在某个模型 checkpoint 上运行一组固定场景，并记录指标，而不是更新权重。

有些框架会用 **orchestrator** 指代更高层的控制器，用来协调多个智能体之间的工作。harness 会驱动一个模型完成执行循环；orchestrator 则把智能体作为单元进行管理，而每个智能体都运行着自己的 harness（见下文「子智能体」）。

## Agent

「Agent」这个词来自强化学习。在强化学习中，智能体只是一个函数：接收 observation，返回 action。环境接收这个 action，并返回新的 observation，如此循环往复。这个循环仍然是 LLM 智能体工作方式的核心。

在 LLM 世界里，这个词的含义被扩展了。智能体是模型加上围绕它的一切，让它不只是响应，还能行动。它把原始文本生成变成一个可以循环行动的系统：接收信息、决定要做什么，并基于结果继续行动。

以 coding agent 为例。system prompt、工具描述，以及模型遵循的输出格式共同构成 scaffolding。调用模型、处理工具调用并决定何时停止的循环就是 harness。在训练时，harness 还会并行运行许多这样的循环，并把结果反馈回去更新模型。

![智能体图示：Agent 内部包含 Harness、Scaffold 和 Model，下方还有 Sub-agent](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agent-glossary/agent-diagram.png)

社区里通常会说 **Agent = Model + Harness**（可参考 [@Vtrivedy10](https://x.com/Vtrivedy10/status/2031408954517971368) 和 [Will Brown 的推文](https://x.com/willccbb/status/2049844685095715289)）。只要不是模型，就是 harness。最容易引发混淆的，是 harness 与 scaffold 之间那一点微妙区别，上面两节正是在处理这个问题。

当人们谈到 Claude Code、Codex 或 Cursor 这类产品时，指的是构建在某个特定模型之上的某个特定 harness，并且二者通常经过共同设计和优化。两个产品即使用的是同一个底层模型，也可能因为 harness 的选择不同而体验完全不同。反过来，在同一个 harness 里换成更好的模型，也会改变体验。模型、harness 和产品是三件不同的事。

## Context Engineering

上下文工程是设计智能体上下文窗口中包含哪些内容：模型在每一步看到什么，system prompt、工具描述、对话历史、检索到的知识等。这不是一次性决策：随着模型运行，前面的轮次会影响后续调用中放入哪些内容，harness 会在整个运行过程中主动管理这些内容。它同时适用于训练和推理，但出错的代价差别很大。在训练时，模型看到的内容会影响它学到什么。如果弄错了，就要重新训练。在推理时，它只是文本：改一个 prompt，然后重新部署即可。[HF Context Engineering Course](https://huggingface.co/learn/context-course/en/unit0/introduction) 对此有深入介绍。

记忆也是这个机制的一部分。**短期记忆** 是单次运行中留在上下文窗口里的内容：对话历史、工具结果、此前的推理。**长期记忆** 会跨会话持久保存，存储在外部，需要时再检索出来，并在相关时注入回上下文。

## Policy

策略是智能体遵循的行为方式：在任意给定情境下，它定义采取每种可能行动的概率。在 LLM 系统中，策略的一部分由模型权重学习得到，但行为也取决于周围的 scaffolding 和 harness。同一个模型可能因为 prompt、工具、记忆和执行循环不同而表现出截然不同的行为。

策略不是智能体。策略定义行为；智能体是在环境中行动的完整系统。把一个 checkpoint 包进 scaffolding 和 harness 并部署出去，就得到了一个智能体，而它的行为就是这个策略。

## Tool Use

工具使用是智能体连接外部世界的方式：API、代码解释器、数据库、Web 搜索、文件系统。模型以结构化格式表达使用工具的意图。现代推理 API 会把这作为一等对象暴露出来：harness 直接接收这个调用，并把它路由到对应函数。结果再被送回上下文，循环继续。

## Skills

Skills 是可复用、结构化的知识包，用于完成多步骤任务。如果说 **tool** 是一个动作（「运行这条命令」），那么 **skill** 会打包完成某个目标所需的一切（「调查这个 bug、形成假设、写出修复」）。它们可以跨智能体移植，并按需加载。tool、skill 和 sub-agent 之间的边界会随框架而变化。[HF Context Engineering Course](https://huggingface.co/learn/context-course/en/unit1/introduction) 对 skills 有深入介绍。

## Sub-agents

子智能体是被另一个智能体调用、用于处理特定子任务的智能体。它有自己的模型和 scaffold，可以独立推理，并返回结果。调用方智能体不需要知道它内部如何工作。这正是 **sub-agent** 区别于 **tool**（函数调用）或 **skill**（打包知识）的地方：子智能体本身可以推理、使用工具，并调用更多子智能体。调用方智能体有时被称为 **orchestrator**。

## Training

无论是在训练还是部署，上面的术语都适用。下面这四个术语则是训练特有的：智能体会运行任务、获得评分，然后其模型权重会被更新。每个面向 LLM 的 RL 训练系统都围绕同一条流水线构建：

![RL 训练流水线图示：RL Environment、Trainer 和 Reward 通过 rollout 与更新后的 policy 相连](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/agent-glossary/rl-pipeline.png)

### RL Environment

环境是任何可以交互的东西：一个有状态对象，接收 action 作为输入，更新自身内部状态，并返回 observation。在 LLM 语境中，action 通常是工具调用。文件系统就是一个简单例子：action `touch foo.txt` 会通过创建文件来更新状态，而 observation 可能是更新后的文件列表。不同框架对环境的定义会有所不同。

我们最近专门发布了一篇相关指南，因此这里不再赘述。完整的类型、框架和示例拆解可以参考 [The Ultimate Guide to RL Environments](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide)。

### Trainer

Trainer 负责让智能体变得更好：它运行许多智能体 episode，对结果打分，并用这些结果更新内部模型的权重。[TRL 的 GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) 是一个具体例子：一个类同时处理 episode 生成、reward 评分和权重更新。

### Rollout

Rollout 是一次从开始到结束的完整智能体运行：智能体看到了什么、做了什么，以及每一步得到了什么 reward。它也会被称为 *trajectory* 或 *trace*，具体取决于语境。这是 RL 算法用于学习的原始数据。

### Reward

Reward 是训练算法用来判断模型是否正在变好的分数。它可以是 *可验证的*（verifiable，例如测试通过/失败、答案匹配），也可以是 *学习得到的*（learned，例如人类偏好、LLM-as-judge）；可以是 *稀疏的*（sparse，即 episode 结束时给一个分数），也可以是 *密集的*（dense，即每一步都给分）。trainer 正是用它来更新内部模型权重。关于每种类型的完整拆解，请参阅 [Adithya](https://huggingface.co/AdithyaSK) 指南中的 [Reward Architecture](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide#dimension-4-reward-architecture) 一节。

**Rubrics（评分标准）** 会把 reward 拆成带权重的多个明确维度，而不是只给一个数字。[OpenEnv](https://github.com/meta-pytorch/OpenEnv) 和 [Verifiers](https://github.com/willccbb/verifiers) 将 rubrics 实现为可以组合的对象（`WeightedSum`、`Sequential`、`Gate`）。

## Learn More

- [@Vtrivedy10: The Anatomy of an Agent Harness](https://x.com/Vtrivedy10/status/2031408954517971368)：详细拆解 harness 的组成部分及其存在原因
- [Agent Harness Engineering](https://www.oreilly.com/radar/agent-harness-engineering/)：围绕 Agent = Model + Harness 的趋同式表述，并包含 coding agent 示例
- [Harness Engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/)：一篇关于完全使用 Codex 智能体构建产品的真实经验文章，涵盖推理阶段的 scaffolding、反馈循环和上下文管理
- [Tool Schema Rendering Atlas](https://huggingface.co/spaces/evalstate/tool-research)（evalstate）：展示不同模型中的工具 schema 如何变成 prompt 文本，以及经过提供方模板处理后每个模型实际看到的内容
- [Simon Willison 的 How coding agents work blog](https://simonwillison.net/guides/agentic-engineering-patterns/how-coding-agents-work/)：以 harness 的视角解释 coding agent 如何工作
- [AI Engineer talks like Harnesses in AI: A Deep Dive](https://www.youtube.com/watch?v=C_GG5g38vLU)：介绍 harness 是什么以及如何构建 harness
- [The Ultimate Guide to RL Environments](https://huggingface.co/spaces/AdithyaSK/rl-environments-guide)：逐框架比较和术语映射
- [Continually improving our agent harness](https://cursor.com/blog/continually-improving-agent-harness)：Cursor 如何把 harness 作为产品持续迭代
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)：经典的 eval harness

*如果某个定义让人觉得不够精确，或发现我们漏掉了某个术语，欢迎告诉我们。*

*感谢 [Pedro Cuenca](https://huggingface.co/pcuenq)、[Quentin Gallouédec](https://huggingface.co/qgallouedec)、[Shaun Smith](https://huggingface.co/evalstate) 和 [Adithya S Kolavi](https://huggingface.co/AdithyaSK) 审阅本文。*
