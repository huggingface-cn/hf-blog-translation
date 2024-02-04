---
title: "使用开源 LLM 充当 LangChain 智能体" 
thumbnail: /blog/assets/open-source-llms-as-agents/thumbnail_open_source_agents.png
authors:
- user: m-ric
- user: Jofthomas
- user: andrewrreed
translators:
- user: MatrixYao
---

# 使用开源 LLM 充当 LangChain 智能体

## 太长不看版

开源 LLM 现已达到一定的性能水平，可堪作为智能体工作流的推理引擎。在我们的测试基准上，[Mixtral](https://huggingface.co/blog/mixtral) 甚至[已超越 GPT-3.5](#结果)，而且我们还可以通过微调轻松地进一步提高其性能。

## 引言

经由[因果语言建模](https://huggingface.co/docs/transformers/tasks/language_modeling)任务训练出的大语言模型（LLM）可以处理很多任务，但在逻辑、计算及搜索等类型的任务上表现不尽人意。最糟糕的是，它们在数学等领域表现不佳而不自知，仍不自量力地想仅凭一己之力完成所有计算。

为了克服这一弱点，方法之一就是将 LLM 集成到一个含有若干可调用工具的系统中，我们称这样的系统为 LLM 智能体（agent）。

本文，我们首先解释了 ReAct 智能体的内在工作原理，然后展示了如何使用最近集成到 LangChain 中的 `ChatHuggingFace` 接口来构建自己的智能体。最后，我们把几个开源 LLM 与 GPT-3.5 和 GPT-4 一起在同一基准测试上进行了比较。

## 目录

- [使用开源 LLM 充当 LangChain 智能体](#使用开源-llm-充当-langchain-智能体)
  - [太长不看版](#太长不看版)
  - [引言](#引言)
  - [目录](#目录)
  - [什么是智能体？](#什么是智能体)
    - [ReAct 智能体内在机制示例](#react-智能体内在机制示例)
    - [智能体系统面临的挑战](#智能体系统面临的挑战)
  - [使用 LangChain 运行智能体](#使用-langchain-运行智能体)
  - [智能体对决：开源 LLM 充当通用推理智能体的表现如何？](#智能体对决开源-llm-充当通用推理智能体的表现如何)
    - [评估](#评估)
    - [模型](#模型)
    - [结果](#结果)

## 什么是智能体？

LLM 智能体的定义相当宽泛：LLM 智能体是所有使用 LLM 作为引擎并基于观察对环境采取相应行动的系统。其使用 `感知⇒反思⇒行动` 的多轮迭代来完成任务，也经常通过规划或知识管理系统来增强性能。如对该领域的全景感兴趣，可参考 [Xi et al., 2023](https://huggingface.co/papers/2309.07864) 这篇论文。

本文重点关注 **ReAct 智能体**。[ReAct](https://huggingface.co/papers/2210.03629) 用“**推理**”和“**行动**”这两个词串联起智能体的工作流。我们通过提示告诉模型可以使用哪些工具，并要求它“一步一步”（即[思维链](https://huggingface.co/papers/2201.11903)）思考并行动，直至获得最终答案。

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png" alt="ReAct 智能体" width=90%>
</p>

### ReAct 智能体内在机制示例

上图看上去很高端，但实现起来其实非常简单。

可以参考一下[这个 notebook](https://colab.research.google.com/drive/1j_vsc28FwZEDocDxVxWJ6Fvxd18FK8Gl?usp=sharing)，这里，我们用 `transformers` 库实现了一个简单的工具调用示例。

我们用下述提示模板循环调用 LLM：

```
Here is a question: "{question}" 
You have access to these tools: {tools_descriptions}. 
You should first reflect with ‘Thought: {your_thoughts}’, then you either:
- call a tool with the proper JSON formatting,
- or your print your final answer starting with the prefix ‘Final Answer:’
```
等 LLM 输出回答后，就用如下方式解析其回答：

- 如果回答中包含字符串 `‘Final Answer:’`，则结束循环并打印答案。
- 否则，LLM 会输出一个工具调用。你可以解析此输出以获取工具名及参数，并使用所述参数调用所述工具。然后，将此次工具调用的输出附加到提示中，并把扩展后的提示输入给 LLM，直到它有足够的信息生成最终答案。

举个例子，当回答问题 `How many seconds are in 1:23:45?` 时，LLM 的输出可能如下所示：

```
Thought: I need to convert the time string into seconds.

Action:
{
    "action": "convert_time",
    "action_input": {
    "time": "1:23:45"
    }
}
```

由于此回答中不含字符串 `‘Final Answer:’`，所以其输出的应该是一个工具调用。此时，我们解析此输出并获取工具调用参数：使用参数 `{"time": "1:23:45"}` 调用工具 `convert_time`。

可以看到，工具返回了 `{'seconds': '5025'}`。

此时，我们将整个过程及结果添加到提示中，新提示就变成了如下这样（比之前稍微复杂一些了）：

```
Here is a question: "How many seconds are in 1:23:45?"
You have access to these tools:
    - convert_time: converts a time given in hours:minutes:seconds into seconds.

You should first reflect with ‘Thought: {your_thoughts}’, then you either:
- call a tool with the proper JSON formatting,
- or your print your final answer starting with the prefix ‘Final Answer:’

Thought: I need to convert the time string into seconds.

Action:
{
    "action": "convert_time",
    "action_input": {
    "time": "1:23:45"
    }
}
Observation: {'seconds': '5025'}
```

➡️ 我们再次调用 LLM，并将这个新提示输入给它。鉴于它在 `Observation` 字段中得到了工具返回的结果，这轮 LLM 很有可能输出如下：

```
Thought: I now have the information needed to answer the question.
Final Answer: There are 5025 seconds in 1:23:45.
``````

至此，任务解决！

### 智能体系统面临的挑战

智能体系统中的 LLM 引擎需要克服以下几个难点：

1. 从候选工具集中选出能实现预期目标的工具：例如当被问到`“大于 30,000 的最小素数是多少？”`时，智能体可以调用 `Search` 工具，并问它`“K2 的高度是多少”，但这么做无济于事。

2. 以规定的参数格式调用工具：例如，当尝试计算 10 分钟内行驶了 3 公里的汽车的速度时，必须调用 `Calculator` 以让其执行“距离”除以“时间”的操作，假设 `Calculator` 工具能接受 JSON 格式的调用： `{"tool": "Calculator", "args": "3km/10min"}` ，看上去很简单，但其实会有很多小陷阱，一步不慎就前功尽弃，例如：
    - 工具名称拼写错误：`“calculator”` 或 `“Compute”` 是无效的
    - 仅给出参数名而未给出参数值：`“args”: “distance/time”`
    - 参数格式未标准化：`“args”："3km in 10minutes”`
  
3. 有效吸收并使用历史信息，无论是原始上下文信息还是前面若干轮工具调用所返回的观察。

那么，在真实场景中如何设置并使用智能体呢？

## 使用 LangChain 运行智能体

我们最近封装了一个 `ChatHuggingFace` 接口，你可以利用它在 [🦜🔗LangChain](https://www.langchain.com/) 中使用开源模型创建智能体。

要创建 ChatModel 并为其提供工具，代码非常简单，你可在 [Langchain 文档](https://python.langchain.com/docs/integrations/chat/huggingface) 中查阅所有内容。

```python
from langchain_community.llms import HuggingFaceHub
from langchain_community.chat_models.huggingface import ChatHuggingFace

llm = HuggingFaceHub(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
)

chat_model = ChatHuggingFace(llm=llm)
```

你可以通过给 `chat_model` 提供 ReAct 风格的提示和工具，将其变成智能体：

```python
from langchain import hub
from langchain.agents import AgentExecutor, load_tools
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import (
    ReActJsonSingleInputOutputParser,
)
from langchain.tools.render import render_text_description
from langchain_community.utilities import SerpAPIWrapper

# setup tools
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# setup ReAct style prompt
prompt = hub.pull("hwchase17/react-json")
prompt = prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools]),
)

# define the agent
chat_model_with_stop = chat_model.bind(stop=["\nObservation"])
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_log_to_str(x["intermediate_steps"]),
    }
    | prompt
    | chat_model_with_stop
    | ReActJsonSingleInputOutputParser()
)

# instantiate AgentExecutor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

agent_executor.invoke(
    {
        "input": "Who is the current holder of the speed skating world record on 500 meters? What is her current age raised to the 0.43 power?"
    }
)
```

智能体第一轮输出如下：

```markdown
Thought: To answer this question, I need to find age of the current speedskating world record holder.  I will use the search tool to find this information.
Action:
{
    "action": "search",
    "action_input": "speed skating world record holder 500m age"
}
Observation: ...
```

## 智能体对决：开源 LLM 充当通用推理智能体的表现如何？

你可在[此处](https://github.com/aymeric-roucher/benchmark_agents/)找到我们使用的基准测试代码。

### 评估

我们想要度量开源 LLM 作为通用推理智能体时的表现。因此，我们选用的问题都是需要依赖逻辑推演以及一些基本工具的使用才能回答出来的。这里，我们将所需工具限制为计算器和互联网搜索。

[最终数据集](https://huggingface.co/datasets/m-ric/agents_small_benchmark) 结合了以下 3 个数据集的样本：

- 为了测试互联网搜索能力，我们从[HotpotQA](https://huggingface.co/datasets/hotpot_qa)中选择了一些问题，该数据集原本是一个检索数据集，但在可以访问互联网时，其可用于通用问答场景。有些问题原先需要结合多个不同来源的信息，对这类问题，我们可以执行多次互联网搜索来综合出最终结果。

- 为了用上计算器，我们添加了来自 [GSM8K](https://huggingface.co/datasets/gsm8k) 的一些问题，该数据集用于测试小学数学四则运算（加、减、乘、除）的能力。

- 我们还从 [GAIA](https://huggingface.co/papers/2311.12983) 中挑选了一些问题，该数据集是面向通用人工智能助手的一个非常困难的基准测试集。原始数据集中的问题会需要用到很多不同的工具，如代码解释器或 pdf 阅读器，我们精心挑选了一些只需使用搜索和计算器的问题。

评估时，我们选用 [Prometheus 格式](https://huggingface.co/kaist-ai/prometheus-13b-v1.0)作为提示格式，并请 GPT-4 对结果以 5 分制李克特量表（Likert scale）形式进行评分。具体使用的提示模板可参阅[此处](https://github.com/aymeric-roucher/benchmark_agents/blob/master/scripts/prompts.py)。

### 模型

我们对下列先进开源模型进行了评估：

- [Llama2-70b-chat](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf)
- [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)
- [OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)
- [Zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)
- [SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0)

上述模型的评估是基于 LangChain 的 [ReAct 实现](https://github.com/langchain-ai/langchain/tree/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/langchain/langchain/agents/react)的。因此，其函数调用模板为：

```json
{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}
```

为了进行比较，我们还使用 LangChain 的 [OpenAI 智能体](https://github.com/langchain-ai/langchain/tree/021b0484a8d9e8cf0c84bc164fb904202b9e4736/libs/langchain/langchain/agents/openai_functions_agent)在相同的数据集上对 GPT-3.5 和 GPT-4 进行了评估。此时，我们使用 OpenAI 自己的函数调用模板，模型已经针对该模板进行了调优，因此我们认为此时 OpenAI 模型性能处于最佳水平。

### 结果

由于并没有专门针对给定调用函数格式进行微调，从这个角度讲，与 OpenAI 智能体相比，开源模型智能体稍有劣势。

尽管如此，还是有一些模型表现让人眼前一亮！ 💪

下面给出了 Mixtral-8x7B 回答问题 `“Which city has a larger population, Guiyang or Tacheng?”` 的一个例子。

```markdown
Thought: To answer this question, I need to find the current populations of both Guiyang and Tacheng. I will use the search tool to find this information.
Action:
{
    "action": "search",
    "action_input": "current population of Guiyang"
}
4.697 million (2016)

Thought: I have obtained the population of Guiyang, which is 4.697 million (as of 2016). Now, I will use the search tool again to find the population of Tacheng.
Action:
{
    "action": "search",
    "action_input": "current population of Tacheng"
}
Population: Xinjiang: Tacheng data was reported at 1,110.000 Persons in 2021. This records a decrease from the previous number of 1,138.638 Persons for 2020.
Population: Xinjiang: Tacheng data is updated yearly, averaging 1,023.000 Persons from Dec 2005 to 2021, with 17 observations.

I have obtained the population of Tacheng, which is approximately 1.11 million (as of 2021). Comparing the two populations, Guiyang has a larger population than Tacheng.

Thought: I now know the final answer
Final Answer: Guiyang has a larger population, which is approximately 4.697 million (as of 2016), compared to Tacheng's population of approximately 1.11 million (as of 2021).
```

以下给出了我们测得的各模型在各数据集上的表现（为了便于阅读，我们把 1-5 分制转换成了 0 - 100%）：

<p align="center">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/benchmark_agents.png" alt="智能体性能基准" width="90%">
</p>

如你所见，部分开源模型在智能体工作流中表现欠佳：虽然我们能预料到较小的 Zephyr-7b 可能不尽如人意，但 Llama2-70b 的表现却让我们大跌眼镜。

👉 但 **Mixtral-8x7B 表现得非常好，甚至超越了 GPT-3.5！** 🏆

而且这仅仅是开箱性能：***与 GPT-3.5 不同，Mixtral 并没有针对智能体工作流场景微调过***（据我们所知），这在一定程度上说明其性能还有进步空间。例如，在 GAIA 上，10% 的失败案例是因为 Mixtral 尝试使用错误的参数格式调用工具。**通过针对函数调用和任务规划技能进行适当的微调，Mixtral 的分数有可能会进一步提高。**

➡️我们强烈建议开源社区针对智能体场景微调 Mixtral，以超越 GPT-4！ 🚀

**最后的话：**

- 虽然本文仅使用了 GAIA 基准的一小部分问题和工具，但该基准似乎有潜力对智能体工作流整体性能进行可靠度量，因为其测例通常需要多步推理以及严格的逻辑。

- 智能体工作流可以提高 LLM 的性能。例如，在 GSM8K 数据集上，[GPT-4 的技术报告](https://arxiv.org/pdf/2303.08774.pdf) 表明 5-样本 CoT 提示的成功率为 92%；但一旦使用计算器工具，零样本的成功率就能提升到 95%。而对 Mixtral-8x7B，[LLM 排行榜](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) 测得其 5-样本成功率仅为 57.6%，但我们在智能体工作流中测得的零样本成功率为 73%。 _（注意，我们只测试了 GSM8K 数据集中的 20 个问题。）_

> 英文原文: <url> https://huggingface.co/blog/open-source-llms-as-agents </url>
> 原文作者：Aymeric Roucher，Joffrey Thomas，Andrew Reed
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。
