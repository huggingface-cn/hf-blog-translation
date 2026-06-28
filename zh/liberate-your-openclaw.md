---
title: "解放 OpenClaw"
thumbnail: /blog/assets/liberate-your-openclaw/thumbnail.png
authors:
- user: clem
- user: burtenshaw
- user: pcuenq
- user: jeffboudier
- user: merve
- user: nielsr
- user: victor
- user: mishig
translators:
- user: HCS9527
---

# 解放 OpenClaw 🦀

Anthropic 正在限制 Pro/Max 订阅用户在开放智能体平台中访问 Claude 模型。Hugging Face 上有很多优秀的开放模型，可以让智能体继续运行，通常成本也低得多。

如果 OpenClaw、Pi 或 Open Code 智能体被切断访问、需要恢复运行，可以通过两种方式迁移到开放模型：

1. 使用通过 Hugging Face Inference Providers 托管的开放模型。
2. 在自有硬件上完全本地运行开放模型。

托管方案是让可用智能体快速恢复运行的最快方式。如果需要隐私、零 API 成本和完全控制，本地方案更合适。

可以直接告诉 Claude Code、Cursor 或常用智能体：_help me move my OpenClaw agents to Hugging Face models_，并附上本文链接。

## Hugging Face Inference Providers

Hugging Face Inference Providers 是一个开放平台，可将请求路由到开源模型服务提供方。如果需要使用性能更好的模型，或者缺少所需硬件，这是合适的选择。

首先，需要在 [Hugging Face token 设置页面](https://huggingface.co/settings/tokens)创建访问令牌（token）。然后可以这样把该 token 添加到 `openclaw`：

```shell
openclaw onboard --auth-choice huggingface-api-key
```

按提示粘贴 Hugging Face token，随后会要求选择一个模型。

推荐使用 [GLM-5](https://huggingface.co/zai-org/GLM-5)，因为它在 [Terminal Bench](https://huggingface.co/datasets/harborframework/terminal-bench-2.0) 上表现很好。当然，也可以在 [Inference Providers 模型列表](https://huggingface.co/inference/models)中选择其他数千个模型。

随时可以在 OpenClaw 配置中输入模型的 `repo_id`，更新 Hugging Face 模型：

```
{
  agents: {
    defaults: {
      model: {
        primary: "huggingface/zai-org/GLM-5:fastest"
      }
    }
  }
}
```

注意：HF PRO 订阅用户每月可获得 2 美元免费额度，可用于 Inference Providers。更多信息见 [HF PRO 页面](https://huggingface.co/pro)。

## 本地设置

本地运行模型可以提供完整的隐私控制、零 API 成本，以及不受速率限制影响的实验能力。

安装 `llama.cpp`。这是一个完全开源的低资源占用推理库。

```shell
# on mac or linux
brew install llama.cpp

# on windows
winget install llama.cpp
```

启动一个带内置 Web UI 的本地服务器：

```shell
llama-server -hf unsloth/Qwen3.5-35B-A3B-GGUF:UD-Q4_K_XL
```

本文示例使用的是 Qwen3.5-35B-A3B，它在 32GB RAM 上运行效果很好。如果有不同需求，请查看目标模型的[硬件兼容性说明](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF)。[还有数千个模型可供选择](https://huggingface.co/models?pipeline_tag=text-generation&library=gguf&sort=trending)。

如果在 llama.cpp 中加载 GGUF，可以使用如下 OpenClaw 配置：

```shell
openclaw onboard --non-interactive \
   --auth-choice custom-api-key \
   --custom-base-url "http://127.0.0.1:8080/v1" \
   --custom-model-id "unsloth-qwen3.5-35b-a3b-gguf" \
   --custom-api-key "llama.cpp" \
   --secret-input-mode plaintext \
   --custom-compatibility openai
```

验证服务器正在运行，并且模型已加载：

```shell
curl http://127.0.0.1:8080/v1/models
```

## 应该选择哪条路径？

如果希望以最快方式恢复一个可用的 OpenClaw 智能体，使用 Hugging Face Inference Providers。如果需要隐私、完整本地控制，并且不想产生 API 费用，使用 `llama.cpp`。

无论选择哪种方式，都不需要依赖闭源托管模型来让 OpenClaw 重新运行起来。
