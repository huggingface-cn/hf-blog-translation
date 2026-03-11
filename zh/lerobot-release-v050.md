---
title: "LeRobot v0.5.0：正式发布"
thumbnail: /blog/assets/lerobot-release-v050/thumbnail.png
authors:
  - user: imstevenpmwork
  - user: pepijn223
  - user: jadechoghari
  - user: CarolinePascal
  - user: lilkm
  - user: nepyope
  - user: Nico-robot
  - user: aractingi
  - user: VirgileBatto
  - user: thomwolf
translators:
- user: chenglu
---

# LeRobot v0.5.0：全面扩展

自 v0.4.0 以来，项目已经合并了 **200+ 个 PR**，并迎来了 **50 多位新贡献者**。因此 **LeRobot v0.5.0** 成为目前规模最大的一次发布 —— 几乎在所有方向上都实现了扩展：支持更多机器人（包括首个类人机器人）、更多策略模型（包括回归的自回归 VLA）、更快的数据集处理、可以直接从 Hub 加载的仿真环境，以及基于 **Python 3.12 与 Transformers v5** 的现代化代码库。无论你是在仿真环境中训练策略，还是在真实硬件上部署，v0.5.0 都提供了大量新能力。

## TL;DR

LeRobot v0.5.0 新增 **Unitree G1 类人机器人完整支持（全身控制模型）**，并引入新的策略，包括 **Pi0-FAST 自回归 VLA** 和 **Real-Time Chunking（实时分块）** 用于实现更快响应的推理。同时还加入 **流式视频编码**，消除了录制任务之间的等待时间。

此外，本版本还推出了 **EnvHub**，允许直接从 Hugging Face Hub 加载仿真环境；集成 **NVIDIA IsaacLab-Arena**；并对代码库进行了全面现代化升级，包括 **Python 3.12+、Transformers v5 以及第三方策略插件系统**。

## 目录

* [LeRobot v0.5.0：全面扩展](#lerobot-v050-scaling-every-dimension)

  * [TL;DR](#tldr)
  * [目录](#table-of-contents)
  * [硬件：支持的机器人越来越多](#hardware-more-robots-than-ever)
    * [Unitree G1 类人机器人](#unitree-g1-humanoid)
    * [OpenArm 与 OpenArm Mini](#openarm--openarm-mini)
    * [更多机器人](#more-robots)
    * [CAN 总线电机](#can-bus-motors)
  * [策略：不断扩展的模型库](#policies-a-growing-model-zoo)
    * [Pi0-FAST：自回归 VLA](#pi0-fast-autoregressive-vlas)
    * [实时分块（RTC）](#real-time-chunking-rtc)
    * [Wall-X](#wall-x)
    * [X-VLA](#x-vla)
    * [SARM](#sarm)
    * [PEFT 支持](#peft-support)
  * [数据集：更快的数据采集与训练](#datasets-faster-recording-faster-training)
    * [流式视频编码](#streaming-video-encoding)
    * [图像训练速度提升 10 倍，编码速度提升 3 倍](#10x-faster-image-training-3x-faster-encoding)
    * [新的数据集工具](#new-dataset-tools)
  * [EnvHub：从 Hub 加载仿真环境](#envhub-environments-from-the-hub)
    * [NVIDIA IsaacLab-Arena](#nvidia-isaaclab-arena)
  * [代码库：现代化基础设施](#codebase-a-modern-foundation)
  * [社区与生态](#community--ecosystem)
  * [总结](#final-thoughts)

## 硬件：支持的机器人数量再创新高

LeRobot v0.5.0 大幅扩展了支持的硬件设备，从机械臂、移动机器人到完整的类人机器人。

### Unitree G1 Humanoid

本次发布中最重要的硬件新增是：**对 Unitree G1 类人机器人的完整支持**。这是 LeRobot 第一次集成类人机器人，而且支持非常全面：

* **运动能力（Locomotion）**：可以行走、导航并在环境中移动。
* **操作能力（Manipulation）**：能够执行精细的物体操作任务。
* **远程操控（Teleoperation）**：通过直观的遥操作界面远程控制 G1。
* **全身控制（Whole-Body Control, WBC）**：同时协调行走和操作，实现复杂的真实世界任务。

G1 的加入标志着 LeRobot 在通用机器人方向迈出了重要一步 —— 从桌面机械臂扩展到 **完整身体的具身智能系统**。你可以按照[文档](https://huggingface.co/docs/lerobot/unitree_g1)自己尝试。

![unitree-boss](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.5.0/unitree_bosswalk.JPG)

### OpenArm & OpenArm Mini

我们新增了对 [**OpenArm**](https://openarm.dev) 机械臂以及其配套 **OpenArm Mini** 遥操作设备的支持。OpenArm 是一款性能出色的机械臂，并且已经实现完整的 LeRobot 集成，而 Mini 则作为它的自然遥操作设备。

两者都支持 **双臂配置（bi-manual）**，可以构建双机械臂系统，从而完成更复杂的操作任务。更多信息可查看[文档](https://huggingface.co/docs/lerobot/openarm)。

### 更多机器人

硬件生态仍在持续扩展：

* [**Earth Rover**](https://shop.frodobots.com/products/miniplus)：LeRobot 首次支持移动机器人，可用于户外导航和地面移动任务。
* [**OMX Robot**](https://ai.robotis.com/omx/hardware_omx.html)：新增的机械臂平台，支持可配置夹爪参数和校准功能。
* **SO-100/SO-101 统一实现**：我们将 SO-100 和 SO-101 的实现整合到一个更简洁的代码库中（包括双臂配置），减少重复代码，更易维护，同时保持原有功能。

### CAN 总线电机

通过 **CAN（Controller Area Network）总线**新增了对电机控制器的支持，从而能够接入更高性能的执行器：

* [**RobStride**](https://github.com/RobStride/Product_Information)：基于 CAN 的电机控制器，适用于高扭矩应用。
* **Damiao**：另一种 CAN 总线电机控制器，进一步扩展兼容硬件范围。

这意味着 LeRobot 现在不仅支持 Dynamixel 和 Feetech，也能够驱动更多 **专业级执行器**。

## 策略模型：不断扩展的模型库

本次发布为 LeRobot 新增 **6 种策略或技术**，进一步推动开源机器人学习的发展。

### Pi0-FAST：自回归 VLA

**Pi0-FAST** 将自回归的 **Vision-Language-Action（VLA）模型**引入 LeRobot，并采用 **FAST（Frequency-space Action Sequence Tokenization）** 方法。

与 Pi0 使用的 flow-matching 方法不同，Pi0-FAST 使用 **基于 Gemma 300M 的自回归动作专家模型**，生成离散化的动作 token，实现：

* **FAST Tokenization**：动作被 token 化，便于自回归解码，使用专门的 [FAST action tokenizer](https://huggingface.co/lerobot/fast-action-tokenizer)。
* **灵活解码**：可以通过温度参数和最大解码步数，在速度与质量之间进行权衡。
* **兼容 RTC**：可与 Real-Time Chunking 结合，实现更快速的推理。

```bash
lerobot-train \
  --policy.type=pi0_fast \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human \
  --policy.device=cuda
```

### Real-Time Chunking (RTC)

**Real-Time Chunking** 是来自 [Physical Intelligence](https://www.pi.website) 的推理阶段技术，可以显著提升 flow-matching 策略的响应速度。

传统方法需要等一个完整动作序列生成后再重新规划，而 RTC 会 **持续融合新的预测与正在执行的动作**，使机器人行为更加平滑、响应更快。

RTC 不是独立策略，而是一个增强模块，可用于 **Pi0 系列、SmolVLA 与 Diffusion** 等策略。

启用方式：

```
--policy.rtc_config.enabled=true
```

在真实机器人部署中（对延迟敏感的场景），这是一个非常重要的改进。更多技术细节见[论文](https://huggingface.co/papers/2506.07339)和[文档](https://huggingface.co/docs/lerobot/rtc)。

### Wall-X

**Wall-X** 是一个新的 VLA 策略，基于 [**Qwen2.5-VL**](https://huggingface.co/collections/Qwen/qwen25-vl) 构建，并使用 flow-matching 进行动作预测。

它将 Qwen2.5-VL 的强大视觉语言理解能力与 flow-matching 控制头结合，实现 **跨机器人形态控制（cross-embodiment control）**。

```bash
pip install lerobot[wall_x]
lerobot-train \
  --policy.type=wall_x \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human
```

### X-VLA

**X-VLA** 将 **基于 Florence2 的 VLA 模型**引入 LeRobot。

该模型基于 Microsoft 的 **Florence-2 视觉语言模型**，为机器人学习提供了另一种基础模型选择，进一步增加模型多样性。

查看[训练指南](https://huggingface.co/docs/lerobot/xvla)和[基础模型](https://huggingface.co/lerobot/xvla-base)。

```bash
pip install lerobot[xvla]
lerobot-train \
  --policy.type=xvla \
  --dataset.repo_id=lerobot/bimanual-so100-handover-cube
```

### SARM

**SARM（Stage-Aware Reward Modeling）** 用于解决机器人学习中一个非常困难的问题：**长时序任务（long-horizon tasks）**。

传统方法通常使用单一线性进度信号，而 SARM 会 **同时预测任务阶段以及阶段内进度**，从而更准确地描述任务进展。

这种方式可以显著提高复杂多步骤操作任务的训练效果。更多信息请查看[文档](https://huggingface.co/docs/lerobot/sarm)。

![sarm-community](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.5.0/sarm_community.gif)

### PEFT 支持

现在你可以使用 **LoRA 等 PEFT 方法对大型 VLA 模型进行微调**，而无需修改核心训练流程。

PEFT 配置在策略层进行管理，可以用较少算力将大型基础模型适配到特定机器人和任务。

详情见[文档](https://huggingface.co/docs/lerobot/peft_training)。

```bash
lerobot-train \
  --policy.type=pi0 \
  --policy.peft_config.use_peft=true \
  --dataset.repo_id=lerobot/aloha_sim_insertion_human
```

## 数据集：录制更快，训练更快

本次发布对数据集处理流程进行了重大优化，使 **数据采集和训练速度显著提升**。

### 流式视频编码

过去在录制数据集时，每个 episode 结束后都需要等待视频编码完成。

**现在不需要等待了。**

通过 **Streaming Video Encoding（流式视频编码）**，视频帧会在采集时实时编码，实现 **episode 之间零等待时间**。

系统还支持 **自动检测硬件编码器**，如果 GPU 提供视频编码能力，会自动使用。

```python
dataset = LeRobotDataset.create(
    repo_id="my/dataset",
    fps=30,
    video_backend="auto",
    streaming_encoding=True,
)
```

### 图像训练速度提升 10 倍，编码速度提升 3 倍

在底层实现中，我们修复了数据访问瓶颈，并重构了图像处理流程：

* **图像训练速度提升 10 倍**：优化图像变换流程并修复隐藏的数据访问瓶颈。
* **编码速度提升 3 倍**：默认启用并行编码，并根据数据类型动态调整压缩级别。
* **更高 CPU 利用率**：录制和数据集创建时资源使用更加高效。

### 新的数据集工具

数据集编辑工具也持续增强：

* **子任务支持**：可以在 episode 中标注子任务，支持层级任务学习。
* **图像转视频**：将现有图像数据集转换为视频格式，提高存储效率，并支持多个 episode 合并到同一视频文件。
* **更多编辑操作**：新增 `info` 数据集检查功能、任务修改工具，以及对拆分、合并、特征编辑等操作的修复。
* **更多配置选项**：可自定义视频编码格式、容差设置和元数据缓冲大小。

## EnvHub：从 Hub 加载仿真环境

**EnvHub** 让 LeRobot 可以 **直接从 Hugging Face Hub 加载仿真环境**。

过去需要在本地安装环境并手动注册，现在只需要指定 Hub 仓库即可：

* 自动下载环境代码
* 自动注册到 Gymnasium
* 直接用于训练和评估

Hub 环境使用 `HubEnvConfig`，会下载并执行远程 `make_env` 函数：

```bash
lerobot-train \
  --env.type=hub \
  --env.hub_path="username/my-custom-env" \
  --policy.type=act
```

这大大降低了分享自定义仿真环境的门槛。只需打包环境并上传到 Hub，其他人就能直接使用。

更多信息见[文档](https://huggingface.co/docs/lerobot/envhub)。

示例教程：
[LeIsaac x LeRobot EnvHub tutorial](https://huggingface.co/docs/lerobot/envhub_leisaac)

### NVIDIA IsaacLab-Arena

我们还集成了 **NVIDIA IsaacLab-Arena**，为 LeRobot 带来 **GPU 加速仿真**。

IsaacLab-Arena 提供了一系列基于 **NVIDIA Isaac Sim** 的操作任务环境，并支持大规模并行环境实例，从而加速强化学习训练。

该集成包括：

* 专门的前处理和后处理流程
* 与 LeRobot 训练流程完全兼容

详情见[文档](https://huggingface.co/docs/lerobot/envhub_isaaclab_arena)。

## 代码库：现代化基础设施

本版本对代码库进行了全面升级：

* **Python 3.12+**：LeRobot 现在要求 **Python 3.12** 作为最低版本，从而能够使用更现代的语法并获得更好的性能。
* **Transformers v5**：项目已经迁移到 **Hugging Face Transformers v5**，以保持与最新模型生态的兼容。
* **第三方策略插件**：类似于 v0.4.0 的硬件插件系统，现在也可以把自定义策略注册为可安装的插件包，例如：`pip install lerobot_policy_mypolicy`，然后通过 `--policy.type=mypolicy` 使用，无需修改核心库代码。具体方法可以参考[文档](https://huggingface.co/docs/lerobot/bring_your_own_policies)。
* **远程 Rerun 可视化**：可以使用 Rerun 远程可视化机器人的遥测数据，并支持图像压缩，从而实现更节省带宽的数据流传输。
* **安装流程改进**：新增 `uv` 的[安装说明](https://huggingface.co/docs/lerobot/installation)，同时进一步明确了安装步骤，并优化了依赖管理。现在顺序安装流程也在文档中有清晰说明。
* **文档版本管理**：文档现在支持版本化，可以始终查阅与你当前安装版本对应的文档。
* **PyTorch 版本更新**：更新了 PyTorch 的版本范围，以支持 **NVIDIA Blackwell GPU**。

## 社区与生态

* **Discord 社区升级**：对 Discord 社区进行了更新，优化了频道结构，使这个最活跃的社区交流平台更加清晰、有序。

* **GitHub README、模板与自动标签**：更新了 README，新增 issue 和 PR 模板、贡献指南，以及自动化标签系统，让社区成员更容易参与和贡献。

* **ICLR 2026 论文录用**：LeRobot 论文已被 [ICLR 2026](https://openreview.net/forum?id=CiZMMAFQR3) 接收。

* **LeRobot Visualizer 更新**：可视化工具进行了升级，新增数据集可视化徽章，并改进了整体功能。[可以在这里体验](https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2Fimstevenpmwork%2Fthanos_picking_power_gem%2Fepisode_0)

* **LeRobot Annotation Studio**：推出了一个 HuggingFace Space，用于给数据集中的每个时刻添加自然语言子任务标注，让数据标注更加方便。[查看项目](https://huggingface.co/spaces/lerobot/annotate)

![visualizer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/lerobot-blog/release-v0.5.0/visualizer.gif)

## 最后

除了上述重点功能之外，v0.5.0 还包含：

* 数百个 bug 修复
* 文档改进
* CI/CD 优化
* 大量开发体验提升

从更严格的类型检查到更健壮的测试基础设施，我们正在持续加强 LeRobot 的基础架构，以支持未来更大规模的发展。

我们也要向 **整个社区表示衷心感谢** —— 所有贡献者、用户和合作伙伴都在推动 LeRobot 不断进步。每一次 bug 报告、PR 和讨论都让这个项目变得更好。

更多内容即将到来 🤗
从这里开始：[https://github.com/huggingface/lerobot](https://github.com/huggingface/lerobot)

— **LeRobot 团队 ❤️**

> [!IMPORTANT]
> 很快就会有一个重大惊喜发布，敬请期待！👕