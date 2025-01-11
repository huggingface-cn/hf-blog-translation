# 在PyTorch中可视化和理解GPU内存

你一定很熟悉这条消息🤬：

```
RuntimeError: CUDA out of memory. Tried to allocate 20.00 MiB (GPU 0; 7.93 GiB total capacity; 6.00 GiB already allocated; 14.88 MiB free; 6.00 GiB reserved in total by PyTorch)

```

尽管很容易看出GPU内存已满，但理解原因以及如何解决可能更加具有挑战性。在本教程中，我们将逐步讲解如何在训练过程中可视化和理解PyTorch中的GPU内存使用情况。我们还将探讨如何估算内存需求以及优化GPU内存使用。

<iframe src="https://qgallouedec-train-memory.hf.space" width="100%" height="600"></iframe>



## 🔎 PyTorch 可视化工具

PyTorch 提供了一个可视化 GPU 内存使用情况的便捷工具：

```
import torch
from torch import nn

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = nn.Linear(10_000, 50_000, device ="cuda")
for _ in range(3):
    inputs = torch.randn(5_000, 10_000, device="cuda")
    outputs = model(inputs)

# Dump memory snapshot history to a file and stop recording
torch.cuda.memory._dump_snapshot("profile.pkl")
torch.cuda.memory._record_memory_history(enabled=None)

```

运行此代码会生成一个 **profile.pkl** 文件，其中包含执行过程中 GPU 内存使用的历史记录。您可以在以下地址可视化此历史记录：[https://pytorch.org/memory_viz](#)。

通过拖放您的 **profile.pkl** 文件，您将看到类似这样的图表：

![Simple profile](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/simple_profile.png)

让我们将该图分解为几个主要部分：

![Simple profile partitioned](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/simple_profile_partitioned.png)

1. **模型创建**：内存增加 2 GB，对应于模型的大小：

   10,000 × 50,000 权重 + 50,000 偏置在 **float32**（4 字节）  
   ⟹ (5 × 10⁸) × 4 字节 = 2 GB。

   这部分内存（以蓝色表示）在整个执行过程中保持不变。

2. **输入张量创建（第 1 次循环）**：内存增加 200 MB，匹配输入张量的大小：

   5,000 × 10,000 元素在 **float32**（4 字节）  
   ⟹ (5 × 10⁷) × 4 字节 = 0.2 GB。

3. **前向传播（第 1 次循环）**：内存为输出张量增加 1 GB：

   5,000 × 50,000 元素在 **float32**（4 字节）  
   ⟹ (25 × 10⁷) × 4 字节 = 1 GB。

4. **输入张量创建（第 2 次循环）**：内存为新的输入张量增加 200 MB。在此时，你可能希望第 2 步中的输入张量被释放。然而，事实并非如此：模型保留了它的激活值，因此即使张量不再被分配给变量 **inputs**，它仍然会被模型的前向传播计算引用。模型保留了它的激活值，因为这些张量是神经网络反向传播过程所必需的。尝试使用 **torch.no_grad()** 查看区别。

5. **前向传播（第 2 次循环）**：为新的输出张量增加 1 GB，如第 3 步所计算。

6. **释放第 1 次循环激活值**：在第 2 次循环的前向传播后，可以释放第 1 次循环的输入张量（第 2 步）。模型的激活值（存储了第 1 次输入张量）被第 2 次循环的输入覆盖。一旦第 2 次循环完成，第 1 次张量不再被引用，其内存可以被释放。

7. **更新输出**：第 3 步的输出张量被重新分配给变量 **output**。之前的张量不再被引用，并被删除，释放了内存。

8. **输入张量创建（第 3 次循环）**：与第 4 步相同。

9. **前向传播（第 3 次循环）**：与第 5 步相同。

10. **释放第 2 次循环激活值**：第 4 步中的输入张量被释放。

11. **再次更新输出**：第 5 步的输出张量被重新分配给变量 **output**，释放了之前的张量。

12. **代码执行结束**：所有内存被释放。

## 📊 训练期间可视化记忆

前面的示例经过了简化。在实际场景中，我们经常训练复杂的模型，而不是单个线性层。此外，前面的示例不包括训练过程。在这里，我们将研究 GPU 内存在真实的大型语言模型 (LLM) 的完整训练循环中的表现。

```
import torch
from transformers import AutoModelForCausalLM

# Start recording memory snapshot history
torch.cuda.memory._record_memory_history(max_entries=100000)

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for _ in range(3):
    inputs = torch.randint(0, 100, (16, 256), device="cuda")  # Dummy input
    loss = torch.mean(model(inputs).logits)  # Dummy loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

# Dump memory snapshot history to a file and stop recording
torch.cuda.memory._dump_snapshot("profile.pkl")
torch.cuda.memory._record_memory_history(enabled=None)

```

💡 **提示**：在进行性能分析时，限制步骤的数量。每个 GPU 内存事件都会被记录，文件可能会变得非常大。例如，上述代码生成了一个 8 MB 的文件。

以下是该示例的内存配置文件：

![Raw training profile](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/raw_training_profile.png)

此图比之前的示例更为复杂，但我们仍然可以逐步将其拆解。注意图中有三个峰值，每个峰值对应训练循环的一次迭代。让我们简化图表，使其更易于理解：

![Colorized training profile](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/colorized_training_profile.png)

1. **模型初始化** (`model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")`)：  
   第一步是将模型加载到 GPU 上。模型参数（以蓝色表示）占用内存并一直保留到训练结束。

2. **前向传播** (`model(inputs)`)：  
   在前向传播过程中，各层的激活值（每层的中间输出）被计算并存储在内存中，用于反向传播。这些激活值以橙色表示，按层逐渐增长，直到最后一层。损失值在橙色区域的峰值处计算。

3. **反向传播** (`loss.backward()`)：  
   梯度（以黄色表示）在此阶段被计算并存储。同时，激活值被丢弃，因为它们不再需要，从而导致橙色区域缩小。黄色区域表示梯度计算所占用的内存。

4. **优化器步骤** (`optimizer.step()`)：  
   梯度用于更新模型参数。最初，优化器本身被初始化（绿色区域）。这种初始化仅发生一次。之后，优化器使用梯度更新模型的参数。为了更新参数，优化器会临时存储中间值（红色区域）。更新完成后，梯度（黄色）和优化器的中间值（红色）都会被丢弃，从而释放内存。

至此，一个训练迭代已完成。该过程会针对剩余的迭代重复，从而在图中产生三个可见的内存峰值。

类似的训练配置通常遵循一致的模式，这使得它们对于估算特定模型和训练循环的 GPU 内存需求非常有用。

## 📐 预估内存需求



从上述部分来看，估算 GPU 内存需求似乎很简单。所需的总内存应对应于内存配置文件中的最高峰值，该峰值出现在 **前向传播** 期间。在这种情况下，内存需求为（蓝色 + 绿色 + 橙色）：**模型参数 + 优化器状态 + 激活值**。

事情真的如此简单吗？实际上，这里存在一个陷阱。内存配置可能会因训练设置而有所不同。例如，将批量大小从 16 减少到 2 会改变这一情况：

```
- inputs = torch.randint(0, 100, (16, 256), device="cuda")  # Dummy input
+ inputs = torch.randint(0, 100, (2, 256), device="cuda")  # Dummy input

```

![Colorized training profile 2](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/colorized_training_profile_2.png)

现在，最高峰值出现在 **优化器步骤** 而不是 **前向传播** 期间。在这种情况下，内存需求变为（蓝色 + 绿色 + 黄色 + 红色）：  
**模型参数 + 优化器状态 + 梯度 + 优化器中间值**

为了对内存估算进行泛化，我们需要考虑所有可能的峰值，无论它们发生在前向传播还是优化器步骤期间：  
**模型参数 + 优化器状态 + max(梯度 + 优化器中间值, 激活值)**

现在我们得到了公式，接下来让我们看看如何估算每个组成部分。



## 模型参数

模型参数是最容易估算的部分：**模型内存 = N × P**

其中：
- **N** 是参数的数量。
- **P** 是精度（以字节为单位，例如 **float32** 为 4 字节）。

例如，一个拥有 15 亿参数且精度为 4 字节的模型需要：

在上述示例中，模型大小为：  
**模型内存 = 1.5 × 10⁹ × 4 字节 = 6 GB**



### 优化器状态

优化器状态所需的内存取决于优化器的类型和模型参数。例如，**AdamW** 优化器为每个参数存储两个动量（第一和第二动量）。因此，优化器状态大小为：  
**优化器状态大小 = 2 × N × P**



### 激活值

激活值所需的内存更难估算，因为它包含了前向传播期间计算的所有中间值。要计算激活值内存，可以使用前向钩子来测量输出的大小：

```
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B").to("cuda")

activation_sizes = []

def forward_hook(module, input, output):
    """
    Hook to calculate activation size for each module.
    """
    if isinstance(output, torch.Tensor):
        activation_sizes.append(output.numel() * output.element_size())
    elif isinstance(output, (tuple, list)):
        for tensor in output:
            if isinstance(tensor, torch.Tensor):
                activation_sizes.append(tensor.numel() * tensor.element_size())

# Register hooks for each submodule
hooks = []
for submodule in model.modules():
    hooks.append(submodule.register_forward_hook(forward_hook))

# Perform a forward pass with a dummy input
dummy_input = torch.zeros((1, 1), dtype=torch.int64, device="cuda")
model.eval()  # No gradients needed for memory measurement
with torch.no_grad():
    model(dummy_input)

# Clean up hooks
for hook in hooks:
    hook.remove()

print(sum(activation_sizes))  # Output: 5065216

```

对于 Qwen2.5-1.5B 模型，每个输入标记会产生 **5,065,216 个激活值**。要估算输入张量的总激活内存，可以使用以下公式：  
**激活内存 = A × B × L × P**

其中：
- **A** 是每个标记的激活值数量。
- **B** 是批量大小。
- **L** 是序列长度。

然而，直接使用这种方法并不总是可行。理想情况下，我们希望通过一种启发式方法来估算激活内存，而无需运行模型。此外，我们可以直观地看到更大的模型会有更多的激活值。这引出了一个问题：  
**模型参数数量和激活值数量之间是否存在关联？**

并非直接关联，因为每个标记的激活值数量取决于模型架构。然而，大型语言模型（LLMs）往往具有类似的结构。通过分析不同模型，我们观察到参数数量和激活值数量之间存在一种大致的线性关系：

![Activations vs. Parameters](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/train_memory/activation_memory_with_global_regression.png)

这种线性关系允许我们使用以下启发式公式来估算激活值：  
**A = 4.6894 × 10⁴ × N + 1.8494 × 10⁶**

尽管这只是一个近似值，但它提供了一种无需对每个模型进行复杂计算即可估算激活内存的实用方法。



### 梯度

梯度更容易估算。梯度所需的内存与模型参数的内存相同：  
**梯度内存 = N × P**



### 优化器中间值

在更新模型参数时，优化器会存储中间值。这些值所需的内存与模型参数相同：  
**优化器中间值内存 = N × P**



### 总内存

总结一下，训练模型所需的总内存为：  
**总内存 = 模型内存 + 优化器状态 + max(梯度, 优化器中间值, 激活值)**

包含以下组成部分：
- **模型内存**：N × P
- **优化器状态**：2 × N × P
- **梯度**：N × P
- **优化器中间值**：N × P
- **激活值**：A × B × L × P，使用以下启发式公式估算：  
  **A = 4.6894 × 10⁴ × N + 1.8494 × 10⁶**

为了简化这一计算，我为你创建了一个小工具：

<iframe src="https://qgallouedec-train-memory.hf.space" width="100%" height="600"></iframe>

🚀 **下一步**

你最初想要了解内存使用情况的动机可能是因为某一天你遇到了内存不足的问题。这篇博客是否为你提供了直接解决该问题的方法？可能没有。然而，现在你对内存使用的工作原理以及如何分析内存使用有了更好的理解，你将能够更好地找到减少内存使用的方法。

如果你想了解如何优化TRL中的内存使用，可以查看文档中的[**Reducing Memory Usage**](https://huggingface.co/docs/trl/main/en/reducing_memory_usage)部分。不过，这些技巧不仅限于TRL，也可以应用于任何基于PyTorch的训练过程。

---

🤝 **致谢**

感谢 [**Kashif Rasul**](https://huggingface.co/kashif) 对这篇博客文章的宝贵反馈和建议。