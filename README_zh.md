# Smooth Grad-CAM for Qwen2.5-VL

基于 SmoothGrad 和 Grad-CAM 的视觉-语言模型注意力可视化工具，专门针对 Qwen2.5-VL 模型优化。

> **说明**：本实现基于 [From-Redundancy-to-Relevance](https://github.com/zhangbaijin/From-Redundancy-to-Relevance/tree/master) 的工作，针对 Qwen2.5-VL 模型进行了适配。

## 📋 目录

- [功能特性](#功能特性)
- [安装要求](#安装要求)
- [快速开始](#快速开始)
- [使用指南](#使用指南)
- [API 文档](#api-文档)
- [可视化模式](#可视化模式)
- [技术细节](#技术细节)
- [常见问题](#常见问题)
- [致谢](#致谢)

## ✨ 功能特性

- **Smooth Grad-CAM 实现**：通过多次加噪采样平均，生成更稳定、更准确的注意力热力图
- **多层可视化对比**：支持同时可视化多个 Transformer 层的注意力模式
- **灵活的可视化模式**：
  - 单层详细分析
  - 多层网格对比
  - 层级进展分析（早期/中期/后期）
  - 自定义层选择
- **非正方形网格支持**：智能处理各种尺寸的视觉 token 网格
- **批量处理能力**：高效处理多层可视化任务

## 📦 安装要求

### 基础依赖

```bash
pip install torch transformers pillow opencv-python numpy matplotlib
```

### 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- Transformers >= 4.35
- CUDA 推荐（CPU 也支持但较慢）

### 模型准备

需要下载 Qwen2.5-VL 模型：

```bash
# 使用 Hugging Face CLI
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir /path/to/model

# 或使用 Python
from transformers import AutoModel, AutoProcessor
model = AutoModel.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
```

## 🚀 快速开始

### 基础用法

```python
from smooth_gradcam import SmoothQwenGradCAM, get_model_layers
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image

# 加载模型
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "path/to/Qwen2.5-VL-3B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)
processor = AutoProcessor.from_pretrained("path/to/Qwen2.5-VL-3B-Instruct")
model.eval()

# 获取目标层
layers, total_layers = get_model_layers(model)
target_layer = layers[-1]  # 使用最后一层

# 创建 Grad-CAM 对象
grad_cam = SmoothQwenGradCAM(
    model=model,
    processor=processor,
    target_layer=target_layer,
    num_samples=10,  # SmoothGrad 采样次数
    noise_std=0.15   # 噪声标准差
)

# 加载图像并生成 CAM
image = Image.open("demo.jpeg").convert("RGB")
prompt = "请详细描述这张图片中的主要内容。"

superimposed_image, heatmap = grad_cam.generate_cam(image, prompt)
```

### 使用测试脚本

```bash
# 基础用法 - 网格可视化
python test_gradcam.py \
    --model_path /path/to/model \
    --image_path demo.jpeg \
    --mode grid

# 单层可视化
python test_gradcam.py \
    --model_path /path/to/model \
    --image_path demo.jpeg \
    --mode single \
    --layer 27

# 多层对比
python test_gradcam.py \
    --model_path /path/to/model \
    --image_path demo.jpeg \
    --mode multi \
    --layers 6 12 18 27

# 阶段性对比
python test_gradcam.py \
    --model_path /path/to/model \
    --image_path demo.jpeg \
    --mode comparative
```

## 📖 使用指南

### 可视化模式详解

#### 1. 单层模式 (Single)

可视化单个 Transformer 层的注意力模式，适合深入分析特定层的行为。

```python
from test_gradcam import single_layer_visualization

single_layer_visualization(
    model, processor, image, prompt,
    layer_idx=27,  # 层索引
    num_samples=10,  # SmoothGrad 采样次数
    noise_std=0.15,  # 噪声强度
    save_dir="results/single"
)
```

#### 2. 网格模式 (Grid)

在一个网格中展示多个层的 CAM 结果，便于快速对比。

```python
from visualization_utils import create_layer_comparison_grid

results = create_layer_comparison_grid(
    model, processor, image, prompt,
    layer_range="auto",  # 可选: "auto", "early", "middle", "late", "all", 或列表
    save_dir="results/grid"
)
```

层范围选项：
- `"auto"`: 自动选择 6 个代表性的层
- `"early"`: 前 1/3 层
- `"middle"`: 中间 1/3 层
- `"late"`: 后 1/3 层
- `"all"`: 所有层（可能较慢）
- `[0, 5, 10, 15]`: 自定义层列表

#### 3. 多层模式 (Multi)

生成详细的多层对比图，包含热力图和叠加图。

```python
from visualization_utils import visualize_multiple_layers

results = visualize_multiple_layers(
    model, processor, image, prompt,
    layer_indices=[6, 12, 18, 27],  # 指定要可视化的层
    save_dir="results/multilayer"
)
```

#### 4. 对比模式 (Comparative)

分别生成早期、中期、后期层的对比分析。

```python
from visualization_utils import create_comparative_analysis

results = create_comparative_analysis(
    model, processor, image, prompt,
    save_dir="results/comparative"
)
```

### 参数调优指南

#### SmoothGrad 参数

- **num_samples** (默认 10): 采样次数越多，结果越平滑但计算越慢
  - 快速预览: 3-5
  - 标准质量: 10
  - 高质量: 20-50

- **noise_std** (默认 0.15): 噪声标准差，控制扰动强度
  - 低噪声 (0.05-0.1): 更接近原始 Grad-CAM
  - 中等噪声 (0.1-0.2): 平衡平滑度和细节
  - 高噪声 (0.2-0.3): 更平滑但可能丢失细节

## 🔧 API 文档

### 核心类

#### SmoothQwenGradCAM

主要的 Grad-CAM 实现类。

```python
class SmoothQwenGradCAM:
    def __init__(self, model, processor, target_layer, num_samples=10, noise_std=0.1):
        """
        Args:
            model: Qwen2.5-VL 模型实例
            processor: 对应的处理器
            target_layer: 要可视化的目标层
            num_samples: SmoothGrad 采样次数
            noise_std: 噪声标准差
        """

    def generate_cam(self, image: Image.Image, prompt: str, use_smooth=True):
        """
        生成 CAM 热力图。

        Args:
            image: PIL 图像
            prompt: 文本提示
            use_smooth: 是否使用 SmoothGrad

        Returns:
            tuple: (叠加图像, 热力图)
        """
```

### 工具函数

#### get_model_layers

```python
def get_model_layers(model):
    """
    获取模型的所有层。

    Returns:
        tuple: (层列表, 总层数)
    """
```

#### create_layer_comparison_grid

```python
def create_layer_comparison_grid(model, processor, image, prompt,
                                layer_range="auto", save_dir="cam_results_grid"):
    """
    创建层对比网格。

    Args:
        layer_range: 层选择策略
        save_dir: 保存目录

    Returns:
        dict: 各层的 CAM 结果
    """
```

## 🔬 技术细节

### Grad-CAM 原理

Grad-CAM (Gradient-weighted Class Activation Mapping) 通过分析目标类别相对于特征图的梯度来生成视觉解释：

1. **前向传播**：获取目标层的特征图
2. **反向传播**：计算目标分数相对于特征图的梯度
3. **加权组合**：使用全局平均池化的梯度作为权重
4. **ReLU 激活**：只保留正向激活

### SmoothGrad 增强

SmoothGrad 通过在输入图像中添加噪声并平均多次运行的结果来减少噪声：

```text
CAM_smooth = 1/n * Σ CAM(image + noise_i)
```

这能够：
- 减少随机噪声
- 突出稳定的注意力模式
- 提供更可靠的可视化

### 视觉 Token 处理

Qwen2.5-VL 使用特殊标记 `<|vision_start|>` 和 `<|vision_end|>` 来标记视觉内容。本实现会自动：

1. 定位视觉 token 范围
2. 提取对应的特征和梯度
3. 处理非正方形网格（如 49×73）

## ❓ 常见问题

### Q: 为什么某些层的热力图看起来很模糊？

A: 早期层通常关注低级特征（边缘、纹理），后期层关注高级语义。可以尝试调整 noise_std 参数。

### Q: 如何选择最佳的层进行可视化？

A: 建议先使用 "auto" 或 "grid" 模式查看多个层，然后选择最有信息量的层进行详细分析。

### Q: 处理速度很慢怎么办？

A:
- 减少 num_samples（如设为 3-5）
- 使用较少的层
- 确保使用 GPU
- 考虑使用较小的模型（如 3B 版本）

### Q: 视觉 token 数量不是正方形怎么处理？

A: 代码会自动寻找最接近的因子对来重塑网格。例如，3577 个 token 会被重塑为 49×73。

## 🙏 致谢

本项目基于以下优秀工作构建：
- [From-Redundancy-to-Relevance](https://github.com/zhangbaijin/From-Redundancy-to-Relevance/tree/master) - 原始实现和方法论

## 📝 引用

如果您在研究中使用了此代码，请引用：

```bibtex
@software{smooth_gradcam_qwen,
  title = {Smooth Grad-CAM for Qwen2.5-VL},
  author = {Chuanyu Qin},
  year = {2024},
  url = {https://github.com/cyuQ1n/smooth-gradcam-qwen}
}

@article{from_redundancy_to_relevance,
  title = {From Redundancy to Relevance: Enhancing Explainability in Multimodal Large Language Models},
  author = {Zhang, Baijin and others},
  year = {2024},
  url = {https://github.com/zhangbaijin/From-Redundancy-to-Relevance}
}
```

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎贡献代码、报告问题或提出建议！请通过 GitHub Issues 或 Pull Requests 参与。

## 👥 致谢名单

- **作者**: Chuanyu Qin
- **AI 助手**: Claude Code
- **原始参考**: [From-Redundancy-to-Relevance](https://github.com/zhangbaijin/From-Redundancy-to-Relevance)

## 📧 联系方式

如有问题或合作意向，请联系：[qincyu21@163.com]