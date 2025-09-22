"""
Visualization Utilities for Grad-CAM
=====================================

This module provides advanced visualization functions for comparing
Grad-CAM results across multiple layers.

Author: Chuanyu Qin
AI Assistant: Claude Code
Reference: https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/master/demo_smooth_grad_threshold.py
Date: 2024
License: MIT
"""

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from smooth_gradcam import SmoothQwenGradCAM, get_model_layers


def create_layer_comparison_grid(model, processor, image, prompt, layer_range="auto", save_dir="cam_results_grid"):
    """
    创建不同层的CAM对比网格图。

    Args:
        model: 模型
        processor: 处理器
        image: PIL图像
        prompt: 文本提示
        layer_range: 层范围选择
            - "auto": 自动选择代表性的层
            - "all": 所有层（可能很慢）
            - "early": 早期层（前1/3）
            - "middle": 中期层（中1/3）
            - "late": 后期层（后1/3）
            - list: 自定义层索引列表，如[0, 5, 10, 15, 20, 27]
        save_dir: 保存结果的目录

    Returns:
        dict: 包含各层CAM结果的字典
    """
    # 获取模型的总层数
    try:
        layers, total_layers = get_model_layers(model)
    except AttributeError as e:
        print(f"错误: {e}")
        return None

    print(f"模型总共有 {total_layers} 层")

    # 根据layer_range选择要可视化的层
    if isinstance(layer_range, list):
        layer_indices = [idx for idx in layer_range if 0 <= idx < total_layers]
    elif layer_range == "auto":
        # 自动选择6个代表性的层
        step = max(1, total_layers // 5)
        layer_indices = [i * step for i in range(min(6, total_layers // step))]
        if total_layers - 1 not in layer_indices:
            layer_indices.append(total_layers - 1)
    elif layer_range == "all":
        layer_indices = list(range(total_layers))
    elif layer_range == "early":
        layer_indices = list(range(0, total_layers // 3))
    elif layer_range == "middle":
        layer_indices = list(range(total_layers // 3, 2 * total_layers // 3))
    elif layer_range == "late":
        layer_indices = list(range(2 * total_layers // 3, total_layers))
    else:
        raise ValueError(f"未知的layer_range: {layer_range}")

    print(f"将可视化以下 {len(layer_indices)} 层: {layer_indices}")

    # 创建进度条样式的输出
    print("\n处理进度:")
    results = {}

    for i, layer_idx in enumerate(layer_indices):
        progress = (i + 1) / len(layer_indices) * 100
        print(f"[{'=' * int(progress/5):<20}] {progress:.1f}% - 处理层 {layer_idx}")

        target_layer = layers[layer_idx]

        # 创建GradCAM对象
        grad_cam = SmoothQwenGradCAM(
            model=model,
            processor=processor,
            target_layer=target_layer,
            num_samples=3 if len(layer_indices) > 10 else 5,
            noise_std=0.1
        )

        # 生成CAM
        superimposed_image, heatmap = grad_cam.generate_cam(image, prompt, use_smooth=True)

        if superimposed_image is not None:
            results[layer_idx] = {
                'superimposed': superimposed_image,
                'heatmap': heatmap
            }

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 创建网格布局
    if results:
        num_results = len(results)
        cols = min(4, num_results)  # 最多4列
        rows = (num_results + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))

        # 处理单行或单列的情况
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        # 填充网格
        for idx, (layer_idx, result) in enumerate(results.items()):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]

            ax.imshow(result['heatmap'])
            ax.set_title(f"Layer {layer_idx}", fontsize=12, fontweight='bold')
            ax.axis('off')

            # 为最后一层添加边框以突出显示
            if layer_idx == total_layers - 1:
                rect = Rectangle((0, 0), result['heatmap'].shape[1]-1,
                                result['heatmap'].shape[0]-1,
                                linewidth=3, edgecolor='red', facecolor='none')
                ax.add_patch(rect)

        # 隐藏多余的子图
        for idx in range(num_results, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')

        plt.suptitle(f"CAM Visualization Across {num_results} Layers", fontsize=16, fontweight='bold')
        plt.tight_layout()

        grid_path = os.path.join(save_dir, "layer_comparison_grid.png")
        plt.savefig(grid_path, dpi=150, bbox_inches='tight')
        print(f"\n网格对比图已保存到: {grid_path}")

        # 尝试显示
        try:
            plt.show()
        except:
            pass
        plt.close()

        # 保存单独的图像
        for layer_idx, result in results.items():
            Image.fromarray(result['heatmap']).save(
                os.path.join(save_dir, f"layer_{layer_idx}_heatmap.png")
            )
            Image.fromarray(result['superimposed']).save(
                os.path.join(save_dir, f"layer_{layer_idx}_superimposed.png")
            )

    return results


def visualize_multiple_layers(model, processor, image, prompt, layer_indices=None, save_dir="cam_results_multilayer"):
    """
    可视化多个层的CAM结果（详细版本，带完整对比）。

    Args:
        model: 模型
        processor: 处理器
        image: PIL图像
        prompt: 文本提示
        layer_indices: 要可视化的层索引列表，None则使用默认值
        save_dir: 保存结果的目录

    Returns:
        dict: 包含各层CAM结果的字典
    """
    # 获取模型的总层数
    try:
        layers, total_layers = get_model_layers(model)
    except AttributeError as e:
        print(f"错误: {e}")
        return None

    print(f"模型总共有 {total_layers} 层")

    # 如果没有指定层索引，则使用默认值
    if layer_indices is None:
        layer_indices = [
            total_layers // 4,      # 早期层
            total_layers // 2,      # 中期层
            3 * total_layers // 4,  # 后期层
            total_layers - 1        # 最后一层
        ]

    # 确保层索引有效
    layer_indices = [idx for idx in layer_indices if 0 <= idx < total_layers]

    print(f"将可视化以下层: {layer_indices}")

    # 存储每层的结果
    results = {}

    for layer_idx in layer_indices:
        print(f"\n正在处理第 {layer_idx} 层...")
        target_layer = layers[layer_idx]

        # 创建GradCAM对象
        grad_cam = SmoothQwenGradCAM(
            model=model,
            processor=processor,
            target_layer=target_layer,
            num_samples=5,
            noise_std=0.1
        )

        # 生成CAM
        superimposed_image, heatmap = grad_cam.generate_cam(image, prompt, use_smooth=True)

        if superimposed_image is not None:
            results[layer_idx] = {
                'superimposed': superimposed_image,
                'heatmap': heatmap
            }
            print(f"第 {layer_idx} 层处理完成")
        else:
            print(f"第 {layer_idx} 层处理失败")

    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)

    # 绘制对比图
    if results:
        num_results = len(results)
        fig = plt.figure(figsize=(6 * num_results, 8))
        gs = gridspec.GridSpec(2, num_results, figure=fig)

        for idx, (layer_idx, result) in enumerate(results.items()):
            # 上排：热力图
            ax1 = fig.add_subplot(gs[0, idx])
            ax1.imshow(result['heatmap'])
            ax1.set_title(f"Layer {layer_idx} - Heatmap")
            ax1.axis('off')

            # 下排：叠加图
            ax2 = fig.add_subplot(gs[1, idx])
            ax2.imshow(result['superimposed'])
            ax2.set_title(f"Layer {layer_idx} - Superimposed")
            ax2.axis('off')

            # 保存单独的图像
            Image.fromarray(result['heatmap']).save(
                os.path.join(save_dir, f"layer_{layer_idx}_heatmap.png")
            )
            Image.fromarray(result['superimposed']).save(
                os.path.join(save_dir, f"layer_{layer_idx}_superimposed.png")
            )

        plt.tight_layout()
        comparison_path = os.path.join(save_dir, "all_layers_comparison.png")
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存到 {comparison_path}")

        # 尝试显示图像
        try:
            plt.show()
        except:
            print("无法显示图像（可能在非GUI环境中）")

        plt.close()

    # 保存原始图像
    image.save(os.path.join(save_dir, "original_image.png"))

    return results


def create_comparative_analysis(model, processor, image, prompt, save_dir="cam_results_comparative"):
    """
    创建早期、中期和后期层的对比分析。

    Args:
        model: 模型
        processor: 处理器
        image: PIL图像
        prompt: 文本提示
        save_dir: 保存结果的目录

    Returns:
        dict: 包含各阶段分析结果的字典
    """
    comparative_results = {}

    # 分别可视化早期、中期、后期层
    for stage, layer_range in [("early", "early"), ("middle", "middle"), ("late", "late")]:
        print(f"\n===== 处理 {stage} 层 =====")
        stage_dir = os.path.join(save_dir, stage)

        results = create_layer_comparison_grid(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            layer_range=layer_range,
            save_dir=stage_dir
        )

        comparative_results[stage] = results
        print(f"{stage} 层处理完成")

    print(f"\n对比分析完成！结果保存在 {save_dir}")
    return comparative_results