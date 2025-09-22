#!/usr/bin/env python
"""
Test Script for Smooth Grad-CAM Visualization
==============================================

This script demonstrates the usage of Smooth Grad-CAM for Qwen2.5-VL models.
It provides various visualization modes for analyzing model attention patterns.

Author: Chuanyu Qin
AI Assistant: Claude Code
Reference: https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/master/demo_smooth_grad_threshold.py
Date: 2024
License: MIT
"""

import os
import argparse
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from smooth_gradcam import SmoothQwenGradCAM, get_model_layers
from visualization_utils import (
    create_layer_comparison_grid,
    visualize_multiple_layers,
    create_comparative_analysis
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Smooth Grad-CAM Visualization for Qwen2.5-VL')

    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to Qwen2.5-VL model')
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to input image')

    # 可选参数
    parser.add_argument('--prompt', type=str,
                        default="请详细描述这张图片中的主要内容。",
                        help='Text prompt for the model')
    parser.add_argument('--mode', type=str, default='grid',
                        choices=['single', 'multi', 'grid', 'comparative'],
                        help='Visualization mode')
    parser.add_argument('--layer', type=int, default=-1,
                        help='Layer index for single mode (-1 for last layer)')
    parser.add_argument('--layers', type=int, nargs='+',
                        help='Layer indices for multi mode')
    parser.add_argument('--layer_range', type=str, default='auto',
                        choices=['auto', 'early', 'middle', 'late', 'all'],
                        help='Layer range for grid mode')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples for SmoothGrad')
    parser.add_argument('--noise_std', type=float, default=0.15,
                        help='Noise standard deviation for SmoothGrad')
    parser.add_argument('--output_dir', type=str, default='cam_results',
                        help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cpu, etc.)')

    return parser.parse_args()


def single_layer_visualization(model, processor, image, prompt, layer_idx=-1,
                               num_samples=10, noise_std=0.15, save_dir="cam_results_single"):
    """
    单层可视化模式。

    Args:
        model: 模型
        processor: 处理器
        image: PIL图像
        prompt: 文本提示
        layer_idx: 层索引（-1表示最后一层）
        num_samples: SmoothGrad采样次数
        noise_std: 噪声标准差
        save_dir: 保存目录
    """
    print("\n===== 单层可视化模式 =====")

    # 获取模型层
    try:
        layers, total_layers = get_model_layers(model)
    except AttributeError as e:
        print(f"错误: {e}")
        return

    # 处理层索引
    if layer_idx == -1:
        layer_idx = total_layers - 1
    elif layer_idx < 0 or layer_idx >= total_layers:
        print(f"错误：层索引 {layer_idx} 超出范围 [0, {total_layers-1}]")
        return

    print(f"可视化第 {layer_idx} 层（共 {total_layers} 层）")
    target_layer = layers[layer_idx]

    # 创建GradCAM对象
    grad_cam = SmoothQwenGradCAM(
        model=model,
        processor=processor,
        target_layer=target_layer,
        num_samples=num_samples,
        noise_std=noise_std
    )

    # 生成CAM
    print("正在生成 CAM...")
    superimposed_image, heatmap = grad_cam.generate_cam(image, prompt)

    # 保存结果
    if superimposed_image is not None:
        os.makedirs(save_dir, exist_ok=True)

        # 保存图像
        Image.fromarray(superimposed_image).save(
            os.path.join(save_dir, "superimposed_result.png")
        )
        Image.fromarray(heatmap).save(
            os.path.join(save_dir, "heatmap.png")
        )
        image.save(os.path.join(save_dir, "original_image.png"))

        print(f"CAM 生成成功！结果已保存到 '{save_dir}' 目录。")

        # 尝试显示图像
        try:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(18, 6))

            axs[0].imshow(image)
            axs[0].set_title("Original Image")
            axs[0].axis('off')

            axs[1].imshow(heatmap)
            axs[1].set_title("Heatmap")
            axs[1].axis('off')

            axs[2].imshow(superimposed_image)
            axs[2].set_title("Superimposed Image")
            axs[2].axis('off')

            plt.savefig(os.path.join(save_dir, "visualization.png"),
                       dpi=150, bbox_inches='tight')
            print(f"可视化图已保存到 '{save_dir}/visualization.png'")

            plt.show()
        except ImportError:
            print("Matplotlib 未安装，无法显示图片。")
        except:
            print("无法显示图片（可能在非GUI环境中）")
    else:
        print("CAM 生成失败。")


def main():
    """主函数"""
    # 如果不是通过命令行运行，使用默认参数
    import sys
    if len(sys.argv) == 1:
        # 使用默认测试参数
        class Args:
            model_path = "/mnt/data/qcy/model/Qwen2.5-VL-3B-Instruct"
            image_path = "demo.jpeg"
            prompt = "请详细描述这张图片中的主要内容。"
            mode = "grid"
            layer = -1
            layers = None
            layer_range = "auto"
            num_samples = 10
            noise_std = 0.15
            output_dir = "cam_results"
            device = "cuda:0"

        args = Args()
    else:
        args = parse_args()

    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(':')[-1] if 'cuda' in args.device else ""

    # 加载模型和处理器
    print("正在加载模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)
    model.eval()
    print("加载完成！")

    # 加载图像
    try:
        raw_image = Image.open(args.image_path).convert("RGB")
    except Exception as e:
        print(f"无法加载图像 {args.image_path}: {e}")
        return

    # 根据模式执行不同的可视化
    if args.mode == "single":
        # 单层可视化
        single_layer_visualization(
            model, processor, raw_image, args.prompt,
            layer_idx=args.layer,
            num_samples=args.num_samples,
            noise_std=args.noise_std,
            save_dir=os.path.join(args.output_dir, "single")
        )

    elif args.mode == "multi":
        # 多层可视化
        print("\n===== 多层可视化模式 =====")
        layer_indices = args.layers if args.layers else [6, 12, 18, 27]
        results = visualize_multiple_layers(
            model, processor, raw_image, args.prompt,
            layer_indices=layer_indices,
            save_dir=os.path.join(args.output_dir, "multilayer")
        )
        print("\n多层可视化完成！")

    elif args.mode == "grid":
        # 网格可视化
        print("\n===== 网格可视化模式 =====")
        print("此模式将创建一个网格对比图，展示不同层的CAM结果\n")

        # 处理层范围参数
        if args.layers:
            layer_range = args.layers
        else:
            layer_range = args.layer_range

        results = create_layer_comparison_grid(
            model, processor, raw_image, args.prompt,
            layer_range=layer_range,
            save_dir=os.path.join(args.output_dir, "grid")
        )
        print("\n网格可视化完成！")

    elif args.mode == "comparative":
        # 对比可视化
        print("\n===== 对比可视化模式 =====")
        print("对比早期、中期和后期层的差异\n")
        results = create_comparative_analysis(
            model, processor, raw_image, args.prompt,
            save_dir=os.path.join(args.output_dir, "comparative")
        )
        print("\n对比可视化完成！")


if __name__ == "__main__":
    main()