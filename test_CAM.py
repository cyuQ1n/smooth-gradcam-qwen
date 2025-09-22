#!/usr/bin/env python
"""
Grad-CAM Test Script for Qwen2.5-VL Model
=========================================

This script tests Grad-CAM visualization for Qwen2.5-VL models,
providing attention pattern analysis for vision-language tasks.

Author: Chuanyu Qin
Reference: https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/master/demo_smooth_grad_threshold.py
Date: 2024
License: MIT
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch.nn.functional as F
import warnings

# 忽略一个特定的 torchvision 警告
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

class SmoothQwenGradCAM:
    """
    适用于 Qwen2.5-VL (transformers) 的 SmoothGrad-CAM 实现。

    这个类通过以下步骤生成视觉解释热力图：
    1.  使用 Hooks 捕获目标层的特征图和梯度。
    2.  首先执行 model.generate() 来获取模型生成的文本序列。
    3.  然后将完整的输入+生成序列通过一次前向传播，以获取用于反向传播的 logits。
    4.  计算生成序列的 logits 之和作为分数，并进行反向传播。
    5.  利用捕获的特征图和梯度计算 CAM。
    6.  通过多次添加噪声并平均热力图来实现 SmoothGrad。
    """
    def __init__(self, model, processor, target_layer, num_samples=10, noise_std=0.1):
        self.model = model
        self.processor = processor
        self.target_layer = target_layer
        self.device = model.device
        self.num_samples = num_samples
        self.noise_std = noise_std

        self.feature_maps = None
        self.gradients = None
        self.hooks = []

    def _add_gaussian_noise_to_pixel_values(self, pixel_values, noise_std):
        """
        向像素值添加高斯噪声，用于 SmoothGrad。
        
        Args:
            pixel_values (torch.Tensor): 原始像素值张量
            noise_std (float): 噪声标准差
            
        Returns:
            torch.Tensor: 添加噪声后的像素值
        """
        noise = torch.randn_like(pixel_values) * noise_std
        noisy_pixel_values = pixel_values + noise
        # 确保像素值在合理范围内（通常是 [-1, 1] 或 [0, 1]）
        noisy_pixel_values = torch.clamp(noisy_pixel_values, -3.0, 3.0)
        return noisy_pixel_values

    def _remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _save_feature_maps(self, module, input, output):
        # 保存目标层的输出特征图
        # output[0] 是 hidden_states, output[1] 是可选的 attentions
        self.feature_maps = output[0]
        # 只在需要梯度时调用 retain_grad
        if self.feature_maps.requires_grad:
            self.feature_maps.retain_grad()

    def _get_visual_token_indices(self, input_ids):
        """
        在 Qwen2.5-VL 的 input_ids 中找到图像 token 的起始和结束位置。
        使用 <|vision_start|> 和 <|vision_end|> 标记来定位视觉内容。
        """
        try:
            # 获取特殊token的ID
            vision_start_token = "<|vision_start|>"
            vision_end_token = "<|vision_end|>"
            
            # 编码特殊token来获取它们的ID
            vision_start_id = self.processor.tokenizer.encode(vision_start_token, add_special_tokens=False)[0]
            vision_end_id = self.processor.tokenizer.encode(vision_end_token, add_special_tokens=False)[0]
            
            # 找到vision token的位置
            vision_start_indices = (input_ids[0] == vision_start_id).nonzero(as_tuple=True)[0]
            vision_end_indices = (input_ids[0] == vision_end_id).nonzero(as_tuple=True)[0]
            
            if len(vision_start_indices) == 0 or len(vision_end_indices) == 0:
                raise ValueError("无法找到 <|vision_start|> 或 <|vision_end|> token")
            
            start_index = vision_start_indices[0].item() + 1  # +1 跳过 vision_start token
            end_index = vision_end_indices[0].item()  # vision_end token之前
            
            print(f"找到视觉token范围: [{start_index}, {end_index})")
            print(f"视觉token数量: {end_index - start_index}")
            
            return start_index, end_index
            
        except (IndexError, RuntimeError) as e:
            # 如果上述方法失败，尝试直接查找token ID
            print(f"使用vision_start/end标记失败: {e}")
            print("尝试查找图像相关的token...")
            
            # 打印token信息用于调试
            tokenized_text = self.processor.tokenizer.convert_ids_to_tokens(input_ids[0])
            print("Input tokens (前50个):", tokenized_text[:50])
            print("Input tokens (后50个):", tokenized_text[-50:])
            
            # 查找可能的图像相关token
            image_related_tokens = []
            for i, token in enumerate(tokenized_text):
                if any(keyword in token.lower() for keyword in ['image', 'vision', 'visual', 'pic']):
                    image_related_tokens.append((i, token))
            
            if image_related_tokens:
                print("找到图像相关tokens:", image_related_tokens)
            
            # 尝试使用固定的范围（基于经验值）
            # 在很多VLM中，图像token通常在序列的前半部分
            seq_len = len(input_ids[0])
            
            # 尝试寻找连续的相似token（可能是图像patch token）
            possible_ranges = []
            current_start = None
            consecutive_count = 0
            
            for i in range(1, seq_len - 1):
                # 检查当前位置是否可能是图像token的开始
                # 通常图像token会有特定的ID范围或者连续性
                current_id = input_ids[0][i].item()
                
                if current_start is None:
                    current_start = i
                    consecutive_count = 1
                else:
                    # 检查是否连续（ID相近或相同）
                    prev_id = input_ids[0][i-1].item()
                    if abs(current_id - prev_id) <= 5:  # 假设图像token ID相近
                        consecutive_count += 1
                    else:
                        if consecutive_count >= 100:  # 如果有超过100个连续token，可能是图像
                            possible_ranges.append((current_start, current_start + consecutive_count))
                        current_start = i
                        consecutive_count = 1
            
            # 检查最后一段
            if consecutive_count >= 100:
                possible_ranges.append((current_start, current_start + consecutive_count))
            
            if possible_ranges:
                print(f"找到可能的图像token范围: {possible_ranges}")
                # 选择最长的范围
                start_index, end_index = max(possible_ranges, key=lambda x: x[1] - x[0])
                print(f"选择范围: [{start_index}, {end_index})")
                return start_index, end_index
            
            # 最后的fallback：使用固定范围
            print("使用fallback方案：假设图像token在序列中间部分")
            estimated_start = max(10, seq_len // 4)  # 跳过前面的系统token
            estimated_end = min(seq_len - 50, estimated_start + 1024)  # 假设1024个图像token
            
            print(f"估计的图像token范围: [{estimated_start}, {estimated_end})")
            return estimated_start, estimated_end

    def _save_gradients(self, module, grad_input, grad_output):
        # 保存反向传播回来的梯度
        self.gradients = grad_output[0]

    def _register_hooks(self):
        """注册钩子"""
        forward_hook = self.target_layer.register_forward_hook(self._save_feature_maps)
        # 使用 register_full_backward_hook 来避免警告
        backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)
        self.hooks = [forward_hook, backward_hook]

    def generate_cam(self, image: Image.Image, prompt: str, use_smooth=True):
        """
        生成 (Smooth)Grad-CAM 热力图。
        
        Args:
            image (PIL.Image.Image): 输入的原始图像。
            prompt (str): 输入的文本 prompt。
            use_smooth (bool): 是否使用 SmoothGrad（加噪平均）。
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (叠加后的图像, 原始热力图)
        """
        # 1. 准备模型输入
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)
        
        prompt_len = inputs['input_ids'].shape[1]

        # 2. 生成完整的回答（不需要钩子）
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        
        # 提取生成的 token 部分
        generated_tokens = generated_ids[0][prompt_len:]
        if len(generated_tokens) == 0:
            print("警告：模型没有生成任何新的 token。无法计算 CAM。")
            return None, None

        # 3. 准备 SmoothGrad 循环
        num_loops = self.num_samples if use_smooth else 1
        accumulated_cam = None

        for i in range(num_loops):
            # 注册钩子（每次循环重新注册）
            self._register_hooks()
            
            try:
                self.model.zero_grad()
                
                # 为 SmoothGrad 添加噪声
                if use_smooth:
                    noisy_pixel_values = self._add_gaussian_noise_to_pixel_values(inputs['pixel_values'].clone(), self.noise_std)
                else:
                    noisy_pixel_values = inputs['pixel_values']

                # 4. 单次前向传播以获取 logits（需要梯度）
                # 将生成的 token 拼接到原始输入后面，形成完整序列
                full_input_ids = torch.cat([inputs['input_ids'], generated_tokens.unsqueeze(0)], dim=1)
                full_attention_mask = torch.ones_like(full_input_ids)
                
                # 确保输入需要梯度
                noisy_pixel_values.requires_grad_(True)
                
                # 准备模型输入参数
                model_inputs = {
                    'input_ids': full_input_ids,
                    'attention_mask': full_attention_mask,
                    'pixel_values': noisy_pixel_values,
                    'output_hidden_states': True
                }
                
                # 添加 image_grid_thw 参数（如果存在）
                if 'image_grid_thw' in inputs:
                    model_inputs['image_grid_thw'] = inputs['image_grid_thw']
                
                # 使用完整的序列进行一次前向传播
                outputs = self.model(**model_inputs)
                
                logits = outputs.logits

                # 5. 计算目标分数并反向传播
                # 目标是让模型预测出我们刚刚生成的 token
                # Logits 的形状是 (batch, seq_len, vocab_size)
                # 我们需要计算从 prompt_len 开始的每个位置上，对应生成 token 的 logit 值
                target_logits = logits[0, prompt_len - 1 : -1, :]
                target_ids = generated_tokens.unsqueeze(1)
                
                # 使用 gather 提取每个位置上目标 token 的 logit
                score = target_logits.gather(1, target_ids).sum()
                score.backward()

                # 6. 计算 CAM
                if self.feature_maps is None or self.gradients is None:
                    print(f"警告：第 {i+1} 次循环中未捕获到特征图或梯度，跳过...")
                    continue
                
                # 定位视觉 token
                vis_start, vis_end = self._get_visual_token_indices(full_input_ids)
                
                # 提取视觉部分的特征图和梯度
                vis_feature_maps = self.feature_maps[:, vis_start:vis_end, :] # (1, 1024, C)
                vis_gradients = self.gradients[:, vis_start:vis_end, :]    # (1, 1024, C)
                
                # Global Average Pooling a^k = 1/Z * sum(d_score/d_A)
                pooled_gradients = torch.mean(vis_gradients, dim=[0, 1]) # (C,)
                
                # L_Grad-CAM = ReLU(sum(a^k * A^k))
                # 权重乘以特征图
                for j in range(vis_feature_maps.shape[-1]):
                    vis_feature_maps[:, :, j] *= pooled_gradients[j]
                
                cam = torch.sum(vis_feature_maps, dim=-1) # (1, 1024)
                cam = F.relu(cam)
                
                # 归一化并 reshape 成 2D 图像
                # cam = cam.squeeze().cpu().numpy()
                cam = cam.squeeze().detach().float().cpu().numpy()
                if np.max(cam) > 0:
                    cam = cam / np.max(cam)

                # 处理非正方形的视觉token网格
                num_patches = cam.shape[0]

                # 尝试找到最接近的因子对
                # 首先尝试完美平方根
                sqrt_patches = int(np.sqrt(num_patches))
                if sqrt_patches * sqrt_patches == num_patches:
                    h = w = sqrt_patches
                else:
                    # 寻找最接近的因子
                    factors = []
                    for i in range(1, int(np.sqrt(num_patches)) + 2):
                        if num_patches % i == 0:
                            factors.append((i, num_patches // i))

                    if factors:
                        # 选择最接近正方形的因子对（差值最小的）
                        h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                    else:
                        # 如果找不到合适的因子，使用近似值
                        # 尝试构造一个略大的网格，多余部分用0填充
                        h = w = int(np.ceil(np.sqrt(num_patches)))
                        # 填充cam数组
                        cam_padded = np.zeros(h * w)
                        cam_padded[:num_patches] = cam
                        cam = cam_padded

                print(f"将 {num_patches} 个patches重塑为 ({h}, {w}) 网格")
                cam = cam.reshape(h, w)
                
                if accumulated_cam is None:
                    accumulated_cam = cam
                else:
                    accumulated_cam += cam
                    
            finally:
                # 清理钩子和缓存
                self._remove_hooks()
                self.feature_maps = None
                self.gradients = None
        
        if accumulated_cam is None:
            print("错误：未能生成任何有效的 CAM")
            return None, None
        
        # 7. 后处理
        avg_cam = accumulated_cam / num_loops
        
        # 调整大小并应用颜色映射
        original_image_np = np.array(image)
        heatmap = cv2.resize(avg_cam, (original_image_np.shape[1], original_image_np.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 叠加到原图
        superimposed_img = heatmap_colored * 0.5 + original_image_np * 0.5
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return superimposed_img, heatmap_colored

def create_layer_comparison_grid(model, processor, image, prompt, layer_range="auto", save_dir="cam_results_grid"):
    """
    创建不同层的CAM对比网格图

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
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import Rectangle

    # 获取模型的总层数
    if hasattr(model.language_model, 'layers'):
        layers = model.language_model.layers
        total_layers = len(layers)
    elif hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        layers = model.language_model.model.layers
        total_layers = len(layers)
    else:
        print("无法获取模型层")
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

        # 创建GradCAM对象（减少采样次数以加快速度）
        grad_cam = SmoothQwenGradCAM(
            model=model,
            processor=processor,
            target_layer=target_layer,
            num_samples=3 if len(layer_indices) > 10 else 5,  # 根据层数调整采样次数
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
        rows = (num_results + cols - 1) // cols  # 向上取整

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1:
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
    可视化多个层的CAM结果（详细版本，带完整对比）

    Args:
        model: 模型
        processor: 处理器
        image: PIL图像
        prompt: 文本提示
        layer_indices: 要可视化的层索引列表，None则使用默认值
        save_dir: 保存结果的目录
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # 获取模型的总层数
    if hasattr(model.language_model, 'layers'):
        layers = model.language_model.layers
        total_layers = len(layers)
    elif hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        layers = model.language_model.model.layers
        total_layers = len(layers)
    else:
        print("无法获取模型层")
        return

    print(f"模型总共有 {total_layers} 层")

    # 如果没有指定层索引，则使用默认值（均匀选择几层）
    if layer_indices is None:
        # 选择早期、中期和后期的几层
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
            num_samples=5,  # 减少采样次数以加快速度
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

        # 原始图像
        original_np = np.array(image)

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
        plt.savefig(os.path.join(save_dir, "all_layers_comparison.png"), dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存到 {save_dir}/all_layers_comparison.png")

        # 尝试显示图像
        try:
            plt.show()
        except:
            print("无法显示图像（可能在非GUI环境中）")

        plt.close()

    return results

# ============== 主程序 ==============
if __name__ == "__main__":
    # --- 配置 ---
    # 模型和处理器路径
    MODEL_PATH = "/mnt/data/qcy/model/Qwen2.5-VL-3B-Instruct" # 请修改为你的模型路径
    # 测试图片和 Prompt
    IMAGE_PATH = "demo.jpeg"
    PROMPT = "请详细描述这张图片中的主要内容。"

    # 可视化模式选择
    # 选项: "single", "multi", "grid", "comparative"
    VISUALIZATION_MODE = "grid"  # 改为 "grid" 来测试网格可视化

    # --- 加载模型和处理器 ---
    print("正在加载模型和处理器...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    model.eval()
    print("加载完成！")

    if VISUALIZATION_MODE == "grid":
        # ========== 网格可视化模式 ==========
        print("\n===== 网格可视化模式 =====")
        print("此模式将创建一个网格对比图，展示不同层的CAM结果\n")

        # 加载图像
        raw_image = Image.open(IMAGE_PATH).convert("RGB")

        # 层范围选择选项:
        # - "auto": 自动选择代表性的层
        # - "early": 早期层（前1/3）
        # - "middle": 中期层（中1/3）
        # - "late": 后期层（后1/3）
        # - [0, 5, 10, 15, 20, 27]: 自定义层列表

        # 执行网格可视化
        results = create_layer_comparison_grid(
            model=model,
            processor=processor,
            image=raw_image,
            prompt=PROMPT,
            layer_range="auto",  # 自动选择代表性的层
            save_dir="cam_results_grid"
        )

        print("\n网格可视化完成！")

    elif VISUALIZATION_MODE == "comparative":
        # ========== 对比可视化模式 ==========
        print("\n===== 对比可视化模式 =====")
        print("对比早期、中期和后期层的差异\n")

        raw_image = Image.open(IMAGE_PATH).convert("RGB")

        # 分别可视化早期、中期、后期层
        for stage, layer_range in [("early", "early"), ("middle", "middle"), ("late", "late")]:
            print(f"\n处理{stage}层...")
            create_layer_comparison_grid(
                model=model,
                processor=processor,
                image=raw_image,
                prompt=PROMPT,
                layer_range=layer_range,
                save_dir=f"cam_results_{stage}"
            )

        print("\n对比可视化完成！")

    elif VISUALIZATION_MODE == "multi":
        # ========== 多层可视化模式 ==========
        print("\n===== 多层可视化模式 =====")

        # 加载图像
        raw_image = Image.open(IMAGE_PATH).convert("RGB")

        # 可以自定义要可视化的层
        # 例如：可视化第6、12、18、27层
        custom_layers = [6, 12, 18, 27]  # 可以根据需要修改

        # 或者让程序自动选择层（传入None）
        # custom_layers = None

        # 执行多层可视化
        results = visualize_multiple_layers(
            model=model,
            processor=processor,
            image=raw_image,
            prompt=PROMPT,
            layer_indices=custom_layers,
            save_dir="cam_results_multilayer"
        )

        print("\n多层可视化完成！")

    else:
        # ========== 单层可视化模式（原始代码）==========
        print("\n===== 单层可视化模式 =====")

        # --- 选择目标层 ---
        # 首先打印模型结构来找到正确的层路径
        print("正在检查模型结构...")
        print("Model components:")
        for name, module in model.named_children():
            print(f"  - {name}: {type(module)}")

        print("\nLanguage model components:")
        for name, module in model.language_model.named_children():
            print(f"  - {name}: {type(module)}")

        # 尝试不同的可能路径来访问transformer层
        try:
            # 尝试直接访问layers
            if hasattr(model.language_model, 'layers'):
                target_layer = model.language_model.layers[27]
                print(f"已选择目标层: language_model.layers[27]")
            # 尝试通过model.layers访问
            elif hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
                target_layer = model.language_model.model.layers[27]
                print(f"已选择目标层: language_model.model.layers[27]")
            # 尝试通过transformer访问
            elif hasattr(model.language_model, 'transformer') and hasattr(model.language_model.transformer, 'h'):
                target_layer = model.language_model.transformer.h[27]
                print(f"已选择目标层: language_model.transformer.h[27]")
            else:
                # 如果都不行，打印所有可能的子模块
                print("无法自动找到transformer层，请检查以下结构：")
                def print_model_structure(module, prefix="", max_depth=3, current_depth=0):
                    if current_depth >= max_depth:
                        return
                    for name, child in module.named_children():
                        print(f"{prefix}{name}: {type(child)}")
                        if 'layer' in name.lower() or 'block' in name.lower() or name in ['h', 'layers']:
                            print_model_structure(child, prefix + "  ", max_depth, current_depth + 1)

                print_model_structure(model.language_model)
                raise AttributeError("无法找到transformer层")

        except (IndexError, AttributeError) as e:
            print(f"错误：{e}")
            print("尝试使用较小的层索引...")

            # 尝试找到实际的层数
            possible_paths = [
                ('language_model.layers', lambda: model.language_model.layers),
                ('language_model.model.layers', lambda: model.language_model.model.layers),
                ('language_model.transformer.h', lambda: model.language_model.transformer.h),
            ]

            for path_name, path_func in possible_paths:
                try:
                    layers = path_func()
                    num_layers = len(layers)
                    print(f"找到 {num_layers} 层在路径 {path_name}")
                    # 选择最后一层或倒数第二层
                    layer_idx = min(27, num_layers - 1)
                    target_layer = layers[layer_idx]
                    print(f"已选择目标层: {path_name}[{layer_idx}]")
                    break
                except:
                    continue
            else:
                print("无法找到任何transformer层，程序退出")
                exit()

        # --- 初始化 Grad-CAM ---
        grad_cam = SmoothQwenGradCAM(
            model=model,
            processor=processor,
            target_layer=target_layer,
            num_samples=10,   # 加噪采样次数，设为 1 则等效于普通 Grad-CAM
            noise_std=0.15    # 噪声标准差
        )

        # --- 加载图像并运行 ---
        print("正在加载图像并生成 CAM...")
        raw_image = Image.open(IMAGE_PATH).convert("RGB")

        superimposed_image, heatmap = grad_cam.generate_cam(raw_image, PROMPT)

        # --- 保存和显示结果 ---
        if superimposed_image is not None:
            output_dir = "cam_results"
            os.makedirs(output_dir, exist_ok=True)

            # 将 PIL 图像和 numpy 数组并排显示
            result_pil = Image.fromarray(superimposed_image)
            heatmap_pil = Image.fromarray(heatmap)

            # 保存文件
            result_pil.save(os.path.join(output_dir, "superimposed_result.png"))
            heatmap_pil.save(os.path.join(output_dir, "heatmap.png"))
            raw_image.save(os.path.join(output_dir, "original_image.png"))

            print(f"CAM 生成成功！结果已保存到 '{output_dir}' 目录。")

            # 尝试显示图片
            try:
                import matplotlib.pyplot as plt
                fig, axs = plt.subplots(1, 3, figsize=(18, 6))
                axs[0].imshow(raw_image)
                axs[0].set_title("Original Image")
                axs[0].axis('off')
                axs[1].imshow(heatmap_pil)
                axs[1].set_title("Heatmap")
                axs[1].axis('off')
                axs[2].imshow(result_pil)
                axs[2].set_title("Superimposed Image")
                axs[2].axis('off')
                plt.savefig(os.path.join(output_dir, "visualization.png"), dpi=150, bbox_inches='tight')
                print(f"可视化图已保存到 '{output_dir}/visualization.png'")
                plt.show()
            except ImportError:
                print("Matplotlib 未安装，无法显示图片。")
            except:
                print("无法显示图片（可能在非GUI环境中）")