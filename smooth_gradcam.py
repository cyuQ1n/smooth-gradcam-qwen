"""
Smooth Grad-CAM for Qwen2.5-VL Model
====================================

Core implementation of Smooth Grad-CAM visualization for Qwen2.5-VL models.
This module provides attention visualization capabilities for vision-language models.

Author: Chuanyu Qin
AI Assistant: Claude Code
Reference: https://github.com/zhangbaijin/From-Redundancy-to-Relevance/blob/master/demo_smooth_grad_threshold.py
Date: 2024
License: MIT
"""

import os
import torch
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import warnings

# 忽略特定的 torchvision 警告
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
        """
        初始化 SmoothQwenGradCAM。

        Args:
            model: Qwen2.5-VL 模型实例
            processor: 对应的处理器
            target_layer: 要可视化的目标层
            num_samples: SmoothGrad 的采样次数（默认10）
            noise_std: 噪声标准差（默认0.1）
        """
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
        # 确保像素值在合理范围内
        noisy_pixel_values = torch.clamp(noisy_pixel_values, -3.0, 3.0)
        return noisy_pixel_values

    def _remove_hooks(self):
        """移除所有注册的钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def _save_feature_maps(self, module, input, output):
        """保存目标层的输出特征图"""
        self.feature_maps = output[0]
        if self.feature_maps.requires_grad:
            self.feature_maps.retain_grad()

    def _save_gradients(self, module, grad_input, grad_output):
        """保存反向传播回来的梯度"""
        self.gradients = grad_output[0]

    def _register_hooks(self):
        """注册前向和反向传播钩子"""
        forward_hook = self.target_layer.register_forward_hook(self._save_feature_maps)
        backward_hook = self.target_layer.register_full_backward_hook(self._save_gradients)
        self.hooks = [forward_hook, backward_hook]

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
            # Fallback方案：使用固定范围
            print(f"使用vision_start/end标记失败: {e}")
            print("使用fallback方案")

            seq_len = len(input_ids[0])
            # 使用经验值估计图像token范围
            estimated_start = max(10, seq_len // 4)
            estimated_end = min(seq_len - 50, estimated_start + 1024)

            print(f"估计的图像token范围: [{estimated_start}, {estimated_end})")
            return estimated_start, estimated_end

    def generate_cam(self, image: Image.Image, prompt: str, use_smooth=True):
        """
        生成 (Smooth)Grad-CAM 热力图。

        Args:
            image (PIL.Image.Image): 输入的原始图像
            prompt (str): 输入的文本 prompt
            use_smooth (bool): 是否使用 SmoothGrad（加噪平均）

        Returns:
            Tuple[np.ndarray, np.ndarray]: (叠加后的图像, 原始热力图)
        """
        # 1. 准备模型输入
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self.device)

        prompt_len = inputs['input_ids'].shape[1]

        # 2. 生成完整的回答
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
            # 注册钩子
            self._register_hooks()

            try:
                self.model.zero_grad()

                # 为 SmoothGrad 添加噪声
                if use_smooth:
                    noisy_pixel_values = self._add_gaussian_noise_to_pixel_values(
                        inputs['pixel_values'].clone(), self.noise_std
                    )
                else:
                    noisy_pixel_values = inputs['pixel_values']

                # 4. 单次前向传播以获取 logits
                full_input_ids = torch.cat([inputs['input_ids'], generated_tokens.unsqueeze(0)], dim=1)
                full_attention_mask = torch.ones_like(full_input_ids)

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

                outputs = self.model(**model_inputs)
                logits = outputs.logits

                # 5. 计算目标分数并反向传播
                target_logits = logits[0, prompt_len - 1 : -1, :]
                target_ids = generated_tokens.unsqueeze(1)
                score = target_logits.gather(1, target_ids).sum()
                score.backward()

                # 6. 计算 CAM
                if self.feature_maps is None or self.gradients is None:
                    print(f"警告：第 {i+1} 次循环中未捕获到特征图或梯度，跳过...")
                    continue

                # 定位视觉 token
                vis_start, vis_end = self._get_visual_token_indices(full_input_ids)

                # 提取视觉部分的特征图和梯度
                vis_feature_maps = self.feature_maps[:, vis_start:vis_end, :]
                vis_gradients = self.gradients[:, vis_start:vis_end, :]

                # Global Average Pooling
                pooled_gradients = torch.mean(vis_gradients, dim=[0, 1])

                # 权重乘以特征图
                for j in range(vis_feature_maps.shape[-1]):
                    vis_feature_maps[:, :, j] *= pooled_gradients[j]

                cam = torch.sum(vis_feature_maps, dim=-1)
                cam = F.relu(cam)

                # 归一化并 reshape 成 2D 图像
                cam = cam.squeeze().detach().float().cpu().numpy()
                if np.max(cam) > 0:
                    cam = cam / np.max(cam)

                # 处理非正方形的视觉token网格
                cam = self._reshape_cam_to_2d(cam)

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

        # 叠加到原图 - 降低热力图权重，让原图更清晰
        superimposed_img = heatmap_colored * 0.3 + original_image_np * 0.7
        superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

        return superimposed_img, heatmap_colored

    def _reshape_cam_to_2d(self, cam):
        """
        将一维CAM数组重塑为二维网格。
        处理非正方形的视觉token网格。

        Args:
            cam (np.ndarray): 一维CAM数组

        Returns:
            np.ndarray: 二维CAM网格
        """
        num_patches = cam.shape[0]

        # 尝试找到最接近的因子对
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
                # 选择最接近正方形的因子对
                h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
            else:
                # 如果找不到合适的因子，使用近似值
                h = w = int(np.ceil(np.sqrt(num_patches)))
                # 填充cam数组
                cam_padded = np.zeros(h * w)
                cam_padded[:num_patches] = cam
                cam = cam_padded

        print(f"将 {num_patches} 个patches重塑为 ({h}, {w}) 网格")
        return cam.reshape(h, w)


def get_model_layers(model):
    """
    获取模型的所有层。

    Args:
        model: Qwen2.5-VL 模型实例

    Returns:
        layers: 模型层列表
        total_layers: 总层数
    """
    if hasattr(model.language_model, 'layers'):
        layers = model.language_model.layers
    elif hasattr(model.language_model, 'model') and hasattr(model.language_model.model, 'layers'):
        layers = model.language_model.model.layers
    elif hasattr(model.language_model, 'transformer') and hasattr(model.language_model.transformer, 'h'):
        layers = model.language_model.transformer.h
    else:
        raise AttributeError("无法找到模型层")

    return layers, len(layers)