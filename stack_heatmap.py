import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_and_normalize_heatmap(npy_path):
    """加载.npy文件并归一化到[0,1]范围"""
    heatmap = np.load(npy_path)
    heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-10)
    return heatmap

def overlay_heatmaps_with_image(original_img, heatmap1, heatmap2, alpha1=0.5, alpha2=0.5, alpha_img=0.7, cmap1='hot', cmap2='cool'):
    """
    将两个热力图与原图叠加
    :param original_img: 原图（RGB格式，0-255范围）
    :param heatmap1: 第一个热力图（归一化到[0,1]）
    :param heatmap2: 第二个热力图（归一化到[0,1]）
    :param alpha1: 热力图1的透明度
    :param alpha2: 热力图2的透明度
    :param alpha_img: 原图的透明度
    :param cmap1: 热力图1的颜色映射
    :param cmap2: 热力图2的颜色映射
    :return: 叠加后的RGB图像（0-255范围）
    """
    # 将热力图转换为RGB（使用matplotlib的颜色映射）
    heatmap1_rgb = plt.get_cmap(cmap1)(heatmap1)[:, :, :3]  # 形状 (H,W,3)
    heatmap2_rgb = plt.get_cmap(cmap2)(heatmap2)[:, :, :3]
    
    # 确保原图与热力图尺寸相同
    if original_img.shape[:2] != heatmap1.shape:
        original_img = cv2.resize(original_img, (heatmap1.shape[1], heatmap1.shape[0]))
    
    # 归一化原图到[0,1]范围
    original_img_normalized = original_img.astype(float) / 255.0
    
    # 三层层叠：原图 + 热力图1 + 热力图2
    overlayed = (
        original_img_normalized * alpha_img + 
        heatmap1_rgb * alpha1 * (1 - alpha_img) + 
        heatmap2_rgb * alpha2 * (1 - alpha_img)
    )
    
    overlayed = np.clip(overlayed, 0, 1)  # 限制范围
    overlayed = (overlayed * 255).astype(np.uint8)  # 转换为0-255
    return overlayed

# 示例用法
if __name__ == "__main__":
    # 1. 加载原图和两个热力图
    original_img = cv2.imread(r"C:\Users\admin\Desktop\CCS-New\CCS\demo_data\chessboard\left1\img\20.png")  # 替换为你的原图路径
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)  # 转为RGB
    
    heatmap1 = load_and_normalize_heatmap(r"C:\Users\admin\Desktop\CCS-New\CCS\demo_data\chessboard\left1\DetectRes\heatmap\20.npy")  # 替换为你的.npy路径
    heatmap2 = load_and_normalize_heatmap(r"C:\Users\admin\Desktop\CCS-New\CCS\train_dataset\train_left_test33\heatmap\20-0.npy")
    
    # 2. 叠加热力图和原图（调整透明度）
    overlayed = overlay_heatmaps_with_image(
        original_img, heatmap1, heatmap2,
        alpha1=0.5,  # 热力图1透明度
        alpha2=0.7,  # 热力图2透明度
        alpha_img=0.6,  # 原图透明度
        cmap1='hot',
        cmap2='cool'
    )
    
    # 3. 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(heatmap1, cmap='hot')
    plt.title("Heatmap 1")
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(heatmap2, cmap='cool')
    plt.title("Heatmap 2")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(overlayed)
    plt.title("Overlayed Result")
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    cv2.imwrite("overlayed_result.png", cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
    plt.show()