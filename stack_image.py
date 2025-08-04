import numpy as np
import matplotlib.pyplot as plt
from PIL import Image  # 用于处理背景图片

# 1. 加载数据
heatmap_data = np.load(r"C:\Users\admin\Desktop\CCS-New\CCS\demo_data\chessboard\left1\DetectRes\heatmap\0.npy")
background_img = Image.open(r"C:\Users\admin\Desktop\CCS-New\CCS\demo_data\chessboard\left1\DetectRes\color_img\0.jpg")  # 替换为你的背景图片路径

# 2. 数据预处理
if heatmap_data.ndim == 3:
    heatmap_data = np.mean(heatmap_data, axis=2) if heatmap_data.shape[2] == 3 else heatmap_data[:,:,0]

# 3. 调整热力图和背景图尺寸一致
background_img = background_img.resize((heatmap_data.shape[1], heatmap_data.shape[0]))
background = np.array(background_img)

# 4. 创建画布
fig, ax = plt.subplots(figsize=(10, 10), frameon=False)

# 5. 先绘制背景图
ax.imshow(background)

# 6. 叠加半透明热力图
heatmap = ax.imshow(heatmap_data,
                   cmap='jet',          # 使用高对比度色标
                   alpha=0.8,           # 设置透明度（0-1之间）
                   interpolation='bilinear')

# 7. 移除所有坐标和边框
ax.set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)

# 8. 保存结果
plt.savefig(r"C:\Users\admin\Desktop\CCS-New\CCS\data\dataset6\overlay.png", 
           dpi=300, 
           bbox_inches='tight',
           pad_inches=0)

plt.show()