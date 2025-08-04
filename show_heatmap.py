import numpy as np
import matplotlib.pyplot as plt

# 加载.npy文件
data = np.load(r"C:\Users\admin\Desktop\CCS-New\CCS\train_dataset\train_left_test23\heatmap\0-0.npy")  # 替换为你的文件路径

# 检查数据维度
print("数据形状:", data.shape)
print("数据类型:", data.dtype)

# 确保是2D数组
if data.ndim == 3:
    # 如果是3D数据（如RGB图像），取第一个通道或转换为灰度
    if data.shape[2] == 3:
        data = np.mean(data, axis=2)  # RGB转灰度
    else:
        data = data[:, :, 0]  # 取第一个通道
elif data.ndim > 3:
    raise ValueError("仅支持2D或3D数据，当前维度: {}".format(data.ndim))

# 创建带自定义色标的图形
plt.figure(figsize=(5, 6))

# 绘制热力图（不显示任何坐标和边框）
plt.imshow(data, cmap='viridis')
plt.axis('off')  # 关闭所有坐标轴
plt.gca().set_axis_off()
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.margins(0,0)

# 保存纯净图像（无任何额外元素）
plt.savefig('pure_heatmap.png', 
           dpi=300, 
           bbox_inches='tight', 
           pad_inches=0, 
           transparent=True)

# 显示图像
plt.show()