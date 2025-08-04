import numpy as np

# 将 'your_file.npy' 替换为您的 .npy 文件的实际路径
file_path = r"C:\Users\admin\Desktop\CCS-New\CCS\train_dataset\train_left_test16\info\1000-0.npy"

try:
    data = np.load(file_path, allow_pickle=True)  # 允许加载包含对象的数组
    print(data)  # 打印数组的形状
except FileNotFoundError:
    print(f"错误：文件 '{file_path}' 未找到。")
except Exception as e:
    print(f"加载文件时发生错误：{e}")