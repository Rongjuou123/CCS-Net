import os
import glob

# --- 您需要根据实际情况修改这些路径 ---
image_folder = r"C:\Users\admin\Desktop\CCS-New\CCS\train_dataset\train_left\img"
# dist_corner_data_folder = r'C:\Users\admin\Desktop\CCS\train_dataset\train_left\dist_corner' # 假设角点数据在另一个文件夹
# ori_corner_data_folder = r'C:\Users\admin\Desktop\CCS\train_dataset\train_left\ori_corner'
heatmap_folder = r"C:\Users\admin\Desktop\CCS-New\CCS\train_dataset\train_left\heatmap"
output_txt_file = r'C:\Users\admin\Desktop\CCS-New\CCS\train_dataset\train_left\train_cornerdect_txt_path.txt' # 这是您的数据集类要读取的txt文件

# 假设图像文件扩展名可以是 .png, .jpg, .jpeg, .bmp
image_extensions = ('*.png', '*.jpg', '*.jpeg', '*.bmp')
# 假设角点数据文件的扩展名是 .npy (请根据您的实际情况修改)
heatmap_extension = '.npy'

# -----------------------------------------

image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(image_folder, ext)))

if not image_files:
    print(f"在文件夹 '{image_folder}' 中没有找到任何支持的图像文件。")
else:
    print(f"找到了 {len(image_files)} 个图像文件。")

with open(output_txt_file, 'w') as f:
    for img_path in image_files:
        # 从图像路径构建对应的角点数据路径
        # 这是一个示例逻辑，您需要根据您的命名规则修改
        base_filename = os.path.splitext(os.path.basename(img_path))[0] # 获取不带扩展名的文件名，例如 'image001'
        
        # 示例1：角点文件与图像文件同名，扩展名不同，在corner_data_folder中
        heatmap_file_path = os.path.join(heatmap_folder, base_filename + heatmap_extension)
        # 示例2：如果角点文件与图像文件在同一图像文件夹，只是后缀不同，例如 image001_corners.npy
        # corner_file_path = os.path.join(image_folder, base_filename + '_corners' + corner_extension)

        # 检查角点文件是否存在
        if os.path.exists(heatmap_file_path):
            f.write(f"{img_path} {heatmap_file_path}\n")
        else:
            print(f"警告：图像 '{img_path}' 没有找到对应的角点文件 '{heatmap_file_path}'，已跳过。")

print(f"已成功生成列表文件: {output_txt_file}")