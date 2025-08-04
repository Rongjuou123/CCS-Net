import os

def batch_rename_jpg(folder_path, prefix="", suffix="", start_num=1, digit=4, dry_run=False):
    """
    批量重命名JPG文件
    
    参数:
        folder_path: 文件夹路径(已在脚本中固定)
        prefix: 文件名前缀(可选)
        suffix: 文件名后缀(可选)
        start_num: 起始序号(默认为1)
        digit: 序号位数(默认为4，如0001)
        dry_run: 试运行模式(只显示将要进行的更改，不实际执行)
    """
    # 获取文件夹中所有JPG文件
    jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.npy')]
    jpg_files.sort()  # 按文件名排序
    
    print(f"找到 {len(jpg_files)} 个JPG文件")
    
    for i, filename in enumerate(jpg_files):
        # 获取文件扩展名
        ext = os.path.splitext(filename)[1]
        
        # 生成新文件名
        num = start_num + i
        new_name = f"{prefix}{str(num).zfill(digit)}{suffix}{ext}"
        
        # 原始文件完整路径
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        if dry_run:
            print(f"[试运行] 将重命名: '{filename}' -> '{new_name}'")
        else:
            # 检查新文件名是否已存在
            if os.path.exists(new_path):
                print(f"警告: '{new_name}' 已存在，跳过重命名 '{filename}'")
                continue
            
            try:
                os.rename(old_path, new_path)
                print(f"已重命名: '{filename}' -> '{new_name}'")
            except Exception as e:
                print(f"重命名 '{filename}' 失败: {e}")

if __name__ == "__main__":
    # 在这里直接设置您的文件夹路径
    TARGET_FOLDER = r"C:\Users\admin\Desktop\CCS-New\CCS\train_dataset\train_left_test25\info"  # 请修改为您的实际路径
    
    # 重命名参数设置
    PREFIX = ""    # 文件名前缀
    SUFFIX = ""          # 文件名后缀
    START_NUM = 0       # 起始序号
    DIGIT = 4            # 序号位数
    DRY_RUN = False      # True为试运行模式，False为实际执行
    
    if not os.path.isdir(TARGET_FOLDER):
        print(f"错误: '{TARGET_FOLDER}' 不是有效的文件夹路径")
    else:
        batch_rename_jpg(
            folder_path=TARGET_FOLDER,
            prefix=PREFIX,
            suffix=SUFFIX,
            start_num=START_NUM,
            digit=DIGIT,
            dry_run=DRY_RUN
        )