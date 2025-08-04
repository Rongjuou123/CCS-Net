import re
import matplotlib.pyplot as plt

# 1. 从txt文件读取日志数据
def parse_log_file(file_path):
    batch_losses = []
    val_losses = []
    
    with open(file_path, 'r') as f:
        for line in f:
            # 提取batch loss
            batch_match = re.search(r'loss \(batch\)=([\d.]+)', line)
            if batch_match:
                batch_losses.append(float(batch_match.group(1)))
            
            # 提取validation loss
            val_match = re.search(r'Validation MSE loss: ([\d.]+)', line)
            if val_match:
                val_losses.append(float(val_match.group(1)))
    
    return batch_losses, val_losses

# 2. 绘制损失曲线
def plot_losses(batch_losses, val_losses):
    plt.figure(figsize=(12, 6))
    
    # Batch Loss（高频细线）
    plt.plot(range(len(batch_losses)), 
             batch_losses, 
             'b-', 
             alpha=0.5, 
             linewidth=1,
             label='Batch Loss')
    
    # Validation Loss（粗线标记关键点）
    val_steps = [i * len(batch_losses) // len(val_losses) for i in range(len(val_losses))]
    plt.plot(val_steps, 
             val_losses, 
             'ro-', 
             linewidth=2,
             markersize=6,
             label='Validation Loss')
    
    # 图表装饰
    plt.title("Training Loss Curves", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss Value", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # 自动调整Y轴范围
    max_loss = max(max(batch_losses), max(val_losses)) * 1.1
    plt.ylim(0, max_loss)
    
    plt.tight_layout()
    plt.show()

# 3. 主程序
if __name__ == "__main__":
    # 替换为您的实际文件路径
    log_file = r"C:\Users\admin\Desktop\CCS-New\CCS\log\train_total.txt"  
    
    # 解析日志文件
    batch_loss, val_loss = parse_log_file(log_file)
    
    # 打印提取的数据量（用于调试）
    print(f"Found {len(batch_loss)} batch losses")
    print(f"Found {len(val_loss)} validation losses")
    
    # 绘制图形
    plot_losses(batch_loss, val_loss)