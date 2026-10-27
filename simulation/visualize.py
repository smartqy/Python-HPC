import os
import numpy as np
import matplotlib.pyplot as plt

# 获取当前脚本所在的目录
script_dir = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(script_dir, "..", "data", "sample")

# 假设数据路径和楼栋 ID 已知
building_ids = ['694','986','1878','1869','3171','4485','5158','5971','6233','6452','7025','7987','8264','9648','9991','10000','10535','29753','49314','50814']
building_id = '1869'

# 示例读取一个 building 的 domain 和 interior
def load_building_data(building_id, base_path):
    domain_path = os.path.join(base_path, f"{building_id}_domain.npy")
    interior_path = os.path.join(base_path, f"{building_id}_interior.npy")

    if not os.path.exists(domain_path):
        raise FileNotFoundError(f"❌ 缺少文件：{domain_path}")
    if not os.path.exists(interior_path):
        raise FileNotFoundError(f"❌ 缺少文件：{interior_path}")

    domain = np.load(domain_path)
    interior = np.load(interior_path)
    return domain, interior

def plot_building_batch(building_ids, base_path, n_cols=2):
    n = len(building_ids)
    n_rows = 2  # 每个建筑显示两行图（温度 & mask）
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 6), dpi=100)

    cmap = plt.cm.magma
    vmin, vmax = 0, 25  # 温度范围固定统一
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for idx, building_id in enumerate(building_ids):
        try:
            domain, interior = load_building_data(building_id, base_path)
        except FileNotFoundError as e:
            print(e)
            continue

        col = idx % n_cols

        # 上排：温度图
        ax1 = axes[0, col]
        im = ax1.imshow(domain, cmap=cmap, norm=norm)
        ax1.set_title(f"ID: {building_id}", fontsize=12)
        ax1.axis('off')

        # 下排：interior mask
        ax2 = axes[1, col]
        ax2.imshow(interior, cmap='gray', vmin=0, vmax=1)  # 更强烈的黑白对比
        ax2.axis('off')

    # 添加 colorbar 到右侧（仅一次）
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Temperature')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 预留右侧 colorbar 空间
    plt.show()

def after_plot_all_buildings(all_u, all_masks, building_ids):
    n_rows = 2  # 每个建筑显示两行图（温度 & mask）
    n_cols = len(building_ids)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 6), dpi=100)

    cmap = plt.cm.magma
    vmin, vmax = 0, 25  # 温度范围固定统一
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for col, bid in enumerate(building_ids):
        u = all_u[col]
        interior = all_masks[col]

        # 温度图
        ax1 = axes[0, col] if n_cols > 1 else axes[0]
        im = ax1.imshow(u, cmap=cmap, norm=norm)
        ax1.set_title(f"Building {bid}", fontsize=10)
        ax1.axis('off')

        # mask 图
        ax2 = axes[1, col] if n_cols > 1 else axes[1]
        ax2.imshow(interior, cmap='gray', vmin=0, vmax=1)  # 更强烈的黑白对比
        ax2.set_title("Interior Mask", fontsize=12)        
        ax2.axis('off')

    # 添加 colorbar 到右侧（仅一次）
    cbar_ax = fig.add_axes([0.92, 0.3, 0.015, 0.4])  # [left, bottom, width, height]
    fig.colorbar(im, cax=cbar_ax, label='Temperature')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 预留右侧 colorbar 空间
    plt.show()

if __name__ == '__main__':

    plot_building_batch(building_ids, base_path, 20)
