import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from data_processing import (
    measure_one_microlens,
    measure_one_ring,
)
def report_microlens_in_ring(
        id,sorted_microlens_params,data,
        filename, ROI_radius=3,
        amp=2,
        power_min=-10,
        power_max=5,
        N_line=6,
        N_point=101,
        figsize=(5,5),
        font_size=6,
        dpi=300):
    # 字体大小配置
    SMALL_SIZE = font_size
    MEDIUM_SIZE = font_size+2
    BIGGER_SIZE = font_size+4

    plt.rc('font', size=SMALL_SIZE)          # 控制默认文本大小
    plt.rc('axes', titlesize=SMALL_SIZE)     # 子图标题的字体大小
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # x和y标签的字体大小
    plt.rc('xtick', labelsize=SMALL_SIZE)    # x轴刻度标签的字体大小
    plt.rc('ytick', labelsize=SMALL_SIZE)    # y轴刻度标签的字体大小
    plt.rc('legend', fontsize=SMALL_SIZE)    # 图例的字体大小
    plt.rc('figure', titlesize=BIGGER_SIZE)  # 图表标题的字体大小
    microlens=sorted_microlens_params[id]
    ring_id=microlens["ring"]
    x,y_mean,y_std=measure_one_microlens(id,sorted_microlens_params,data,N_line=N_line,N_point=N_point)
    ROI_index=np.where(np.abs(x)<ROI_radius)
    ROI_y_mean=np.mean(y_mean[ROI_index])

    ring_x,ring_y_mean,ring_y_std=measure_one_ring(ring_id,sorted_microlens_params,data,N_line=N_line,N_point=N_point)
    # 绘图

    gs = gridspec.GridSpec(1, 2, width_ratios=[2, 1])  # 为左边的图分配更大的空间

    fig=plt.figure(figsize=figsize)
    # 光焦度曲线
    ax1=fig.add_subplot(gs[0])
    ax1.plot(x,y_mean,'r')
    ax1.plot(ring_x,ring_y_mean,'b')
    ax1.fill_between(ring_x, ring_y_mean-amp*ring_y_std, ring_y_mean+amp*ring_y_std, alpha=0.3)
    ax1.set_xlabel("Distance from the center (pixel) ")
    ax1.set_ylabel("Sphere power (D)")
    ax1.legend([f"Microlens {id+1}",
                f"Ring mean {ring_id+1}",
                f"Ring 95%CI {ring_id+1}"
                ])
    ax1.grid()
    ax1.set_yticks(np.arange(power_min, power_max, 0.5))
    ax1.set_aspect(2)  # 设置子图的纵横比为1:1


    # 位置标记
    ax2=fig.add_subplot(gs[1])
    center_x, center_y = microlens["center"]
    radius = microlens["radius"]
    ax2.imshow(data, cmap="gray", interpolation='nearest')
    ax2.imshow(data, cmap="gray", interpolation='nearest')
    circle = plt.Circle((center_y, center_x), radius, color='r', fill=False)
    ax2.add_patch(circle)
    ax2.text(center_y, center_x, str(id+1), color="black", fontsize=font_size, ha='center', va='center')
    ax2.axis('off')


    plt.savefig(filename,dpi=dpi,bbox_inches='tight')
    plt.close()
    return ROI_y_mean 