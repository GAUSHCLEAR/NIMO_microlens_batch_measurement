import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import numpy as np
from data_processing import (
    measure_one_microlens,
    measure_one_ring,
    check_all_microlens
)
import matplotlib.pyplot as plt

def report_checked_microlens(sorted_microlens_params, data, power_color_dict,radius=10, dpi=75,threshold=0.5):
    checked_microlens=check_all_microlens(sorted_microlens_params,data,power_color_dict,radius=radius,threshold=threshold)
    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.imshow(data, cmap="gray", interpolation='nearest')
    ax.axis('off')
    for i, microlens in enumerate(checked_microlens):
        center_x, center_y = microlens["center"]
        radius = microlens["radius"]
        color=microlens["color"] if microlens["color"]!="warning" else "red"
        ax.add_patch(plt.Circle((center_y, center_x), radius, color=color, fill=False))
        ax.text(center_y, center_x, str(i), color=color, fontsize=6, ha='center', va='center')
    return fig
        
def report_whole_picture(sorted_microlens_params, data, filename, dpi=75):
    # 创建一个Figure对象
    fig, axs = plt.subplots(1, 2, figsize=(10, 5), dpi=dpi)

    # 绘制第一个子图
    axs[0].imshow(data, cmap="gray", interpolation='nearest')
    axs[0].axis('off')

    # 绘制第二个子图
    axs[1].imshow(data, cmap="gray", interpolation='nearest')
    axs[1].axis('off')

    # 在第二个子图上画圆并标记序号
    for i, microlens in enumerate(sorted_microlens_params):
        center_x, center_y = microlens["center"]
        radius = microlens["radius"]
        ring = microlens["ring"]
        ring_color = ['r', 'g', 'b']
        axs[1].add_patch(plt.Circle((center_y, center_x), radius, color=ring_color[ring % 3], fill=False))
        axs[1].text(center_y, center_x, str(i), color=ring_color[ring % 3], fontsize=6, ha='center', va='center')

    # 返回Figure对象
    return fig

