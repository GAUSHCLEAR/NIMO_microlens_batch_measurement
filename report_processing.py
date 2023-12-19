import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import traceback
import PySimpleGUI as sg

import numpy as np
from data_processing import (
    measure_one_microlens,
    measure_one_ring,
    check_all_microlens,
    update_microlens_with_common_power,
)
import matplotlib.pyplot as plt

def report_checked_microlens(sorted_microlens_params, data, power_color_dict,radius=10, dpi=75,threshold=0.5):
    try:
        checked_microlens=check_all_microlens(sorted_microlens_params,data,power_color_dict,radius=radius,threshold=threshold)
        # checked_microlens = update_microlens_with_common_power(checked_microlens)

        # fig, ax = plt.subplots(figsize=(6, 5), dpi=dpi)
        fig,ax=plt.subplots()
        fig.patch.set_facecolor('none')
        ax.imshow(data, cmap="gray", interpolation='nearest')
        ax.axis('off')
        for i, microlens in enumerate(checked_microlens):
            center_x, center_y = microlens["center"]
            radius = microlens["radius"]
            color=microlens["color"] if microlens["color"]!="warning" else "red"
            alpha=0.4 if microlens["color"]!="warning" else 1.0
            ax.add_patch(plt.Circle((center_y, center_x), radius, color=color, fill=True,alpha=alpha))
            ax.text(center_y, center_x, str(i), color="black", fontsize=8, ha='center', va='center')
        # power_color_dict作为legend
        # power_color_dict是: {power:color}字典
        ax.legend(handles=[plt.Circle((0, 0), 0.1, color=color) for color in power_color_dict.values()],
            labels=[f"{power:.2f} D" for power in power_color_dict.keys()],
            loc='upper right',
            #   fontsize=8,
            bbox_to_anchor=(1.3, 1))  # 将图例向右移动
    except:
        error_str = traceback.format_exc()
        sg.popup('发生错误', error_str)
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

