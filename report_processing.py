import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import traceback
# import PySimpleGUI as sg

import numpy as np
from data_processing import (
    measure_one_microlens,
    measure_one_ring,
    measure_one_microlens_center_area,
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
        pass 
        # error_str = traceback.format_exc()
        # sg.popup('发生错误', error_str)
    return fig,checked_microlens
        
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

def report_one_microlens(id,sorted_microlens_params,data,radius, mm_per_point, N_line=6,N_point=100):
    x,y_mean,y_std=measure_one_microlens(id,sorted_microlens_params,data,N_line=N_line,N_point=N_point)
    x=x*mm_per_point
    power=measure_one_microlens_center_area(id,sorted_microlens_params,data,radius=radius)
    fig,ax=plt.subplots()
    ax.plot(x,y_mean)
    ax.fill_between(x,y_mean-y_std,y_mean+y_std,alpha=0.5)
    # 画一条虚线y=power
    ax.plot([x[0],x[-1]],[power,power],"--")
    # 在图中虚线上标注power
    ax.text(x[0],power,f"{power:.2f}D",ha="left",va="bottom")
    ax.set_xlabel("x/mm")
    ax.set_ylabel("power")
    ax.set_title(f"ID={id}, mean power={power:.2f}D")
    # 画一个矩形框
    ax.add_patch(plt.Rectangle((-radius*mm_per_point,power-0.5),2*radius*mm_per_point,1,fill=False,linestyle="--"))

    return fig