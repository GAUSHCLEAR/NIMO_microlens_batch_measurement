import PySimpleGUI as sg

import matplotlib.pyplot as plt
import os
from data_processing import *
from report_processing import *
import os 

os.environ['OMP_NUM_THREADS'] = '1'

# 超参数
image_center=(224,220)
ring_num=4

# 设定layout
# 选择csv文件
layout = [
    [sg.Text('选择csv文件')],
    [sg.Input(key='csv_file'), sg.FileBrowse()],
    [sg.Button('确定'), sg.Button('取消')]
]

# 设定窗口
window = sg.Window('csv文件选择', layout)

# 事件循环
while True:
    event, values = window.read()
    if event in (None, '取消'):
        break
    if event == '确定':
        filename=values['csv_file']
        data = read_data(filename)
        binary_image = detect_edge(data,threshold=1.0)
        # Apply the function to the binary_image
        microlenses, microlens_only_image = label_microlens(
            binary_image,
            min_area=15*15,
            )
        
        microlens_params=microlens_centers_radius(microlenses)
        sorted_microlens_params=cluster_rings(
                microlens_params,
                image_center=(220,220),
                ring_num=4,
                max_ring = 6,
                threshold=10)
        print("figure drawing")
        plt.figure(figsize=(10, 10))
        plt.subplot(121)
        plt.imshow(data, cmap="gray", interpolation='nearest')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(data, cmap="gray", interpolation='nearest')
        # 在microlens_only_image上画圆并标记序号
        for i, microlens in enumerate(sorted_microlens_params):
            center_x, center_y = microlens["center"]
            radius = microlens["radius"]
            ring=microlens["ring"]
            # print(ring)
            ring_color=['r','g','b']
            plt.gca().add_patch(plt.Circle((center_y, center_x), radius, color='r', fill=False))
            plt.text(center_y, center_x, str(i+1), color=ring_color[ring % 3], fontsize=6, ha='center', va='center')
            plt.axis('off')
        plt.show()
        print("figure saving")
        # break
window.close()