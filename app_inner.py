import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
import os
from data_processing import *
from report_processing import *
import os 

os.environ['OMP_NUM_THREADS'] = '1'

# 超参数
image_center=(224,220)
ring_num=4

# 创建一个绘图函数
def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

# 设定layout
# 选择csv文件
# 选择是内圈，中圈还是外圈
# 绘图也放在layout里面
layout = [
    [sg.Text('选择csv文件')],
    [sg.Input(key='csv_file'), sg.FileBrowse()],
    [sg.Text('选择是内圈，中圈还是外圈')],
    [sg.Radio('内圈', "RADIO1", default=True, key='inner_ring'), sg.Radio('中圈', "RADIO1", key='middle_ring'), sg.Radio('外圈', "RADIO1", key='outer_ring')],
    [sg.Button('确定'), sg.Button('取消')],
    [sg.Canvas(key='-CANVAS-')],


]

# 设定窗口
window = sg.Window('微透镜测量', layout)

# 事件循环
while True:
    event, values = window.read()
    if event in (None, '取消'):
        break
    if event == '确定':
        filename=values['csv_file']

        data = read_data(filename)
        if values['inner_ring']:
            image_center=(220,220)
            ring_num=4
        elif values['middle_ring']:
            image_center=(600,220)
            ring_num=14
        elif values['outer_ring']:
            image_center=(1000,220)
            ring_num=14

        binary_image = detect_edge(data,threshold=1.0)
        # Apply the function to the binary_image
        microlenses, microlens_only_image = label_microlens(
            binary_image,
            min_area=15*15,
            )
        
        microlens_params=microlens_centers_radius(microlenses)
        sorted_microlens_params=cluster_rings(
                microlens_params,
                image_center=image_center,
                ring_num=ring_num,
                max_ring = ring_num+2,
                threshold=10)
        print("figure drawing")
        fig = report_whole_picture(sorted_microlens_params, data, "")
        # 检查是否已存在图表，如果是，则先清除
        if 'fig_agg' in locals():
            fig_agg.get_tk_widget().forget()
            plt.close('all')

        # 将图表绘制到PySimpleGUI窗口中
        fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

        # break
window.close()