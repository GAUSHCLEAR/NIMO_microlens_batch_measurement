import PySimpleGUI as sg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import matplotlib.pyplot as plt
import os
from data_processing import *
from report_processing import *
import os 
import warnings
warnings.filterwarnings('ignore')

sg.theme('SystemDefaultForReal')   # Add a touch of color
sg.set_options(font=("Default", 16))  # Set default font size

os.environ['OMP_NUM_THREADS'] = '1'
add_power_list=[3,3.5,4,4.5,5,5.5]
# add_power_list=[3.5,4.5,5]

# 创建一个绘图函数
def draw_figure(canvas, figure):
    figure.patch.set_facecolor('none')
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def build_addpower_color_list(add_power_list):
    color_list=["#98D2EB","#F9EBAE","#C5A3FF","#FFB1BB","#87CEFA","#FFA500"]
    choose_color_list=color_list[:len(add_power_list)]
    addpower_color_dict=dict(zip(add_power_list,choose_color_list))
    color_layout_list=[
        [sg.Text('■',background_color=color,text_color=color,size=(10,1) ), 
         sg.Input(key=color, default_text=f"{addpower:.2f}",size=(20,1))] for addpower,color in addpower_color_dict.items()]
    return color_layout_list,addpower_color_dict

color_layout_list,addpower_color_dict=build_addpower_color_list(add_power_list)

# 设定layout
# 选择csv文件
# 选择是内圈，中圈还是外圈
# 绘图也放在layout里面
left_column = [
    [sg.Text('1. 选择csv文件')],
    [sg.Input(key='csv_file',size=(20,1)), sg.FileBrowse('选择文件', file_types=(('CSV Files', '*.csv'),),size=(10,1))],
    [sg.Text('2. 选择是内圈，中圈还是外圈')],
    [sg.Radio('内圈', "RADIO1", default=True, key='inner_ring'), sg.Radio('中圈', "RADIO1", key='middle_ring'), sg.Radio('外圈', "RADIO1", key='outer_ring')],
    [sg.Text('3. 测量参数')],
    [sg.Text('处方焦度',size=(10,1)), 
     sg.Input(key='Rx',size=(20,1))],
    [sg.Text('允差',size=(10,1)), 
     sg.Input(key='measure_threshold', default_text='0.5',size=(20,1))],
    [sg.Text('测量直径',size=(10,1)), sg.Input(key='diameter', default_text='0.7',size=(20,1))],
    [sg.Text('4. 加光颜色设定')]]+color_layout_list+[
    [sg.Button('确定'), sg.Button('取消')],
]

right_column = [
    [sg.Canvas(key='-CANVAS-',size=(512,512))],
]

layout = [
    [
        sg.Column(left_column),
        sg.Column(right_column),
    ]
]

# 设定窗口
window = sg.Window('微透镜测量', layout,
        # size=(1024,512),
        resizable=True
        )

# 事件循环
while True:
    event, values = window.read()
    if event in (None, '取消'):
        break
    if event == '确定' :
        # filename  
        filename=values['csv_file']
        if not os.path.exists(filename):
            sg.popup("文件不存在")
            continue

        data = read_data(filename)
        point_per_mm=data.shape[0]/17 # 17mm
        mm_per_point=1/point_per_mm

        # power_color_dict
        Rx=float(values['Rx']) if values['Rx'] else 0.00
        color_value_list=[float(values[color]) if values[color] else addpower for addpower,color in addpower_color_dict.items()]
        power_color_dict={
            Rx+color_value:color for color,color_value in zip(addpower_color_dict.values(),color_value_list)
        }

        # measure_threshold
        measure_threshold=float(values['measure_threshold']) if values['measure_threshold'] else 0.5
        # diameter
        diameter=float(values['diameter']) if values['diameter'] else 0.7 
        semi_diameter=diameter/2 * point_per_mm

        if values['inner_ring']:
            image_center=(17/2*point_per_mm,17/2*point_per_mm)
            # print(image_center)
            ring_num=4
        elif values['middle_ring']:
            image_center=((17/2+11)*point_per_mm,17/2*point_per_mm)
            # print(image_center)
            ring_num=14
        elif values['outer_ring']:
            image_center=((17/2+16)*point_per_mm,17/2*point_per_mm)
            # print(image_center)
            ring_num=14
        else:
            sg.popup("请选择是内圈，中圈还是外圈")
            continue

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
        # print("figure drawing")
        fig = report_checked_microlens(sorted_microlens_params, data, power_color_dict,radius=semi_diameter, dpi=75,threshold=measure_threshold)
        # 检查是否已存在图表，如果是，则先清除
        if 'fig_agg' in locals():
            fig_agg.get_tk_widget().forget()
            plt.close('all')

        # 将图表绘制到PySimpleGUI窗口中
        fig_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)

        # break
window.close()