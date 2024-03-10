import streamlit as st
import matplotlib.pyplot as plt
import os
from data_processing import *
from report_processing import *
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='微透镜测量', layout='wide')
add_power_list = [4.5, 5, 5.5, 6, 6.5, 7]

def build_addpower_color_list(add_power_list):
    # color_list = ["#98D2EB", "#F9EBAE", "#C5A3FF", "#FFB1BB", "#87CEFA", "#FFA500"]
    color_list = ['#FF7F00', '#FFFF00', '#00FF00', '#00FFFF', '#0000FF', '#7F00FF']
    choose_color_list = color_list[:len(add_power_list)]
    addpower_color_dict = dict(zip(add_power_list, choose_color_list))
    return addpower_color_dict

addpower_color_dict = build_addpower_color_list(add_power_list)

# Sidebar for inputs
# st.sidebar.header('1. 选择csv文件')
filename = st.sidebar.file_uploader('1.选择csv文件', type='csv')

# st.sidebar.header('2. 选择是内圈，中圈还是外圈')
ring_choice = st.sidebar.radio('2. 选择是内圈，中圈还是外圈', ['内圈', '中圈', '外圈'])


st.sidebar.header('3. 测量参数')
# Rx = st.sidebar.number_input('处方焦度', value=0.0, step=0.01)
Rx=0.0
measure_threshold = st.sidebar.number_input('允差', value=0.25, step=0.01)
measure_threshold=measure_threshold/2
diameter = st.sidebar.number_input('测量直径', value=0.7, step=0.01)

st.sidebar.header('4. 加光颜色设定')
color_value_dict = {}
for addpower, color in addpower_color_dict.items():
    col1, col2  = st.sidebar.columns([1,3])

    choose_color= col1.color_picker("颜色", color,label_visibility='hidden')
    color_value = col2.number_input(f"加光度", value=float(addpower), step=0.01,label_visibility='hidden')
    color_value_dict[color] = color_value
# Main Panel
# st.title('微透镜测量')
st.sidebar.header('5. 一组微透镜测量')
ring_number_list=st.sidebar.text_input("需要测量的微透镜列表",value="0,1,3-10")
if ring_number_list:
    ring_number_list=parse_number_range(ring_number_list)

# Process and display data
if st.sidebar.button('确定') and filename is not None:
    data = read_data(filename)
    point_per_mm=data.shape[0]/17 # 17mm
    mm_per_point=1/point_per_mm

    semi_diameter = diameter / 2 * point_per_mm


    if ring_choice == '内圈':
        image_center=(17/2*point_per_mm,17/2*point_per_mm)
        ring_num=4
    elif ring_choice == '中圈':
        image_center=((17/2+11)*point_per_mm,17/2*point_per_mm)
        ring_num=7
    elif ring_choice == '外圈':
        image_center=((17/2+16)*point_per_mm,17/2*point_per_mm)
        ring_num=7
    else:
        pass 
    binary_image = detect_edge(data,threshold=1.0)
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
    sorted_microlens_params=calculate_each_lens_Rx(data,sorted_microlens_params)
    # report
    report=measure_list_of_microlens(ring_number_list,
    point_per_mm,sorted_microlens_params,data,Rx)
    st.markdown(report)

    # Display the figure using st.pyplot
    power_color_dict={
            Rx+color_value:color for color,color_value in color_value_dict.items()
        }
    fig,checked_microlens = report_checked_microlens(sorted_microlens_params, data, power_color_dict,radius=semi_diameter, dpi=75,Rx=Rx,threshold=measure_threshold)    # ... your plotting code ...
    st.pyplot(fig)
    # 从checked_microlens挑选出color='warning'的microlens
    # 找到它们的id
    # 逐一绘制report_one_microlens(id,sorted_microlens_params,data,radius, mm_per_point, N_line=6,N_point=100)
    warning_id=[i for i,microlens in enumerate(checked_microlens) if microlens["color"]=="warning"]
    warning_id_list=warning_id
    # # 将warning_id显示到text_input中，以逗号分隔
    # warning_id_str=",".join([str(i) for i in warning_id])
    # warning_id_strlist=st.text_input("需要关注的微透镜列表",value=warning_id_str)
    # warning_id_list=[int(i) for i in warning_id_strlist.split(",")]
    # print(warning_id_list)
    # if st.button("显示需要关注的微透镜"):
    if True:
        # print("显示需要关注的微透镜")
        # print(warning_id_list)
        for id in warning_id_list:
            print(f"绘制微透镜{id}")
            one_lens_fig=report_one_microlens(id,sorted_microlens_params,data,radius=semi_diameter, mm_per_point=mm_per_point, N_line=6,N_point=100)
            st.pyplot(one_lens_fig)


# Add more elements as needed
