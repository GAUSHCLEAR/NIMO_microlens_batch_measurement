import streamlit as st
import matplotlib.pyplot as plt
import os
from data_processing import *
from report_processing import *
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title='微透镜批量测量', layout='wide')

# Sidebar for inputs
# st.sidebar.header('1. 选择csv文件')
filename = st.sidebar.file_uploader('1.选择csv文件', type='csv')

# st.sidebar.header('2. 选择是内圈，中圈还是外圈')
ring_choice = st.sidebar.radio('2. 选择是内圈，中圈还是外圈', ['内圈', '中圈', '外圈'])

if ring_choice == '内圈':
    default_ring_num=2
elif ring_choice == '中圈':
    default_ring_num=7
elif ring_choice == '外圈':
    default_ring_num=7
else:
    pass 

ring_num=st.sidebar.number_input('3. 选择测量的微透镜数量', value=default_ring_num, step=1)

if st.sidebar.button('确定') and filename is not None:
    data = read_data(filename)
    point_per_mm=data.shape[0]/17 # 17mm
    mm_per_point=1/point_per_mm
    diameter=0.7 
    semi_diameter = diameter / 2 * point_per_mm

    if ring_choice == '内圈':
        image_center=(17/2*point_per_mm,17/2*point_per_mm)
    elif ring_choice == '中圈':
        image_center=((17/2+11)*point_per_mm,17/2*point_per_mm)
    elif ring_choice == '外圈':
        image_center=((17/2+16)*point_per_mm,17/2*point_per_mm)
    else:
        pass 
    binary_image = detect_edge(data,threshold=0.8)
    microlenses, microlens_only_image = label_microlens(
    binary_image,
    min_area=10*10,
    )
    microlens_params=microlens_centers_radius(microlenses)
    sorted_microlens_params=cluster_rings(
            microlens_params,
            image_center=image_center,
            ring_num=ring_num,
            max_ring = ring_num+2,
            threshold=10)
    fig=report_whole_picture(sorted_microlens_params, data,"",dpi=75)
    st.pyplot(fig)

    sorted_microlens_params=calculate_each_lens_Rx(data,sorted_microlens_params)
    ring_number_list_list=[]
    for id, microlens in enumerate(sorted_microlens_params):
        # 根据microlens['ring']的值，将microlens分组
        # 把同一个ring的id 放在一个list里
        ring_number_list_list.append([i for i in range(len(sorted_microlens_params)) if sorted_microlens_params[i]['ring']==microlens['ring']])
    # 去掉重复的list
    ring_number_list_list = list(set([tuple(t) for t in ring_number_list_list]))
    # 按照ring的顺序排序
    ring_number_list_list.sort(key=lambda x: sorted_microlens_params[x[0]]['ring'])

    report_text=""
    for ring_number_list in ring_number_list_list:
        mean_list=[]
        std_list=[]

        diameter_list=[0.7,0.5,0.3,0.1]
        for d in diameter_list:
            power_list=[]
            maxpower_list=[]
            for i in ring_number_list:
                measure_radius=d/2*point_per_mm
                power=measure_one_microlens_center_area(i,sorted_microlens_params,data,radius=measure_radius)

                max_power=measure_one_microlens_max(i,sorted_microlens_params,data,radius=measure_radius)

                power_list.append(power)
                maxpower_list.append(max_power)
                # fig.show()

            mean_power=np.mean(power_list)
            std_power=np.std(power_list)
            max_power_mean=np.mean(maxpower_list)
            max_power_std=np.std(maxpower_list)

            mean_list.append(mean_power)
            std_list.append(std_power)
            report_text+=f"{d}\t{mean_power:.3f}\t{std_power:.3f}\n" 
        report_text+=f"0.0\t{max_power_mean:.3f}\t{max_power_std:.3f}\n\n"
    
    st.text_area("测量结果",report_text)