import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
from data_processing import *
from report_processing import *
import warnings
warnings.filterwarnings('ignore')
point_per_mm=26.058823529411764

if 'session_state' not in st.session_state:
    st.session_state['session_state'] = {
        'report_plot': None,
        'report_text': None,
        'data_measure_csv': None,
    }

st.set_page_config(page_title='微透镜自动测量', layout='wide')

# 读取设计样板和测量数据文件
filename_design = st.sidebar.file_uploader('1.选择设计样板csv文件', type='csv')
if filename_design:
    data_origin=pd.read_csv(filename_design)

filename_measure = st.sidebar.file_uploader('2.选择测量数据csv文件', type='csv')
if filename_measure:
    data = read_data(filename_measure)
    point_per_mm=data.shape[0]/17 # 17mm
    mm_per_point=1/point_per_mm

measure_button=st.sidebar.button('3.开始测量')


if measure_button and (filename_design is not None) and (filename_measure is not None):
    # 微透镜识别与定位位置
    binary_image = detect_edge(data,threshold=1.0)
    microlenses, microlens_only_image = label_microlens(
        binary_image,
        min_area=15*15,
        )
    microlens_params=microlens_centers_radius(microlenses)
    # 微透镜邻域测量
    sorted_microlens_params=calculate_each_lens_Rx(data,microlens_params, fix_raidus=0.6*point_per_mm)
    # 微透镜加光度测量
    sorted_microlens_params=measure_microlens_in_diamter(sorted_microlens_params,data
    ,point_per_mm,diameter_list=[0.7,0.5,0.3,0.1])
    # 微透镜对齐
    aligned_coords=align_microlens(sorted_microlens_params,data_origin,point_per_mm)
    # 更新微透镜参数
    sorted_microlens_params=update_microlens_params_after_align(sorted_microlens_params,aligned_coords,data_origin)
    
    # 生成报告

    report_plot=report_align_location(data_origin,data,aligned_coords,sorted_microlens_params)

    st.session_state['session_state']['report_plot'] = report_plot

    
    ring_param_list=analysis_ring(sorted_microlens_params)
    report_text=generate_ring_report(ring_param_list)

    st.session_state['session_state']['report_text'] = report_text

    # 显示报告
    if st.session_state['session_state']['report_plot'] is not None:
        st.pyplot(st.session_state['session_state']['report_plot'])

    if st.session_state['session_state']['report_plot'] is not None:
        st.text_area("测量结果",st.session_state['session_state']['report_text'])

    # 保存测量结果
    
    data_measure_csv = generate_report_csv(data_origin,sorted_microlens_params)
    
    st.download_button(
        label="下载测量结果",
        data=data_measure_csv,
        file_name='测量结果.csv',
        mime='text/csv',
    )


