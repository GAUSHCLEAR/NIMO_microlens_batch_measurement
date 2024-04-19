import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import traceback
# import PySimpleGUI as sg

import numpy as np
from data_processing import (
    measure_one_microlens,
    measure_one_ring,
    measure_one_microlens_max,
    measure_one_microlens_center_area,
    check_all_microlens,
    update_microlens_with_common_power,
    calculate_each_lens_Rx,
)
import matplotlib.pyplot as plt
from point_set_registration import (
    alignment_by_coordinates,
    alignment_by_powers,
    find_nearest_index,
)

def report_checked_microlens(sorted_microlens_params, data, power_color_dict,radius=10, dpi=75,Rx=0,threshold=0.5):
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
            labels=[f"add {power-Rx:.2f} D" for power in power_color_dict.keys()],
            loc='upper right',
            #   fontsize=8,
            bbox_to_anchor=(1.4, 1))  # 将图例向右移动
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
    Rx=sorted_microlens_params[id]['Rx']
    power=measure_one_microlens_center_area(id,sorted_microlens_params,data,radius=radius)
    fig,ax=plt.subplots()
    ax.plot(x,y_mean)
    ax.fill_between(x,y_mean-y_std,y_mean+y_std,alpha=0.5)
    # 画一条虚线y=Rx+power
    ax.plot([x[0],x[-1]],[Rx+power,Rx+power],"--")
    # 在图中虚线上标注power
    ax.text(x[0],Rx+power,f"{power:.2f}D",ha="left",va="bottom")
    # 画y=Rx
    ax.plot([x[0],x[-1]],[Rx,Rx],"--")
    ax.text(x[0],Rx,f"{Rx:.2f}D",ha="left",va="bottom")

    ax.set_xlabel("x/mm")
    ax.set_ylabel("power")
    ax.set_title(f"ID={id}, mean add power={power:.2f}D")
    # 画一个矩形框
    ax.add_patch(plt.Rectangle((-radius*mm_per_point,Rx+power-0.5),2*radius*mm_per_point,1,fill=False,linestyle="--"))

    return fig

def parse_number_range(s: str):
    result = []  # 结果列表
    parts = s.split(",")  # 以逗号分割字符串
    for part in parts:
        if '-' in part:  # 如果部分包含短划线，表示这是一个范围
            start, end = map(int, part.split('-'))  # 分割起始和结束值并转换为整数
            result.extend(range(start, end + 1))  # 将范围内的所有数字添加到结果列表中
        else:
            result.append(int(part))  # 如果不包含短划线，直接添加数字
    return result

def measure_list_of_microlens(ring_number_list,
    point_per_mm,sorted_microlens_params,data,Rx):
    report_text=f"|测量直径|平均值|标准差|周边光焦度|\n|---|---|---|---|\n"
    mean_list=[]
    std_list=[]



    diameter_list=[0.7,0.5,0.3,0.1]
    for d in diameter_list:
        power_list=[]
        maxpower_list=[]
        base_power_list=[]
        for i in ring_number_list:
            measure_radius=d/2*point_per_mm
            power=measure_one_microlens_center_area(i,sorted_microlens_params,data,radius=measure_radius)
            
            max_power=measure_one_microlens_max(i,sorted_microlens_params,data,radius=measure_radius)

            base_power=sorted_microlens_params[i]['Rx']

            power_list.append(power)
            maxpower_list.append(max_power)
            
            base_power_list.append(base_power)
            # fig.show()

        mean_power=np.mean(power_list)
        std_power=np.std(power_list)
        mean_base_power=np.mean(base_power_list)
        max_power_mean=np.mean(maxpower_list)
        max_power_std=np.std(maxpower_list)

        mean_list.append(mean_power)
        std_list.append(std_power)
        report_text+=f"|{d}|{mean_power:.3f}|{std_power:.3f}|{mean_base_power:.3f}|\n"
    report_text+=f"|0.0|{max_power_mean:.3f}|{max_power_mean-Rx:.3f}|{max_power_std:.3f}\n"
    return report_text

def measure_microlens_in_diamter(sorted_microlens_params,data
    ,point_per_mm,diameter_list=[0.7,0.5,0.3,0.1]):
    for i in range(len(sorted_microlens_params)):
        for d in diameter_list:
            measure_radius=d/2*point_per_mm
            power=measure_one_microlens_center_area(i,sorted_microlens_params,data,radius=measure_radius)
            max_power=measure_one_microlens_max(i,sorted_microlens_params,data,radius=measure_radius)

            sorted_microlens_params[i][f'power at {d:.1f}']=power
            sorted_microlens_params[i][f'power at {0.0}']=max_power
    return sorted_microlens_params



def align_microlens(sorted_microlens_params,data_origin,point_per_mm):
    x_measurement = np.array([microlens['center'][1] for microlens in sorted_microlens_params]).T/point_per_mm
    y_measurement = np.array([microlens['center'][0] for microlens in sorted_microlens_params]).T/point_per_mm
    p_measurement = np.array([microlens['power at 0.7'] for microlens in sorted_microlens_params]).T
    x_measurement -= np.mean(x_measurement)
    y_measurement -= np.mean(y_measurement)
    aligned_coords=alignment_by_coordinates(
        data_origin['x'],data_origin['y'],
        x_measurement,y_measurement
        )
    aligned_coords=alignment_by_powers(
            data_origin['x'],data_origin['y'], data_origin['p'],
            aligned_coords[:,0], aligned_coords[:,1], p_measurement,
    ) 
    return aligned_coords 

def update_microlens_params_after_align(sorted_microlens_params,final_coords,data_origin):
    id_list = [find_nearest_index(data_origin[['x','y']].to_numpy(), coord) for coord in final_coords]
    ring_list= data_origin['ring'][id_list].to_list()
    for i in range(len(ring_list)):
        sorted_microlens_params[i]['ring']=ring_list[i]
        sorted_microlens_params[i]['id']=id_list[i]
    # 按照id_list的升序排序sorted_microlens_params
    # 排序id_list，并得到索引
    sorted_id_list = sorted(range(len(id_list)), key=lambda k: id_list[k])
    # 按照sorted_id_list的顺序，重新排列sorted_microlens_params
    sorted_microlens_params=[sorted_microlens_params[i] for i in sorted_id_list]
    for i in range(len(sorted_microlens_params)):
        power_measured=sorted_microlens_params[i]['power at 0.7']
        power_origin=data_origin.iloc[sorted_microlens_params[i]['id']]['p']
        power_diff=power_measured-power_origin
        sorted_microlens_params[i]['power difference']=power_diff
    return sorted_microlens_params

def report_align_location(data_origin,data,aligned_coords,sorted_microlens_params,dpi=75):
    # 创建一个Figure对象
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # 绘制第一个子图
    axs[0].scatter(data_origin['y'], data_origin['x'], c='blue', label='Origin Data', alpha=0.7)

    axs[0].scatter(aligned_coords[:,0], -aligned_coords[:,1], c='red', label='Measured Data (After ICP)', alpha=0.7)
    axs[0].axis('equal')

    # axs[0].imshow(data, cmap="gray", interpolation='nearest')
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

def analysis_ring(sorted_microlens_params):
    ring_list = set([microlens['ring'] for microlens in sorted_microlens_params])
    ring_param_list=[]

    for i,ring in enumerate(ring_list):
        ring_params={}
        ring_microlens_params=[microlens for microlens in sorted_microlens_params if microlens['ring']==ring]
        ring_params['ring']=ring+1
        ring_params['lens count']=len(ring_microlens_params)
        ring_params['power at 0.7']=[microlens['power at 0.7'] 
            for microlens in ring_microlens_params]
        ring_params['power at 0.3']=[microlens['power at 0.3'] 
            for microlens in ring_microlens_params]
        ring_params['power at 0.1']=[microlens['power at 0.1']
            for microlens in ring_microlens_params]
        ring_params['power at 0.0']=[microlens['power at 0.0']
            for microlens in ring_microlens_params]
        ring_params['Rx']=[microlens['Rx'] for microlens in ring_microlens_params]
        ring_params['power difference']=[microlens['power difference'] for microlens in ring_microlens_params]

        ring_params['power at 0.7 mean']=np.mean(ring_params['power at 0.7'])
        ring_params['power at 0.3 mean']=np.mean(ring_params['power at 0.3'])
        ring_params['power at 0.1 mean']=np.mean(ring_params['power at 0.1'])
        ring_params['power at 0.0 mean']=np.mean(ring_params['power at 0.0'])
        ring_params['Rx mean']=np.mean(ring_params['Rx'])
        ring_params['power difference abs mean']=np.mean(np.abs(ring_params['power difference']))

        ring_params['power at 0.7 std']=np.std(ring_params['power at 0.7'])
        ring_params['power at 0.3 std']=np.std(ring_params['power at 0.3'])
        ring_params['power at 0.1 std']=np.std(ring_params['power at 0.1'])
        ring_params['power at 0.0 std']=np.std(ring_params['power at 0.0'])
        ring_params['Rx std']=np.std(ring_params['Rx'])
        ring_params['power difference abs std']=np.std(np.abs(ring_params['power difference']))

        ring_params['power at 0.7 range']=np.max(ring_params['power at 0.7'])-np.min(ring_params['power at 0.7'])
        ring_params['power at 0.3 range']=np.max(ring_params['power at 0.3'])-np.min(ring_params['power at 0.3'])
        ring_params['power at 0.1 range']=np.max(ring_params['power at 0.1'])-np.min(ring_params['power at 0.1'])
        ring_params['power at 0.0 range']=np.max(ring_params['power at 0.0'])-np.min(ring_params['power at 0.0'])
        ring_params['Rx range']=np.max(ring_params['Rx'])-np.min(ring_params['Rx'])
        ring_params['power difference abs max']=np.max(np.abs(ring_params['power difference']))
        ring_param_list.append(ring_params)
    return ring_param_list

def generate_ring_report(ring_param_list):
    report_text="" 
    for ring in ring_param_list:
        report_text +=f"Ring {ring['ring']}\t Lens Count: {ring['lens count']}\n"
        report_text +="\tMean\tStd\tRange\n"
        report_text +=f"0.7\t{ring['power at 0.7 mean']:.3f}\t{ring['power at 0.7 std']:.3f}\t{ring['power at 0.7 range']:.3f}\n"
        report_text +=f"0.3\t{ring['power at 0.3 mean']:.3f}\t{ring['power at 0.3 std']:.3f}\t{ring['power at 0.3 range']:.3f}\n"
        report_text +=f"Rx\t{ring['Rx mean']:.3f}\t{ring['Rx std']:.3f}\t{ring['Rx range']:.3f}\n"
        report_text +=f"diff\t{ring['power difference abs mean']:.3f}\t{ring['power difference abs std']:.3f}\t{ring['power difference abs max']:.3f}\n"
        report_text +="\n"
    return(report_text)
