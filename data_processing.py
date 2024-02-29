import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2
from sklearn.cluster import KMeans
import numpy as np
from scipy.optimize import minimize
import random
# from collections import Counter
from typing import List, Dict
from collections import Counter

import matplotlib.pyplot as plt
def read_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    data[:, 0] = np.NaN
    data[0, :] = np.NaN
    blank_value=np.nanmean(data[:10, :10])+0.1
    data[data <= blank_value] = np.NaN
    return data
def detect_edge(data,threshold=0.8):
    gradient_x, gradient_y = np.gradient(data)
    gradient_magnitude=np.sqrt(gradient_x**2+gradient_y**2)
    binary_image = np.where(gradient_magnitude >= threshold, 1, 0) 
    return binary_image

def label_microlens(binary_image, 
                    min_area=10*10, max_area=30*30):
    labeled_image, num_labels = label(1 - binary_image)
    filtered_regions = []
    for label_value in range(1, num_labels + 1):
        region = labeled_image == label_value
        if min_area < region.sum() < max_area:
            filtered_regions.append(region)
    circle_image = cv2.UMat(np.zeros_like(binary_image, dtype=np.uint8))
    blurred_image = cv2.GaussianBlur(binary_image.astype(np.uint8)*255, (3, 3), 0)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=10,param1=50, param2=30, minRadius=5, maxRadius=15)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center_x, center_y, radius = circle
            cv2.circle(circle_image, (center_x, center_y), radius, 1, thickness=-1)
    circular_regions = [region for region in filtered_regions if circle_image.get()[region].sum() > 0]
    circular_image = np.zeros_like(binary_image)
    for region in circular_regions:
        circular_image[region] = 1
    return circular_regions, circular_image

def label_microlens_test(binary_image, 
                    minRadius, 
                    maxRadius):
    labeled_image, num_labels = label(1 - binary_image)
    filtered_regions = []
    # no idea why
    min_area = np.pi * minRadius**2/4 
    max_area = np.pi * maxRadius**2/4
    for label_value in range(1, num_labels + 1):
        region = labeled_image == label_value
        if min_area < region.sum() < max_area:
            filtered_regions.append(region)
    circle_image = cv2.UMat(np.zeros_like(binary_image, dtype=np.uint8))
    blurred_image = cv2.GaussianBlur(binary_image.astype(np.uint8)*255, (3, 3), 0)
    circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1, minDist=10,param1=50, param2=30, minRadius=int(minRadius), maxRadius=int(maxRadius))
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center_x, center_y, radius = circle
            cv2.circle(circle_image, (center_x, center_y), radius, 1, thickness=-1)
    circular_regions = [region for region in filtered_regions if circle_image.get()[region].sum() > 0]
    circular_image = np.zeros_like(binary_image)
    for region in circular_regions:
        circular_image[region] = 1
    return circular_regions, circular_image

def microlens_centers_radius(regions,
        min_radius=5,
        max_radius=15,):
    microlens = []
    for region in regions:
        center, radius = cv2.minEnclosingCircle(np.argwhere(region))
        if min_radius < radius < max_radius:
            microlens.append({"center": center, "radius": radius})
    return microlens

def rename_labels(sorted_microlens_params):
    # Find all unique ring values
    unique_rings =list(dict.fromkeys([param["ring"] for param in sorted_microlens_params]))

    # Create a mapping from old ring values to new ones
    ring_mapping = {old: new for new, old in enumerate(unique_rings)}

    # Update the ring values
    for param in sorted_microlens_params:
        param["ring"] = ring_mapping[param["ring"]]

    return sorted_microlens_params


def distance(p1, p2):
    """Calculate the Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_circle_center_from_three_points(p1, p2, p3):
    """Calculate the center of a circle passing through three points."""
    A = np.array([
        [2 * (p1[0] - p3[0]), 2 * (p1[1] - p3[1])],
        [2 * (p2[0] - p3[0]), 2 * (p2[1] - p3[1])]
    ])
    b = np.array([
        p1[0]**2 - p3[0]**2 + p1[1]**2 - p3[1]**2,
        p2[0]**2 - p3[0]**2 + p2[1]**2 - p3[1]**2
    ])
    try:
        center = np.linalg.solve(A, b)
        return center
    except np.linalg.LinAlgError:
        return None

def find_concentric_circle_center(circle_centers, iterations=1000, radius_threshold=2, max_alpha=1.0,plot=False):
    """Find the possible center of concentric circles from a set of points."""
    def update_center_with_weight(centers, new_center):
        """Update the list of centers with the new center, considering the radius threshold."""
        for i, (center, weight) in enumerate(centers):
            if distance(center, new_center) <= radius_threshold:
                avg_center = weighted_average_point(center, weight, new_center, 1)
                centers[i] = (avg_center, weight + 1)
                return centers
        centers.append((new_center, 1))
        return centers

    def weighted_average_point(p1, weight1, p2, weight2):
        """Calculate the weighted average of two points."""
        x = (p1[0] * weight1 + p2[0] * weight2) / (weight1 + weight2)
        y = (p1[1] * weight1 + p2[1] * weight2) / (weight1 + weight2)
        return (x, y)

    weighted_centers = []
    for _ in range(iterations):
        selected_points = random.sample(circle_centers, 3)
        center = calculate_circle_center_from_three_points(*selected_points)
        if center is not None:
            weighted_centers = update_center_with_weight(weighted_centers, center)

    if plot:

        # Plotting the results
        plt.figure(figsize=(5, 5))
        circle_centers_array = np.array(circle_centers)
        plt.scatter(circle_centers_array[:, 0], circle_centers_array[:, 1], c='blue', label='Circle Centers')
        for center, weight in weighted_centers:
            alpha = min(0.1 + 0.1 * weight, max_alpha)
            circle = plt.Circle(center, 1.5, color='red', alpha=alpha, fill=True)
            plt.gca().add_artist(circle)
        plt.axis('equal')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.title('Possible Centers of Concentric Circles')
        plt.legend()
        plt.show()
    top_weighted_centers = sorted(weighted_centers, key=lambda x: x[1], reverse=True)
    return top_weighted_centers[0][0]

def cluster_rings(
        microlens_params,
        image_center=None,
        ring_num=None,
        max_ring = 6,
        threshold=10
        ):
    centers = [microlens["center"] for microlens in microlens_params]
    if image_center is None:
        image_center=[600,250]
    distances = [np.linalg.norm(np.array(center) - np.array(image_center)) for center in centers]
    sorted_microlens_params = [microlens_params[i] for i in np.argsort(distances)]
    sorted_distances = [distances[i] for i in np.argsort(distances)]
    
    if ring_num is None:
        for k in range(1,max_ring+1):
            kmeans = KMeans(n_clusters=k,n_init=10)
            kmeans.fit(np.array(sorted_distances).reshape(-1, 1))
            cluster_labels = kmeans.labels_
            inertia_=kmeans.inertia_
            print("k=",k,"inertia_=",inertia_)
            if inertia_ < threshold:
                # print("rings=",k)
                break 
    else:
        kmeans = KMeans(n_clusters=ring_num,n_init=10)
        kmeans.fit(np.array(sorted_distances).reshape(-1, 1))
        cluster_labels = kmeans.labels_
    
    for i, microlens in enumerate(sorted_microlens_params):
        microlens["ring"] = cluster_labels[i]

    return rename_labels(sorted_microlens_params)

## 试图反复两次进行聚类，第一次聚类后，将聚类中心作为新的中心，再次聚类
## 但是效果不好，还是采用上面的方法
# def estimate_circle_center(points):
#     # 初始猜测
#     x_m = np.mean(points, axis=0)

#     def calc_R(xc, yc):
#         """计算半径"""
#         return np.sqrt((points[:,0] - xc)**2 + (points[:,1] - yc)**2)

#     def f_2(c):
#         """最小化的函数"""
#         Ri = calc_R(*c)
#         return np.sum((Ri - Ri.mean())**2)

#     center_estimate = x_m
#     result = minimize(f_2, center_estimate, method='BFGS')
#     center = result.x

#     # 确保返回的是一个长度为2的数组或元组
#     if len(center) == 2:
#         return center
#     else:
#         # 如果返回的不是两个值，则打印错误信息并返回None
#         print("Error in estimation. Result:", center)
#         return None

# def cluster_and_get_labels(microlens_params, image_center, ring_num, max_ring, threshold):
#     centers = [microlens["center"] for microlens in microlens_params]
#     distances = [np.linalg.norm(np.array(center) - np.array(image_center)) for center in centers]
#     sorted_indices = np.argsort(distances)
#     sorted_distances = [distances[i] for i in sorted_indices]
#     sorted_microlens_params = [microlens_params[i] for i in sorted_indices]

#     if ring_num is None:
#         for k in range(1, max_ring + 1):
#             kmeans = KMeans(n_clusters=k, n_init=10)
#             kmeans.fit(np.array(sorted_distances).reshape(-1, 1))
#             inertia_ = kmeans.inertia_
#             print("k=", k, "inertia_=", inertia_)
#             if inertia_ < threshold:
#                 break
#     else:
#         k = ring_num
#         kmeans = KMeans(n_clusters=k, n_init=10)
#         kmeans.fit(np.array(sorted_distances).reshape(-1, 1))
    
#     return kmeans.labels_, sorted_indices

# def cluster_rings(microlens_params, image_center=None, ring_num=None, max_ring=6, threshold=10):
#     if image_center is None:
#         image_center = [600, 250]

#     # 初次聚类
#     cluster_labels, sorted_indices = cluster_and_get_labels(microlens_params, image_center, ring_num, max_ring, threshold)

#     # 重新计算每个圆环的圆心
#     new_centers = []
#     for k in set(cluster_labels):
#         cluster_points = np.array([microlens_params[i]["center"] for i in sorted_indices if cluster_labels[i] == k])
#         if len(cluster_points) > 2:  # 需要至少3个点来确定一个圆
#             new_center = estimate_circle_center(cluster_points)
#             new_centers.append(new_center)

#     # 检查 new_centers 中是否有异常值
#     new_centers = [nc for nc in new_centers if np.all(np.isfinite(nc))]

#     # 如果找到新的圆心，则计算平均位置；否则使用原始 image_center
#     if new_centers:
#         new_image_center = np.mean(new_centers, axis=0)
#     else:
#         new_image_center = image_center

#     # 使用新的 image_center 重新聚类
#     cluster_labels, _ = cluster_and_get_labels(microlens_params, new_image_center, ring_num, max_ring, threshold)

#     # 更新微透镜参数并返回
#     for i, microlens in enumerate(microlens_params):
#         microlens["ring"] = cluster_labels[i]

#     print("new_image_center", new_image_center)

#     return microlens_params



def measure_one_microlens_center_area(id, microlens_params, data, radius=10):
    center = microlens_params[id]["center"]
    x,y = np.ogrid[-center[0]:data.shape[0]-center[0], -center[1]:data.shape[1]-center[1]]
    mask = x*x + y*y <= radius*radius
    sub_data = data[mask]
    power = np.mean(sub_data)
    Rx=microlens_params[id]['Rx']
    add_power=power-Rx
    return add_power 


def measure_one_microlens_max(id, microlens_params, data, radius=10):
    center = microlens_params[id]["center"]
    x,y = np.ogrid[-center[0]:data.shape[0]-center[0], -center[1]:data.shape[1]-center[1]]
    mask = x*x + y*y <= radius*radius
    sub_data = data[mask]
    power= np.max(sub_data)
    Rx=microlens_params[id]['Rx']
    add_power=power-Rx
    return add_power






def measure_line_data(center,radius,theta, data):
    start_x = int(center[0] - radius * np.cos(np.radians(theta)))
    start_y = int(center[1] - radius * np.sin(np.radians(theta)))
    end_x = int(center[0] + radius * np.cos(np.radians(theta)))
    end_y = int(center[1] + radius * np.sin(np.radians(theta)))
    line_data = []
    for i in range(int(radius * 2)):
        x = int(start_x + (end_x - start_x) * i / (radius * 2))
        y = int(start_y + (end_y - start_y) * i / (radius * 2))
        line_data.append(data[x, y])
    return line_data

def interp_data(line_data,radius,N=101):
    x_interp = np.linspace(0, 2*radius, N)
    y_interp = np.interp(x_interp, np.arange(len(line_data)), line_data)
    return x_interp-radius,y_interp




def measure_one_microlens(id, microlens_params, data,N_line=6,N_point=101):
    center = microlens_params[id]["center"]
    radius = microlens_params[id]["radius"]*2
    line_datas=[]
    for theta in np.arange(0,180,N_line):
        line_data=measure_line_data(center,radius,theta, data)
        x_interp, y_interp=interp_data(line_data,radius,N=N_point)
        line_datas.append(y_interp)
    line_datas=np.array(line_datas)
    line_datas_mean=np.mean(line_datas,axis=0)
    line_datas_std=np.std(line_datas,axis=0)
    return x_interp,line_datas_mean,line_datas_std



def measure_one_ring(ring_id,microlens_params,data,N_line=20,N_point=101):
    # ring_id=microlens_params[sample_microlens_id]['ring']
    microlens_ids=[i for i in range(len(microlens_params)) if microlens_params[i]["ring"]==ring_id]

    radius_in_the_ring=np.array([microlens_params[i]["radius"] for i in microlens_ids])
    max_radius=np.max(radius_in_the_ring)
    x_interp = np.linspace(-2*max_radius,2*max_radius, N_point)
    line_datas=[]
    for microlens_id in microlens_ids:
        x,line_data_mean,_=measure_one_microlens(microlens_id,microlens_params,data,N_line=N_line,N_point=N_point)
        y_interp = np.interp(x_interp, x, line_data_mean)
        line_datas.append(y_interp)
    y_mean=np.mean(line_datas,axis=0)
    y_std=np.std(line_datas,axis=0)
    return x_interp,y_mean,y_std

def power_check(power,power_color_dict,threshold=0.5):
    # 检查power与power_color_dict中的key的差值，
    # 如果差值的绝对值<阈值，就返回power_color_dict中的key, value
    # 否则返回最接近的key，并返回"warning"
    for key in power_color_dict.keys():
        if abs(power-key)<threshold:
            return key,power_color_dict[key]
    return min(power_color_dict.keys(), key=lambda x:abs(x-power)), "warning"

def check_all_microlens(sorted_microlens_params,data, power_color_dict,radius=10,threshold=0.5):
    # 检查所有的microlens的power是否在power_color_dict中
    # 如果不在，就返回warning
    # 如果在，就返回power_color_dict中的key, value
    microlens_params_list=sorted_microlens_params.copy()
    for id, microlens_params in enumerate(microlens_params_list):
        power=measure_one_microlens_center_area(id,sorted_microlens_params,data,radius=radius)        
        round_power,power_color=power_check(power,power_color_dict,threshold=threshold)
        microlens_params['id']=id
        microlens_params["measured_power"]=power
        microlens_params["round_power"]=round_power
        microlens_params["color"]=power_color
    return microlens_params_list

def get_most_common_round_power(checked_microlens: List[Dict]) -> Dict[int, float]:
    ring_round_powers = {}
    for microlens in checked_microlens:
        ring = microlens['ring']
        round_power = microlens['round_power']
        if ring not in ring_round_powers:
            ring_round_powers[ring] = []
        ring_round_powers[ring].append(round_power)

    # Find the most common round power in each ring
    most_common_round_power = {}
    for ring, powers in ring_round_powers.items():
        most_common_power = Counter(powers).most_common(1)[0][0]
        most_common_round_power[ring] = most_common_power

    return most_common_round_power

def update_microlens_with_common_power(checked_microlens: List[Dict]) -> List[Dict]:
    """
    Function to update the checked_microlens list with the most common round_power for each ring.
    It also changes the color to 'warning' if the microlens' round_power is different from the most common round_power.

    :param checked_microlens: List of dictionaries, each representing a microlens.
    :return: The updated list of dictionaries with 'most_common_round_power' added and color updated if necessary.
    """
    # First, get the most common round power for each ring
    most_common_round_power = get_most_common_round_power(checked_microlens)

    # Update checked_microlens with the most common round power and change color if necessary
    for microlens in checked_microlens:
        ring = microlens['ring']
        microlens['most_common_round_power'] = most_common_round_power[ring]
        if microlens['round_power'] != most_common_round_power[ring]:
            microlens['color'] = 'warning'

    return checked_microlens

def calculate_each_lens_Rx(data,sorted_microlens_params):
    # 计算每个微透镜周边的基础镜片的屈光度
    # 这样可以用来处理基础镜片为柱镜的情况
    background=data.copy()
    for microlens in sorted_microlens_params:
        center=microlens['center']
        radius=microlens['radius']*1.5
        # 在background中，把center为中心,radius为半径内的数据替换成np.nan
        
        # 创建一个表示半径的二维数组
        y, x = np.ogrid[-center[0]:background.shape[0]-center[0], -center[1]:background.shape[1]-center[1]]
        mask = x*x + y*y <= radius*radius

        # 使用这个数组来更新background
        background[mask] = np.nan
    for microlens in sorted_microlens_params:
        center = microlens['center']
        radius = microlens['radius'] * 3

        # 创建一个表示半径的二维数组
        y, x = np.ogrid[-center[0]:background.shape[0]-center[0], -center[1]:background.shape[1]-center[1]]
        mask = x*x + y*y <= radius*radius

        # 创建一个新的掩码，表示background中哪些值不是np.nan
        not_nan_mask = ~np.isnan(background)

        # 将新的掩码和原来的mask进行逻辑与操作
        final_mask = mask & not_nan_mask

        # 使用这个结果来计算中位数，用中位数比较稳定
        if np.any(final_mask):
            microlens['Rx'] = np.nanmedian(background[final_mask])
        else:
            microlens['Rx'] = np.nan
    return sorted_microlens_params
