import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2
from sklearn.cluster import KMeans
import numpy as np


def read_data(filename, blank_value=-15):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    data[:, 0] = np.NaN
    data[0, :] = np.NaN
    data[data < blank_value] = np.NaN
    return data
def detect_edge(data,threshold=0.8):
    gradient_x, gradient_y = np.gradient(data)
    gradient_magnitude=np.sqrt(gradient_x**2+gradient_y**2)
    binary_image = np.where(gradient_magnitude >= threshold, 1, 0) 
    return binary_image

def label_microlens(binary_image, min_area=10*10, max_area=30*30):
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

def microlens_centers_radius(regions):
    img_size=regions[0].shape
    microlens = []
    for region in regions:
        center, radius = cv2.minEnclosingCircle(np.argwhere(region))
        microlens.append({"center": center, "radius": radius})
    return microlens

def cluster_rings(microlens_params, microlenses,rings_number = 4):
    img_size=microlenses[0].shape
    centers = [microlens["center"] for microlens in microlens_params]
    image_center = (img_size[0] // 2, img_size[1] // 2)
    distances = [np.linalg.norm(np.array(center) - np.array(image_center)) for center in centers]
    sorted_microlens_params = [microlens_params[i] for i in np.argsort(distances)]
    sorted_distances = [distances[i] for i in np.argsort(distances)]
    kmeans = KMeans(n_clusters=rings_number)
    kmeans.fit(np.array(sorted_distances).reshape(-1, 1))
    cluster_labels = kmeans.labels_
    for i, microlens in enumerate(sorted_microlens_params):
        microlens["ring"] = cluster_labels[i]
    return sorted_microlens_params

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

def measure_one_microlens(microlens_params, id, data,N_line=6,N_point=101):
    center = microlens_params[id]["center"]
    radius = microlens_params[id]["radius"]*2
    line_datas=[]
    for theta in np.arange(0,180,20):
        line_data=measure_line_data(center,radius,theta, data)
        x_interp, y_interp=interp_data(line_data,radius,N=101)
        line_datas.append(y_interp)
    line_datas=np.array(line_datas)
    line_datas_mean=np.mean(line_datas,axis=0)
    line_datas_std=np.std(line_datas,axis=0)
    return x_interp,line_datas_mean,line_datas_std

def measure_one_ring(sample_microlens_id,microlens_params,data,N_line=6,N_point=101):
    ring_id=microlens_params[sample_microlens_id]['ring']
    microlens_ids=[i for i in range(len(microlens_params)) if microlens_params[i]["ring"]==ring_id]

    radius_in_the_ring=np.array([microlens_params[i]["radius"] for i in microlens_ids])
    max_radius=np.max(radius_in_the_ring)
    x_interp = np.linspace(-2*max_radius,2*max_radius, N_point)
    line_datas=[]
    for microlens_id in microlens_ids:
        x,line_data_mean,_=measure_one_microlens(microlens_params,microlens_id,data,N_line=N_line,N_point=N_point)
        y_interp = np.interp(x_interp, x, line_data_mean)
        line_datas.append(y_interp)
    y_mean=np.mean(line_datas,axis=0)
    y_std=np.std(line_datas,axis=0)
    return x_interp,y_mean,y_std