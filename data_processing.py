import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
import cv2
from sklearn.cluster import KMeans
import numpy as np
# from scipy.optimize import minimize
import random
# from collections import Counter
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

# def microlens_centers_radius(regions):
#     microlens = []
#     for region in regions:
#         center, radius = cv2.minEnclosingCircle(np.argwhere(region))
#         microlens.append({"center": center, "radius": radius})
#     return microlens

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
        ring_num=None,
        max_ring = 6,
        threshold=10,
        iterations=1000, 
        radius_threshold=2, 
        max_alpha=1.0,
        plot=False):
    centers = [microlens["center"] for microlens in microlens_params]
    # image_center=find_concentric_circle_center(
    #     centers,iterations=iterations, radius_threshold=radius_threshold, max_alpha=max_alpha,plot=plot)
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