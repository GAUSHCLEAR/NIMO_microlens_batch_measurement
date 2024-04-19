import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

# 生成数据集，用于测试点集配准算法
def generate_dataset(X,Y,
        x0,y0,R,
        error=0.3,drop_rate=0.1
        ):
    df=pd.DataFrame({'x':X,'y':Y})
    df_in_circle=df[(df['x']-x0)**2+(df['y']-y0)**2<R**2].copy()
    df_in_circle.loc[:, 'x'] += np.random.uniform(-error, error, len(df_in_circle))
    df_in_circle.loc[:, 'y'] += np.random.uniform(-error, error, len(df_in_circle))
    df_in_circle=df_in_circle.sample(frac=1-drop_rate)
    return df_in_circle

# 平移和旋转点集，用于测试点集配准算法
def transform_dataset(x0,y0,theta,df_in_circle):
    # 平移df_in_circle到原点
    df_in_circle = df_in_circle.copy()
    df_in_circle.loc[:, 'x'] -= x0
    df_in_circle.loc[:, 'y'] -= y0
    # 绕圆心旋转df_in_circle中的点theta角
    x_new=df_in_circle['x']*np.cos(theta)-df_in_circle['y']*np.sin(theta)
    y_new=df_in_circle['x']*np.sin(theta)+df_in_circle['y']*np.cos(theta)
    df_in_circle['x']=x_new
    df_in_circle['y']=y_new
    return df_in_circle

# 为ICP算法找到可能的初始配准点
def find_best_circle_match_with_rotation_2d(
        x_origin, y_origin, x_measure, y_measure, 
        circle_diameter=17, angle_steps=12):
    # Radius from diameter
    radius = circle_diameter / 2

    # Best match initial values
    best_match_score = np.inf
    best_center = None
    best_rotation = None

    # Generate a grid of potential circle centers within the range of data_origin points
    x_min, y_min = np.min(x_origin) - 2*radius, np.min(y_origin) - 2*radius
    x_max, y_max = np.max(x_origin) + 2*radius, np.max(y_origin) + 2*radius
    print(x_min, y_min, x_max, y_max)
    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 75), np.linspace(y_min, y_max, 75))
    grid_centers = np.vstack([grid_x.ravel(), grid_y.ravel()]).T

    # Array of angles to try for rotation
    angles = np.linspace(0, 2 * np.pi, angle_steps)

    # Iterate over potential centers and angles
    for center in grid_centers:
        for angle in angles:
            # Calculate rotation matrix for 2D
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            rotated_measure_coords = np.dot(np.vstack([x_measure, y_measure]).T, rotation_matrix.T) + center

            # Points within the circle at this center
            distances = np.sqrt((x_origin - center[0])**2 + (y_origin - center[1])**2)
            within_circle = np.vstack([x_origin[distances <= radius], y_origin[distances <= radius]]).T

            # Continue only if there are enough points to match
            if len(within_circle) >= len(x_measure):
                # Calculate distance matrix between rotated measured data and points within the circle
                distance_matrix = cdist(rotated_measure_coords, within_circle)

                # Find the minimum matching distance for each measured point
                min_distances = np.min(distance_matrix, axis=1)
                match_score = np.sum(min_distances)

                # Update best match if this one is better
                if match_score < best_match_score:
                    best_match_score = match_score
                    best_center = center
                    best_rotation = angle
    best_center = best_center if best_center is not None else np.array([0, 0])
    best_rotation = best_rotation if best_rotation is not None else 0
    return best_center, best_rotation, best_match_score

def improved_weighted_icp(
        x_origin,y_origin,
        x_measure,y_measure,
        initial_center, initial_angle):
    origin_coords = np.vstack([x_origin,y_origin]).T
    measure_coords = np.vstack([x_measure,y_measure]).T
    # Apply initial transformation based on template matching results
    initial_translation = initial_center
    initial_rotation_matrix = np.array([
        [np.cos(initial_angle), -np.sin(initial_angle)],
        [np.sin(initial_angle), np.cos(initial_angle)]
    ])
    measure_coords = np.dot(measure_coords, initial_rotation_matrix.T) + initial_translation

    # Function to apply rotation and translation
    def transform_points(coords, angle, translation):
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])
        return coords @ rotation_matrix + translation

    # Calculate the total weighted distance between point sets
    def weighted_distance(translation_angle):
        angle, tx, ty = translation_angle
        transformed_coords = transform_points(measure_coords, angle, [tx, ty])
        tree = KDTree(origin_coords)
        distances, indices = tree.query(transformed_coords)
        weighted_dist = np.sum(distances ** 2)
        return weighted_dist

    # Optimization to minimize the weighted distance
    result = minimize(weighted_distance, 
            [0.0,0.0,0.0], method='L-BFGS-B')

    # Final transformation parameters
    final_angle, final_tx, final_ty = result.x
    final_transformation = transform_points(measure_coords, final_angle, [final_tx, final_ty])

    # Check for convergence
    # if result.success :
    #     print("Convergence achieved.")
    
    return final_transformation, (final_angle + initial_angle, final_tx + initial_translation[0], final_ty + initial_translation[1])

def alignment_by_coordinates(x_origin, y_origin, x_measure, y_measure):
    # # 平移到原点
    # x_measure -= np.mean(x_measure)
    # x_measure -= np.mean(x_measure)

    best_center, best_rotation, _ = find_best_circle_match_with_rotation_2d(
        x_origin,y_origin,x_measure,y_measure) 
    
    print(f"Initial center: {best_center}, initial rotation: {best_rotation}")
    final_coords, transformation_params = improved_weighted_icp(
        x_origin, y_origin,x_measure, y_measure, 
        best_center, best_rotation)
    return final_coords

def find_symmetric_rotations(
        # data_origin, final_coords_df, 
        x_origin,y_origin,
        x_measure,y_measure,
        steps=360):
    origin_coords = origin_coords = np.vstack([x_origin,y_origin]).T
    measure_coords = np.vstack([x_measure,y_measure]).T
    # measure_coords = final_coords_df[['x', 'y']].to_numpy()
    
    angle_step = 2 * np.pi / steps
    angles = np.arange(0, 2 * np.pi, angle_step)
    distances = []

    for angle in angles:
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotated_x = measure_coords[:, 0] * cos_angle - measure_coords[:, 1] * sin_angle
        rotated_y = measure_coords[:, 0] * sin_angle + measure_coords[:, 1] * cos_angle
        rotated_coords = np.column_stack((rotated_x, rotated_y))

        tree = KDTree(origin_coords)
        dist, _ = tree.query(rotated_coords)
        total_distance = np.sum(dist)
        distances.append(total_distance)

    # Convert distances to a numpy array for easier analysis
    distances = np.array(distances)

    # Find local minima: a point is a local minimum if it's less than both its neighbors
    local_minima_indices = np.where((distances < np.roll(distances, 1)) & (distances < np.roll(distances, -1)))[0]
    local_minima_angles = angles[local_minima_indices]

    return local_minima_angles, distances[local_minima_indices]

def find_best_rotation_angles(
        x_origin,y_origin,
        x_measure,y_measure,
        n_clusters=2):
    # 运行查找对称旋转函数
    local_minima_angles, local_minima_distances = find_symmetric_rotations(
    x_origin,y_origin,x_measure,y_measure)

    # 聚类分析
    data = np.vstack((local_minima_angles, local_minima_distances)).T
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 确定最佳匹配的群组（距离最小的群组）
    best_cluster_index = np.argmin(centers[:, 1])
    best_angles = local_minima_angles[labels == best_cluster_index]

    return best_angles

def evaluate_p_mismatch(
        x_origin,y_origin,p_origin,
        x_measure,y_measure,p_measure,
        best_angles
        ):

    origin_coords = np.vstack([x_origin,y_origin]).T
    origin_p = np.array(p_origin)
    measure_coords = np.vstack([x_measure,y_measure]).T
    measure_p = np.array(p_measure)

    min_p_error = float('inf')
    best_angle_for_p = None

    for angle in best_angles:
        cos_angle = np.cos(angle)
        sin_angle = np.sin(angle)
        rotated_x = measure_coords[:, 0] * cos_angle - measure_coords[:, 1] * sin_angle
        rotated_y = measure_coords[:, 0] * sin_angle + measure_coords[:, 1] * cos_angle
        rotated_coords = np.column_stack((rotated_x, rotated_y))

        tree = KDTree(origin_coords)
        _, indices = tree.query(rotated_coords)
        
        # Calculate p value mismatches
        p_error = np.sum(np.abs(measure_p - origin_p[indices]))
        
        if p_error < min_p_error:
            min_p_error = p_error
            best_angle_for_p = angle

    return best_angle_for_p, min_p_error

def apply_best_rotation(
        x,y,
        best_angle):
    measure_coords = np.column_stack([x,y])

    # 计算旋转矩阵
    cos_angle = np.cos(best_angle)
    sin_angle = np.sin(best_angle)

    # 应用旋转
    rotated_x = measure_coords[:, 0] * cos_angle - measure_coords[:, 1] * sin_angle
    rotated_y = measure_coords[:, 0] * sin_angle + measure_coords[:, 1] * cos_angle

    return rotated_x, rotated_y

def alignment_by_powers(
        x_origin, y_origin, p_origin,
        x_measure, y_measure, p_measure,
):

    best_angles=find_best_rotation_angles(
        x_origin,y_origin,
        x_measure, y_measure,
    )
    best_angle_for_p, min_p_error = evaluate_p_mismatch(
        x_origin,y_origin,p_origin,
        x_measure, y_measure,p_measure,
        best_angles)
    # 应用最优旋转角度
    rotated_x, rotated_y = apply_best_rotation(
        x_measure, y_measure,
        best_angle_for_p)
    return np.array([rotated_x, rotated_y]).T

def find_nearest_index(coordinate_origin, point):
    distances = np.sum((coordinate_origin - point)**2, axis=1)
    return np.argmin(distances)