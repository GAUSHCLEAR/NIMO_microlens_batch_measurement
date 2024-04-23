import numpy as np 
import matplotlib.pyplot as plt

def circle_intersection_area(r, R, x):
    # 将单个数字输入转换为 NumPy 数组
    if isinstance(r, (int, float)):
        r = np.array([r])
        R = np.array([R])
        x = np.array([x])

    # 计算两个圆是否相交
    non_intersecting = (x >= r + R) | (x <= np.abs(R - r))

    # 计算一个圆是否在另一个圆内部
    one_inside_another = x <= np.abs(R - r)

    # 初始化相交面积数组
    areas = np.zeros_like(r, dtype=float)

    # 对于完全在另一个圆内的情况，返回较小圆的面积
    areas[one_inside_another] = np.pi * np.minimum(r[one_inside_another], R[one_inside_another])**2

    # 对于相交的情况，计算相交面积
    intersecting = ~non_intersecting
    part1 = r[intersecting]**2 * np.arccos((x[intersecting]**2 + r[intersecting]**2 - R[intersecting]**2) / (2 * x[intersecting] * r[intersecting]))
    part2 = R[intersecting]**2 * np.arccos((x[intersecting]**2 + R[intersecting]**2 - r[intersecting]**2) / (2 * x[intersecting] * R[intersecting]))
    part3 = 0.5 * np.sqrt((-x[intersecting] + r[intersecting] + R[intersecting]) * (x[intersecting] + r[intersecting] - R[intersecting]) * (x[intersecting] - r[intersecting] + R[intersecting]) * (x[intersecting] + r[intersecting] + R[intersecting]))

    areas[intersecting] = part1 + part2 - part3

    # 如果原始输入是单个数字，则返回单个数字结果
    if areas.size == 1:
        return areas[0]
    return areas

# 计算微透镜密度
# 给定点(x,y)，计算从该点出发，到所有微透镜圆心的距离
# 给定点(x,y)，有半径为R=4的圆，计算该圆与所有微透镜圆心的相交面积s(x,y)
def calculate_density(x,y, 
                centers, lensletDiameter, R=2):
    # 计算距离
    distance = np.sqrt((centers[:,0] - x)**2 + (centers[:,1] - y)**2)
    # 计算相交面积
    r= lensletDiameter/2 * np.ones_like(distance)
    Radius= R * np.ones_like(distance)
    areas = circle_intersection_area(r, Radius, distance)
    # 计算微透镜密度
    density = np.sum(areas) / np.pi / R**2
    return density

def pattern_analysis(x_points,y_points,pupil_diameter,lens_Diameter,lensletDiameter):
    x = np.linspace(-lens_Diameter/2, lens_Diameter/2, 201)
    y = np.linspace(-lens_Diameter/2, lens_Diameter/2, 201)
    X, Y = np.meshgrid(x, y)
    centers=np.array([x_points,y_points]).T
    # 计算密度
    density = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            density[i,j] = calculate_density(X[i,j], Y[i,j], centers, lensletDiameter, R=pupil_diameter/2)
    return density

def draw_pattern_analysis(density,filename):
    # 分别画出密度图和密度直方图
    plt.figure()
    fig,ax=plt.subplots(2,1,figsize=(6,12))
    ax[0].imshow(density, cmap='RdYlGn_r', vmin=0, vmax=1)
    # ax[0].colorbar()
    # ax1.colorbar()
    # 横纵坐标,-lensletZoneDiameter/2, lensletZoneDiameter/2
    ticks_N=5
    # ax[0].set_xticks(np.linspace(-lens_Diameter/2,lens_Diameter/2, 200, ticks_N))
    # ax[0].set_yticks(np.linspace(-lens_Diameter/2,lens_Diameter, 200, ticks_N))
    # 设定x,y轴的范围
    # ax[0].set_xlim(-lens_Diameter/2,lens_Diameter/2)
    # ax[0].set_ylim(-lens_Diameter/2,lens_Diameter/2)
    # 无坐标
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    # 保存图片
    # plt.savefig('density.png', dpi=300, bbox_inches='tight')
    density_without_0 = density[density >= 0.01]
    # 密度直方图
    # 横坐标[0, 0.7]
    ax[1].hist(density_without_0.flatten(), bins=30, density=True, alpha=0.5,
            histtype='stepfilled', color='steelblue',
            edgecolor='none')
    ax[1].set_xlim(0, 1)
    ax[1].set_xticks(np.arange(0, 1.1, 0.1))
    ax[1].set_ylim(0, 15)

    # plt.savefig('density_hist.png', dpi=300, bbox_inches='tight')
    fig.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    fig.show()