import os
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.spatial import ConvexHull


# 加载STL模型
def load_stl(file_path):
    return mesh.Mesh.from_file(file_path)

# 保存STL模型
def save_stl(vertices, file_path):
    output_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    output_mesh.vectors = vertices
    output_mesh.save(file_path)

# 计算牙齿的惯性矩和纵向长轴
def compute_long_axis(model):
    vertices = model.vectors.reshape(-1, 3)
    centroid = np.mean(vertices, axis=0)  # 计算质心
    inertia_tensor = np.zeros((3, 3))

    # 计算惯性矩
    for vertex in vertices:
        relative_position = vertex - centroid
        inertia_tensor[0, 0] += (relative_position[1]**2 + relative_position[2]**2)
        inertia_tensor[1, 1] += (relative_position[0]**2 + relative_position[2]**2)
        inertia_tensor[2, 2] += (relative_position[0]**2 + relative_position[1]**2)
        inertia_tensor[0, 1] -= relative_position[0] * relative_position[1]
        inertia_tensor[1, 0] = inertia_tensor[0, 1]
        inertia_tensor[0, 2] -= relative_position[0] * relative_position[2]
        inertia_tensor[2, 0] = inertia_tensor[0, 2]
        inertia_tensor[1, 2] -= relative_position[1] * relative_position[2]
        inertia_tensor[2, 1] = inertia_tensor[1, 2]

    # 特征向量提取
    eigvals, eigvecs = np.linalg.eig(inertia_tensor)
    long_axis = eigvecs[:, np.argmax(eigvals)]  # 最大特征值对应的特征向量

    return centroid, long_axis

# 计算模型与平面相交的截面
def get_intersection_section(model, plane_point, plane_normal):
    section_points = []
    for triangle in model.vectors:
        intersection = []
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i + 1) % 3]
            t = np.dot(plane_point - p1, plane_normal) / np.dot(p2 - p1, plane_normal)
            if 0 <= t <= 1:
                intersection.append(p1 + t * (p2 - p1))
        if len(intersection) == 2:  # 每个三角形的交点最多为两个
            section_points.extend(intersection)

    return np.array(section_points)

# 计算截面面积（凸包体积）
def compute_section_area(section_points):
    if len(section_points) < 3:
        return 0
    hull = ConvexHull(section_points, qhull_options='QJ')  # 使用扰动解决共面问题
    return hull.volume

# 找最大截面平面
def find_max_section(model, center, long_axis):
    z_coords = model.vectors[:, :, 2].flatten()
    z_min, z_max = np.min(z_coords), np.max(z_coords)

    max_area = 0
    max_section_points = None
    max_plane_point = None

    for z in np.linspace(z_min, z_max, 100):
        plane_point = center + z * long_axis
        section_points = get_intersection_section(model, plane_point, long_axis)
        section_area = compute_section_area(section_points)

        if section_area > max_area:
            max_area = section_area
            max_section_points = section_points
            max_plane_point = plane_point

    return max_section_points, max_plane_point

# 分割模型
def split_model(model, plane_point, plane_normal):
    above_vertices = []
    below_vertices = []

    for triangle in model.vectors:
        above = []
        below = []

        for vertex in triangle:
            if np.dot(vertex - plane_point, plane_normal) > 0:
                above.append(vertex)
            else:
                below.append(vertex)

        if len(above) == 3:
            above_vertices.append(triangle)
        elif len(below) == 3:
            below_vertices.append(triangle)

    return np.array(above_vertices), np.array(below_vertices)

# 计算投影并生成渐变色图
# 计算投影并生成颜色区间渐变图
def plot_projection_on_section(vertices, section_point, long_axis, output_path, label):
    # 投影到最大截面
    projection_points = []
    z_values = []  # 存储高度差
    for vertex in vertices:
        projection_point = vertex - np.dot(vertex - section_point, long_axis) * long_axis
        projection_points.append(projection_point[:2])  # 只取X、Y坐标
        z_values.append(projection_point[2])  # Z轴值

    projection_points = np.array(projection_points)
    z_values = np.array(z_values)

    # 设置颜色区间
    levels = [-2.0, -0.75, -0.25, -0.09, 0.25, 0.75, 1.0, 2.0]
    colors_list = ['#1f3b7b', '#1976d2', '#4fc3f7', '#90ee90', '#ffff00', '#ffa500', '#ff4500']  # 从深蓝到红色

    # 自定义颜色映射
    cmap = colors.ListedColormap(colors_list)
    norm = colors.BoundaryNorm(boundaries=levels, ncolors=len(colors_list))

    # 绘制投影图
    plt.figure(figsize=(8, 8))
    plt.scatter(projection_points[:, 0], projection_points[:, 1], c=z_values, cmap=cmap, norm=norm, s=0.5)
    plt.title(f"{label} Model Projection")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar(label="Height (Z-axis)", boundaries=levels, ticks=levels)
    plt.axis("equal")
    plt.grid(True)
    plt.savefig(os.path.join(output_path, f"{label}_projection_continuous.png"))
    plt.close()


if __name__ == "__main__":
    input_path = os.path.join("input", "below_twenty", "2.stl")
    upper_output_path = os.path.join("output", "upper")
    lower_output_path = os.path.join("output", "below")
    projection_output_path = os.path.join("output", "projection")

    os.makedirs(upper_output_path, exist_ok=True)
    os.makedirs(lower_output_path, exist_ok=True)
    os.makedirs(projection_output_path, exist_ok=True)

    # 加载 STL 模型
    model = load_stl(input_path)

    # 计算中心和主轴
    center, long_axis = compute_long_axis(model)
    print(f"Center: {center}, Long Axis: {long_axis}")

    # 找最大截面
    max_section_points, max_plane_point = find_max_section(model, center, long_axis)

    # 切割模型
    above, below = split_model(model, max_plane_point, long_axis)

    # 保存切割结果
    save_stl(above, os.path.join(upper_output_path, "crown_above.stl"))
    save_stl(below, os.path.join(lower_output_path, "crown_below.stl"))

    # 绘制并保存投影图
    plot_projection_on_section(above.reshape(-1, 3), max_plane_point, long_axis, projection_output_path, "upper")
    plot_projection_on_section(below.reshape(-1, 3), max_plane_point, long_axis, projection_output_path, "below")

    print("完成所有操作！")
