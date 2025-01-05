import os
import numpy as np
from stl import mesh
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
    long_axis = eigvecs[:, np.argmax(eigvals)]
    long_axis /= np.linalg.norm(long_axis)  # 规范化主轴

    return centroid, long_axis

# 计算模型与平面相交的截面
def get_intersection_section(model, plane_point, plane_normal):
    section_points = []
    for triangle in model.vectors:
        intersection = []
        for i in range(3):
            p1 = triangle[i]
            p2 = triangle[(i + 1) % 3]
            edge_vector = p2 - p1
            denom = np.dot(edge_vector, plane_normal)
            if abs(denom) > 1e-6:  # 避免平行情况
                t = np.dot(plane_point - p1, plane_normal) / denom
                if 0 <= t <= 1:
                    intersection.append(p1 + t * edge_vector)
        if len(intersection) == 2:
            section_points.extend(intersection)

    return np.array(section_points)

# 提取最大环的点集
def extract_largest_loop(section_points):
    if len(section_points) < 3:
        return None

    try:
        hull = ConvexHull(section_points)
        largest_loop = section_points[hull.vertices]
        return largest_loop
    except Exception as e:
        print(f"Error in ConvexHull: {e}")
        return None

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
        if len(section_points) == 0:
            continue

        largest_loop = extract_largest_loop(section_points)
        if largest_loop is None:
            continue

        try:
            section_area = ConvexHull(largest_loop).volume
            if section_area > max_area:
                max_area = section_area
                max_section_points = largest_loop
                max_plane_point = plane_point
        except Exception as e:
            print(f"Error calculating section area: {e}")
            continue

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
        elif len(above) > 0 and len(below) > 0:
            intersection = []
            for i in range(len(above)):
                for j in range(len(below)):
                    edge_vector = below[j] - above[i]
                    denom = np.dot(edge_vector, plane_normal)
                    if abs(denom) > 1e-6:
                        t = np.dot(plane_point - above[i], plane_normal) / denom
                        if 0 <= t <= 1:
                            intersection.append(above[i] + t * edge_vector)
            above_vertices.append(intersection)
            below_vertices.append(intersection)

    return np.array(above_vertices), np.array(below_vertices)

# 主程序
if __name__ == "__main__":
    from pathlib import Path

    input_path = Path("input/thirty_forty/310.stl")
    upper_output_path = Path("output/upper")
    lower_output_path = Path("output/below")
    upper_output_path.mkdir(parents=True, exist_ok=True)
    lower_output_path.mkdir(parents=True, exist_ok=True)

    # 加载 STL 模型
    model = load_stl(str(input_path))

    # 计算中心和主轴
    center, long_axis = compute_long_axis(model)
    print(f"Center: {center}, Long Axis: {long_axis}")

    # 找最大截面
    max_section_points, max_plane_point = find_max_section(model, center, long_axis)
    if max_section_points is None or max_plane_point is None:
        print("未找到有效的截面")
    else:
        # 切割模型
        above, below = split_model(model, max_plane_point, long_axis)

        # 保存切割结果
        save_stl(above, str(upper_output_path / "crown_above.stl"))
        save_stl(below, str(lower_output_path / "crown_below.stl"))

        print("完成所有操作！")
