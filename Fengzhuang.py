import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

# ========== STL 相关处理函数 ===========
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

# 绘制热力图，生成连续光滑的图
def plot_heatmap_on_section(vertices, section_point, long_axis, output_path, label):
    # 计算垂直于 long_axis 的点到平面的距离
    long_axis = long_axis / np.linalg.norm(long_axis)  # 单位化
    distances = np.dot(vertices - section_point, long_axis)  # 点到平面的距离

    # **归一化高度值**
    min_distance, max_distance = distances.min(), distances.max()
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)

    # 设置颜色区间（归一化后为[0, 1]范围）
    levels = np.linspace(0, 1, 100)  # 更细致的颜色渐变
    cmap = plt.cm.get_cmap('RdYlBu_r')  # 使用红-黄-蓝渐变颜色

    # 计算点的投影到平面
    arbitrary_vec = np.array([1, 0, 0]) if abs(long_axis[0]) < 0.9 else np.array([0, 1, 0])
    v1 = np.cross(long_axis, arbitrary_vec)
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(long_axis, v1)

    projection_points = []
    for vertex in vertices:
        relative_position = vertex - section_point
        x = np.dot(relative_position, v1)
        y = np.dot(relative_position, v2)
        projection_points.append([x, y])
    projection_points = np.array(projection_points)

    # 使用插值创建平滑的二维网格
    x = projection_points[:, 0]
    y = projection_points[:, 1]
    z = normalized_distances  # 使用归一化后的高度值

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 500),
        np.linspace(y.min(), y.max(), 500)
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')  # 使用 cubic 插值

    # 绘制等高线图
    plt.figure(figsize=(8, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)
    plt.colorbar(contour, label="Normalized Distance to Section Plane (Z-axis)")
    plt.title(f"{label} Model Heatmap")
    plt.xlabel("X-axis (projected)")
    plt.ylabel("Y-axis (projected)")
    plt.axis("equal")
    plt.grid(True)

    # 保存图像
    plt.savefig(os.path.join(output_path, f"{label}_heatmap.png"))
    plt.close()

# ========== GUI 部分 ===========
class STLProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("STL File Processing Tool")
        self.root.geometry("1080x800")
        self.root.resizable(True, True)

        # 确保关闭界面时结束进程
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.selected_file = None
        self.save_directory = None

        self.create_widgets()

    def create_widgets(self):
        # 左侧区域预留空间
        self.canvas_frame = tk.Frame(self.root, bg="white", width=800, height=800)
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右侧控制区域
        self.control_frame = tk.Frame(self.root, bg="lightgray", width=280)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y)

        # 按钮：选择STL文件
        self.btn_select_file = tk.Button(self.control_frame, text="选择STL文件", command=self.select_stl_file)
        self.btn_select_file.pack(pady=20, fill=tk.X, padx=20)

        # 按钮：选择生成文件存储位置
        self.btn_select_save = tk.Button(self.control_frame, text="选择生成文件存储位置", command=self.select_save_directory)
        self.btn_select_save.pack(pady=20, fill=tk.X, padx=20)

        # 按钮：处理文件
        self.btn_process_file = tk.Button(self.control_frame, text="处理文件", command=self.process_file)
        self.btn_process_file.pack(pady=20, fill=tk.X, padx=20)

        # 按钮：打开投影图
        self.btn_open_heatmap = tk.Button(self.control_frame, text="打开投影图", command=self.open_heatmap)
        self.btn_open_heatmap.pack(pady=20, fill=tk.X, padx=20)

    def select_stl_file(self):
        self.selected_file = filedialog.askopenfilename(filetypes=[("STL Files", "*.stl")])
        if self.selected_file:
            messagebox.showinfo("文件已选择", f"已选择文件: {self.selected_file}")

    def select_save_directory(self):
        self.save_directory = filedialog.askdirectory()
        if self.save_directory:
            messagebox.showinfo("保存路径已选择", f"文件将存储到: {self.save_directory}")

    def process_file(self):
        if not self.selected_file:
            messagebox.showwarning("警告", "请先选择STL文件！")
            return
        if not self.save_directory:
            messagebox.showwarning("警告", "请先选择存储位置！")
            return

        # 创建进度窗口
        progress_window = tk.Toplevel(self.root)
        progress_window.title("处理进度")
        progress_window.geometry("400x150")

        progress_label = tk.Label(progress_window, text="正在处理，请稍候...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="determinate")
        progress_bar.pack(pady=10, fill=tk.X, padx=20)

        progress_bar['value'] = 0
        self.root.update_idletasks()

        try:
            # 调用合并的方法
            self.segment_and_generate_heatmap(self.selected_file, self.save_directory, progress_label, progress_bar)

            progress_window.destroy()
            messagebox.showinfo("完成", "文件处理完成！")

        except Exception as e:
            progress_window.destroy()
            messagebox.showerror("错误", f"处理失败: {str(e)}")


    def segment_and_generate_heatmap(self, file_path, save_directory, progress_label, progress_bar):
        # 加载 STL 模型
        progress_label.config(text="加载 STL 模型...")
        self.update_progress(progress_bar, 10)
        model = load_stl(file_path)

        # 计算中心和主轴
        progress_label.config(text="计算中心和主轴...")
        self.update_progress(progress_bar, 20)
        center, long_axis = compute_long_axis(model)
        print(f"Center: {center}, Long Axis: {long_axis}")

        # 找最大截面
        progress_label.config(text="计算最大截面...")
        self.update_progress(progress_bar, 40)
        max_section_points, max_plane_point = find_max_section(model, center, long_axis)

        # 切割模型
        progress_label.config(text="切割模型...")
        self.update_progress(progress_bar, 60)
        above, below = split_model(model, max_plane_point, long_axis)

        # 保存切割结果
        progress_label.config(text="保存切割结果...")
        self.update_progress(progress_bar, 80)
        upper_output_path = os.path.join(save_directory, "crown_above.stl")
        lower_output_path = os.path.join(save_directory, "crown_below.stl")
        save_stl(above, upper_output_path)
        save_stl(below, lower_output_path)
        print(f"模型已保存到: {upper_output_path} 和 {lower_output_path}")

        # 绘制并保存热力图
        progress_label.config(text="生成投影热力图...")
        self.update_progress(progress_bar, 100)
        plot_heatmap_on_section(above.reshape(-1, 3), max_plane_point, long_axis, save_directory, "upper")
        plot_heatmap_on_section(below.reshape(-1, 3), max_plane_point, long_axis, save_directory, "below")
        print(f"热力图已保存到: {save_directory}")


    def update_progress(self, progress_bar, value):
        progress_bar['value'] = value
        progress_bar.update_idletasks()

        

    def open_heatmap(self):
        heatmap_file = filedialog.askopenfilename(filetypes=[("PNG Files", "*.png")])
        if not heatmap_file:
            messagebox.showwarning("警告", "请先选择投影图文件！")
            return

        try:
            for widget in self.canvas_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(6, 6))
            img = plt.imread(heatmap_file)
            ax.imshow(img)
            ax.axis('off')

            canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        except Exception as e:
            messagebox.showerror("错误", f"无法打开投影图: {str(e)}")

    def on_close(self):
        self.root.destroy()
        os._exit(0)

if __name__ == "__main__":
    root = tk.Tk()
    app = STLProcessingApp(root)
    root.mainloop()