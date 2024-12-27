import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
from stl import mesh
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata

# ========== STL 相关处理函数 ===========
def load_stl(file_path):
    return mesh.Mesh.from_file(file_path)

def save_stl(vertices, file_path):
    output_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    output_mesh.vectors = vertices
    output_mesh.save(file_path)

def compute_long_axis(model):
    vertices = model.vectors.reshape(-1, 3)
    centroid = np.mean(vertices, axis=0)
    inertia_tensor = np.zeros((3, 3))

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

    eigvals, eigvecs = np.linalg.eig(inertia_tensor)
    long_axis = eigvecs[:, np.argmax(eigvals)]

    return centroid, long_axis

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

def plot_heatmap_on_section(vertices, section_point, long_axis, output_path, label):
    long_axis = long_axis / np.linalg.norm(long_axis)
    distances = np.dot(vertices - section_point, long_axis)

    min_distance, max_distance = distances.min(), distances.max()
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)

    levels = np.linspace(0, 1, 100)
    cmap = plt.cm.get_cmap('RdYlBu_r')

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

    x = projection_points[:, 0]
    y = projection_points[:, 1]
    z = normalized_distances

    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 500),
        np.linspace(y.min(), y.max(), 500)
    )
    grid_z = griddata((x, y), z, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 8))
    contour = plt.contourf(grid_x, grid_y, grid_z, levels=levels, cmap=cmap)
    plt.colorbar(contour, label="Normalized Distance to Section Plane (Z-axis)")
    plt.title(f"{label} Model Heatmap")
    plt.xlabel("X-axis (projected)")
    plt.ylabel("Y-axis (projected)")
    plt.axis("equal")
    plt.grid(True)

    plt.savefig(os.path.join(output_path, f"{label}_heatmap.png"))
    plt.close()

# ========== GUI 部分 ===========
class STLProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("STL File Processing Tool")
        self.root.geometry("1080x800")
        self.root.resizable(True, True)

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

        progress_window = tk.Toplevel(self.root)
        progress_window.title("处理进度")
        progress_window.geometry("400x150")

        progress_label = tk.Label(progress_window, text="正在处理，请稍候...")
        progress_label.pack(pady=10)

        progress_bar = ttk.Progressbar(progress_window, mode="indeterminate")
        progress_bar.pack(pady=10, fill=tk.X, padx=20)
        progress_bar.start()

        self.root.update_idletasks()

        try:
            # 模拟处理逻辑
            import time
            time.sleep(3)  # 模拟耗时处理

            progress_bar.stop()
            progress_window.destroy()
            messagebox.showinfo("完成", "文件处理完成！")
        except Exception as e:
            progress_bar.stop()
            progress_window.destroy()
            messagebox.showerror("错误", f"处理失败: {str(e)}")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = STLProcessingApp(root)
    root.mainloop()
