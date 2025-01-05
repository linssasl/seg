from PIL import Image, ImageTk
import tkinter as tk

# 高度和颜色的范围映射表
COLOR_HEIGHT_MAP = [
    ((245, 0, 0), (255, 10, 10), (1.0, 2.0)),
    ((245, 50, 0), (255, 70, 10), (0.75, 1.0)),
    ((245, 100, 0), (255, 120, 10), (0.5, 0.75)),
    ((245, 150, 0), (255, 170, 10), (0.25, 0.5)),
    ((245, 200, 0), (255, 220, 10), (0.09, 0.25)),
    ((0, 245, 0), (10, 255, 10), (-0.09, 0.09)),
    ((0, 200, 20), (10, 210, 30), (-0.25, -0.09)),
    ((0, 100, 245), (10, 120, 255), (-0.5, -0.25)),
    ((0, 50, 245), (10, 70, 255), (-0.75, -0.5)),
    ((0, 0, 245), (10, 10, 255), (-1.0, -0.75)),
    ((0, 0, 245), (10, 0, 255), (-2.0, -1.0))
]

def is_color_in_range(rgb, min_rgb, max_rgb):
    """判断RGB值是否在范围内"""
    return all(min_rgb[i] <= rgb[i] <= max_rgb[i] for i in range(3))

def get_height_from_color(rgb):
    """根据颜色RGB值获取对应的高度范围"""
    for min_rgb, max_rgb, height_range in COLOR_HEIGHT_MAP:
        if is_color_in_range(rgb, min_rgb, max_rgb):
            return height_range
    return None

# 绘制颜色竖条图示
def draw_color_bar(canvas):
    bar_width = 50  # 竖条宽度
    bar_height = 30  # 每个颜色的高度
    x0, x1 = 10, 10 + bar_width
    y = 10

    for min_rgb, max_rgb, height_range in COLOR_HEIGHT_MAP:
        # 使用最小RGB值显示颜色块
        color_hex = f"#{min_rgb[0]:02x}{min_rgb[1]:02x}{min_rgb[2]:02x}"
        # 绘制矩形颜色块
        canvas.create_rectangle(x0, y, x1, y + bar_height, fill=color_hex, outline="black")
        # 绘制文字
        canvas.create_text(x1 + 10, y + bar_height / 2, anchor='w', text=f"{height_range[0]:.2f} ~ {height_range[1]:.2f}", font=('Helvetica', 10))
        y += bar_height

# 获取点击位置的色调和高度
def get_color_and_height(event, image, label):
    x = event.x
    y = event.y

    if x < 50:  # 防止点击颜色竖条
        return

    # 获取该位置的RGB值
    try:
        r, g, b = image.getpixel((x, y))
    except IndexError:
        return

    # 获取对应高度
    height_range = get_height_from_color((r, g, b))

    # 更新标签显示色调和高度
    if height_range:
        label.config(text=f"Color at ({x}, {y}): RGB({r}, {g}, {b}), Height: {height_range[0]:.2f} ~ {height_range[1]:.2f}")
    else:
        label.config(text=f"Color at ({x}, {y}): RGB({r}, {g}, {b}), Height: Not Found")

# 创建主窗口
root = tk.Tk()
root.title("Image Color and Height Picker")

# 加载图片，使用相对路径
image_path = r'Element/color.jpg' 
image = Image.open(image_path)
img_tk = ImageTk.PhotoImage(image)

# 创建一个Canvas来显示图片和颜色竖条
canvas = tk.Canvas(root, width=image.width + 100, height=image.height)
canvas.pack()

# 显示颜色竖条
draw_color_bar(canvas)

# 显示图片
canvas.create_image(50, 0, anchor='nw', image=img_tk)

# 创建一个标签来显示色调和高度
info_label = tk.Label(root, text="Click on the image to see the color and height", font=('Helvetica', 12))
info_label.pack()

# 绑定鼠标点击事件
canvas.bind("<Button-1>", lambda event: get_color_and_height(event, image, info_label))

# 运行主循环
root.mainloop()
