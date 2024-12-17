from PIL import Image, ImageTk
import tkinter as tk

# 读取图片文件
def load_image(image_path):
    return Image.open(image_path)

# 获取点击位置的色调
def get_color(event, image, img_tk, label):
    # 将点击位置转换为图片坐标
    x = event.x
    y = event.y
    # 获取该位置的RGB值
    r, g, b = image.getpixel((x, y))
    # 更新标签显示色调
    label.config(text=f"Color at ({x}, {y}): RGB({r}, {g}, {b})")

# 创建主窗口
root = tk.Tk()
root.title("Image Color Picker")

# 加载图片，使用相对路径
image_path = r'Element\color.jpg' 
image = load_image(image_path)
img_tk = ImageTk.PhotoImage(image)

# 创建一个Canvas来显示图片
canvas = tk.Canvas(root, width=image.width, height=image.height)
canvas.pack()
canvas.create_image(0, 0, anchor='nw', image=img_tk)

# 创建一个标签来显示色调
color_label = tk.Label(root, text="Click on the image to see the color", font=('Helvetica', 12))
color_label.pack()

# 绑定鼠标点击事件
canvas.bind("<Button-1>", lambda event: get_color(event, image, img_tk, color_label))

# 运行主循环
root.mainloop()