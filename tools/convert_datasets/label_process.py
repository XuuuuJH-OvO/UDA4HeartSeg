# import os
# # import numpy as np
# # from PIL import Image
# #
# # # 1. 定义调色板（根据你的标签类别自定义）
# # palette = np.array([
# #     [0, 0, 0],  # 类别0：黑色（背景）
# #     [128, 128, 128],  # 类别1：红色
# #     [255, 255, 255],  # 类别2：绿色
# # ], dtype=np.uint8)
# #
# # # 2. 设置输入输出路径（修改为你实际的路径）
# # input_dir = "/home/server4090/lyx/data/shenzhen_heart/model_data/A4C/test/labels"  # 标签图像所在文件夹
# # output_dir = "/home/server4090/lyx/data/shenzhen_heart/model_data/A4C/test/labelss"  # RGB输出文件夹
# #
# # # 3. 创建输出文件夹（如果不存在）
# # os.makedirs(output_dir, exist_ok=True)
# #
# # # 4. 处理所有图像
# # for filename in os.listdir(input_dir):
# #     if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
# #         # 读取标签图像
# #         label_path = os.path.join(input_dir, filename)
# #         label_img = Image.open(label_path)
# #         label_array = np.array(label_img)
# #
# #         # 创建RGB图像
# #         rgb_array = np.zeros((*label_array.shape, 3), dtype=np.uint8)
# #
# #         # 应用调色板
# #         for label_id, color in enumerate(palette):
# #             rgb_array[label_array == label_id] = color
# #
# #         # 保存结果
# #         output_path = os.path.join(output_dir, filename)
# #         Image.fromarray(rgb_array).save(output_path)
# #         print(f"Converted: {filename}")
# #
# # print("All label images have been converted to RGB images!")
import numpy as np
import matplotlib.pyplot as plt

# 1. 定义Tanh函数
def tanh(x):
    return np.tanh(x)

# 2. 生成输入数据（范围：-5到5，间隔0.1）
x = np.arange(-5, 5, 0.1)
y = tanh(x)

# 3. 绘制图像
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Tanh Function', color='blue', linewidth=2)

# 4. 添加标题和标签
plt.title('Tanh Activation Function', fontsize=14)
plt.xlabel('Input (x)', fontsize=12)
plt.ylabel('Output (tanh(x))', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 5. 标注关键点
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)  # x轴
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)  # y轴
plt.text(0.5, 0.1, 'S-shaped Curve', fontsize=10, color='red')
plt.text(3, 0.9, 'Saturates to +1', fontsize=10, color='green')
plt.text(-4, -0.9, 'Saturates to -1', fontsize=10, color='green')

# 6. 显示图例和图像
plt.legend(loc='upper left', fontsize=10)
plt.show()