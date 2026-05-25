import gzip
import numpy as np
import os
import glob
import shutil

import nibabel as nib
import numpy as np

from nii2npy import sequence_nii2_sequence_npy
from dicom2npy_png import sequence_dicom2_sequence_npy

# 将文件夹中的a2c提取出来
# import os
# import shutil
#
# src_dir = '/home/server4090/lyx/data/shenzhen_heart/model_data/label'  # 源文件夹路径
# # /home/server4090/lyx/data/shenzhen_heart/2021.8.2-DATASET-HEART     2021.4.21-DATASET-HEART
# dst_dir = '/home/server4090/lyx/data/shenzhen_heart/model_data/cur'  # 目标文件夹路径
#
# for root, dirs, files in os.walk(src_dir):
#     for file in files:
#         if file == "a3c.nii.gz":
#             src_file = os.path.join(root, file)  # 源文件路径
#             dst_file = os.path.join(dst_dir, os.path.basename(root) + ".nii.gz")  # 目标文件路径
#             shutil.copy(src_file, dst_file)  # 复制文件到目标文件夹中



# sequence_nii_path = '/home/server4090/lyx/data/shenzhen_heart/model_data/cur/'
# sequence_mask_npy_path = '/home/server4090/lyx/data/shenzhen_heart/model_data/npy/'

# 将.nii.gz 转换为 .npy文件
# sequence_nii2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)
#
# 将.DCM 转换为 .npy文件
# sequence_dicom2_sequence_npy(sequence_nii_path, sequence_mask_npy_path)



# import os
# import numpy as np
# from PIL import Image
# #
# # 指定文件夹路径
# folder_path = "/home/server4090/lyx/data/shenzhen_heart/model_data/npy"
#
# # 遍历文件夹中所有子文件夹
# for root, dirs, files in os.walk(folder_path):
#     for file in files:
#         # 判断文件是否为.npy文件
#         if file.endswith(".npy"):
#             # 读取.npy文件
#             np_array = np.load(os.path.join(root, file))
#             # 转化为PIL.Image对象
#             image = Image.fromarray(np_array)
#             # 保存为.png文件
#             image.save(os.path.join(root, file.replace(".npy", ".png")))
#             os.remove(os.path.join(root, file))




#
# import os
# import shutil
#
# # 指定原始文件夹路径
# source_folder_path = '/home/server4090/lyx/data/shenzhen_heart/model_data/npy'             #"/home/server4090/lyx/data/shenzhen_heart/model_data/npy"
#
# # 指定目标文件夹路径
# target_folder_path = "/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/labels"
#
# # 遍历文件夹中所有子文件夹
# for root, dirs, files in os.walk(source_folder_path):
#     for file in files:
#         # 判断文件是否为.png文件
#         if file.endswith(".png"):
#             # 将文件移动到目标文件夹中
#             shutil.move(os.path.join(root, file), os.path.join(target_folder_path, file))


# from PIL import Image
# import os
#
# # 定义输入和输出文件夹路径
# input_folder = '/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/labels'
# output_folder = '/home/server4090/lyx/data/shenzhen_heart/model_data/A3C/train/labels'
# target_size = (256, 256)  # 目标尺寸
#
# # 确保输出文件夹存在
# if not os.path.exists(output_folder):
#     os.makedirs(output_folder)
#
# # 遍历输入文件夹中的所有文件
# for filename in os.listdir(input_folder):
#     # 检查文件是否为图片文件
#     if filename.endswith('.png'):
#         # 打开图片
#         img_path = os.path.join(input_folder, filename)
#         img = Image.open(img_path)
#
#         # 调整图片大小为 target_size，使用双线性插值方法
#         img_resized = img.resize(target_size, Image.NEAREST)
#
#         # 保存调整后的图片到输出文件夹
#         output_path = os.path.join(output_folder, filename)
#         img_resized.save(output_path)



# 删除特定字符
# import os
#
# # 获取当前文件夹路径
# dir_path = '/home/server4090/lyx/data/a4c/images'
#
# # 遍历当前文件夹下的所有文件
# for filename in os.listdir(dir_path):
#     # 如果文件名中包含0
#     if '.png' in filename:
#         # 新文件名为去掉0的文件名
#         new_filename = filename.replace('.png', '', 1)
#         # 使用rename()函数将原文件名改为新的文件名
#         os.rename(os.path.join(dir_path, filename), os.path.join(dir_path, new_filename))


# import os
# import shutil
#
# # 获取当前文件夹的路径
# folder_path = '/home/server3090/lyx/data/shenzhen_heart/2022.12.14-DATASET-HEART'
#
# # 遍历当前文件夹及其所有子文件夹
# for root, dirs, files in os.walk(folder_path):
#     # 遍历子文件夹中的所有文件
#     for file_name in files:
#         # 如果文件名以"a3c_nii.gz"结尾
#         if file_name.endswith("a3c.nii.gz"):
#             # 构建文件的完整路径
#             old_file_path = os.path.join(root, file_name)
#             new_file_path = os.path.join(root, "a3c_nii.gz")
#             # 将原文件名改为"a3c_nii.gz"
#             os.rename(old_file_path, new_file_path)

# import os
# import glob
#
# # 指定要删除文件的目录路径
# folder_path = '/home/server4090/lyx/data/CAMUS_A2C/val/images/'  # 将此处替换为你的文件夹路径
#
# # 查找文件夹下包含 'a2c' 名字的所有文件
# files_to_delete = glob.glob(os.path.join(folder_path, '*a4c*'))
#
# # 删除找到的文件
# for file_path in files_to_delete:
#     try:
#         os.remove(file_path)
#         print(f"Deleted: {file_path}")
#     except Exception as e:
#         print(f"Error deleting {file_path}: {e}")

