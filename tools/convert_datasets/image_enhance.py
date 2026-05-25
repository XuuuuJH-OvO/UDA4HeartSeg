import os
import os.path as path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# 1,反色变换
# 假设原始图像的灰度范围是[0,L],L表示该图像最大的灰度值
# 则反色变换为output = L - input

def image_inverse(input):#图像颜色反转
    value_max = np.max(input)
    output = value_max - input
    return output



def laplacian(image):#拉普拉斯算子增强
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return image_lap
  


def hist(image):#直方图均衡增强
    r, g, b = cv2.split(image)
    r1 = cv2.equalizeHist(r)
    g1 = cv2.equalizeHist(g)
    b1 = cv2.equalizeHist(b)
    image_equal_clo = cv2.merge([r1, g1, b1])
    return image_equal_clo



def clahe(image):
    b, g, r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    image_clahe = cv2.merge([b, g, r])
    return image_clahe



'''
gray_img = np.asarray(Image.open('191.jpg').convert('L'))
img = cv2.imread('191.jpg')
inv_img = clahe(img)

plt.subplot(121)
plt.title('original')
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)

plt.subplot(122)
plt.title('inverse')
plt.imshow(inv_img, cmap='gray', vmin=0, vmax=255)
plt.show()


'''

root_dir = "/home/server4090/lyx/data/CAMUS/A2C/images/"#直接到class的上一级
output_dir ='/home/server4090/lyx/data/A2C/sharpen'
class_index =os.listdir(root_dir)
for filename in os.listdir(root_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # 可根据需要修改文件格式
        input_image_path = os.path.join(root_dir, filename)
        output_image_path = os.path.join(output_dir, filename)

        # 读取图像
        image = cv2.imread(input_image_path, 0)  # 以灰度图像的形式读取
        sharped_image = laplacian(image)
        cv2.imwrite(output_image_path, sharped_image)
            
            
