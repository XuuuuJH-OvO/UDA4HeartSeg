import numpy as np
import os  # 遍历文件夹
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像
import cv2
import pydicom

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
import os
from matplotlib import pylab as plt
import cv2
from scipy import ndimage as ndi

def image_fill(image):
    """

    选取最大连通分量，根据图像特征，漫水算法的起始点是图像的中心。
这是一个图像处理函数，用于选取输入图像中的最大连通分量并返回该连通分量的标签图像。
具体实现过程为：先将输入图像复制一份，然后根据图像特征，使用漫水算法从图像中心开始填充一个掩膜，得到一个标记了最大连通分量的副本图像。
接着通过将原始图像和副本图像相减的方式得到最大连通分量，最后将最大连通分量以外的部分设置为黑色，得到标签图像。

    :param image: [h,w]
    :return: 选取最大连通分量之后的label。[h,w]
    """

    ori = image.copy()
    src = image.copy()  # 先创建一个副本
    mask = np.zeros([src.shape[0] + 2, src.shape[1] + 2], np.uint8)  # 根据副本形状建一个掩膜， 注意，长和宽必须要+2，类型只能是uint8
    cv2.floodFill(src, mask, (src.shape[0]//2, src.shape[1]//2), (0, 255, 255), (0, 0, 0), (0, 0, 0), cv2.FLOODFILL_FIXED_RANGE)
    # img, mask, seed, newvalue(BGR),(loDiff1,loDiff2,loDiff3),(upDiff1,upDiff2,upDiff3),flag
    output = ori - src
    blank_pic = np.zeros([src.shape[0], src.shape[1]], np.uint8)
    mask = cv2.circle(blank_pic, (int(src.shape[0] / 2), int(src.shape[1] / 2)), int(src.shape[0] / 2), (1, 0, 0), -1)
    output = output * mask

    return output

def fill_hole(image):
    ## 对比mask_fill和binary_fill_holes，该算法的运行速度最快
    ## 输入必须是uint8的图像，若无法运行要记得这一点
    #该函数用于填补二值图像中的空洞。它利用OpenCV中的floodFill函数和位运算实现。它首先找到一个背景像素作为种子点，
    #然后使用floodFill函数将其周围的所有背景像素都涂成前景像素，最后使用位运算将涂色过的图像与原始图像进行或运算，从而填补空洞。
    src = image.copy()  # 先创建一个副本
    mask = np.zeros([src.shape[0] + 2, src.shape[1] + 2], np.uint8)  # 根据副本形状建一个掩膜， 注意，长和宽必须要+2，类型只能是uint8

    #将种子点设为背景
    isbreak = False
    for i in range(src.shape[0]):
        for j in range(src.shape[1]):
            if src[i, j] == 0:
                seedpoint = (i, j)
                isbreak = True
                break
        if isbreak:
            break

    cv2.floodFill(src, mask, seedpoint, 255)
    img_floofill_inv = cv2.bitwise_not(src)
    im_out = image | img_floofill_inv

    return im_out

def draw_fig(img_mat, title='img_mat'):
    plt.close()
    plt.title(title)
    plt.imshow(img_mat, cmap='gray')
    plt.show()
    plt.close()

def erode_dilate_process(origin_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
    erode_res = cv2.erode(origin_mask, kernel)  # 腐蚀结果
    dilate_res = cv2.dilate(origin_mask, kernel)  # 膨胀结果

    draw_fig(origin_mask, title='origin_mask')
    draw_fig(erode_res, title='erode_mask')
    draw_fig(dilate_res, title='dilate_mask')


def open_close_process(origin_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iterations = 20  # 执行开闭运算的次数
    open_res = cv2.morphologyEx(origin_mask, cv2.MORPH_OPEN, kernel, iterations)
    #     close_res = cv2.morphologyEx(origin_mask, cv2.MORPH_CLOSE, kernel, iterations)

    draw_fig(origin_mask, title='origin_mask')
    draw_fig(open_res, title='open_res')


#     draw_fig(close_res, title='close_res')

def post_process(origin_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    iterations = 20  # 执行开闭运算的次数
    open_res = cv2.morphologyEx(origin_mask, cv2.MORPH_OPEN, kernel, iterations)
    #     draw_fig(origin_mask, title='origin_mask')
    #     draw_fig(open_res, title='open_res')

    marker = ndi.label(np.uint8(open_res), output=np.uint32)[0]
    label_list, label_num = np.unique(marker, return_counts=True)
    #     print(label_list, label_num)

    max_label_num = max(label_num[1:])
    max_label_num_index = np.where(label_num == max_label_num)

    new_label_list = list(label_list.copy())
    new_label_list.remove(0)
    new_label_list.remove(int(max_label_num_index[0][0]))

    if len(new_label_list) != 0:
        for detele_element in new_label_list:
            marker = np.where(marker == detele_element, 0, marker)
    else:
        pass

    marker = np.where(marker > 0, 1, marker)
    return marker