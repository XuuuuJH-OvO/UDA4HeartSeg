#dicom文件转png图像或者npy文件，每一个dicom文件输出为一个文件夹的png或者npy文件。
#sequence2oneFolder函数的作用是将子文件夹的png或者npy文件转到一个文件夹下。

# import SimpleITK as sitk
import os
import pydicom
import shutil
import numpy as np
import cv2
import shutil


#dicom文件转npy文件，每一个dicom文件转出来对应一个文件夹的npy文件
# 如果目标子文件夹已创建，会直接跳过该case
# pydicom.dcmread(dicom_file_path)异常，会pydicom.dcmread(dicom_file_path, force=True)，并打印该case名称
def sequence_dicom2_sequence_png(dicom_path, sequence_img_path):
    for dicom_file in os.listdir(dicom_path):
        print('dicom_file name: ', dicom_file)
        dicom_file_path = os.path.join(dicom_path, dicom_file)

        dicom_file_prefix = os.path.splitext(dicom_file)[0]
        sub_suquence_img_path = sequence_img_path + dicom_file_prefix + '/'
        if os.path.isdir(sub_suquence_img_path) is False:
            os.mkdir(sub_suquence_img_path)
        else:
            continue

        try:
            dicom_parse = pydicom.dcmread(dicom_file_path)
        except pydicom.errors.InvalidDicomError as e:
            print('wrong dicom_file name: ', dicom_file)
            print(e)
            # dicom_parse = pydicom.dcmread(dicom_file_path, force=True)
            shutil.rmtree(sub_suquence_img_path)
            continue


        img_array = np.array(dicom_parse.pixel_array)

        z,h,w,c = img_array.shape
        for step in range(z):
            cv2.imwrite(sub_suquence_img_path + dicom_file_prefix + f'-{step+1}.png', img_array[step,:,:,0])

#dicom文件转npy文件，每一个dicom文件转出来对应一个文件夹的npy文件
# 如果目标子文件夹已创建，会认为该case已转换，并直接跳过该case
# pydicom.dcmread(dicom_file_path)异常，会pydicom.dcmread(dicom_file_path, force=True)，并打印该case名称
def sequence_dicom2_sequence_npy(dicom_path, sequence_img_npy_path):
    for dicom_file in os.listdir(dicom_path):
        print('dicom_file name: ', dicom_file)
        dicom_file_path = os.path.join(dicom_path, dicom_file)

        dicom_file_prefix = os.path.splitext(dicom_file)[0]
        sub_suquence_img_npy_path = sequence_img_npy_path + dicom_file_prefix + '/'
        if os.path.isdir(sub_suquence_img_npy_path) is False:
            os.mkdir(sub_suquence_img_npy_path)
        else:
            continue
        try:
            dicom_parse = pydicom.dcmread(dicom_file_path)
        except pydicom.errors.InvalidDicomError as e:
            print('wrong dicom_file name: ', dicom_file)
            print(e)
            # dicom_parse = pydicom.dcmread(dicom_file_path, force=True)
            # dicom_parse.file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
            shutil.rmtree(sub_suquence_img_npy_path)
            continue

        img_array = np.array(dicom_parse.pixel_array)

        z,h,w,c = img_array.shape
        for step in range(z):
            np.save(sub_suquence_img_npy_path + dicom_file_prefix + f'-{step+1}.npy', img_array[step,:,:,0])

#sequence2oneFolder函数的作用是将子文件夹的png或者npy文件转到一个文件夹下。
def sequence2oneFolder(sequence_img_path, img_path):
    for subfolder in os.listdir(sequence_img_path):
        subfolder_path = os.path.join(sequence_img_path, subfolder) + '/'
        for img_file in os.listdir(subfolder_path):
            img_file_path = os.path.join(subfolder_path, img_file)
            target_file_path = os.path.join(img_path, img_file)
            shutil.copy(img_file_path, target_file_path)

if __name__ == "__main__":
    # 二腔心切面，包括健康和不健康
    sequence_image_dicom = '/home/server4090/lyx/data/shenzhen_heart/model_data/label/'
    sequence_image_npy = '/home/server4090/lyx/data/shenzhen_heart/model_data/cur/'
    sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)
    # sequence_image_dicom = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_image_dicom/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_image_npy/'
    # sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)

    # 三腔心切面，包括健康和不健康
    # sequence_image_dicom = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_image_dicom/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_image_npy/'
    # sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)
    #
    # sequence_image_dicom = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_image_dicom/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_image_npy/'
    # sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)

    # 四腔心切面，包括健康和不健康
    # sequence_image_dicom = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_image_dicom/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_image_npy/'
    # sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)

    # sequence_image_dicom = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_image_dicom/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_image_npy/'
    # sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)

    # 短轴切面，包括健康和不健康
    # sequence_image_dicom = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_image_dicom/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/health/sequence_image_npy/'
    # sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)

    # sequence_image_dicom = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_image_dicom/'
    # sequence_image_npy = '/data/dwl/children_heart/children_heart_paper/model_adopt/minor_axis/unhealth/sequence_image_npy/'
    # sequence_dicom2_sequence_npy(sequence_image_dicom, sequence_image_npy)