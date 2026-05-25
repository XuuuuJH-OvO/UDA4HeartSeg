import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk

nii_path = '/data/dwl/children_heart/children_heart_paper/nii_to_add_frame/origin_nii/A3CV.nii.gz'
added_nii_path = '/data/dwl/children_heart/children_heart_paper/nii_to_add_frame/added_nii/'

def npy2nii(oneP_npy_path, nii_path):
    oneP_npy_list = []
    Patient_name = oneP_npy_path.split('/')[-2]
    for each_npy_file in os.listdir(oneP_npy_path):
        each_npy_file_path = os.path.join(oneP_npy_path, each_npy_file)
        each_npy_mat = np.load(each_npy_file_path)
        oneP_npy_list.append(each_npy_mat)
    oneP_npy_list = np.stack(oneP_npy_list, 0)
    print(oneP_npy_list.shape)
    oneP_nii_file = sitk.GetImageFromArray(oneP_npy_list)
    sitk.WriteImage(oneP_nii_file,nii_path + Patient_name + '.nii.gz')

def add_nii_frame(nii_path, added_nii_path, dcm_frame, nii_frame):
    ori_mask = nib.load(nii_path)
    nii_file_name = os.path.split(nii_path)[1].split('.')[0]
    mask_arr = ori_mask.dataobj
    width, height, deep = mask_arr.shape
    mask_arr_np = np.array(mask_arr)

    added_frame = dcm_frame - nii_frame
    added_frame_mat = np.zeros((width,height, added_frame), dtype=np.uint16)
    # added_nii_mat = np.concatenate((mask_arr_np,added_frame_mat), axis=2)
    added_nii_mat = np.concatenate((added_frame_mat, mask_arr_np), axis=2)
    added_nii_mat = np.swapaxes(added_nii_mat, 0, 2)
    add_nii_mat_file = sitk.GetImageFromArray(added_nii_mat)
    sitk.WriteImage(add_nii_mat_file, added_nii_path + nii_file_name + '-m' + '.nii.gz')
    # print(mask_arr_np.dtype)
    # print(added_frame_mat.dtype)
    print(added_nii_mat.shape, added_nii_mat.dtype)

def change_nii_value(nii_path, added_nii_path, origin_value, changed_value):
    ori_mask = nib.load(nii_path)
    nii_file_name = os.path.split(nii_path)[1].split('.')[0]
    mask_arr = ori_mask.dataobj
    width, height, deep = mask_arr.shape
    mask_arr_np = np.array(mask_arr)

    mask_arr_np = np.where(mask_arr_np == origin_value, changed_value, mask_arr_np)


    mask_arr_np = np.swapaxes(mask_arr_np, 0, 2)
    mask_arr_np_file = sitk.GetImageFromArray(mask_arr_np)
    sitk.WriteImage(mask_arr_np_file, added_nii_path + nii_file_name + '-m' + '.nii.gz')
    # print(mask_arr_np.dtype)
    # print(added_frame_mat.dtype)
    print(mask_arr_np.shape, mask_arr_np.dtype)


if __name__ == "__main__":
    nii_path = 'E:/shenzhen_heart/2022.12.14- DATASET-HEART/outnpy/15-psaxout/pasxout.nii.gz'
    added_nii_path = '/data/dwl/children_heart/children_heart_paper/modifiedd_nii/1/'
    oneP_npy_path = 'E:/shenzhen_heart/2022.12.14- DATASET-HEART/outnpy/15-psaxout'
    npy2nii(oneP_npy_path, nii_path)