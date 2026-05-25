import numpy as np
import os



def merge_two_mask(mask_1, mask_2, mask_merge):
    mask_1_list = os.listdir(mask_1)
    # mask_1_list = ['81-2cavity']
    for subfolder_name in mask_1_list:
        subfolder_path = mask_1 + subfolder_name + '/'

        mask_2_subfolder_name = subfolder_name + 'f'
        mask_2_subfolder_path = mask_2 + mask_2_subfolder_name + '/'

        mask_merge_subfolder_path = mask_merge + subfolder_name + '/'
        if os.path.exists(mask_merge_subfolder_path) is False:
            os.mkdir(mask_merge_subfolder_path)
        else:
            continue

        for subsubfile_name in os.listdir(subfolder_path):
            subsubfile_path = subfolder_path + subsubfile_name

            subsubfile_name_split = subsubfile_name.split('-')
            mask_2_subsub_file_name = subsubfile_name_split[0] + '-' + subsubfile_name_split[1] + 'f-' + subsubfile_name_split[2]
            mask_2_subsub_file_path = mask_2_subfolder_path + mask_2_subsub_file_name

            subsubfile_mat = np.load(subsubfile_path)
            mask_2_subsub_file_mat = np.load(mask_2_subsub_file_path)
            subsubfile_mat_value = np.unique(subsubfile_mat)
            mask_2_subsub_file_mat_value = np.unique(mask_2_subsub_file_mat)
            if not (subsubfile_mat_value == np.array([0,1])).all():
                subsubfile_mat = np.where(subsubfile_mat > 0, 1, subsubfile_mat)
                print('!', end='')
            # print(mask_2_subsub_file_mat_value)
            try:
                if not ((mask_2_subsub_file_mat_value == np.array([0,3])).all()):
                    mask_2_subsub_file_mat = np.where(mask_2_subsub_file_mat > 0, 3, mask_2_subsub_file_mat)
                    print('!', end='')
            except (AttributeError):
                print('-------------subsubfile_name', subsubfile_name)
                mask_2_subsub_file_mat = np.where(mask_2_subsub_file_mat > 0, 3, mask_2_subsub_file_mat)

            merge_mask_mat = subsubfile_mat + mask_2_subsub_file_mat
            merge_mask_mat = np.where(merge_mask_mat == 4, 0, merge_mask_mat)

            mask_merge_subsub_file_path = mask_merge_subfolder_path + subsubfile_name
            np.save(mask_merge_subsub_file_path, merge_mask_mat)

        print(f'{subsubfile_name}', 'subsubfile_mat', subsubfile_mat_value, subsubfile_mat.shape, end='; ')
        print('mask_2_subsub_file_mat', mask_2_subsub_file_mat_value, mask_2_subsub_file_mat.shape, end='; ')
        print('merge_mask_mat', np.unique(merge_mask_mat), merge_mask_mat.shape)

def merge_two_mask_unhealth(mask_1, mask_2, mask_merge):
    mask_1_list = os.listdir(mask_1)
    # mask_1_list = ['81-2cavity']
    for subfolder_name in mask_1_list:
        subfolder_path = mask_1 + subfolder_name + '/'

        mask_2_subfolder_name = subfolder_name.split('-')[0] + '-' + 'A4CA'
        mask_2_subfolder_path = mask_2 + mask_2_subfolder_name + '/'

        mask_merge_subfolder_path = mask_merge + subfolder_name + '/'
        if os.path.exists(mask_merge_subfolder_path) is False:
            os.mkdir(mask_merge_subfolder_path)
        else:
            continue

        for subsubfile_name in os.listdir(subfolder_path):
            subsubfile_path = subfolder_path + subsubfile_name

            subsubfile_name_split = subsubfile_name.split('-')
            mask_2_subsub_file_name = subsubfile_name_split[0] + '-A4CA-' + subsubfile_name_split[2]
            mask_2_subsub_file_path = mask_2_subfolder_path + mask_2_subsub_file_name

            subsubfile_mat = np.load(subsubfile_path)
            mask_2_subsub_file_mat = np.load(mask_2_subsub_file_path)
            subsubfile_mat_value = np.unique(subsubfile_mat)
            mask_2_subsub_file_mat_value = np.unique(mask_2_subsub_file_mat)
            try:
                if not (subsubfile_mat_value == np.array([0,1])).all():
                    subsubfile_mat = np.where(subsubfile_mat > 0, 1, subsubfile_mat)
                    print('!', end='')
            except (AttributeError):
                print('-------------subsubfile_name', subsubfile_name)
                mask_2_subsub_file_mat = np.where(mask_2_subsub_file_mat > 0, 1, mask_2_subsub_file_mat)
            # print(mask_2_subsub_file_mat_value)
            try:
                if not ((mask_2_subsub_file_mat_value == np.array([0,3])).all()):
                    mask_2_subsub_file_mat = np.where(mask_2_subsub_file_mat > 0, 3, mask_2_subsub_file_mat)
                    print('!', end='')
            except (AttributeError):
                print('-------------subsubfile_name', subsubfile_name)
                mask_2_subsub_file_mat = np.where(mask_2_subsub_file_mat > 0, 3, mask_2_subsub_file_mat)

            merge_mask_mat = subsubfile_mat + mask_2_subsub_file_mat
            merge_mask_mat = np.where(merge_mask_mat == 4, 0, merge_mask_mat)

            mask_merge_subsub_file_path = mask_merge_subfolder_path + subsubfile_name
            np.save(mask_merge_subsub_file_path, merge_mask_mat)

        print(f'{subsubfile_name}', 'subsubfile_mat', subsubfile_mat_value, subsubfile_mat.shape, end='; ')
        print('mask_2_subsub_file_mat', mask_2_subsub_file_mat_value, mask_2_subsub_file_mat.shape, end='; ')
        print('merge_mask_mat', np.unique(merge_mask_mat), merge_mask_mat.shape)

# def merge_two_mask_short(mask_1, mask_2, mask_merge):
#     mask_1_list = os.listdir(mask_1)
#     # mask_1_list = ['81-2cavity']
#     for subfolder_name in mask_1_list:
#         subfolder_path = mask_1 + subfolder_name + '/'
#
#         mask_2_subfolder_name = subfolder_name.split('-')[0] + '-' + 'out'
#         mask_2_subfolder_path = mask_2 + mask_2_subfolder_name + '/'
#
#         mask_merge_subfolder_path = mask_merge + subfolder_name + '/'
#         if os.path.exists(mask_merge_subfolder_path) is False:
#             os.mkdir(mask_merge_subfolder_path)
#         else:
#             continue
#
#         for subsubfile_name in os.listdir(subfolder_path):
#             subsubfile_path = subfolder_path + subsubfile_name
#
#             subsubfile_name_split = subsubfile_name.split('-')
#             mask_2_subsub_file_name = subsubfile_name_split[0] + '-' + 'out-' + subsubfile_name_split[2]
#             mask_2_subsub_file_path = mask_2_subfolder_path + mask_2_subsub_file_name
#
#             subsubfile_mat = np.load(subsubfile_path)
#             mask_2_subsub_file_mat = np.load(mask_2_subsub_file_path)
#             subsubfile_mat_value = np.unique(subsubfile_mat)
#             mask_2_subsub_file_mat_value = np.unique(mask_2_subsub_file_mat)
#             if not (subsubfile_mat_value == np.array([0,1])).all():
#                 subsubfile_mat = np.where(subsubfile_mat > 0, 1, subsubfile_mat)
#                 print('!', end='')
#             # print(mask_2_subsub_file_mat_value)
#             try:
#                 if not ((mask_2_subsub_file_mat_value == np.array([0,2])).all()):
#                     mask_2_subsub_file_mat = np.where(mask_2_subsub_file_mat > 0, 2, mask_2_subsub_file_mat)
#                     print('!', end='')
#             except (AttributeError):
#                 print('-------------subsubfile_name', subsubfile_name)
#                 mask_2_subsub_file_mat = np.where(mask_2_subsub_file_mat > 0, 2, mask_2_subsub_file_mat)
#
#             merge_mask_mat = subsubfile_mat + mask_2_subsub_file_mat
#             merge_mask_mat = np.where(merge_mask_mat == 4, 0, merge_mask_mat)
#
#             mask_merge_subsub_file_path = mask_merge_subfolder_path + subsubfile_name
#             np.save(mask_merge_subsub_file_path, merge_mask_mat)
#
#         print(f'{subsubfile_name}', 'subsubfile_mat', subsubfile_mat_value, subsubfile_mat.shape, end='; ')
#         print('mask_2_subsub_file_mat', mask_2_subsub_file_mat_value, mask_2_subsub_file_mat.shape, end='; ')
#         print('merge_mask_mat', np.unique(merge_mask_mat), merge_mask_mat.shape)

if __name__ == '__main__':
    # mask_1 = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy/'
    # mask_2 = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy_LA_final/'
    # mask_merge = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/health/sequence_mask_npy_merge/'
    # merge_two_mask(mask_1, mask_2, mask_merge)

    # mask_1 = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy/'
    # mask_2 = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy_LA_final/'
    # mask_merge = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/health/sequence_mask_npy_merge/'
    # merge_two_mask(mask_1, mask_2, mask_merge)

    # mask_1 = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_npy/'
    # mask_2 = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_npy_LA_final/'
    # mask_merge = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/health/sequence_mask_npy_merge/'
    # merge_two_mask(mask_1, mask_2, mask_merge)

    mask_1 = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_npy/'
    mask_2 = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_npy_LA/'
    mask_merge = '/data/dwl/children_heart/children_heart_paper/model_adopt/four_cavity/unhealth/sequence_mask_npy_merge/'
    merge_two_mask_unhealth(mask_1, mask_2, mask_merge)

    # mask_1 = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_npy/'
    # mask_2 = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_npy_LA/'
    # mask_merge = '/data/dwl/children_heart/children_heart_paper/model_adopt/three_cavity/unhealth/sequence_mask_npy_merge/'
    # merge_two_mask_unhealth(mask_1, mask_2, mask_merge)

    # mask_1 = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_npy/'
    # mask_2 = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_npy_LA/'
    # mask_merge = '/data/dwl/children_heart/children_heart_paper/model_adopt/two_cavity/unhealth/sequence_mask_npy_merge/'
    # merge_two_mask_unhealth(mask_1, mask_2, mask_merge)

