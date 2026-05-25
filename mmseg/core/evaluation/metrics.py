# Obtained from: https://github.com/open-mmlab/mmsegmentation/tree/v0.16.0

from collections import OrderedDict
from math import ceil, floor

import mmcv
import numpy as np
import torch
import cv2
from scipy.spatial.distance import directed_hausdorff



def f_score(precision, recall, beta=1):
    """calcuate the f-score value.

    Args:
        precision (float | torch.Tensor): The precision value.
        recall (float | torch.Tensor): The recall value.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.

    Returns:
        [torch.tensor]: The f-score value.
    """
    score = (1 + beta**2) * (precision * recall) / (
        (beta**2 * precision) + recall)
    return score


def intersect_and_union(pred_label,
                        label,
                        num_classes,
                        ignore_index,
                        label_map=dict(),
                        reduce_zero_label=False):
    """Calculate intersection and Union.

    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. The parameter will
            work only when label is str. Default: False.

     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """

    if isinstance(pred_label, str):
        pred_label = torch.from_numpy(np.load(pred_label))
    else:
        pred_label = torch.from_numpy((pred_label))

    if isinstance(label, str):
        label = torch.from_numpy(
            mmcv.imread(label, flag='unchanged', backend='pillow'))
    else:
        label = torch.from_numpy(label)

    if label_map is not None:
        for old_id, new_id in label_map.items():
            label[label == old_id] = new_id
    if reduce_zero_label:
        label[label == 0] = 255
        label = label - 1
        label[label == 254] = 255

    mask = (label != ignore_index)
    pred_label = pred_label[mask]
    label = label[mask]

    intersect = pred_label[pred_label == label]
    area_intersect = torch.histc(
        intersect.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_pred_label = torch.histc(
        pred_label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_label = torch.histc(
        label.float(), bins=(num_classes), min=0, max=num_classes - 1)
    area_union = area_pred_label + area_label - area_intersect
    return area_intersect, area_union, area_pred_label, area_label


def total_intersect_and_union(results,
                              gt_seg_maps,
                              num_classes,
                              ignore_index,
                              label_map=dict(),
                              reduce_zero_label=False):
    """Calculate Total Intersection and Union.

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
         ndarray: The intersection of prediction and ground truth histogram
             on all classes.
         ndarray: The union of prediction and ground truth histogram on all
             classes.
         ndarray: The prediction histogram on all classes.
         ndarray: The ground truth histogram on all classes.
    """
    num_imgs = len(results)
    assert len(gt_seg_maps) == num_imgs
    total_area_intersect = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_union = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_pred_label = torch.zeros((num_classes, ), dtype=torch.float64)
    total_area_label = torch.zeros((num_classes, ), dtype=torch.float64)
    for i in range(num_imgs):
        area_intersect, area_union, area_pred_label, area_label = \
            intersect_and_union(
                results[i], gt_seg_maps[i], num_classes, ignore_index,
                label_map, reduce_zero_label)
        total_area_intersect += area_intersect
        total_area_union += area_union
        total_area_pred_label += area_pred_label
        total_area_label += area_label
    return total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label


def mean_iou(results,
             gt_seg_maps,
             num_classes,
             ignore_index,
             nan_to_num=None,
             label_map=dict(),
             reduce_zero_label=False):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]:
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <IoU> ndarray: Per category IoU, shape (num_classes, ).
    """
    iou_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mIoU'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return iou_result


def mean_dice(results,
              gt_seg_maps,
              num_classes,
              ignore_index,
              nan_to_num=None,
              label_map=dict(),
              reduce_zero_label=False):
    """Calculate Mean Dice (mDice)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.

     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Acc> ndarray: Per category accuracy, shape (num_classes, ).
            <Dice> ndarray: Per category dice, shape (num_classes, ).
    """

    dice_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mDice'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return dice_result


def hausdorff_distance(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False):

    hd = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['HD'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return hd


def mean_absolute_distance(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False):

    mad = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['MAD'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label)
    return mad

def mean_fscore(results,
                gt_seg_maps,
                num_classes,
                ignore_index,
                nan_to_num=None,
                label_map=dict(),
                reduce_zero_label=False,
                beta=1):
    """Calculate Mean Intersection and Union (mIoU)

    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
        beta (int): Determines the weight of recall in the combined score.
            Default: False.


     Returns:
        dict[str, float | ndarray]: Default metrics.
            <aAcc> float: Overall accuracy on all images.
            <Fscore> ndarray: Per category recall, shape (num_classes, ).
            <Precision> ndarray: Per category precision, shape (num_classes, ).
            <Recall> ndarray: Per category f-score, shape (num_classes, ).
    """
    fscore_result = eval_metrics(
        results=results,
        gt_seg_maps=gt_seg_maps,
        num_classes=num_classes,
        ignore_index=ignore_index,
        metrics=['mFscore'],
        nan_to_num=nan_to_num,
        label_map=label_map,
        reduce_zero_label=reduce_zero_label,
        beta=beta)
    return fscore_result




def eval_metrics(results,
                 gt_seg_maps,
                 num_classes,
                 ignore_index,
                 metrics=['mIoU'],
                 nan_to_num=None,
                 label_map=dict(),
                 reduce_zero_label=False,
                 beta=1):
    """Calculate evaluation metrics
    Args:
        results (list[ndarray] | list[str]): List of prediction segmentation
            maps or list of prediction result filenames.
        gt_seg_maps (list[ndarray] | list[str]): list of ground truth
            segmentation maps or list of label filenames.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        metrics (list[str] | str): Metrics to be evaluated, 'mIoU' and 'mDice'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        label_map (dict): Mapping old labels to new labels. Default: dict().
        reduce_zero_label (bool): Wether ignore zero label. Default: False.
     Returns:
        float: Overall accuracy on all images.
        ndarray: Per category accuracy, shape (num_classes, ).
        ndarray: Per category evaluation metrics, shape (num_classes, ).
    """
    if isinstance(metrics, str):
        metrics = [metrics]
    allowed_metrics = ['mIoU', 'mDice', 'mFscore', 'HD', 'MAD']
    if not set(metrics).issubset(set(allowed_metrics)):
        raise KeyError('metrics {} is not supported'.format(metrics))

    total_area_intersect, total_area_union, total_area_pred_label, \
        total_area_label = total_intersect_and_union(
            results, gt_seg_maps, num_classes, ignore_index, label_map,
            reduce_zero_label)
    all_acc = total_area_intersect.sum() / total_area_label.sum()
    ret_metrics = OrderedDict({'aAcc': all_acc})
    '''
    HD, MAD = [0, 0], [0, 0]
    iters = 0
    error = 0
    for i in range(len(results) - 1):
        target_1 = np.uint8(results[i] == 1)
        target_2 = np.uint8(results[i] == 2)
        gt_1 = np.uint8(gt_seg_maps[i] == 1)
        gt_2 = np.uint8(gt_seg_maps[i] == 2)
        # _, axe = plt.subplots(2,1)
        # axe[0].imshow(target_1)
        # axe[1].imshow(gt_1)
        # plt.show()

        contour_I1, _ = cv2.findContours(target_1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contour_I2, _ = cv2.findContours(gt_1, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contour_J1, _ = cv2.findContours(target_2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        contour_J2, _ = cv2.findContours(gt_2, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        if contour_I1[0] == [] or contour_J1[0] == []:
            print('no contour')
            continue
        if len(contour_I1) > 1 or len(contour_I2) > 1 or len(contour_J1) > 1 or len(contour_J2) > 1:

            if len(contour_I1) > 1:
                temp = [0]
                length = []
                for i in range(len(contour_I1)):
                    length.append(len(contour_I1[i]))
                a = list.index(length, max(length))
                temp[0] = contour_I1[a]
                contour_I1 = temp
                error += 1
                # print(len(temp[0]))
            if len(contour_I2) > 1:
                length = []
                temp = [[]]
                for i in range(len(contour_I2)):
                    length.append(len(contour_I2[i]))
                a = list.index(length, max(length))
                temp[0] = contour_I2[a]
                contour_I2 = temp
                # print(len(temp[0]))
            if len(contour_J1) > 1:
                length = []
                temp = [[]]
                for i in range(len(contour_J1)):
                    length.append(len(contour_J1[i]))
                a = list.index(length, max(length))
                temp[0] = contour_J1[a]
                contour_J1 = temp
                error += 1
                # print(len(temp[0]))
            if len(contour_J2) > 1:
                length = []
                temp = [[]]
                for i in range(len(contour_J2)):
                    length.append(len(contour_J2[i]))
                a = list.index(length, max(length))
                temp[0] = contour_J2[a]
                contour_J2 = temp
            # continue
        # HD_1 = D.directed_hausdorff(contour_I1[0].squeeze(), contour_I2[0].squeeze())
        point_I1 = contour_I1[0].squeeze()
        point_I2 = contour_I2[0].squeeze()
        point_J1 = contour_J1[0].squeeze()
        point_J2 = contour_J2[0].squeeze()

        hd_target_to_gt_1 = directed_hausdorff(point_I1, point_J1)[0]
        hd_gt_to_target_1 = directed_hausdorff(point_J1, point_I1)[0]
        hd_target_to_gt_2 = directed_hausdorff(point_I2, point_J2)[0]
        hd_gt_to_target_2 = directed_hausdorff(point_J2, point_I2)[0]

        HD_1 = max(hd_target_to_gt_1, hd_gt_to_target_1)
        HD_2 = max(hd_target_to_gt_2, hd_gt_to_target_2)

        distance_matrix_1 = torch.cdist(torch.Tensor(point_I1), torch.Tensor(point_J1), p=2)
        distance_matrix_2 = torch.cdist(torch.Tensor(point_I2), torch.Tensor(point_J2), p=2)

        MAD_1 = (distance_matrix_1.min(1)[0].mean() + distance_matrix_1.min(0)[0].mean()) / 2
        MAD_2 = (distance_matrix_2.min(1)[0].mean() + distance_matrix_2.min(0)[0].mean()) / 2

        HD[0] += HD_1
        HD[1] += HD_2
        MAD[0] += MAD_1
        MAD[1] += MAD_2
        iters += 1
    Hausdorff_distance = torch.tensor([0, HD[0] / iters, HD[1] / iters])
    Mean_absolute_distance = torch.tensor([0, MAD[0] / iters, MAD[1] / iters])
    if error != 0:
        print('\nError = {}'.format(error))
    '''

    for metric in metrics:
        if metric == 'mIoU':
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['IoU'] = iou
            ret_metrics['Acc'] = acc
        elif metric == 'mDice':
            dice = 2 * total_area_intersect / (
                total_area_pred_label + total_area_label)
            iou = total_area_intersect / total_area_union
            acc = total_area_intersect / total_area_label
            ret_metrics['Dice'] = dice
            ret_metrics['Acc'] = acc
            ret_metrics['IoU'] = iou
        elif metric == 'mFscore':
            precision = total_area_intersect / total_area_pred_label
            recall = total_area_intersect / total_area_label
            f_value = torch.tensor(
                [f_score(x[0], x[1], beta) for x in zip(precision, recall)])
            ret_metrics['Fscore'] = f_value
            ret_metrics['Precision'] = precision
            ret_metrics['Recall'] = recall
        # elif metric == 'HD':
        #     dice = 2 * total_area_intersect / (
        #             total_area_pred_label + total_area_label)
        #     iou = total_area_intersect / total_area_union
        #     acc = total_area_intersect / total_area_label
        #     ret_metrics['Dice'] = dice
        #     ret_metrics['Acc'] = acc
        #     ret_metrics['IoU'] = iou
        #     ret_metrics['HD'] = Hausdorff_distance
        #     ret_metrics['MAD'] = Mean_absolute_distance


    ret_metrics = {
        metric: value.numpy()
        for metric, value in ret_metrics.items()
    }
    if nan_to_num is not None:
        ret_metrics = OrderedDict({
            metric: np.nan_to_num(metric_value, nan=nan_to_num)
            for metric, metric_value in ret_metrics.items()
        })
    return ret_metrics

# def torch2D_Hausdorff_distance(x, y):  # Input be like (Batch,width,height)
#     x = torch.Tensor(x[0].squeeze())
#     y = torch.Tensor(y[0].squeeze())
#     x = x.float()
#     y = y.float()
#     distance_matrix = torch.cdist(x, y, p=2)  # p=2 means Euclidean Distance
#     a = distance_matrix.min(1)[0]
#     num_1 = floor(len(torch.unique(distance_matrix.min(0)[0])) * 0.8)
#     num_2 = floor(len(torch.unique(distance_matrix.min(1)[0])) * 0.8)
#     v = sorted(torch.unique(distance_matrix.min(0)[0]))
#     t = sorted(torch.unique(distance_matrix.min(1)[0]))
#     distance_x = sorted(torch.unique(distance_matrix.min(0)[0]))[num_1]
#     distance_y = sorted(torch.unique(distance_matrix.min(1)[0]))[num_2]
#     value1 = distance_matrix.min(1)[0].max(0, keepdim=True)[0]
#     value2 = distance_matrix.min(0)[0].max(0, keepdim=True)[0]
#
#     value3 = torch.mean(distance_matrix.min(1)[0])
#     value4 = torch.mean(distance_matrix.min(0)[0])
#
#     HDvalue = max(distance_x, distance_y)
#     # HDvalue = torch.cat((value1, value2), dim=0)
#     MADvalue = max(value3, value4)
#
#     # return HDvalue.max(0)[0], MADvalue
#     return HDvalue, MADvalue