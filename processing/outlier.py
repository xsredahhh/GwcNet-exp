import numpy as np
import cv2

def gt_outlier_remove(gt, mask):
    # gt是 h*w*3，numpy
    x_cor = gt[:, :, 0]
    y_cor = gt[:, :, 1]
    z_cor = gt[:, :, 2]
    z_mask = mask[:, :, 2]
    h, w = z_cor.shape
    gt_plus = np.zeros([h + 4, w + 4], dtype=np.float32)
    gt_plus[2:h + 2, 2:w + 2] = gt
    mask_plus = np.zeros([h + 4, w + 4], dtype=np.float32)
    mask_plus[2:h + 2, 2:w + 2] = z_mask
    mask_list = []
    gt_diff_list = []
    for dy in range(5):
        for dx in range(5):
            if dy == 2 and dx == 2:
                continue
            tmp_gt = gt_plus[dy: dy + h, dx: dx + w]
            tmp_mask = mask_plus[dy: dy + h, dx: dx + w] * z_mask
            mask_list.append(tmp_mask)
            gt_diff_list.append(np.abs(tmp_gt - gt) * tmp_mask)
    gt_diff_array = np.array(gt_diff_list)
    gt_diff_sum = np.sum(gt_diff_array, axis=0)
    mask_array = np.array(mask_list)
    mask_count = np.sum(mask_array, axis=0) + 0.001
    gt_diff_avg = gt_diff_sum / mask_count
    mask_final = (gt_diff_avg < 8).astype(np.float32) * z_mask

    # gt_up = gt_plus[2:h+2, 1:w+1]
    # gt_down = gt_plus[0:h, 1:w+1]
    # gt_left = gt_plus[1:h+1, 2:w+2]
    # gt_right = gt_plus[1:h+1, 0:w]
    #
    # mask_up = mask_plus[2:h + 2, 1:w + 1] * mask
    # mask_down = mask_plus[0:h, 1:w + 1] * mask
    # mask_left = mask_plus[1:h + 1, 2:w + 2] * mask
    # mask_right = mask_plus[1:h + 1, 0:w] * mask
    # diff_up = mask_up * np.abs(gt_up - gt)
    # diff_down = mask_down * np.abs(gt_down - gt)
    # diff_left = mask_left * np.abs(gt_left - gt)
    # diff_right = mask_right * np.abs(gt_right - gt)
    # diff_sum = diff_down + diff_left + diff_right + diff_up
    # mask_count = mask_up + mask_left + mask_right + mask_down + 0.0001
    # diff_avg = diff_sum / mask_count
    # mask_grad = (diff_avg < 10).astype(np.float32)
    gt_grad = gt * mask_final
